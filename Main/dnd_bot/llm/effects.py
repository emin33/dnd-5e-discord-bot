"""Proposed Effects Schema - Constrained effect types for narrator output.

The narrator can propose effects, but only the orchestrator can execute them.
This ensures:
1. Creative output doesn't have unilateral write access to state
2. All effects are validated before application
3. Idempotency is enforced at the effect level

Effect Categories:
- Scene: spawn_object, reveal_object, add_npc, start_combat
- Transfer: transfer_item, grant_currency, consume_resource
- Combat: apply_damage, apply_healing, add_condition, remove_condition
- Meta: set_flag, log_memory
"""

from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field


class EffectType(str, Enum):
    """Constrained set of effect types the narrator can propose."""

    # Scene effects - modify world state
    SPAWN_OBJECT = "spawn_object"           # Add object to scene (loot, item in world)
    REVEAL_OBJECT = "reveal_object"         # Make hidden object visible
    ADD_NPC = "add_npc"                     # Introduce an NPC to the scene
    REMOVE_ENTITY = "remove_entity"         # Remove entity from scene
    START_COMBAT = "start_combat"           # Initiate combat with entities

    # Transfer effects - move items/currency between entities
    TRANSFER_ITEM = "transfer_item"         # Move item: scene→player, npc→player, etc.
    GRANT_CURRENCY = "grant_currency"       # Give gold/currency
    CONSUME_RESOURCE = "consume_resource"   # Use up ammunition, rations, etc.

    # Combat effects - HP and conditions
    APPLY_DAMAGE = "apply_damage"           # Deal damage to target
    APPLY_HEALING = "apply_healing"         # Heal target
    ADD_CONDITION = "add_condition"         # Apply condition (poisoned, prone, etc.)
    REMOVE_CONDITION = "remove_condition"   # Remove condition

    # DM-initiated mechanics
    REQUEST_ROLL = "request_roll"           # DM requests a roll from the player

    # Entity tracking
    REF_ENTITY = "ref_entity"              # Narrator referenced an existing roster entity

    # Meta effects - game state tracking
    SET_FLAG = "set_flag"                   # Quest progress, discovered facts
    LOG_MEMORY = "log_memory"               # Add fact to memory system


class ProposedEffect(BaseModel):
    """A single proposed effect from the narrator.

    The narrator proposes effects; the orchestrator validates and executes.
    """

    effect_type: EffectType

    # Common fields
    target: Optional[str] = None            # Target entity ID or name (e.g., "player:alice", "npc:merchant")
    source: Optional[str] = None            # Source entity (for transfers, damage sources)

    # For spawn_object / reveal_object
    object_name: Optional[str] = None       # Name of the object
    object_description: Optional[str] = None
    object_properties: Optional[dict] = None  # item_index, value, magical, etc.

    # For add_npc
    npc_name: Optional[str] = None
    npc_description: Optional[str] = None
    npc_disposition: Optional[str] = None   # friendly, neutral, hostile
    monster_index: Optional[str] = None     # SRD monster index if applicable

    # For transfer_item
    item_name: Optional[str] = None
    item_index: Optional[str] = None
    quantity: int = 1
    from_entity: Optional[str] = None       # "scene", "npc:merchant", "player:bob"
    to_entity: Optional[str] = None         # "player:alice", "scene", etc.

    # For grant_currency
    copper: int = 0
    silver: int = 0
    electrum: int = 0
    gold: int = 0
    platinum: int = 0

    # For consume_resource
    resource_name: Optional[str] = None     # "Arrow", "Ration", etc.

    # For damage/healing
    amount: Optional[int] = None
    damage_type: Optional[str] = None       # slashing, fire, etc.

    # For conditions
    condition: Optional[str] = None         # poisoned, prone, etc.
    duration_rounds: Optional[int] = None

    # For set_flag / log_memory
    flag_name: Optional[str] = None
    flag_value: Optional[Union[str, int, bool]] = None
    memory_text: Optional[str] = None

    # For request_roll - DM-initiated uncertainty
    roll_type: Optional[str] = None         # "ability_check", "saving_throw", "skill_check"
    ability: Optional[str] = None           # "dexterity", "wisdom", etc.
    skill: Optional[str] = None             # "perception", "stealth", etc.
    dc: Optional[int] = None                # Difficulty class
    roll_reason: Optional[str] = None       # "to notice the hidden trap", "to resist the poison"

    # For ref_entity — narrator declares which roster entity it referenced
    ref_entity_id: Optional[str] = None     # Slugified ID from roster (e.g. "tavern-keeper")
    ref_alias_used: Optional[str] = None    # Alias used in prose if different from canonical name

    # Confirmation semantics
    requires_confirmation: bool = False     # If True, player must accept/decline
    confirmation_prompt: Optional[str] = None  # "Accept the merchant's gift?"

    # Reason for the effect (for logging/debugging)
    reason: Optional[str] = None


class NarratorOutput(BaseModel):
    """Structured output from the narrator.

    Contains both the narrative prose and any proposed mechanical effects.
    """

    narrative: str = Field(..., description="The narrative prose to display to players")
    proposed_effects: list[ProposedEffect] = Field(
        default_factory=list,
        description="Mechanical effects the narrator is proposing"
    )

    # For pending mechanics - narrator knows a roll is needed but hasn't happened yet
    awaiting_resolution: bool = False
    resolution_type: Optional[str] = None   # "skill_check", "attack_roll", "saving_throw"


class EffectValidationResult(BaseModel):
    """Result of validating a proposed effect."""

    effect: ProposedEffect
    valid: bool
    rejection_reason: Optional[str] = None

    # If valid, the effect may be modified (e.g., canonical IDs assigned)
    modified_effect: Optional[ProposedEffect] = None


class EffectExecutionResult(BaseModel):
    """Result of executing a validated effect."""

    effect: ProposedEffect
    success: bool
    error: Optional[str] = None

    # Details of what happened
    details: dict = Field(default_factory=dict)

    # Idempotency
    idempotency_key: Optional[str] = None
    was_duplicate: bool = False


# JSON schema for narrator structured output
def get_narrator_output_schema() -> dict:
    """Get JSON schema for narrator structured output."""
    return NarratorOutput.model_json_schema()


# Helper to build idempotency key
def build_effect_idempotency_key(
    campaign_id: str,
    message_id: str,
    effect_index: int,
) -> str:
    """Build idempotency key for an effect.

    Format: campaign_id:message_id:effect_index
    """
    return f"{campaign_id}:{message_id}:{effect_index}"


class EffectValidator:
    """Validates proposed effects before execution.

    Checks:
    - Plausibility (does the source/target exist?)
    - Legality (can this happen now?)
    - Conflicts (does this contradict other effects?)
    """

    def __init__(self, scene_registry=None, session=None):
        self.scene_registry = scene_registry
        self.session = session

    def validate(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate a single proposed effect."""
        # Type-specific validation
        validators = {
            EffectType.SPAWN_OBJECT: self._validate_spawn_object,
            EffectType.ADD_NPC: self._validate_add_npc,
            EffectType.TRANSFER_ITEM: self._validate_transfer_item,
            EffectType.GRANT_CURRENCY: self._validate_grant_currency,
            EffectType.APPLY_DAMAGE: self._validate_apply_damage,
            EffectType.APPLY_HEALING: self._validate_apply_healing,
            EffectType.ADD_CONDITION: self._validate_add_condition,
            EffectType.START_COMBAT: self._validate_start_combat,
            EffectType.REQUEST_ROLL: self._validate_request_roll,
        }

        validator = validators.get(effect.effect_type, self._validate_default)
        return validator(effect)

    def _validate_spawn_object(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate spawn_object effect."""
        if not effect.object_name:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="spawn_object requires object_name",
            )
        return EffectValidationResult(effect=effect, valid=True)

    def _validate_add_npc(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate add_npc effect."""
        if not effect.npc_name:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="add_npc requires npc_name",
            )
        return EffectValidationResult(effect=effect, valid=True)

    def _validate_transfer_item(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate transfer_item effect.

        Validates that:
        1. item_name is specified
        2. to_entity is specified
        3. from_entity references an existing scene object or known NPC
        """
        if not effect.item_name:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="transfer_item requires item_name",
            )
        if not effect.to_entity:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="transfer_item requires to_entity",
            )

        # Validate source exists (if scene registry available)
        from_entity = effect.from_entity or ""
        if self.scene_registry and from_entity:
            # "scene" means transfer from scene object
            if from_entity == "scene":
                # Check if the item exists as a scene object
                scene_object = self.scene_registry.get_by_name(effect.item_name)
                if not scene_object:
                    return EffectValidationResult(
                        effect=effect,
                        valid=False,
                        rejection_reason=f"Scene object '{effect.item_name}' does not exist. Must spawn_object first.",
                    )
            elif from_entity.startswith("npc:"):
                # Check if NPC exists in scene
                npc_name = from_entity.split(":", 1)[1]
                npc = self.scene_registry.get_by_name(npc_name)
                if not npc:
                    # NPC might be implied by context, allow with warning
                    import structlog
                    logger = structlog.get_logger()
                    logger.warning(
                        "transfer_from_unknown_npc",
                        npc_name=npc_name,
                        item=effect.item_name,
                    )
                    # Still allow - NPC may have been established narratively

        return EffectValidationResult(effect=effect, valid=True)

    def _validate_grant_currency(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate grant_currency effect."""
        total = effect.copper + effect.silver + effect.electrum + effect.gold + effect.platinum
        if total <= 0:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="grant_currency requires positive currency amount",
            )
        return EffectValidationResult(effect=effect, valid=True)

    def _validate_apply_damage(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate apply_damage effect."""
        if not effect.amount or effect.amount <= 0:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="apply_damage requires positive amount",
            )
        if not effect.target:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="apply_damage requires target",
            )
        return EffectValidationResult(effect=effect, valid=True)

    def _validate_apply_healing(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate apply_healing effect."""
        if not effect.amount or effect.amount <= 0:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="apply_healing requires positive amount",
            )
        return EffectValidationResult(effect=effect, valid=True)

    def _validate_add_condition(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate add_condition effect."""
        if not effect.condition:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="add_condition requires condition",
            )
        if not effect.target:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="add_condition requires target",
            )
        return EffectValidationResult(effect=effect, valid=True)

    def _validate_start_combat(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate start_combat effect."""
        # Check if already in combat
        if self.session and hasattr(self.session, 'state'):
            from ..game.session import SessionState
            if self.session.state == SessionState.COMBAT:
                return EffectValidationResult(
                    effect=effect,
                    valid=False,
                    rejection_reason="Already in combat",
                )
        return EffectValidationResult(effect=effect, valid=True)

    def _validate_request_roll(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate request_roll effect.

        Validates that:
        1. roll_type is specified and valid
        2. ability is specified for checks/saves
        3. DC is reasonable (1-40)
        """
        valid_roll_types = {"ability_check", "saving_throw", "skill_check"}
        if not effect.roll_type or effect.roll_type not in valid_roll_types:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=f"request_roll requires roll_type in {valid_roll_types}",
            )

        # Skill checks need either skill or ability
        if effect.roll_type == "skill_check" and not effect.skill and not effect.ability:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="skill_check requires skill or ability",
            )

        # Saving throws and ability checks need ability
        if effect.roll_type in ("saving_throw", "ability_check") and not effect.ability:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=f"{effect.roll_type} requires ability",
            )

        # Validate DC range
        if effect.dc is not None and (effect.dc < 1 or effect.dc > 40):
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=f"DC {effect.dc} out of valid range (1-40)",
            )

        return EffectValidationResult(effect=effect, valid=True)

    def _validate_default(self, effect: ProposedEffect) -> EffectValidationResult:
        """Default validation - accept the effect."""
        return EffectValidationResult(effect=effect, valid=True)


class EffectExecutor:
    """Executes validated effects against game state.

    Uses existing tool functions to apply effects, ensuring
    single source of truth for state mutations.
    """

    def __init__(
        self,
        scene_registry=None,
        session=None,
        inventory_repo=None,
        applied_effects_store=None,  # For idempotency
    ):
        self.scene_registry = scene_registry
        self.session = session
        self.inventory_repo = inventory_repo
        self.applied_effects = applied_effects_store or set()

    async def execute(
        self,
        effect: ProposedEffect,
        idempotency_key: Optional[str] = None,
    ) -> EffectExecutionResult:
        """Execute a single validated effect."""
        import structlog
        logger = structlog.get_logger()

        # Check idempotency
        if idempotency_key and idempotency_key in self.applied_effects:
            return EffectExecutionResult(
                effect=effect,
                success=True,
                was_duplicate=True,
                idempotency_key=idempotency_key,
                details={"message": "Effect already applied"},
            )

        # Execute based on type
        executors = {
            EffectType.SPAWN_OBJECT: self._execute_spawn_object,
            EffectType.ADD_NPC: self._execute_add_npc,
            EffectType.TRANSFER_ITEM: self._execute_transfer_item,
            EffectType.GRANT_CURRENCY: self._execute_grant_currency,
            EffectType.APPLY_DAMAGE: self._execute_apply_damage,
            EffectType.APPLY_HEALING: self._execute_apply_healing,
            EffectType.ADD_CONDITION: self._execute_add_condition,
            EffectType.REMOVE_CONDITION: self._execute_remove_condition,
            EffectType.START_COMBAT: self._execute_start_combat,
            EffectType.SET_FLAG: self._execute_set_flag,
            EffectType.LOG_MEMORY: self._execute_log_memory,
            EffectType.CONSUME_RESOURCE: self._execute_consume_resource,
            EffectType.REQUEST_ROLL: self._execute_request_roll,
            EffectType.REF_ENTITY: self._execute_ref_entity,
        }

        executor = executors.get(effect.effect_type)
        if not executor:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error=f"No executor for effect type: {effect.effect_type}",
            )

        try:
            result = await executor(effect)

            # Record for idempotency
            if idempotency_key and result.success:
                self.applied_effects.add(idempotency_key)
                result.idempotency_key = idempotency_key

            logger.info(
                "effect_executed",
                effect_type=effect.effect_type.value,
                success=result.success,
                idempotency_key=idempotency_key,
            )

            return result

        except Exception as e:
            logger.error(
                "effect_execution_failed",
                effect_type=effect.effect_type.value,
                error=str(e),
            )
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error=str(e),
            )

    async def _execute_spawn_object(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Spawn an object in the scene registry."""
        if not self.scene_registry:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="No scene registry available",
            )

        from ..models.npc import SceneEntity

        entity = SceneEntity(
            name=effect.object_name,
            description=effect.object_description or "",
            entity_type="object",
            disposition="neutral",
            properties=effect.object_properties or {},
        )
        self.scene_registry.register_entity(entity)

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"entity_id": entity.id, "object_name": effect.object_name},
        )

    async def _execute_add_npc(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Add an NPC to the scene registry."""
        if not self.scene_registry:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="No scene registry available",
            )

        from ..models.npc import SceneEntity

        entity = SceneEntity(
            name=effect.npc_name,
            description=effect.npc_description or "",
            entity_type="npc",
            disposition=effect.npc_disposition or "neutral",
            monster_index=effect.monster_index,
        )
        self.scene_registry.register_entity(entity)

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"entity_id": entity.id, "npc_name": effect.npc_name},
        )

    async def _execute_transfer_item(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Transfer an item between entities."""
        if not self.inventory_repo:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="No inventory repository available",
            )

        # Parse target entity
        to_entity = effect.to_entity or ""
        if to_entity.startswith("player"):
            # Get player character ID
            char_id = await self._resolve_player_character_id(to_entity)
            if not char_id:
                return EffectExecutionResult(
                    effect=effect,
                    success=False,
                    error=f"Could not resolve player: {to_entity}",
                )

            from ..models import InventoryItem
            item = InventoryItem(
                character_id=char_id,
                item_index=effect.item_index or effect.item_name.lower().replace(" ", "-"),
                item_name=effect.item_name,
                quantity=effect.quantity,
            )
            await self.inventory_repo.add_item(item)

            return EffectExecutionResult(
                effect=effect,
                success=True,
                details={"item": effect.item_name, "quantity": effect.quantity, "to": to_entity},
            )

        return EffectExecutionResult(
            effect=effect,
            success=False,
            error=f"Unsupported target entity type: {to_entity}",
        )

    async def _execute_grant_currency(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Grant currency to a player."""
        if not self.inventory_repo:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="No inventory repository available",
            )

        char_id = await self._resolve_player_character_id(effect.target or "player")
        if not char_id:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="Could not resolve target player",
            )

        # Convert all to gold for now (can enhance later)
        total_gp = effect.gold + (effect.platinum * 10) + (effect.electrum * 0.5) + (effect.silver * 0.1) + (effect.copper * 0.01)
        await self.inventory_repo.add_gold(char_id, int(total_gp))

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"gold": effect.gold, "target": effect.target},
        )

    async def _execute_apply_damage(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Apply damage to a target."""
        # This would integrate with combat system
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"amount": effect.amount, "type": effect.damage_type, "target": effect.target},
        )

    async def _execute_apply_healing(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Apply healing to a target."""
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"amount": effect.amount, "target": effect.target},
        )

    async def _execute_add_condition(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Add a condition to a target."""
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"condition": effect.condition, "target": effect.target},
        )

    async def _execute_remove_condition(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Remove a condition from a target."""
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"condition": effect.condition, "target": effect.target},
        )

    async def _execute_start_combat(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Start combat - signals orchestrator to trigger combat."""
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"reason": effect.reason, "triggers_combat": True},
        )

    async def _execute_set_flag(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Set a game flag."""
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"flag": effect.flag_name, "value": effect.flag_value},
        )

    async def _execute_log_memory(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Log a memory fact."""
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"memory": effect.memory_text},
        )

    async def _execute_consume_resource(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Consume a resource (ammunition, etc.)."""
        if not self.inventory_repo:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="No inventory repository available",
            )

        char_id = await self._resolve_player_character_id(effect.target or "player")
        if not char_id:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="Could not resolve target player",
            )

        # Find and consume the resource
        items = await self.inventory_repo.get_all_items(char_id)
        for item in items:
            if effect.resource_name and effect.resource_name.lower() in item.item_name.lower():
                if item.quantity >= effect.quantity:
                    await self.inventory_repo.remove_item(item.id, effect.quantity)
                    return EffectExecutionResult(
                        effect=effect,
                        success=True,
                        details={"resource": effect.resource_name, "consumed": effect.quantity},
                    )

        return EffectExecutionResult(
            effect=effect,
            success=False,
            error=f"Resource not found or insufficient: {effect.resource_name}",
        )

    async def _execute_request_roll(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Request a roll from the player.

        This effect signals the orchestrator that the narrator wants the player
        to make a roll. The orchestrator will handle the actual roll resolution.
        """
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={
                "roll_type": effect.roll_type,
                "ability": effect.ability,
                "skill": effect.skill,
                "dc": effect.dc,
                "reason": effect.roll_reason,
                "triggers_roll": True,  # Signal to orchestrator
            },
        )

    async def _execute_ref_entity(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Record that the narrator referenced an existing roster entity.

        Lightweight — updates mention tracking on the SceneEntity.
        If the narrator used an alias, records it for future name promotion.
        """
        entity_id = effect.ref_entity_id
        if not entity_id:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="ref_entity missing entity_id",
            )

        # Try to find entity in scene registry by slug ID
        entity = None
        if self.scene_registry:
            entity = self.scene_registry.get_by_name(entity_id)

        if entity:
            # Update mention tracking
            from datetime import datetime
            entity.mention_count = getattr(entity, 'mention_count', 0) + 1
            entity.last_mentioned_at = datetime.utcnow()

            # Record alias if provided and different from canonical
            alias = effect.ref_alias_used
            if alias and alias.lower() != entity.name.lower():
                if not hasattr(entity, 'aliases') or entity.aliases is None:
                    entity.aliases = []
                if alias not in entity.aliases:
                    entity.aliases.append(alias)

            logger.info(
                "entity_referenced",
                entity_id=entity_id,
                entity_name=entity.name,
                alias_used=alias,
            )
        else:
            logger.debug(
                "ref_entity_not_in_scene",
                entity_id=entity_id,
            )

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={
                "entity_id": entity_id,
                "alias_used": effect.ref_alias_used,
                "found_in_scene": entity is not None,
            },
        )

    async def _resolve_player_character_id(self, entity_ref: str) -> Optional[str]:
        """Resolve a player entity reference to character ID.

        Supports canonical formats:
        - "pc:<character_name>" (e.g., "pc:Thorin")
        - "player:<character_name>" (legacy)
        - "player" (shorthand for current/first player)
        """
        if not self.session:
            return None

        # Handle "pc:name" or "player:name" format
        if ":" in entity_ref:
            prefix, name = entity_ref.split(":", 1)
            if prefix.lower() not in ("pc", "player"):
                # Not a player reference
                return None
        else:
            name = entity_ref

        # Find by name or get first player
        for player in self.session.players.values():
            if player.character:
                if name.lower() in ("player", "current") or name.lower() in player.character.name.lower():
                    return player.character.id

        # Fallback: first player (for "player" shorthand)
        if name.lower() in ("player", "current"):
            for player in self.session.players.values():
                if player.character:
                    return player.character.id

        return None

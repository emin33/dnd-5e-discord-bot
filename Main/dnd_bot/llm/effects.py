"""Proposed Effects Schema - Constrained effect types for narrator output.

The narrator can propose effects, but only the orchestrator can execute them.
This ensures:
1. Creative output doesn't have unilateral write access to state
2. All effects are validated before application
3. Idempotency is enforced at the effect level

Effect Categories:
- Scene: spawn_object, add_npc, remove_entity, start_combat, change_location
- Transfer: transfer_item, grant_currency, consume_resource
- Damage (legacy INTENTS fallback only): apply_damage
- Entity/player tracking: ref_entity, update_entity, update_player
- Meta: set_flag, request_roll

The narrator tool surface (schemas, tiers, converters) lives in
``dnd_bot.llm.tool_registry`` — one declarative entry per tool.
"""

from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field


class EffectType(str, Enum):
    """Constrained set of effect types the narrator can propose."""

    # Scene effects - modify world state
    SPAWN_OBJECT = "spawn_object"           # Add object to scene (loot, item in world)
    ADD_NPC = "add_npc"                     # Introduce an NPC to the scene
    REMOVE_ENTITY = "remove_entity"         # Remove entity from scene
    START_COMBAT = "start_combat"           # Initiate combat with entities

    # Transfer effects - move items/currency between entities
    # (legacy INTENTS fallback producers only; the tool path uses update_player)
    TRANSFER_ITEM = "transfer_item"         # Move item: scene→player, npc→player, etc.
    GRANT_CURRENCY = "grant_currency"       # Give gold/currency
    CONSUME_RESOURCE = "consume_resource"   # Use up ammunition, rations, etc.

    # Damage (legacy INTENTS fallback producer only; executor is honestly
    # unimplemented — player damage flows through UPDATE_PLAYER, combat
    # damage through the combat engine)
    APPLY_DAMAGE = "apply_damage"           # Deal damage to target

    # DM-initiated mechanics
    REQUEST_ROLL = "request_roll"           # DM requests a roll from the player

    # Entity tracking
    REF_ENTITY = "ref_entity"              # Narrator referenced an existing roster entity
    UPDATE_ENTITY = "update_entity"        # Narrator changed something about an existing entity

    # Player state mutations (consolidated tool — replaces apply_damage and the player-side of transfer_item/grant_currency)
    UPDATE_PLAYER = "update_player"        # Narrator changed something about the player(s)

    # Scene navigation
    CHANGE_LOCATION = "change_location"    # Party moved to a new named location

    # Meta effects - game state tracking
    SET_FLAG = "set_flag"                   # Quest progress, discovered facts


class ProposedEffect(BaseModel):
    """A single proposed effect from the narrator.

    The narrator proposes effects; the orchestrator validates and executes.
    """

    effect_type: EffectType

    # Common fields
    target: Optional[str] = None            # Target entity ID or name (e.g., "player:alice", "npc:merchant")
    source: Optional[str] = None            # Source entity (for transfers, damage sources)

    # For spawn_object
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

    # For set_flag
    flag_name: Optional[str] = None
    flag_value: Optional[Union[str, int, bool]] = None
    memory_text: Optional[str] = None       # unused since LOG_MEMORY was deleted (ProposedEffect slim-down is a follow-up)

    # For request_roll - DM-initiated uncertainty
    roll_type: Optional[str] = None         # "ability_check", "saving_throw", "skill_check"
    ability: Optional[str] = None           # "dexterity", "wisdom", etc.
    skill: Optional[str] = None             # "perception", "stealth", etc.
    dc: Optional[int] = None                # Difficulty class
    roll_reason: Optional[str] = None       # "to notice the hidden trap", "to resist the poison"

    # For ref_entity — narrator declares which roster entity it referenced
    ref_entity_id: Optional[str] = None     # Slugified ID from roster (e.g. "tavern-keeper")
    ref_alias_used: Optional[str] = None    # Alias used in prose if different from canonical name
    dialogue_indices: list[int] = Field(default_factory=list)  # Which quotes this entity speaks (1-indexed)
    dialogue_emotions: list[str] = Field(default_factory=list)  # Emotion per dialogue line (same order)

    # For update_entity — narrator declares a meaningful change to an existing roster entity.
    # All update_* fields are optional. The narrator emits ONLY the fields that
    # actually changed in the just-narrated turn. At least one update_* field
    # must be set or the effect is rejected as a no-op.
    update_entity_id: Optional[str] = None       # Required: which entity changed
    update_importance: Optional[bool] = None     # Promote/demote importance (None = unchanged)
    update_disposition: Optional[str] = None     # New disposition (None = unchanged)
    update_status: Optional[str] = None          # alive | wounded | unconscious | dead | fled | captured (None = unchanged)
    update_description_addition: Optional[str] = None  # Short clause appended to description; None = unchanged
    update_add_items: list[str] = Field(default_factory=list)    # Items the entity now holds (added to NPC.inventory)
    update_remove_items: list[str] = Field(default_factory=list) # Items the entity gave away / lost / used

    # For update_player — narrator declares a change to the player's state.
    # All player_* fields are optional; at least one mutation must be set
    # or the effect is rejected as a no-op. Use list[dict] for items so the
    # narrator can pass {name, quantity, source/destination} per item.
    player_item_grant: list[dict] = Field(default_factory=list)   # [{"name", "quantity", "source"}]
    player_item_remove: list[dict] = Field(default_factory=list)  # [{"name", "quantity", "destination"}]
    player_currency_delta: dict = Field(default_factory=dict)     # {"gp": 50, "sp": -10}, etc.
    player_hp_delta: Optional[int] = None                          # Negative = damage, positive = heal
    player_hp_reason: Optional[str] = None                         # "wall trap dart", "potion of healing"
    player_damage_type: Optional[str] = None                       # When hp_delta < 0: fire, poison, slashing, etc.
    player_add_conditions: list[str] = Field(default_factory=list) # ["poisoned", "prone"]
    player_remove_conditions: list[str] = Field(default_factory=list)
    player_spell_slot_used: Optional[int] = None                   # Slot level consumed (1-9)

    # For change_location — narrator declares the party moved to a new named area.
    location_name: Optional[str] = None          # Short canonical name (2-4 words preferred)
    location_description: Optional[str] = None   # Brief sentence describing the new location

    # Confirmation semantics
    requires_confirmation: bool = False     # If True, player must accept/decline
    confirmation_prompt: Optional[str] = None  # "Accept the merchant's gift?"

    # Reason for the effect (for logging/debugging)
    reason: Optional[str] = None


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
            EffectType.START_COMBAT: self._validate_start_combat,
            EffectType.REQUEST_ROLL: self._validate_request_roll,
            EffectType.UPDATE_ENTITY: self._validate_update_entity,
            EffectType.UPDATE_PLAYER: self._validate_update_player,
            EffectType.CHANGE_LOCATION: self._validate_change_location,
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

    def _validate_update_entity(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate update_entity effect.

        Rejects no-op calls — the narrator must include at least one
        change field beyond entity_id. Tool-error feedback shapes the
        model's next call (see prompt-engineering research).
        """
        if not effect.update_entity_id:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="update_entity requires entity_id",
            )

        # At least one change field must be present
        has_change = any([
            effect.update_importance is not None,
            effect.update_disposition is not None,
            effect.update_status is not None,
            effect.update_description_addition,
            effect.update_add_items,
            effect.update_remove_items,
        ])
        if not has_change:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=(
                    "update_entity called with no change fields. Pass at least "
                    "one of: importance, disposition, status, description_addition, "
                    "add_items, remove_items. Do not call update_entity to merely "
                    "reference an existing entity — use ref_entity for that."
                ),
            )

        # Validate disposition enum
        if effect.update_disposition is not None:
            valid_dispositions = {"friendly", "neutral", "unfriendly", "hostile", "allied"}
            if effect.update_disposition.lower() not in valid_dispositions:
                return EffectValidationResult(
                    effect=effect,
                    valid=False,
                    rejection_reason=(
                        f"update_entity disposition '{effect.update_disposition}' "
                        f"not in {valid_dispositions}"
                    ),
                )

        # Validate status enum
        if effect.update_status is not None:
            valid_status = {"alive", "wounded", "unconscious", "dead", "fled", "captured"}
            if effect.update_status.lower() not in valid_status:
                return EffectValidationResult(
                    effect=effect,
                    valid=False,
                    rejection_reason=(
                        f"update_entity status '{effect.update_status}' "
                        f"not in {valid_status}"
                    ),
                )

        return EffectValidationResult(effect=effect, valid=True)

    def _validate_update_player(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate update_player effect.

        Rejects no-op calls (no mutation field set). Validates that hp_delta,
        currency_delta, and item entries are well-formed. The narrator must
        emit ONLY the fields it's actually changing.
        """
        # At least one mutation must be set
        has_mutation = any([
            effect.player_item_grant,
            effect.player_item_remove,
            effect.player_currency_delta,
            effect.player_hp_delta is not None,
            effect.player_add_conditions,
            effect.player_remove_conditions,
            effect.player_spell_slot_used is not None,
        ])
        if not has_mutation:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=(
                    "update_player called with no mutation fields. Pass at "
                    "least one of: item_grant, item_remove, currency_delta, "
                    "hp_delta, add_conditions, remove_conditions, "
                    "spell_slot_used. Empty calls are no-ops."
                ),
            )

        # HP delta sanity
        if effect.player_hp_delta is not None:
            if not isinstance(effect.player_hp_delta, int):
                return EffectValidationResult(
                    effect=effect,
                    valid=False,
                    rejection_reason="update_player hp_delta must be an integer",
                )
            if effect.player_hp_delta == 0:
                return EffectValidationResult(
                    effect=effect,
                    valid=False,
                    rejection_reason=(
                        "update_player hp_delta=0 is a no-op; omit the field "
                        "entirely if HP didn't change."
                    ),
                )
            # Damage requires damage_type for downstream resistance/vulnerability
            if effect.player_hp_delta < 0 and not effect.player_damage_type:
                return EffectValidationResult(
                    effect=effect,
                    valid=False,
                    rejection_reason=(
                        "update_player hp_delta < 0 (damage) requires "
                        "damage_type (fire / poison / piercing / etc.)"
                    ),
                )

        # Currency delta sanity
        if effect.player_currency_delta:
            valid_denoms = {"cp", "sp", "ep", "gp", "pp"}
            for k, v in effect.player_currency_delta.items():
                if k not in valid_denoms:
                    return EffectValidationResult(
                        effect=effect,
                        valid=False,
                        rejection_reason=(
                            f"update_player currency_delta has invalid "
                            f"denomination '{k}'. Use cp, sp, ep, gp, pp."
                        ),
                    )
                if not isinstance(v, int):
                    return EffectValidationResult(
                        effect=effect,
                        valid=False,
                        rejection_reason=(
                            f"update_player currency_delta values must be "
                            f"integers; got {type(v).__name__} for '{k}'."
                        ),
                    )

        # Item entries must be dicts with at least "name"
        for slot_name, entries in (
            ("item_grant", effect.player_item_grant),
            ("item_remove", effect.player_item_remove),
        ):
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("name"):
                    return EffectValidationResult(
                        effect=effect,
                        valid=False,
                        rejection_reason=(
                            f"update_player {slot_name} entries must be objects "
                            f"with a 'name' field. Optionally include 'quantity' "
                            f"and a source/destination."
                        ),
                    )

        return EffectValidationResult(effect=effect, valid=True)

    def _validate_change_location(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate change_location effect.

        location_name must be present and conform to the short-name format
        (2-4 words, no sentence structure).
        """
        if not effect.location_name:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="change_location requires location_name",
            )

        # Format check: short name, no commas/periods, no "behind/inside/near"
        name = effect.location_name.strip()
        word_count = len(name.split())
        if word_count > 5:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=(
                    f"change_location name '{name}' is too long ({word_count} words). "
                    "Use 2-4 words; invent a short name if needed (e.g. 'shrine clearing')."
                ),
            )

        if "," in name or name.endswith("."):
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=(
                    f"change_location name '{name}' looks like a sentence. "
                    "Use a short canonical name (e.g. 'the tavern', 'north gate')."
                ),
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
        # Set per-turn by the orchestrator so update_player targets the acting
        # PC rather than guessing in a multiplayer session (audit #1 / Option C).
        self.acting_character_id: Optional[str] = None
        # EffectType → handler registration. Built once here (not inline in
        # execute()) so tests can cross-check it against the tool registry's
        # emittable effect types — a converter-producible type with no row
        # here is exactly the silent-no-op drift the audit flagged.
        self._executors = {
            EffectType.SPAWN_OBJECT: self._execute_spawn_object,
            EffectType.ADD_NPC: self._execute_add_npc,
            EffectType.REMOVE_ENTITY: self._execute_remove_entity,
            EffectType.TRANSFER_ITEM: self._execute_transfer_item,
            EffectType.GRANT_CURRENCY: self._execute_grant_currency,
            EffectType.APPLY_DAMAGE: self._execute_apply_damage,
            EffectType.START_COMBAT: self._execute_start_combat,
            EffectType.SET_FLAG: self._execute_set_flag,
            EffectType.CONSUME_RESOURCE: self._execute_consume_resource,
            EffectType.REQUEST_ROLL: self._execute_request_roll,
            EffectType.REF_ENTITY: self._execute_ref_entity,
            EffectType.UPDATE_ENTITY: self._execute_update_entity,
            EffectType.UPDATE_PLAYER: self._execute_update_player,
            EffectType.CHANGE_LOCATION: self._execute_change_location,
        }

    def handled_effect_types(self) -> set[EffectType]:
        """The EffectTypes this executor has a registered handler for."""
        return set(self._executors)

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

        executor = self._executors.get(effect.effect_type)
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
                exc_info=True,
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

        # Auto-assign TTS voice from catalog (non-blocking, best-effort)
        try:
            from ..immersion.voice_assigner import assign_voice
            # Get character TTS provider from active profile
            char_provider = None
            try:
                from ..config import get_profile
                char_provider = get_profile().immersion.character_tts_provider or None
            except Exception:
                pass
            voice_id = await assign_voice(
                npc_description=effect.npc_description or "",
                scene_registry=self.scene_registry,
                npc_id=entity.npc_id,
                provider=char_provider,
            )
            if voice_id:
                entity.voice_id = voice_id
        except Exception as e:
            import structlog
            structlog.get_logger().debug("voice_auto_assign_skipped", error=str(e), exc_info=True)

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={
                "entity_id": entity.id,
                "npc_name": effect.npc_name,
                "dialogue_indices": effect.dialogue_indices,
                "dialogue_emotions": effect.dialogue_emotions,
            },
        )

    async def _execute_remove_entity(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Remove an entity (NPC, creature, or object) from the scene registry.

        Wired by the Step-1 registry cut: REMOVE_ENTITY used to have producers
        (INTENTS fallback) and a world-state sync branch but no executor row,
        making it a silent end-to-end no-op (audit Duplication P0,
        effects.py:686). The world-state side (scene-item removal) stays in
        the orchestrator's REMOVE_ENTITY sync branch; the KG bridge clears
        the entity's location edge once execution succeeds.
        """
        if not self.scene_registry:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="No scene registry available",
            )

        target = (effect.target or "").strip()
        if not target:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="remove_entity requires a target entity id or name",
            )

        removed = self.scene_registry.remove_by_name(target)
        if removed is None:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error=f"remove_entity target '{target}' not in scene registry",
            )

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={
                "entity_id": removed.id,
                "entity_name": removed.name,
                "reason": effect.reason,
            },
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
        """apply_damage has NO live implementation — fail honestly.

        Audit May #11 / 2026-06-09 (success-reporting no-op executors): this
        used to return success=True while mutating nothing, so the world-state
        sync recorded damage that never landed on anyone's HP. Only the legacy
        INTENTS text fallback still produces this type; player damage flows
        through UPDATE_PLAYER (hp_delta) and combat damage through the combat
        engine.
        """
        return EffectExecutionResult(
            effect=effect,
            success=False,
            error=(
                "apply_damage is not executable: narrator-declared player "
                "damage must use update_player (hp_delta); combat damage is "
                "owned by the combat engine"
            ),
        )

    async def _execute_start_combat(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Start combat - signals orchestrator to trigger combat."""
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"reason": effect.reason, "triggers_combat": True},
        )

    async def _execute_set_flag(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Acknowledge a flag change — a sync-applied signal effect.

        Not a stub: the actual write (world_state.global_flags[flag] = value)
        happens in the orchestrator's _sync_effect_to_world_state SET_FLAG
        branch, which runs only when this returns success — the same division
        of labor as change_location.
        """
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"flag": effect.flag_name, "value": effect.flag_value},
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
        import structlog
        _logger = structlog.get_logger()

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

            _logger.info(
                "entity_referenced",
                entity_id=entity_id,
                entity_name=entity.name,
                alias_used=alias,
                dialogue_indices=effect.dialogue_indices,
                dialogue_emotions=effect.dialogue_emotions,
            )
        else:
            _logger.debug(
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

    async def _execute_update_entity(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Apply a narrator-declared update to an existing scene entity.

        Updates the SceneEntity in-place. Each optional field on the effect
        is applied only if set (None means "no change"). The orchestrator
        consumes this effect as authoritative — extractor-derived updates
        for the same entity should defer to whatever this records.
        """
        import structlog
        _logger = structlog.get_logger()

        entity_id = effect.update_entity_id
        if not entity_id or not self.scene_registry:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="update_entity needs entity_id and scene_registry",
            )

        entity = self.scene_registry.get_by_name(entity_id)
        if entity is None:
            _logger.warning(
                "update_entity_target_not_found",
                entity_id=entity_id,
            )
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error=f"update_entity target '{entity_id}' not in scene registry",
            )

        applied: dict = {}

        if effect.update_disposition is not None:
            entity.disposition = effect.update_disposition.lower()
            applied["disposition"] = entity.disposition

        if effect.update_status is not None:
            entity.status = effect.update_status.lower()
            applied["status"] = effect.update_status.lower()

        if effect.update_importance is not None:
            entity.important = bool(effect.update_importance)
            applied["important"] = bool(effect.update_importance)

        if effect.update_description_addition:
            existing = getattr(entity, "description", "") or ""
            addition = effect.update_description_addition.strip()
            if addition and addition not in existing:
                entity.description = (existing + " " + addition).strip()
                applied["description_appended"] = addition

        # Inventory deltas are recorded here for the log; the actual NPCState
        # inventory mutation happens in the orchestrator's
        # _sync_effect_to_world_state pass (NPCState lives on WorldState,
        # not on SceneEntity).
        if effect.update_add_items:
            applied["items_added"] = list(effect.update_add_items)
        if effect.update_remove_items:
            applied["items_removed"] = list(effect.update_remove_items)

        _logger.info(
            "entity_updated_by_narrator",
            entity_id=entity_id,
            applied=applied,
        )

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"entity_id": entity_id, "applied": applied},
        )

    def _resolve_update_player_character(self):
        """Resolve the LIVE session Character object an update_player targets.

        Single-authority refactor (Stage A): we return and mutate the session's
        OWN Character instance — the same object `sync_player` reads for the
        narrator party snapshot and `_sync_session_characters` reconciles at
        end — rather than a fresh get_by_id copy. That kills the stale-copy
        clobber (DF-1) and the mid-turn HP incoherence (DF-11) at the source.

        update_player carries no explicit target (the narrator addresses "you"
        = the acting player). Resolution refuses to guess in ambiguous
        multiplayer so narrated damage never lands on the wrong PC:
          1. acting_character_id threaded in by the orchestrator this turn
          2. the sole player in the session (unambiguous)
          3. None — caller falls back to log-only
        """
        if not self.session:
            return None
        characters = [p.character for p in self.session.players.values() if p.character]
        if self.acting_character_id:
            for c in characters:
                if c.id == self.acting_character_id:
                    return c
            return None  # acting id set but not in session → don't guess
        if len(characters) == 1:
            return characters[0]
        return None

    async def _execute_update_player(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Apply and PERSIST a narrator-declared update to the player's state.

        Audit #1: this used to only build a log dict and defer to a
        "downstream sync layer" that never existed, so out-of-combat
        damage/heal/loot/currency/conditions were silently dropped. It now
        applies to the Character + currency/inventory and persists:
          - HP / conditions / spell-slot: mutate the Character, then a single
            character_repo.update() persists all three.
          - currency / items: applied through inventory_repo (separate tables).

        The world-state side (transfer log, NPC-inventory mirror) stays in the
        orchestrator's UPDATE_PLAYER branch — correct division of labor.
        """
        import structlog
        _logger = structlog.get_logger()

        applied: dict = {}
        persisted = False

        # Single authority (Stage A): resolve and mutate the LIVE session object.
        character = self._resolve_update_player_character()
        if character is None:
            _logger.warning(
                "update_player_no_target",
                reason="ambiguous or no player in session; recorded but not persisted",
                fields=[k for k in (
                    "hp" if effect.player_hp_delta is not None else None,
                    "currency" if effect.player_currency_delta else None,
                    "items" if (effect.player_item_grant or effect.player_item_remove) else None,
                ) if k],
            )
            # Preserve the log-only contract so the world-state branch still runs.
            return EffectExecutionResult(
                effect=effect, success=True,
                details={"applied": {}, "narrator_authoritative": True, "persisted": False},
            )

        from ..data.repositories.character_repo import get_character_repo
        from ..models.common import Condition
        from ..models.character import CharacterCondition

        char_repo = await get_character_repo()
        char_id = character.id

        character_dirty = False

        # --- HP delta: damage drains temp HP first; heal clamps to max ---
        if effect.player_hp_delta is not None and effect.player_hp_delta != 0:
            delta = effect.player_hp_delta
            if delta < 0:
                dmg = -delta
                absorbed = min(character.hp.temporary, dmg)
                character.hp.temporary -= absorbed
                character.hp.current = max(0, character.hp.current - (dmg - absorbed))
            else:
                character.hp.current = min(character.hp.maximum, character.hp.current + delta)
            applied["hp_delta"] = delta
            if effect.player_damage_type:
                applied["damage_type"] = effect.player_damage_type
            character_dirty = True

        # --- Conditions ---
        for cond_name in effect.player_add_conditions:
            try:
                cond = Condition(cond_name.strip().lower())
            except ValueError:
                continue
            if not any(c.condition == cond for c in character.conditions):
                character.conditions.append(
                    CharacterCondition(condition=cond, source=effect.player_hp_reason or "narrator")
                )
                applied.setdefault("conditions_added", []).append(cond.value)
                character_dirty = True
        if effect.player_remove_conditions:
            remove_set = set()
            for cond_name in effect.player_remove_conditions:
                try:
                    remove_set.add(Condition(cond_name.strip().lower()))
                except ValueError:
                    continue
            if remove_set:
                before = len(character.conditions)
                character.conditions = [c for c in character.conditions if c.condition not in remove_set]
                if len(character.conditions) != before:
                    applied["conditions_removed"] = [c.value for c in remove_set]
                    character_dirty = True

        # --- Spell slot expenditure ---
        if effect.player_spell_slot_used is not None and 1 <= effect.player_spell_slot_used <= 9:
            level = effect.player_spell_slot_used
            if character.spell_slots.expend_slot(level):  # no-op + False if none left
                applied["spell_slot_used"] = level
                character_dirty = True

        # Single write persists HP + conditions + spell slots together.
        if character_dirty:
            await char_repo.update(character)
            persisted = True

        # --- Currency (separate table) ---
        if effect.player_currency_delta and self.inventory_repo:
            try:
                currency = await self.inventory_repo.get_currency(char_id)
                denom = {"cp": "copper", "sp": "silver", "ep": "electrum", "gp": "gold", "pp": "platinum"}
                for k, v in effect.player_currency_delta.items():
                    field = denom.get(k.strip().lower()[:2]) or denom.get(k.strip().lower())
                    if field and isinstance(v, int):
                        setattr(currency, field, max(0, getattr(currency, field) + v))
                await self.inventory_repo.update_currency(currency)
                applied["currency_delta"] = effect.player_currency_delta
                persisted = True
            except Exception as e:
                _logger.error("persist_failed", entity="currency", character_id=char_id, error=str(e), exc_info=True)

        # --- Item grants / removals (separate table) ---
        if effect.player_item_grant and self.inventory_repo:
            from ..models.inventory import InventoryItem
            granted = []
            for entry in effect.player_item_grant:
                name = (entry.get("name") or "").strip()
                if not name:
                    continue
                qty = int(entry.get("quantity", 1) or 1)
                try:
                    await self.inventory_repo.add_item(InventoryItem(
                        character_id=char_id,
                        item_index=name.lower().replace(" ", "-"),
                        item_name=name,
                        quantity=qty,
                    ))
                    granted.append({"name": name, "quantity": qty})
                    persisted = True
                except Exception as e:
                    _logger.error("persist_failed", entity="inventory_item", character_id=char_id, item=name, error=str(e), exc_info=True)
            if granted:
                applied["items_granted"] = granted
        if effect.player_item_remove and self.inventory_repo:
            removed = []
            for entry in effect.player_item_remove:
                name = (entry.get("name") or "").strip()
                if not name:
                    continue
                qty = int(entry.get("quantity", 1) or 1)
                try:
                    existing = await self.inventory_repo.get_item_by_index(
                        char_id, name.lower().replace(" ", "-")
                    )
                    if existing:
                        await self.inventory_repo.remove_item(existing.id, qty)
                        removed.append({"name": name, "quantity": qty})
                        persisted = True
                except Exception as e:
                    _logger.error("persist_failed", entity="inventory_item", character_id=char_id, item=name, error=str(e), exc_info=True)
            if removed:
                applied["items_removed"] = removed

        _logger.info(
            "player_updated_by_narrator",
            character_id=char_id,
            applied=applied,
            persisted=persisted,
        )

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={"applied": applied, "narrator_authoritative": True, "persisted": persisted},
        )

    async def _execute_change_location(self, effect: ProposedEffect) -> EffectExecutionResult:
        """Record a narrator-declared location change.

        The actual world-state mutation happens in the orchestrator's
        state-application path (which already handles location_change from
        the state extractor). This effect signals "narrator authoritatively
        declared a move" so the orchestrator can prefer it over the extractor's
        parsing.
        """
        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={
                "location_name": effect.location_name,
                "location_description": effect.location_description or "",
                "narrator_authoritative": True,  # Orchestrator should prefer this over extractor
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

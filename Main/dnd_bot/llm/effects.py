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
import re
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
    # Production-only anchor for a durable off-scene NPC returning through
    # add_npc. Narrator schemas never expose this field; the orchestrator
    # resolves it against the campaign graph before execution.
    npc_canonical_id: Optional[str] = None
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
_CURRENCY_FIELDS = ("copper", "silver", "electrum", "gold", "platinum")


def _inventory_match_key(value: str) -> str:
    """Punctuation/space-insensitive form for matching item names to rows."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())


def resolve_inventory_row(rows: list, name: str, item_index: str = ""):
    """Find the inventory row a removal names, from ALL of the player's rows.

    ``get_item_by_index`` alone is far too narrow to address a row for
    removal: it filters ``equipped = 0`` and matches only ``item_index``,
    so it silently misses every equipped row (starting equipment auto-equips
    all weapons and armor) and every row whose stored SRD index is not the
    slug of its display name — 88 of 237 SRD equipment entries, including
    all seven packs, "Rations (1 day)", "Crossbow, light" and "Thieves'
    Tools". Removals addressed by name through it returned success while
    removing nothing, and the narration still claimed the item was gone.

    Resolution order: caller-supplied index, exact index, exact normalized
    name, then substring. Unequipped rows win ties so a spare is consumed
    before the wielded one.
    """
    def _ranked(candidates: list):
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda row: (bool(getattr(row, "equipped", False)), -int(getattr(row, "quantity", 1) or 1)),
        )[0]

    if item_index:
        hit = _ranked([r for r in rows if r.item_index == item_index])
        if hit is not None:
            return hit

    slug = (name or "").strip().lower().replace(" ", "-")
    if slug:
        hit = _ranked([r for r in rows if r.item_index == slug])
        if hit is not None:
            return hit

    target = _inventory_match_key(name)
    if not target:
        return None
    hit = _ranked([r for r in rows if _inventory_match_key(r.item_name) == target])
    if hit is not None:
        return hit
    return _ranked([
        r for r in rows
        if target in _inventory_match_key(r.item_name)
        or _inventory_match_key(r.item_name) in target
    ])


def build_effect_idempotency_key(
    campaign_id: str,
    message_id: str,
    effect_index: int,
) -> str:
    """Build idempotency key for an effect.

    Format: campaign_id:message_id:effect_index
    """
    return f"{campaign_id}:{message_id}:{effect_index}"


def _session_world_state(session):
    """Return the session's WorldState however it is currently attached."""
    if session is None:
        return None
    world_state = getattr(session, "world_state", None)
    if world_state is None:
        world_store = getattr(session, "world_store", None)
        world_state = getattr(world_store, "state", None)
    return world_state


def resolve_world_npc(session, entity_id: str):
    """Resolve a durable NPC after its scene-scoped view has departed.

    This establishes identity only. The orchestrator's WorldStateStore
    remains the sole writer for the authoritative world projection, and
    DeltaBridge owns graph mutation. A graph-only match therefore makes
    ``update_entity`` executable without incorrectly rematerializing an
    off-screen NPC into the current scene.

    Validator and executor share this one function on purpose: an id the
    validator accepts and the executor cannot resolve is the validate-then-
    die class the id-resolution work set out to eliminate.
    """
    world_state = _session_world_state(session)
    if world_state is not None:
        world_npc = (
            world_state.npcs.get(entity_id)
            or world_state._find_npc(entity_id)
        )
        if world_npc is not None:
            return world_npc

    # Departed-but-real identities. ``_is_known_entity`` has always accepted
    # these, so resolution must too — otherwise a narrator update to a dead
    # NPC validates and then dies at execution (post-merge review, seam 4).
    dead_npcs = getattr(session, "campaign_dead_npcs", {}) if session else {}
    if dead_npcs:
        dead = dead_npcs.get(entity_id)
        if dead is None:
            from ..game.identity import resolve_unique_identity

            dead = resolve_unique_identity(entity_id, list(dead_npcs.values()))
        if dead is not None:
            return dead

    knowledge_graph = (
        getattr(session, "knowledge_graph", None) if session else None
    )
    if knowledge_graph is None:
        return None
    graph_entity = knowledge_graph.get_entity(entity_id)
    if graph_entity is None:
        resolver = getattr(knowledge_graph, "resolve_entity_reference", None)
        if callable(resolver):
            graph_entity = resolver(entity_id)
    entity_type = getattr(
        getattr(graph_entity, "entity_type", None), "value", None
    )
    return graph_entity if graph_entity is not None and entity_type == "npc" else None


def resolve_world_reference(session, entity_id: str):
    """Resolve a canonical NPC, current location, or active scene item."""
    npc = resolve_world_npc(session, entity_id)
    if npc is not None:
        return ("npc", npc)

    world_state = _session_world_state(session)
    if world_state is None:
        return None

    from ..game.knowledge.models import slugify

    query_slug = slugify(entity_id)
    location = (world_state.current_location or "").strip()
    if location and query_slug == slugify(location):
        return ("location", location)
    # Historical/adjacent locations are not ambient prompt seeds, but an
    # explicit narrator reference to a known place is still legitimate.
    for known_location in world_state.connected_locations:
        if query_slug and query_slug == slugify(known_location):
            return ("location", known_location)
    for name in world_state.scene_items:
        if entity_id == name or (query_slug and query_slug == slugify(name)):
            return ("item", name)

    # Explicit references may resolve through the durable campaign graph
    # after an entity leaves scene-scoped WorldState. This lookup does not
    # make it an ambient prompt seed.
    knowledge_graph = (
        getattr(session, "knowledge_graph", None) if session else None
    )
    if knowledge_graph is not None:
        graph_entity = knowledge_graph.get_entity(entity_id)
        if graph_entity is None:
            resolver = getattr(
                knowledge_graph, "resolve_entity_reference", None
            )
            if callable(resolver):
                graph_entity = resolver(entity_id)
        if graph_entity is not None:
            graph_type = getattr(
                getattr(graph_entity, "entity_type", None), "value", None
            ) or str(getattr(graph_entity, "entity_type", "entity"))
            return (graph_type, graph_entity)
    return None


def world_reference_update_kind(scene_registry, session, entity_id: str):
    """Return the world-reference type an ``update_entity`` target resolves to.

    ``None`` means "do not gate": either the target is a live scene entity
    (its own long-standing contract) or nothing about its type is knowable
    from the collaborators at hand. Validator and executor both call this so
    they cannot disagree about what a target is.
    """
    if scene_registry is not None and scene_registry.get_by_name(entity_id):
        return None
    if resolve_world_npc(session, entity_id) is not None:
        return "npc"
    reference = resolve_world_reference(session, entity_id)
    return reference[0] if reference else None


# Attitude toward the party and carried inventory are person semantics.
# Nothing downstream writes them for a non-NPC: WorldStateStore's
# UPDATE_ENTITY branch is NPC-gated and DeltaBridge would stamp the raw
# property onto whatever node the id names.
_PERSON_ONLY_UPDATE_FIELDS = (
    ("disposition", "update_disposition"),
    ("add_items", "update_add_items"),
    ("remove_items", "update_remove_items"),
)

# A place is not a creature and holds no inventory. ``update_status`` is a
# creature-liveness enum (alive/wounded/dead/fled/captured) whose 'dead'
# value DeltaBridge translates into ``alive=false`` on the node.
_LOCATION_ALLOWED_UPDATE_FIELDS = (
    ("description_addition", "update_description_addition"),
    ("importance", "update_importance"),
)


def update_entity_target_conflict(effect, target_kind) -> Optional[str]:
    """Return why *effect*'s change fields cannot target *target_kind*.

    Post-merge review, seam 1: the world-reference fallback made LOCATION and
    item nodes reachable by ``update_entity``, and the executor applied the
    NPC-only field family to them regardless — DeltaBridge then stamped
    disposition/status onto a place. Abstains (returns None) whenever the
    target kind is unknown.
    """
    if not target_kind or target_kind == "npc":
        return None

    if target_kind == "location":
        allowed = {name for name, _attr in _LOCATION_ALLOWED_UPDATE_FIELDS}
        offending = [
            name
            for name, attr in (
                ("disposition", "update_disposition"),
                ("status", "update_status"),
                ("add_items", "update_add_items"),
                ("remove_items", "update_remove_items"),
            )
            if name not in allowed and getattr(effect, attr, None) not in (None, [], "")
        ]
        if offending:
            return (
                f"update_entity target '{effect.update_entity_id}' is a location; "
                f"{', '.join(offending)} describes a creature, not a place. Use "
                "description_addition for a place, or target the NPC directly."
            )
        return None

    offending = [
        name
        for name, attr in _PERSON_ONLY_UPDATE_FIELDS
        if getattr(effect, attr, None) not in (None, [], "")
    ]
    if offending:
        return (
            f"update_entity target '{effect.update_entity_id}' is a "
            f"{target_kind}, not an NPC; {', '.join(offending)} applies only to "
            "a creature. Target the NPC directly."
        )
    return None


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
        # Type-specific validators, hoisted from validate() so tests can
        # introspect coverage (mirrors EffectExecutor._executors /
        # handled_effect_types). Types without a row ride
        # _validate_default deliberately — the exhaustiveness test keeps
        # an explicit allowlist of those.
        self._validators = {
            EffectType.REF_ENTITY: self._validate_ref_entity,
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

    def validated_effect_types(self) -> set[EffectType]:
        """The EffectTypes with a type-specific validator row."""
        return set(self._validators)

    def validate(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate a single proposed effect."""
        validator = self._validators.get(effect.effect_type, self._validate_default)
        return validator(effect)

    def _validate_ref_entity(self, effect: ProposedEffect) -> EffectValidationResult:
        """Reject structurally incomplete roster references before execution."""
        if not effect.ref_entity_id or not effect.ref_entity_id.strip():
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="ref_entity requires entity_id from the roster",
            )
        if not self._is_known_entity(effect.ref_entity_id):
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=(
                    f"ref_entity target '{effect.ref_entity_id}' is not a known entity"
                ),
            )
        conflict = self._alias_canonical_conflict(
            effect.ref_entity_id, (effect.ref_alias_used or "").strip()
        )
        if conflict:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=(
                    f"ref_entity alias '{effect.ref_alias_used}' is the "
                    f"canonical name of a different entity ('{conflict}'); "
                    "reference that entity_id instead or drop the alias"
                ),
            )
        return EffectValidationResult(effect=effect, valid=True)

    def _alias_canonical_conflict(
        self, entity_id: str, alias: str
    ) -> Optional[str]:
        """Return the id of a DIFFERENT NPC whose canonical name is *alias*.

        This is the production form of the long-form audit's
        ``tool_reference_identity_grounding`` gate: a narrator ref that
        addresses one roster id while displaying another entity's proper
        name splits one person across two identities (the ``Lyra <- Elara``
        misbinding). Conservative on every axis — generic labels, aliases
        the target itself owns, and ambiguous ownership all abstain.
        """
        from ..game.identity import identity_keys, is_generic_npc_label

        if not alias or is_generic_npc_label(alias):
            return None
        alias_keys = identity_keys(alias)
        if not alias_keys:
            return None

        # Collect the ref target's own ids and labels; an alias the target
        # already owns (name, prior alias, promoted name) is never a
        # conflict.
        target_ids: set[str] = {entity_id}
        target_keys: set[str] = set(identity_keys(entity_id))
        targets = []
        if self.scene_registry is not None:
            targets.append(self.scene_registry.get_by_id(entity_id))
            targets.append(self.scene_registry.get_by_name(entity_id))
        world_state = (
            getattr(self.session, "world_state", None) if self.session else None
        )
        if world_state is None and self.session is not None:
            world_store = getattr(self.session, "world_store", None)
            world_state = getattr(world_store, "state", None)
        if world_state is not None:
            targets.append(
                world_state.npcs.get(entity_id)
                or world_state._find_npc(entity_id)
            )
        graph = (
            getattr(self.session, "knowledge_graph", None)
            if self.session
            else None
        )
        if graph is not None:
            target = graph.get_entity(entity_id)
            if target is None:
                resolver = getattr(graph, "resolve_entity_reference", None)
                target = resolver(entity_id) if callable(resolver) else None
            targets.append(target)
        for target in targets:
            if target is None:
                continue
            for id_attr in ("id", "node_id"):
                value = str(getattr(target, id_attr, "") or "").strip()
                if value:
                    target_ids.add(value)
            for label in (
                getattr(target, "name", ""),
                *(getattr(target, "aliases", None) or []),
            ):
                target_keys.update(identity_keys(str(label or "")))
        if alias_keys & target_keys:
            return None

        # Canonical owners of the alias, matched on NAME only (alias-to-alias
        # matching would over-reject once pollution exists). Only a
        # PROPER-named NPC can claim ownership: "the apothecary" must not
        # veto an alias of "apothecary" on the shop she works in.
        owners: set[str] = set()
        if world_state is not None:
            for npc_id, npc in world_state.npcs.items():
                if is_generic_npc_label(npc.name):
                    continue
                if identity_keys(npc.name) & alias_keys:
                    owners.add(npc_id)
        if graph is not None:
            for node in (getattr(graph, "_entities", {}) or {}).values():
                node_type = getattr(
                    getattr(node, "entity_type", None), "value", ""
                )
                if node_type != "npc":
                    continue
                node_name = str(getattr(node, "name", "") or "")
                if is_generic_npc_label(node_name):
                    continue
                if identity_keys(node_name) & alias_keys:
                    owners.add(str(getattr(node, "node_id", "") or ""))
        owners -= target_ids
        owners.discard("")
        if len(owners) == 1:
            return owners.pop()
        return None

    def _is_known_entity(self, entity_id: str) -> bool:
        """Resolve an entity across the live scene, WorldState, and campaign KG.

        A validator constructed without live collaborators remains a structural
        validator (used by NarrationStrategy before execution). The orchestrator
        injects both collaborators, so invented roster IDs fail at validation
        instead of reaching the executor.
        """
        if self.scene_registry is None and self.session is None:
            return True

        if self.scene_registry is not None:
            if self.scene_registry.get_by_id(entity_id):
                return True
            if self.scene_registry.get_by_name(entity_id):
                return True

        world_state = getattr(self.session, "world_state", None) if self.session else None
        if world_state is None and self.session is not None:
            world_store = getattr(self.session, "world_store", None)
            world_state = getattr(world_store, "state", None)
        if world_state is not None:
            if entity_id in world_state.npcs or world_state._find_npc(entity_id):
                return True
            from ..game.knowledge.models import slugify

            query_slug = slugify(entity_id)
            if any(
                entity_id == item_id or (query_slug and query_slug == slugify(item_id))
                for item_id in world_state.scene_items
            ):
                return True

        dead_npcs = getattr(self.session, "campaign_dead_npcs", {}) if self.session else {}
        if entity_id in dead_npcs:
            return True

        graph = getattr(self.session, "knowledge_graph", None) if self.session else None
        if graph is not None:
            entity = graph.get_entity(entity_id)
            if entity is None:
                resolver = getattr(graph, "resolve_entity_reference", None)
                entity = resolver(entity_id) if callable(resolver) else None
            if entity is not None:
                return True
        return False

    def _validate_spawn_object(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate spawn_object effect."""
        if not effect.object_name:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="spawn_object requires object_name",
            )
        return EffectValidationResult(effect=effect, valid=True)

    def _graph_dead_npcs(self) -> tuple[list, bool]:
        """``(death facts, answered)`` from the knowledge graph.

        The second value is why this returns a tuple. "The graph knows of no
        dead" and "the graph could not be asked" look identical as an empty
        list, and treating them the same fails OPEN on precisely the topology
        a sourcebook creates: hydration keeps the book's dead off the roster,
        nothing has written a campaign dead-NPC row yet, so a broken graph
        leaves every death source empty and `add_npc Old Bram` sails through.

        No graph at all is a legitimate answer — plenty of sessions have
        none, and their deaths live in the roster. A graph that RAISES is
        not: it means the one store that would know is unavailable.
        """
        graph = getattr(self.session, "knowledge_graph", None)
        reader = getattr(graph, "dead_npcs", None)
        if not callable(reader):
            return [], True
        try:
            return list(reader()), True
        except Exception as exc:
            # Imported here, like every other logging site in this module —
            # there is no module-level logger to reach for.
            import structlog

            structlog.get_logger().warning(
                "graph_dead_npc_read_failed", error=str(exc)
            )
            return [], False

    def _validate_add_npc(self, effect: ProposedEffect) -> EffectValidationResult:
        """Validate add_npc effect."""
        if not effect.npc_name:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason="add_npc requires npc_name",
            )
        from ..game.identity import is_generic_npc_label

        if is_generic_npc_label(effect.npc_name):
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=(
                    "add_npc requires a proper established name; generic roles "
                    f"must enter through StateDelta placeholders ({effect.npc_name!r})"
                ),
            )
        if self.session:
            from ..game.identity import resolve_unique_identity

            dead_npcs = []
            world_state = getattr(self.session, "world_state", None)
            if world_state is not None:
                dead_npcs.extend(
                    npc for npc in world_state.npcs.values() if not npc.alive
                )
            dead_npcs.extend(
                getattr(self.session, "campaign_dead_npcs", {}).values()
            )
            # The graph too. It is the one campaign-wide store carrying BOTH
            # authored and played-in deaths, and an authored-dead NPC is in
            # neither list above: hydration deliberately refuses to restore
            # the dead, so a book's corpse never reaches world_state.npcs at
            # all. Without this, the very first thing a sourcebook opening
            # could do is put the ferryman it killed back on stage alive.
            graph_dead, graph_answered = self._graph_dead_npcs()
            dead_npcs.extend(graph_dead)
            # A death we can SEE outranks one we merely cannot rule out, so
            # this runs first: when the answer is known, say which NPC and
            # why rather than blaming the graph.
            dead = resolve_unique_identity(effect.npc_name, dead_npcs)
            if dead is not None:
                return EffectValidationResult(
                    effect=effect,
                    valid=False,
                    rejection_reason=(
                        f"Dead NPC '{dead.name}' cannot be reintroduced with "
                        "add_npc; an authoritative resurrection mechanic is required"
                    ),
                )
            if not graph_answered:
                # Refuse rather than guess. "The dead stay dead" is stated to
                # outrank scene continuity, and with the graph unavailable
                # this NPC cannot be shown to be alive. The cost is bounded:
                # ref_entity still works, so the scene continues with the
                # cast it has until the graph recovers.
                return EffectValidationResult(
                    effect=effect,
                    valid=False,
                    rejection_reason=(
                        f"Cannot verify whether '{effect.npc_name}' is dead - "
                        "the knowledge graph is unavailable; refusing to "
                        "introduce an NPC that canon may record as dead"
                    ),
                )
            # Someone already standing here is REFERENCED, not introduced.
            # `ensure_npc` collapses a duplicate to the existing NPCState, so
            # this is not the last line of defence — but it is the only one
            # that reports, and the executor is skipped entirely when there
            # is no store. It matters most right after a sourcebook install,
            # where the whole authored cast is on the roster before the
            # narrator has written a word: an opening that "introduces" Mara
            # Venn is describing someone the party can already see.
            if world_state is not None:
                living = [npc for npc in world_state.npcs.values() if npc.alive]
                present = resolve_unique_identity(effect.npc_name, living)
                if present is not None:
                    return EffectValidationResult(
                        effect=effect,
                        valid=False,
                        rejection_reason=(
                            f"'{present.name}' is already in the scene; use "
                            "ref_entity to reference them rather than add_npc"
                        ),
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
        if not self._is_known_entity(effect.update_entity_id):
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=(
                    f"update_entity target '{effect.update_entity_id}' is not a known entity"
                ),
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

        # Target-kind gate. Shares one resolver with the executor so an
        # accepted target is always an executable one.
        conflict = update_entity_target_conflict(
            effect,
            world_reference_update_kind(
                self.scene_registry, self.session, effect.update_entity_id
            ),
        )
        if conflict:
            return EffectValidationResult(
                effect=effect,
                valid=False,
                rejection_reason=conflict,
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

            # Narrator tools describe changes; they do not authorize a
            # resurrection transition. Keep that capability behind the
            # deterministic game mechanic (WorldState.revive_npc).
            if effect.update_status.lower() == "alive" and self.session:
                # Resolves through the shared helper so the departed-roster
                # NPCs seam 4 made executable cannot slip past this guard.
                npc = resolve_world_npc(self.session, effect.update_entity_id)
                if npc is not None and not getattr(npc, "alive", True):
                    return EffectValidationResult(
                        effect=effect,
                        valid=False,
                        rejection_reason=(
                            f"Dead NPC '{npc.name}' cannot be revived by narrator "
                            "tool; an authoritative resurrection mechanic is required"
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

        # Stage C: mint/resolve the canonical NPCState up front so the
        # SceneEntity links to it via npc_id. That WorldState UUID is the
        # shared cross-store key — the KG node keys on it and the DB row
        # later adopts it, so all five stores converge on ONE id instead of
        # minting three unrelated UUIDs for one NPC in one turn. No store
        # (a sessionless apply) → npc_id stays None, unchanged from before.
        canonical_id = None
        store = getattr(self.session, "world_store", None) if self.session else None
        if store is not None:
            npc_state = store.ensure_npc(
                name=effect.npc_name or "Unknown",
                disposition=effect.npc_disposition or "neutral",
                description=effect.npc_description or "",
                canonical_id=effect.npc_canonical_id,
            )
            canonical_id = npc_state.id

        entity = SceneEntity(
            name=effect.npc_name,
            description=effect.npc_description or "",
            entity_type="npc",
            disposition=effect.npc_disposition or "neutral",
            monster_index=effect.monster_index,
            npc_id=canonical_id,
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
            # Scene rescoping may legitimately evict a tracked object from
            # the transient registry before a later authoritative destruction
            # call.  If the durable WorldState still knows the item, accept
            # the effect so the store sync can remove that projection.  Do
            # not extend this to off-scene NPCs: their lifecycle belongs to
            # update_entity, and REMOVE_ENTITY does not delete durable NPCs.
            world_reference = self._resolve_known_world_reference(target)
            if world_reference and world_reference[0] == "item":
                return EffectExecutionResult(
                    effect=effect,
                    success=True,
                    details={
                        "entity_id": target,
                        "entity_name": world_reference[1],
                        "reason": effect.reason,
                        "found_in_scene": False,
                        "found_in_world": True,
                    },
                )
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

        Not a stub: the actual write to the world state's global flags
        happens in ``WorldStateStore.apply_effect``'s SET_FLAG branch
        (Step 4; was the orchestrator's sync chain), which runs only when
        this returns success — the same division of labor as
        change_location.
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
        alias = (effect.ref_alias_used or "").strip()
        if self.scene_registry:
            entity = self.scene_registry.get_by_name(entity_id)
            if entity is None and alias:
                entity = self.scene_registry.get_by_name(alias)
        world_reference = self._resolve_known_world_reference(entity_id)
        if world_reference is None and alias:
            # The tool contract carries the exact display name used in prose.
            # Models sometimes pluralize or punctuate the roster slug while
            # still returning a correct canonical alias.  Resolve that alias
            # only through known state/graph catalogs; never trust it as a new
            # entity declaration.
            world_reference = self._resolve_known_world_reference(alias)

        if entity:
            # Update mention tracking
            from datetime import datetime
            entity.mention_count = getattr(entity, 'mention_count', 0) + 1
            entity.last_mentioned_at = datetime.utcnow()

            # Record alias if provided and different from canonical
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
        elif world_reference is None:
            _logger.warning("ref_entity_target_not_found", entity_id=entity_id)
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error=f"ref_entity target '{entity_id}' is not a known entity",
            )
        else:
            _logger.debug("ref_entity_known_off_scene", entity_id=entity_id)

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={
                "entity_id": entity_id,
                "alias_used": alias or None,
                "found_in_scene": entity is not None,
                "found_in_world": world_reference is not None,
                "world_reference_type": world_reference[0] if world_reference else None,
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
        if not entity_id:
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error="update_entity needs entity_id",
            )

        entity = (
            self.scene_registry.get_by_name(entity_id)
            if self.scene_registry
            else None
        )
        world_npc = self._resolve_known_world_npc(entity_id)
        # Validation accepts scene items and non-NPC graph entities
        # (_is_known_entity), so execution must resolve them too — a
        # narrator update to a known item ('carved-wooden-door') otherwise
        # passes validation and then dies here (soak 20260723_230351,
        # turns 16/23/69). Identity-only, same contract as the world_npc
        # path: no scene materialization, WorldStateStore stays the writer.
        world_reference = (
            self._resolve_known_world_reference(entity_id)
            if entity is None and world_npc is None
            else None
        )
        if entity is None and world_npc is None and world_reference is None:
            _logger.warning(
                "update_entity_target_not_found",
                entity_id=entity_id,
            )
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error=f"update_entity target '{entity_id}' is not a known entity",
            )

        # A LOCATION or item reached through the world-reference fallback must
        # not carry NPC-only semantics into DeltaBridge. The validator applies
        # the same rule via the same helpers, so this is a fail-closed backstop
        # rather than a second opinion.
        conflict = update_entity_target_conflict(
            effect,
            None if entity is not None or world_npc is not None
            else (world_reference[0] if world_reference else None),
        )
        if conflict:
            _logger.warning(
                "update_entity_target_kind_conflict",
                entity_id=entity_id,
                target_kind=world_reference[0] if world_reference else None,
            )
            return EffectExecutionResult(
                effect=effect,
                success=False,
                error=conflict,
            )

        applied: dict = {}

        if effect.update_disposition is not None:
            # Pydantic does not validate ordinary attribute assignment by
            # default. Assigning the raw tool string here poisoned the typed
            # SceneEntity and caused the next turn's ``.value`` access to
            # crash. Preserve the model invariant at the mutation boundary.
            from ..models.npc import Disposition
            disposition = Disposition(effect.update_disposition.lower())
            if entity is not None:
                entity.disposition = disposition
            applied["disposition"] = disposition.value

        if effect.update_status is not None:
            if entity is not None:
                entity.status = effect.update_status.lower()
            applied["status"] = effect.update_status.lower()

        if effect.update_importance is not None:
            if entity is not None:
                entity.important = bool(effect.update_importance)
            applied["important"] = bool(effect.update_importance)

        # ``applied`` is a receipt, not a wish list: a key here asserts that
        # some writer will carry the change. Pre-fix, a target that resolved
        # only through the world reference read ``getattr(None, "description")``
        # — always '' — so the dedup never fired and every re-execution
        # re-reported the same append that nothing had written (review, seam 3).
        existing_description, description_writer = self._description_write_target(
            entity, entity_id, world_reference
        )
        if effect.update_description_addition:
            addition = effect.update_description_addition.strip()
            if addition and addition not in existing_description:
                if entity is not None:
                    entity.description = (
                        existing_description + " " + addition
                    ).strip()
                if description_writer:
                    applied["description_appended"] = addition
                else:
                    _logger.debug(
                        "update_entity_description_has_no_writer",
                        entity_id=entity_id,
                        target_kind=(
                            world_reference[0] if world_reference else "npc"
                        ),
                    )

        # Inventory deltas are recorded here for the log; the actual NPCState
        # inventory mutation happens in the orchestrator's
        # _sync_effect_to_world_state pass (NPCState lives on WorldState,
        # not on SceneEntity) — so only claim them when that NPCState exists.
        inventory_writer = self._world_state_npc(entity_id) is not None
        if effect.update_add_items and inventory_writer:
            applied["items_added"] = list(effect.update_add_items)
        if effect.update_remove_items and inventory_writer:
            applied["items_removed"] = list(effect.update_remove_items)

        _logger.info(
            "entity_updated_by_narrator",
            entity_id=entity_id,
            applied=applied,
        )

        return EffectExecutionResult(
            effect=effect,
            success=True,
            details={
                "entity_id": entity_id,
                "applied": applied,
                "found_in_scene": entity is not None,
                "found_in_world": (
                    world_npc is not None or world_reference is not None
                ),
                "world_reference_type": (
                    world_reference[0] if world_reference else None
                ),
            },
        )

    def _world_state_npc(self, entity_id: str):
        """The NPCState ``WorldStateStore.apply_effect`` would resolve, if any.

        The store is the writer for NPC description/inventory, and it resolves
        strictly through ``WorldState.npcs``. A graph-only or departed-roster
        identity therefore has no inventory writer even though it is a real NPC.
        """
        world_state = _session_world_state(self.session)
        if world_state is None:
            return None
        return world_state.npcs.get(entity_id) or world_state._find_npc(entity_id)

    def _description_write_target(self, entity, entity_id: str, world_reference):
        """Return ``(existing_description, a_writer_exists)`` for this target.

        Mirrors exactly what ``WorldStateStore.apply_effect`` will do, so the
        dedup compares against the text that actually persists and the receipt
        only claims appends something records.
        """
        if entity is not None:
            return (getattr(entity, "description", "") or "", True)

        npc_state = self._world_state_npc(entity_id)
        if npc_state is not None:
            return (getattr(npc_state, "description", "") or "", True)

        world_state = _session_world_state(self.session)
        if world_state is not None and world_reference is not None:
            kind, target = world_reference
            if kind == "item" and isinstance(target, str):
                if target in world_state.scene_items:
                    return (world_state.scene_items[target] or "", True)
            elif kind == "location" and isinstance(target, str):
                current = (world_state.current_location or "").strip()
                if current and target == current:
                    return (world_state.location_description or "", True)
        return ("", False)

    def _resolve_known_world_npc(self, entity_id: str):
        """Resolve a durable NPC after its scene-scoped view has departed."""
        return resolve_world_npc(self.session, entity_id)

    def _resolve_known_world_reference(self, entity_id: str):
        """Resolve a canonical NPC, current location, or active scene item."""
        return resolve_world_reference(self.session, entity_id)

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
        # Spending breaks larger coins (a 2gp payment from a platinum purse
        # must succeed), and the receipt records what ACTUALLY moved per
        # denomination rather than what was requested. The old per-field
        # ``max(0, cur + v)`` could neither make change nor report the
        # shortfall: it clamped at zero and still receipted the full delta.
        if effect.player_currency_delta and self.inventory_repo:
            try:
                currency = await self.inventory_repo.get_currency(char_id)
                before = {
                    field: getattr(currency, field)
                    for field in _CURRENCY_FIELDS
                }
                denom = {"cp": "copper", "sp": "silver", "ep": "electrum", "gp": "gold", "pp": "platinum"}
                copper_per = {"copper": 1, "silver": 10, "electrum": 50, "gold": 100, "platinum": 1000}
                copper_out = 0
                for k, v in effect.player_currency_delta.items():
                    field = denom.get(str(k).strip().lower()[:2]) or denom.get(str(k).strip().lower())
                    if not field or not isinstance(v, int) or v == 0:
                        continue
                    if v > 0:
                        setattr(currency, field, getattr(currency, field) + v)
                    else:
                        copper_out += -v * copper_per[field]

                if copper_out:
                    if currency.total_in_copper >= copper_out:
                        currency.remove_currency(copper_out)
                    else:
                        _logger.warning(
                            "insufficient_currency_for_delta",
                            character_id=char_id,
                            requested=dict(effect.player_currency_delta),
                            have_copper=currency.total_in_copper,
                            need_copper=copper_out,
                        )

                await self.inventory_repo.update_currency(currency)
                code_for = {v: k for k, v in denom.items()}
                effective = {
                    code_for[field]: getattr(currency, field) - before[field]
                    for field in _CURRENCY_FIELDS
                    if getattr(currency, field) != before[field]
                }
                if effective:
                    applied["currency_delta"] = effective
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
            unresolved = []
            try:
                rows = await self.inventory_repo.get_all_items(char_id)
            except Exception as e:
                rows = []
                _logger.error("inventory_read_failed", character_id=char_id, error=str(e), exc_info=True)
            for entry in effect.player_item_remove:
                name = (entry.get("name") or "").strip()
                if not name:
                    continue
                qty = int(entry.get("quantity", 1) or 1)
                # Address the ROW, not a name-derived index: see
                # resolve_inventory_row for why the index lookup silently
                # missed equipped and SRD-indexed rows.
                target = resolve_inventory_row(
                    rows, name, str(entry.get("item_index") or "")
                )
                if target is None:
                    unresolved.append(name)
                    _logger.warning(
                        "item_remove_unresolved",
                        character_id=char_id,
                        item=name,
                        reason="no inventory row matched; nothing removed",
                    )
                    continue
                try:
                    await self.inventory_repo.remove_item(target.id, qty)
                    rows = [r for r in rows if r.id != target.id] + (
                        [target] if target.quantity > qty else []
                    )
                    removed.append({"name": target.item_name, "quantity": qty})
                    persisted = True
                except Exception as e:
                    _logger.error("persist_failed", entity="inventory_item", character_id=char_id, item=name, error=str(e), exc_info=True)
            if removed:
                applied["items_removed"] = removed
            if unresolved:
                # Surfaced in the receipt so a claimed-but-impossible removal
                # is visible instead of passing as a silent no-op.
                applied["items_remove_unresolved"] = unresolved

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

"""Narrator tool registry — the single authority for the tool→effect path.

REFACTOR_PLAN.md Step 1: effect-type wiring used to be hand-maintained in
five drifted dispatch sites (AUDIT_QUALITY_2026_06_09, Duplication P0).
This module makes "add a narrator tool" a one-entry change: each
:class:`NarratorToolSpec` co-locates everything the pipeline needs to
know about one tool —

- ``name``          the tool-call name the narrator emits
- ``schema``        the OpenAI/Ollama-format JSON schema sent to the model
- ``tiers``         which narrator tool tiers include it (core ⊂ core_plus ⊂ full)
- ``converter``     tool args → :class:`ProposedEffect`
- ``effect_types``  the EffectType(s) the converter can emit — cross-checked
                    by tests against EffectExecutor registration and the
                    orchestrator's world-state sync so a silent no-op type
                    can never ship again
- ``world_sync``    whether successful execution must be mirrored into
                    WorldState by ``_sync_effect_to_world_state`` (False for
                    pure signal effects the orchestrator consumes directly,
                    e.g. start_combat / request_roll)

Consumers:
- ``narrator_tools.get_narrator_tools_for_tier`` reads :func:`tools_for_tier`
- ``narrator_tools.tool_calls_to_effects`` reads :func:`convert_tool_call`
- ``tests/unit/test_tool_registry.py`` enforces the exhaustiveness contracts

The legacy INTENTS text fallback (``intents.py``) is the OTHER producer of
ProposedEffects; it is plan-deferred and intentionally not migrated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import structlog

from .effects import EffectType, ProposedEffect

logger = structlog.get_logger()


# ── Tier names ─────────────────────────────────────────────────────────
# Smaller / less reliable narrators get a smaller tool surface so they pay
# attention to the tools they have. The state and entity extractors act as
# fallback for whatever the narrator's tier doesn't cover.
KNOWN_TIERS: tuple[str, ...] = ("core", "core_plus", "full")

_ALL_TIERS = ("core", "core_plus", "full")
_PLUS_TIERS = ("core_plus", "full")
_FULL_ONLY = ("full",)


@dataclass(frozen=True)
class NarratorToolSpec:
    """One declarative entry per narrator tool. See module docstring."""

    name: str
    schema: dict
    tiers: tuple[str, ...]
    converter: Callable[[dict], Optional[ProposedEffect]]
    effect_types: tuple[EffectType, ...]
    world_sync: bool = True


# Registration order is canonical: it defines the schema order sent to the
# model per tier (required-first, common-first — same order the hardcoded
# NARRATOR_TOOLS list used).
_REGISTRY: dict[str, NarratorToolSpec] = {}


def register(spec: NarratorToolSpec) -> None:
    """Register a tool spec. Name collisions are a programming error."""
    if spec.name in _REGISTRY:
        raise ValueError(f"narrator tool '{spec.name}' already registered")
    schema_name = spec.schema.get("function", {}).get("name")
    if schema_name != spec.name:
        raise ValueError(
            f"narrator tool '{spec.name}' schema declares name '{schema_name}'"
        )
    unknown = set(spec.tiers) - set(KNOWN_TIERS)
    if unknown:
        raise ValueError(f"narrator tool '{spec.name}' has unknown tiers {unknown}")
    _REGISTRY[spec.name] = spec


def get_spec(name: str) -> Optional[NarratorToolSpec]:
    """Look up one tool spec by tool-call name."""
    return _REGISTRY.get(name)


def all_specs() -> list[NarratorToolSpec]:
    """All registered specs, in registration (= schema) order."""
    return list(_REGISTRY.values())


def tools_for_tier(tier: str) -> Optional[list[dict]]:
    """Schema list for a tier, in registration order. None for unknown tiers
    (the narrator_tools wrapper owns the warn-and-fall-back-to-core policy)."""
    if tier not in KNOWN_TIERS:
        return None
    return [spec.schema for spec in _REGISTRY.values() if tier in spec.tiers]


def convert_tool_call(name: str, args: dict) -> Optional[ProposedEffect]:
    """Convert a single narrator tool call to a ProposedEffect.

    Unknown tool names are logged and dropped (None), matching the legacy
    ``_convert_tool_call`` contract.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        logger.warning("unknown_narrator_tool", tool_name=name)
        return None
    return spec.converter(args)


def emittable_effect_types() -> set[EffectType]:
    """Every EffectType any registered converter can emit — the set the
    exhaustiveness tests cross-check against EffectExecutor + world sync."""
    types: set[EffectType] = set()
    for spec in _REGISTRY.values():
        types.update(spec.effect_types)
    return types


# ======================================================================
# Converters (tool args → ProposedEffect)
# ======================================================================

# Ability names for detecting save vs check
_ABILITIES = {"strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"}


def _convert_ref_entity(args: dict) -> ProposedEffect:
    return ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id=args.get("entity_id", ""),
        ref_alias_used=args.get("alias_used"),
        dialogue_indices=args.get("dialogue_indices", []),
        dialogue_emotions=args.get("dialogue_emotions", []),
    )


def _convert_add_npc(args: dict) -> ProposedEffect:
    # Sanitize disposition -- models sometimes invent values outside the enum
    raw_disposition = args.get("disposition", "neutral").lower()
    valid_dispositions = {"friendly", "neutral", "unfriendly", "hostile", "allied"}
    disposition = raw_disposition if raw_disposition in valid_dispositions else "neutral"

    # Pass gender hint for voice assignment
    gender = args.get("gender", "")
    desc = args.get("description", "") or ""
    if gender:
        desc = f"[{gender}] {desc}"

    return ProposedEffect(
        effect_type=EffectType.ADD_NPC,
        npc_name=args.get("name", ""),
        npc_description=desc,
        npc_disposition=disposition,
        source=f"npc:{args.get('npc_id', 'unknown')}",
        dialogue_indices=args.get("dialogue_indices", []),
        dialogue_emotions=args.get("dialogue_emotions", []),
    )


def _convert_spawn_object(args: dict) -> ProposedEffect:
    return ProposedEffect(
        effect_type=EffectType.SPAWN_OBJECT,
        object_name=args.get("name", ""),
        object_description=args.get("description"),
        source=f"scene:{args.get('object_id', 'unknown')}",
    )


def _convert_update_player(args: dict) -> ProposedEffect:
    # Consolidated player-state mutation: items, currency, HP, conditions,
    # spell slots. Each field is independent and optional. Validator
    # rejects no-op calls.
    item_grant_raw = args.get("item_grant") or []
    item_remove_raw = args.get("item_remove") or []
    currency_delta = args.get("currency_delta") or {}
    hp_delta = args.get("hp_delta")
    damage_type = args.get("damage_type")
    hp_reason = args.get("hp_reason")
    add_conditions = args.get("add_conditions") or []
    remove_conditions = args.get("remove_conditions") or []
    spell_slot_used = args.get("spell_slot_used")

    # Normalize item entries — accept both string ("dagger") and dict
    # ({"name": "dagger", "quantity": 1, "source": "scene:..."}) shapes.
    def _norm_item(entry):
        if isinstance(entry, str):
            return {"name": entry}
        if isinstance(entry, dict) and entry.get("name"):
            return entry
        return None

    item_grant = [e for e in (_norm_item(x) for x in item_grant_raw) if e]
    item_remove = [e for e in (_norm_item(x) for x in item_remove_raw) if e]

    # Sanitize currency dict — accept only valid denoms with int values
    valid_denoms = {"cp", "sp", "ep", "gp", "pp"}
    sanitized_currency = {
        k: v for k, v in currency_delta.items()
        if k in valid_denoms and isinstance(v, int)
    } if isinstance(currency_delta, dict) else {}

    return ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER,
        player_item_grant=item_grant,
        player_item_remove=item_remove,
        player_currency_delta=sanitized_currency,
        player_hp_delta=hp_delta if isinstance(hp_delta, int) else None,
        player_damage_type=(damage_type or None) if isinstance(damage_type, str) else None,
        player_hp_reason=(hp_reason or None) if isinstance(hp_reason, str) else None,
        player_add_conditions=[c.lower() for c in add_conditions if isinstance(c, str)],
        player_remove_conditions=[c.lower() for c in remove_conditions if isinstance(c, str)],
        player_spell_slot_used=spell_slot_used if isinstance(spell_slot_used, int) else None,
    )


def _convert_request_roll(args: dict) -> ProposedEffect:
    ability_or_skill = args.get("ability_or_skill", "").lower()
    roll_type_raw = args.get("roll_type", "check")

    # Determine if this is a save, ability check, or skill check
    if roll_type_raw == "save":
        roll_type = "saving_throw"
        ability = ability_or_skill
        skill = None
    elif ability_or_skill in _ABILITIES:
        roll_type = "ability_check"
        ability = ability_or_skill
        skill = None
    else:
        roll_type = "skill_check"
        ability = None
        skill = ability_or_skill

    return ProposedEffect(
        effect_type=EffectType.REQUEST_ROLL,
        target=args.get("target", "player"),
        roll_type=roll_type,
        ability=ability,
        skill=skill,
        dc=args.get("dc", 10),
        roll_reason=args.get("reason", ""),
    )


def _convert_start_combat(args: dict) -> ProposedEffect:
    return ProposedEffect(
        effect_type=EffectType.START_COMBAT,
        reason=args.get("reason", "Combat begins!"),
    )


def _convert_change_location(args: dict) -> ProposedEffect:
    return ProposedEffect(
        effect_type=EffectType.CHANGE_LOCATION,
        location_name=args.get("location_name", "").strip() or None,
        location_description=(args.get("description") or "").strip() or None,
    )


def _convert_update_entity(args: dict) -> ProposedEffect:
    # Only set fields that the narrator actually included. Missing
    # fields stay None (semantics: "no change").
    disposition = args.get("disposition")
    status = args.get("status")
    importance = args.get("importance")
    desc_add = args.get("description_addition") or None
    add_items = args.get("add_items") or []
    remove_items = args.get("remove_items") or []

    return ProposedEffect(
        effect_type=EffectType.UPDATE_ENTITY,
        update_entity_id=args.get("entity_id", "").strip() or None,
        update_importance=importance if isinstance(importance, bool) else None,
        update_disposition=disposition.lower() if isinstance(disposition, str) else None,
        update_status=status.lower() if isinstance(status, str) else None,
        update_description_addition=desc_add,
        update_add_items=[s.strip().lower() for s in add_items if isinstance(s, str) and s.strip()],
        update_remove_items=[s.strip().lower() for s in remove_items if isinstance(s, str) and s.strip()],
    )


# ======================================================================
# Tool registrations — one block per tool: schema + tier + converter.
# Order matters (it is the schema order sent to the model).
# ======================================================================

register(NarratorToolSpec(
    name="ref_entity",
    tiers=_ALL_TIERS,
    converter=_convert_ref_entity,
    effect_types=(EffectType.REF_ENTITY,),
    schema={
        "type": "function",
        "function": {
            "name": "ref_entity",
            "description": (
                "Declare that you referenced an existing entity from the roster "
                "in your prose. Call this for EVERY roster entity your narration "
                "mentions, even briefly — the system uses these calls to keep "
                "entity recall and dialogue routing in sync. Do NOT call for the "
                "player or party PCs (they are not roster entries), for entities "
                "you are introducing for the first time (use add_npc), or for "
                "objects (use spawn_object). If the entity speaks quoted dialogue, "
                "list which quotes by order of appearance (1-indexed) and the "
                "delivery emotion for each.\n\n"
                "Example calls:\n"
                "- After 'Marta nods grimly': {entity_id: 'marta'}\n"
                "- After 'Kael whispers \"They\\'re inside\"': "
                "{entity_id: 'kael', dialogue_indices: [1], "
                "dialogue_emotions: ['whispering']}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The ID from the roster [id: ...] tag",
                    },
                    "alias_used": {
                        "type": "string",
                        "description": "Name used in prose if different from roster name",
                    },
                    "dialogue_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Rough order of when this entity speaks (1-indexed). E.g. if they speak first and third, use [1, 3].",
                    },
                    "dialogue_emotions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Emotion/delivery for this entity's dialogue. E.g. ['desperate', 'calm']. Options: happy, sad, angry, scared, excited, whispering, shouting, nervous, calm, sarcastic, desperate, commanding.",
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
))

register(NarratorToolSpec(
    name="add_npc",
    tiers=_ALL_TIERS,
    converter=_convert_add_npc,
    effect_types=(EffectType.ADD_NPC,),
    schema={
        "type": "function",
        "function": {
            "name": "add_npc",
            "description": (
                "Introduce a NEW NPC into the scene for the first time. Call "
                "this when your prose brings an unnamed character on stage — "
                "a merchant greeting the party, a guard stopping them, a "
                "stranger at the bar. Do NOT call for entities already in "
                "the roster (use ref_entity), for objects/items (use "
                "spawn_object), for the player or party PCs, or for purely "
                "atmospheric mentions ('a few patrons sit in the corner') "
                "where no specific named NPC is meant to be tracked. Every "
                "NPC must have a proper invented name — never generic roles "
                "like 'Merchant' or 'Old Woman'.\n\n"
                "Example call after 'A burly dwarf behind the forge looks up "
                "as you enter, soot covering his missing left eye': "
                "{npc_id: 'blacksmith_1', name: 'Korin Ironeye', "
                "disposition: 'neutral', gender: 'male', description: 'A burly "
                "dwarven blacksmith with soot-covered arms and a missing left eye'}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "npc_id": {
                        "type": "string",
                        "description": "Short ID (e.g. merchant_1, guard_2)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Proper name for the NPC (e.g. 'Grom', 'Elara Swiftblade', 'Captain Voss'). NEVER use generic roles like 'Merchant' or descriptions like 'Old Woman'.",
                    },
                    "disposition": {
                        "type": "string",
                        "enum": ["friendly", "neutral", "unfriendly", "hostile", "allied"],
                        "description": "NPC's ATTITUDE toward the party (not their emotional state -- use dialogue_emotions for that). friendly = wants to help/cooperate, neutral = indifferent, unfriendly = dislikes party, hostile = will fight, allied = fights alongside party.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Physical appearance, role, and personality (e.g. 'A burly dwarven blacksmith with soot-covered arms and a missing left eye')",
                    },
                    "gender": {
                        "type": "string",
                        "enum": ["male", "female"],
                        "description": "The NPC's gender.",
                    },
                    "dialogue_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Rough order of when this NPC speaks (1-indexed). E.g. if they speak first and third, use [1, 3].",
                    },
                    "dialogue_emotions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Emotion/delivery for this NPC's dialogue. E.g. ['desperate', 'scared']. Options: happy, sad, angry, scared, excited, whispering, shouting, nervous, calm, sarcastic, desperate, commanding.",
                    },
                },
                "required": ["npc_id", "name", "disposition", "description"],
            },
        },
    },
))

register(NarratorToolSpec(
    name="spawn_object",
    tiers=_ALL_TIERS,
    converter=_convert_spawn_object,
    effect_types=(EffectType.SPAWN_OBJECT,),
    schema={
        "type": "function",
        "function": {
            "name": "spawn_object",
            "description": (
                "Place a discoverable object into the scene — a piece of "
                "treasure, an item the party can pick up, an interactable "
                "feature (lever, chest, body). Call this when your prose "
                "introduces a SPECIFIC object the player can engage with. "
                "Do NOT call for objects already held by an NPC (use "
                "offer_item or transfer_item), for ambient scenery that "
                "the player won't interact with ('cobwebs', 'old furniture'), "
                "for objects already in the scene from a previous turn, or "
                "for things mentioned only in passing.\n\n"
                "Example call after 'On the altar rests a small jade dagger, "
                "its hilt wrapped in dark leather': "
                "{object_id: 'jade_dagger_1', name: 'jade dagger', "
                "description: 'A small jade dagger with a dark leather-wrapped hilt'}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {
                        "type": "string",
                        "description": "Short snake_case ID unique to this scene (e.g. 'jade_dagger_1', 'chest_1', 'guard_corpse').",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short display name as the player would refer to it (e.g. 'jade dagger', 'iron chest').",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-sentence physical description, used to seed the object's record so future turns can reference it.",
                    },
                },
                "required": ["object_id", "name"],
            },
        },
    },
))

register(NarratorToolSpec(
    name="update_player",
    tiers=_FULL_ONLY,
    converter=_convert_update_player,
    effect_types=(EffectType.UPDATE_PLAYER,),
    schema={
        "type": "function",
        "function": {
            "name": "update_player",
            "description": (
                "Record any change to the PLAYER's state that your prose "
                "described in this turn. One tool, multiple optional "
                "fields — set ONLY the fields that actually changed. "
                "Covers items, currency, HP (damage and healing), "
                "conditions, and spell-slot consumption.\n\n"
                "Call when your prose described any of:\n"
                "- The player picking up, receiving, dropping, or handing "
                "off an item.\n"
                "- The player gaining or spending coin.\n"
                "- The player taking environmental damage (trap, fall, "
                "poison cloud, fire) or being healed (potion, divine "
                "intervention, NPC tending wounds).\n"
                "- The player gaining or losing a status condition "
                "(poisoned, prone, frightened, etc.).\n"
                "- The player consuming a spell slot outside the combat "
                "engine.\n\n"
                "Do NOT call when:\n"
                "- An NPC merely OFFERS an item the player hasn't accepted "
                "yet — describe the offer in prose and wait. Only call "
                "this once the player has accepted IN FICTION.\n"
                "- Combat damage from an attack roll — the combat engine "
                "handles those automatically.\n"
                "- The player asks about something but takes no action.\n"
                "- Nothing about the player's state actually changed; "
                "empty calls are rejected.\n\n"
                "All fields are optional and independent. Pass at least "
                "one mutation field. Absent fields mean 'no change.'\n\n"
                "Example calls:\n"
                "- After 'You grab the jade dagger off the altar': "
                "{item_grant: [{name: 'jade dagger', source: 'scene:jade_dagger_1'}]}\n"
                "- After 'The mayor presses fifty gold into your palm': "
                "{currency_delta: {gp: 50}}\n"
                "- After 'You hand the merchant 12 silver for the rope': "
                "{currency_delta: {sp: -12}, item_grant: [{name: 'rope', source: 'npc:merchant'}]}\n"
                "- After 'A poisoned dart fires from the wall and pierces "
                "your shoulder': {hp_delta: -6, damage_type: 'poison', "
                "hp_reason: 'wall trap dart', add_conditions: ['poisoned']}\n"
                "- After 'You uncork the potion and feel warmth flood "
                "through you': {hp_delta: 8, hp_reason: 'potion of healing', "
                "item_remove: [{name: 'potion of healing'}]}\n"
                "- After 'You hand the relic to the innkeeper for safekeeping': "
                "{item_remove: [{name: 'ancient relic', destination: 'npc:innkeeper'}]}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_grant": {
                        "type": "array",
                        "description": (
                            "Items the player gained this turn. Each entry "
                            "is {name, quantity?, source?}. 'source' is "
                            "where it came from — 'scene:<object_id>' for "
                            "scene pickups, 'npc:<id>' for items received "
                            "from an NPC. Omit if not applicable."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "quantity": {"type": "integer"},
                                "source": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    },
                    "item_remove": {
                        "type": "array",
                        "description": (
                            "Items the player lost / spent / handed off this "
                            "turn. Each entry is {name, quantity?, "
                            "destination?}. 'destination' is where it went — "
                            "'npc:<id>' for handing to an NPC. Omit if not "
                            "applicable."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "quantity": {"type": "integer"},
                                "destination": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    },
                    "currency_delta": {
                        "type": "object",
                        "description": (
                            "Net change to the player's purse, keyed by "
                            "denomination. Positive values are gains, "
                            "negative are spends. E.g. {gp: 50} for fifty "
                            "gold gained, {sp: -12} for twelve silver "
                            "spent, {gp: 2, sp: -5} for mixed."
                        ),
                        "properties": {
                            "cp": {"type": "integer"},
                            "sp": {"type": "integer"},
                            "ep": {"type": "integer"},
                            "gp": {"type": "integer"},
                            "pp": {"type": "integer"},
                        },
                    },
                    "hp_delta": {
                        "type": "integer",
                        "description": (
                            "Change to the player's HP. Negative = damage "
                            "(from world hazards, NOT combat attacks); "
                            "positive = healing. If negative, "
                            "damage_type is required."
                        ),
                    },
                    "damage_type": {
                        "type": "string",
                        "description": (
                            "Required when hp_delta < 0. One of fire, cold, "
                            "lightning, thunder, acid, poison, necrotic, "
                            "radiant, force, psychic, bludgeoning, piercing, "
                            "slashing."
                        ),
                    },
                    "hp_reason": {
                        "type": "string",
                        "description": (
                            "One short clause describing what changed the "
                            "HP, used for the combat/heal log. E.g. 'wall "
                            "trap dart', 'potion of healing'."
                        ),
                    },
                    "add_conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Status conditions the player just gained "
                            "(poisoned, prone, frightened, charmed, "
                            "blinded, deafened, paralyzed, stunned, "
                            "incapacitated, restrained, grappled). "
                            "Lowercase."
                        ),
                    },
                    "remove_conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Conditions removed this turn.",
                    },
                    "spell_slot_used": {
                        "type": "integer",
                        "description": (
                            "Spell-slot level consumed (1-9), only when "
                            "the player cast outside the combat engine "
                            "(narrative spell use). Combat spells are "
                            "tracked by the engine."
                        ),
                    },
                },
            },
        },
    },
))

register(NarratorToolSpec(
    name="request_roll",
    tiers=_FULL_ONLY,
    converter=_convert_request_roll,
    effect_types=(EffectType.REQUEST_ROLL,),
    # Signal effect: the orchestrator consumes the roll request directly;
    # there is nothing to mirror into WorldState.
    world_sync=False,
    schema={
        "type": "function",
        "function": {
            "name": "request_roll",
            "description": (
                "Pause your prose and ask the player for a dice roll before "
                "describing the outcome. Call this when the action's success "
                "is uncertain and the rules layer needs to roll — Perception "
                "to spot a hidden trap, Stealth to sneak past a guard, "
                "Constitution save vs poison gas, Strength check to force a "
                "stuck door. After the player rolls, you'll see the result "
                "in the next turn and narrate accordingly. Do NOT call for "
                "in-combat attack rolls (the combat engine handles those), "
                "for damage rolls (mechanics handles damage), for trivial "
                "actions where success is automatic ('open the unlocked door'), "
                "for narrative-only beats with no mechanical outcome, or "
                "when the player has already rolled in a prior turn for this "
                "exact action.\n\n"
                "Example call when the player says 'I try to spot anything "
                "off about the merchant': {target: 'player', roll_type: "
                "'check', ability_or_skill: 'insight', dc: 14, reason: "
                "'reading the merchant for deception'}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Who rolls. Usually 'player'.",
                    },
                    "roll_type": {
                        "type": "string",
                        "enum": ["save", "check", "skill"],
                        "description": (
                            "'save' for saving throws (DEX save vs trap, CON "
                            "save vs poison). 'check' for raw ability checks "
                            "without a skill (raw STR to lift a portcullis). "
                            "'skill' for skill-based checks like Perception, "
                            "Stealth, Persuasion."
                        ),
                    },
                    "ability_or_skill": {
                        "type": "string",
                        "description": (
                            "Lowercase name. For saves/checks: ability name "
                            "(strength, dexterity, constitution, intelligence, "
                            "wisdom, charisma). For skills: skill name "
                            "(perception, stealth, insight, athletics, etc.)."
                        ),
                    },
                    "dc": {
                        "type": "integer",
                        "description": (
                            "Difficulty class. Standard scale: 10 easy, 13 "
                            "moderate, 15 hard, 18 very hard, 20+ heroic. "
                            "Pick the value that matches how hard the action "
                            "is in fiction."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "One short clause describing what the roll resolves, shown to the player. E.g. 'spotting the tripwire', 'forcing the rusted hatch open'.",
                    },
                },
                "required": ["target", "roll_type", "ability_or_skill", "dc", "reason"],
            },
        },
    },
))

register(NarratorToolSpec(
    name="start_combat",
    tiers=_PLUS_TIERS,
    converter=_convert_start_combat,
    effect_types=(EffectType.START_COMBAT,),
    # Signal effect: consumed by the orchestrator as combat_triggered; the
    # combat pipeline owns all resulting state.
    world_sync=False,
    schema={
        "type": "function",
        "function": {
            "name": "start_combat",
            "description": (
                "Declare that the situation has escalated to combat. This "
                "hands the scene off to the deterministic combat engine, "
                "which will run round-by-round initiative, attack rolls, "
                "and damage. The narrator does NOT narrate combat itself.\n\n"
                "Call this at the precipitating moment — when your prose "
                "captures the beat where hostility crosses into committed "
                "attack. The hostile party (or the player) has chosen to "
                "engage and the next thing that happens is a fight. Examples "
                "of when to call:\n"
                "- A hostile NPC draws their weapon with clear intent to "
                "strike and steps toward the party.\n"
                "- An ambush springs (assassins drop from rafters, archers "
                "rise from the brush).\n"
                "- The player explicitly attacks ('I lunge at the goblin "
                "with my sword') — combat begins regardless of the NPC's "
                "prior disposition.\n"
                "- An NPC who's been on edge gets pushed past the breaking "
                "point (insulted, threatened, betrayed) and commits to "
                "attacking.\n"
                "- A creature reveals itself in mid-strike (predator "
                "lunging from cover).\n\n"
                "Do NOT call when:\n"
                "- The scene is tense but no one has committed yet "
                "(weapons drawn in a standoff, threats exchanged, hostile "
                "posture without action).\n"
                "- An NPC enters the scene hostile but hasn't engaged yet "
                "(still might be talked down).\n"
                "- The player is asking about a fight without committing "
                "to an action.\n"
                "- Combat is already underway from a previous turn (this "
                "tool fires once, at onset).\n"
                "- Your prose ends with the precipitating moment but you "
                "want the player to choose how to respond — that's still "
                "valid; just call start_combat and stop the prose there. "
                "The combat engine takes the next round.\n\n"
                "When you call this tool, your prose should END at the "
                "precipitating moment — do NOT narrate blows landing, "
                "blocks, or counter-attacks. That's the combat engine's "
                "job once start_combat fires."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "One short sentence describing what crossed the "
                            "line into combat, used for the combat log. "
                            "E.g., 'The bandit leader snarled and drew his "
                            "cutlass when the party refused to pay.' Or: "
                            "'The player attacked the goblin who had been "
                            "stonewalling them.'"
                        ),
                    },
                },
                "required": ["reason"],
            },
        },
    },
))

register(NarratorToolSpec(
    name="change_location",
    tiers=_PLUS_TIERS,
    converter=_convert_change_location,
    effect_types=(EffectType.CHANGE_LOCATION,),
    schema={
        "type": "function",
        "function": {
            "name": "change_location",
            "description": (
                "Declare that the party has moved to a new named location in "
                "this turn's narration. Call this when your prose brings the "
                "party into a distinct named area — entering a building, "
                "walking to a new district, descending into a dungeon level, "
                "stepping out of one place into another. Do NOT call when "
                "describing the surroundings of an existing location, narrating "
                "an in-place action, or when the party is still in the same "
                "place as last turn. The location_name field becomes the "
                "canonical name in world state, so keep it short and reusable "
                "(2-4 words, the same name you would say to refer to it later)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": (
                            "Short canonical name for the location, 2-4 words. "
                            "Use a proper name when one exists ('Thornfield', "
                            "'the Rusty Compass'). Otherwise invent a short "
                            "name that captures the place's character "
                            "('shrine clearing', 'north gate', 'old mill'). "
                            "NEVER write a sentence or description here — "
                            "no commas, no 'behind/inside/near' phrases."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "One sentence describing what the new location "
                            "looks/feels like, used to seed the world-state "
                            "entry. Example: 'A low-ceilinged stone chamber "
                            "lit by guttering torches, the air thick with "
                            "old incense.'"
                        ),
                    },
                },
                "required": ["location_name"],
            },
        },
    },
))

register(NarratorToolSpec(
    name="update_entity",
    tiers=_FULL_ONLY,
    converter=_convert_update_entity,
    effect_types=(EffectType.UPDATE_ENTITY,),
    schema={
        "type": "function",
        "function": {
            "name": "update_entity",
            "description": (
                "Declare a meaningful change to an existing tracked entity "
                "(an NPC, creature, or object already in the roster). Call "
                "this ONLY when your prose changed something specific about "
                "the entity in this turn — their disposition shifted, they "
                "became important to the campaign, they died/fled/were "
                "captured, new descriptive detail was revealed, or items "
                "moved into or out of their possession.\n\n"
                "Do NOT call to merely reference an existing entity (use "
                "ref_entity for that), to reaffirm existing state, to "
                "summarize, or for atmospheric flavor without state change.\n\n"
                "Pass ONLY the fields you are actually changing — every "
                "field is independent and absent fields mean 'no change'. "
                "Calling with only entity_id and no change fields is invalid "
                "and will be rejected; use ref_entity instead if you only "
                "referenced the entity.\n\n"
                "Use add_items / remove_items when an NPC's holdings change. "
                "This is what lets the world remember 'the innkeeper is "
                "holding my relic' twenty turns later. For player-side item "
                "movement use update_player; this tool is for the NPC side.\n\n"
                "Example calls:\n"
                "- After 'Brother Kael's robes drop to show the cult symbol': "
                "{entity_id: 'kael', disposition: 'hostile', importance: true}\n"
                "- After 'The captain slumps lifeless against the wall': "
                "{entity_id: 'captain_halloran', status: 'dead'}\n"
                "- After 'Marta tucks the herbal potion into her apron pocket': "
                "{entity_id: 'marta', add_items: ['herbal potion']}\n"
                "- After 'The innkeeper accepts the relic and locks it in his strongbox': "
                "{entity_id: 'innkeeper', add_items: ['ancient relic'], importance: true}\n"
                "- After 'The bandit drops his cutlass and runs': "
                "{entity_id: 'bandit_lead', remove_items: ['cutlass'], status: 'fled'}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": (
                            "ID of the existing entity from the roster "
                            "[id: ...] tag. Required."
                        ),
                    },
                    "importance": {
                        "type": "boolean",
                        "description": (
                            "Set to true when the entity has just become "
                            "plot-significant (a recurring quest-giver "
                            "revealed, a key ally promoted from background). "
                            "Set to false to demote. Omit unless importance "
                            "actually changed in this turn."
                        ),
                    },
                    "disposition": {
                        "type": "string",
                        "enum": ["friendly", "neutral", "unfriendly", "hostile", "allied"],
                        "description": (
                            "New disposition toward the party. Omit unless "
                            "disposition actually shifted in this turn."
                        ),
                    },
                    "status": {
                        "type": "string",
                        "enum": ["alive", "wounded", "unconscious", "dead", "fled", "captured"],
                        "description": (
                            "New status of the entity. Omit unless status "
                            "changed. 'wounded' means visibly injured but "
                            "still active; 'unconscious' means down but "
                            "alive; 'dead' means killed; 'fled' means left "
                            "the scene under their own power; 'captured' "
                            "means restrained or in custody."
                        ),
                    },
                    "description_addition": {
                        "type": "string",
                        "description": (
                            "One short clause to append to the entity's "
                            "description, capturing newly-revealed detail "
                            "(a scar, a possession, an accent, a mannerism). "
                            "Omit if no new detail was revealed. Keep under "
                            "60 characters."
                        ),
                    },
                    "add_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Items the entity now holds (added to their "
                            "inventory). Use lowercase item names: "
                            "['ancient relic', 'sealed letter']. Omit if "
                            "no items were added this turn."
                        ),
                    },
                    "remove_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Items the entity gave away, lost, used, or "
                            "had taken from them this turn. Same format as "
                            "add_items. Omit if no items were removed."
                        ),
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
))

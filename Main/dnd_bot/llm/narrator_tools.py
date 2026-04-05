"""Narrator Tool Definitions — structured intent emission via tool calling.

Instead of parsing PROSE/INTENTS text blocks, the narrator emits
tool calls alongside its prose content. This is more reliable than
text formatting, especially on smaller local models.

The tool schemas map 1:1 to existing ProposedEffect types so the
entire downstream pipeline (validation, execution, world state sync)
is unchanged.

Two tool sets are exported:
- NARRATOR_TOOLS: Full set (9 tools) for capable models (Haiku, Sonnet)
- NARRATOR_TOOLS_CORE: Critical subset (3 tools) for local models
  that lose track of tools in long conversations. The state extractor
  and triage system handle the remaining effect types.
"""

from .effects import EffectType, ProposedEffect

import structlog

logger = structlog.get_logger()


# ======================================================================
# Tool Definitions (OpenAI/Ollama compatible format)
# ======================================================================

NARRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ref_entity",
            "description": (
                "Declare that you referenced an existing entity from the roster "
                "in your prose. Use for EVERY roster entity your narration mentions."
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
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_npc",
            "description": "Introduce a NEW NPC to the scene for the first time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "npc_id": {
                        "type": "string",
                        "description": "Short ID (e.g. merchant_1, guard_2)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Display name",
                    },
                    "disposition": {
                        "type": "string",
                        "enum": ["friendly", "neutral", "unfriendly", "hostile"],
                        "description": "NPC's attitude toward the party",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief physical/personality description",
                    },
                },
                "required": ["npc_id", "name", "disposition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_object",
            "description": "Place an object in the scene (treasure, item, interactable).",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {
                        "type": "string",
                        "description": "Short ID (e.g. dagger_1, chest_1)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Object name",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description",
                    },
                },
                "required": ["object_id", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "offer_item",
            "description": "NPC offers an item to the player (requires player confirmation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_entity": {
                        "type": "string",
                        "description": "Who is offering (e.g. npc:merchant)",
                    },
                    "to_entity": {
                        "type": "string",
                        "description": "Who receives (e.g. player)",
                    },
                    "item_name": {
                        "type": "string",
                        "description": "Item being offered",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "How many (default 1)",
                    },
                },
                "required": ["from_entity", "to_entity", "item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grant_currency",
            "description": "NPC gives currency to a target (requires player confirmation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Who receives (e.g. player)",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Amount of currency",
                    },
                    "denomination": {
                        "type": "string",
                        "enum": ["cp", "sp", "ep", "gp", "pp"],
                        "description": "Currency type",
                    },
                },
                "required": ["target", "amount", "denomination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_item",
            "description": "Direct item transfer (player picks up item, purchase completion).",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_entity": {
                        "type": "string",
                        "description": "Source (e.g. scene:dagger_1, npc:merchant)",
                    },
                    "to_entity": {
                        "type": "string",
                        "description": "Destination (e.g. player)",
                    },
                    "item_name": {
                        "type": "string",
                        "description": "Item being transferred",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "How many (default 1)",
                    },
                },
                "required": ["from_entity", "to_entity", "item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_damage",
            "description": "Environmental or trap damage (not combat attacks).",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Who takes damage (e.g. player)",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Damage amount",
                    },
                    "damage_type": {
                        "type": "string",
                        "description": "Type (fire, poison, bludgeoning, etc.)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "What caused the damage",
                    },
                },
                "required": ["target", "amount", "damage_type", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_roll",
            "description": "Request a dice roll from the player.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Who rolls (e.g. player)",
                    },
                    "roll_type": {
                        "type": "string",
                        "enum": ["save", "check", "skill"],
                        "description": "Type of roll",
                    },
                    "ability_or_skill": {
                        "type": "string",
                        "description": "Ability (constitution, dexterity) or skill (perception, stealth)",
                    },
                    "dc": {
                        "type": "integer",
                        "description": "Difficulty class (1-40)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the roll is needed",
                    },
                },
                "required": ["target", "roll_type", "ability_or_skill", "dc", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_combat",
            "description": "Initiate combat encounter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why combat starts",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]


# Core subset for local models (Ollama/Qwen) — fewer tools means the
# model pays more attention to each one in long conversations.
# The state extractor and triage system handle the remaining types.
NARRATOR_TOOLS_CORE = [
    t for t in NARRATOR_TOOLS
    if t["function"]["name"] in ("ref_entity", "add_npc", "spawn_object")
]


# ======================================================================
# Tool Call → ProposedEffect Converter
# ======================================================================

# Ability names for detecting save vs check
_ABILITIES = {"strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"}

# Currency denomination → ProposedEffect field mapping
_CURRENCY_FIELDS = {"cp": "copper", "sp": "silver", "ep": "electrum", "gp": "gold", "pp": "platinum"}


def tool_calls_to_effects(tool_calls: list[dict]) -> list[ProposedEffect]:
    """Convert narrator tool calls to ProposedEffect objects.

    Each tool call maps 1:1 to a ProposedEffect. Unknown tool names
    are logged and skipped.
    """
    effects = []

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        try:
            effect = _convert_tool_call(name, args)
            if effect:
                effects.append(effect)
        except Exception as e:
            logger.warning("tool_call_conversion_failed", tool=name, error=str(e))

    return effects


def _convert_tool_call(name: str, args: dict) -> ProposedEffect | None:
    """Convert a single tool call to a ProposedEffect."""

    if name == "ref_entity":
        return ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id=args.get("entity_id", ""),
            ref_alias_used=args.get("alias_used"),
        )

    elif name == "add_npc":
        return ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name=args.get("name", ""),
            npc_description=args.get("description"),
            npc_disposition=args.get("disposition", "neutral"),
            source=f"npc:{args.get('npc_id', 'unknown')}",
        )

    elif name == "spawn_object":
        return ProposedEffect(
            effect_type=EffectType.SPAWN_OBJECT,
            object_name=args.get("name", ""),
            object_description=args.get("description"),
            source=f"scene:{args.get('object_id', 'unknown')}",
        )

    elif name == "offer_item":
        return ProposedEffect(
            effect_type=EffectType.TRANSFER_ITEM,
            from_entity=args.get("from_entity", ""),
            to_entity=args.get("to_entity", "player"),
            item_name=args.get("item_name", ""),
            quantity=args.get("quantity", 1),
            requires_confirmation=True,
            confirmation_prompt=f"Accept {args.get('item_name', 'the item')}?",
        )

    elif name == "grant_currency":
        denom = args.get("denomination", "gp")
        amount = args.get("amount", 0)
        field = _CURRENCY_FIELDS.get(denom, "gold")
        currency = {field: amount}
        return ProposedEffect(
            effect_type=EffectType.GRANT_CURRENCY,
            target=args.get("target", "player"),
            requires_confirmation=True,
            confirmation_prompt=f"Accept {amount}{denom}?",
            **currency,
        )

    elif name == "transfer_item":
        return ProposedEffect(
            effect_type=EffectType.TRANSFER_ITEM,
            from_entity=args.get("from_entity", ""),
            to_entity=args.get("to_entity", "player"),
            item_name=args.get("item_name", ""),
            quantity=args.get("quantity", 1),
        )

    elif name == "apply_damage":
        return ProposedEffect(
            effect_type=EffectType.APPLY_DAMAGE,
            target=args.get("target", "player"),
            amount=args.get("amount", 0),
            damage_type=args.get("damage_type", ""),
            reason=args.get("reason", ""),
        )

    elif name == "request_roll":
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

    elif name == "start_combat":
        return ProposedEffect(
            effect_type=EffectType.START_COMBAT,
            reason=args.get("reason", "Combat begins!"),
        )

    else:
        logger.warning("unknown_narrator_tool", tool_name=name)
        return None

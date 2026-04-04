"""Intent Mini-Language Parser.

The narrator outputs a simple INTENTS block alongside prose. This module
parses those intents into ProposedEffect objects.

This is NOT LLM inference - it's deterministic string parsing.
The narrator explicitly signals intents at creation time.

## INTENTS Mini-Language Spec

Format: One intent per line, starting with intent type.

### Intent Types:

spawn_object <object_id> "<name>" ["<description>"]
    Creates an object in the scene (not inventory).
    Example: spawn_object ruby_1 "Ruby" "A glittering red gem"

add_npc <npc_id> "<name>" <disposition> ["<description>"]
    Introduces an NPC to the scene.
    Disposition: friendly | neutral | unfriendly | hostile
    Example: add_npc merchant_1 "Grizzled Merchant" neutral "A weathered trader"

offer_item <from>-><to> "<item>" [qty=N] [confirm]
    NPC offers item to player (requires confirmation by default).
    Example: offer_item npc:merchant->player "Healing Potion" confirm

grant_currency <target> <amount><denomination> [confirm]
    Grant currency. Denominations: cp, sp, ep, gp, pp
    Example: grant_currency player 15gp confirm

transfer_item <from>-><to> "<item>" [qty=N]
    Direct transfer (player pickup, purchase completion).
    Example: transfer_item scene:ruby_1->player "Ruby"

apply_damage <target> <amount> <type> "<reason>"
    Environmental/trap damage (not combat).
    Example: apply_damage player 5 fire "Touched the brazier"

request_roll <target> <roll_type> <ability_or_skill> dc=<N> "<reason>"
    DM-initiated roll request.
    Roll types: save, check, skill
    Example: request_roll player save constitution dc=15 "Resist the poison"
    Example: request_roll player skill perception dc=12 "Notice the tracks"

start_combat "<reason>"
    Initiates combat.
    Example: start_combat "The bandits attack!"

remove_entity <entity_id>
    Removes entity from scene.
    Example: remove_entity npc:merchant

set_flag <flag_name> <value>
    Sets a game flag.
    Example: set_flag quest_accepted true

NONE
    Explicitly indicates no mechanical effects.

## Parsing Rules

- Lines starting with # are comments (ignored)
- Empty lines are ignored
- Unrecognized lines are logged and skipped
- Quoted strings use double quotes, escaped with \"
- Entity references: player, pc:<name>, npc:<name>, scene, scene:<id>, creature:<name>
"""

import re
import shlex
from typing import Optional
from dataclasses import dataclass, field

import structlog

from .effects import ProposedEffect, EffectType

logger = structlog.get_logger()


@dataclass
class ParsedIntent:
    """A single parsed intent before conversion to ProposedEffect."""
    intent_type: str
    args: list[str]
    raw_line: str


@dataclass
class IntentParseResult:
    """Result of parsing an INTENTS block."""
    effects: list[ProposedEffect] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_intents: str = ""
    had_none: bool = False  # Explicitly had NONE marker


def validate_narrator_format(response: str) -> bool:
    """
    Validate that narrator response follows the PROSE:/INTENTS: contract.

    Accepts two formats:
    - Text markers: PROSE: ... INTENTS: ... (Qwen/Groq/OpenRouter)
    - XML tags: <prose>...</prose> <intents>...</intents> (Claude/Anthropic)

    Returns True if response contains valid prose output in either format.
    """
    if not response:
        return False

    # Normalize and strip code block wrappers
    normalized = response.strip()

    # Strip leading code block markers (```, ```text, etc.)
    if normalized.startswith("```"):
        first_newline = normalized.find("\n")
        if first_newline != -1:
            normalized = normalized[first_newline + 1:]
        if normalized.rstrip().endswith("```"):
            normalized = normalized.rstrip()[:-3]
        normalized = normalized.strip()

    # Accept XML format: <prose>...</prose>
    if "<prose>" in normalized.lower():
        import structlog
        structlog.get_logger().info("narrator_format_ok", format="xml")
        return True

    # Accept text marker format: PROSE: (case-insensitive)
    if normalized.upper().startswith("PROSE:"):
        import structlog
        structlog.get_logger().info("narrator_format_ok", format="text")
        return True

    return False


def strip_planning_text_fallback(text: str) -> tuple[str, bool]:
    """
    Last-resort fallback to strip obvious planning text.

    Only use this after reprompting has failed. Returns (cleaned_text, did_strip).
    The did_strip flag should trigger logging for prompt improvement.

    WARNING: This can accidentally strip legitimate narration like dialogue
    starting with "We are..." or "Let me...". Use sparingly.
    """
    # Very conservative patterns - only obviously meta content
    meta_patterns = [
        "we are given", "we are to ", "we are not to", "we must narrate",
        "we need to", "we should ", "we cannot ", "we don't ",
        "the resolution ", "authorized reveal", "important:",
        "(1/2)", "(2/2)", "let's write:", "revised prose:",
    ]

    lines = text.split("\n")
    cleaned_lines = []
    stripped_count = 0
    in_meta = True

    for line in lines:
        line_lower = line.lower().strip()

        # Check if this line is obviously meta (not narrative)
        is_meta = False
        if in_meta and line_lower:
            if any(pattern in line_lower for pattern in meta_patterns):
                is_meta = True
                stripped_count += 1

        if is_meta:
            continue
        elif line_lower:  # Non-empty, non-meta line
            in_meta = False
            cleaned_lines.append(line)
        elif not in_meta:  # Empty line after real content starts
            cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    did_strip = stripped_count > 0

    return result, did_strip


def extract_intents_block(response: str) -> tuple[str, str]:
    """
    Extract PROSE and INTENTS blocks from narrator response.

    Supports two formats:
    - Text markers: PROSE: ... INTENTS: ... (Qwen/Groq/OpenRouter)
    - XML tags: <prose>...</prose> <intents>...</intents> (Claude/Anthropic)

    Returns:
        Tuple of (prose, intents_block)
    """
    prose = ""
    intents = ""

    # Normalize line endings
    response = response.replace("\r\n", "\n")

    # Strip code block wrappers (LLMs sometimes wrap output in ```)
    response = response.strip()
    if response.startswith("```"):
        first_newline = response.find("\n")
        if first_newline != -1:
            response = response[first_newline + 1:]
        if response.rstrip().endswith("```"):
            response = response.rstrip()[:-3]
        response = response.strip()

    # Try XML format first: <prose>...</prose> <intents>...</intents>
    prose_xml = re.search(r'<prose>(.*?)</prose>', response, re.DOTALL | re.IGNORECASE)
    if prose_xml:
        prose = prose_xml.group(1).strip()
        intents_xml = re.search(r'<intents>(.*?)</intents>', response, re.DOTALL | re.IGNORECASE)
        if intents_xml:
            intents = intents_xml.group(1).strip()
        import structlog
        structlog.get_logger().debug(
            "narrator_xml_format_parsed",
            prose_length=len(prose),
            has_intents=bool(intents),
        )
        return prose, intents

    # Find INTENTS marker with various formats:
    # - "INTENTS:" (preferred)
    # - "INTENTS" (without colon, on its own line)
    # - "## INTENTS" (markdown header)
    # - "**INTENTS**" (bold)
    intents_pattern = re.compile(
        r'^(?:\*\*)?(?:##?\s*)?INTENTS(?:\*\*)?:?\s*$',
        re.MULTILINE | re.IGNORECASE
    )

    # Find PROSE marker with various formats
    prose_pattern = re.compile(
        r'^(?:\*\*)?(?:##?\s*)?PROSE(?:\*\*)?:?\s*$',
        re.MULTILINE | re.IGNORECASE
    )

    # Search for INTENTS marker
    intents_match = intents_pattern.search(response)

    if intents_match:
        # Split at the INTENTS marker
        before_intents = response[:intents_match.start()]
        intents_section = response[intents_match.end():]

        # Check if there's a PROSE: marker before INTENTS
        prose_match = prose_pattern.search(before_intents)
        if prose_match:
            prose = before_intents[prose_match.end():].strip()
        else:
            prose = before_intents.strip()

        # Intents section goes until end or another marker
        intents = intents_section.strip()

        # Remove any trailing markers (PROSE, markdown separators, code blocks)
        for marker in ["PROSE:", "PROSE", "---", "```"]:
            if marker in intents:
                intents = intents.split(marker)[0].strip()
    else:
        # No INTENTS marker - check for PROSE: marker anyway
        prose_match = prose_pattern.search(response)
        if prose_match:
            prose = response[prose_match.end():].strip()
        else:
            prose = response.strip()
        intents = ""

    return prose, intents


def parse_intents(intents_block: str) -> IntentParseResult:
    """
    Parse an INTENTS block into ProposedEffect objects.

    This is deterministic string parsing, NOT LLM inference.
    """
    result = IntentParseResult(raw_intents=intents_block)

    if not intents_block.strip():
        return result

    lines = intents_block.strip().split("\n")

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Handle bullet points
        if line.startswith("- "):
            line = line[2:].strip()
        elif line.startswith("* "):
            line = line[2:].strip()

        # Check for NONE marker
        if line.upper() == "NONE":
            result.had_none = True
            continue

        # Parse the intent
        try:
            effect = _parse_intent_line(line)
            if effect:
                result.effects.append(effect)
        except Exception as e:
            result.errors.append(f"Failed to parse: {line!r} - {e}")
            logger.warning("intent_parse_error", line=line, error=str(e))

    return result


def _parse_intent_line(line: str) -> Optional[ProposedEffect]:
    """Parse a single intent line into a ProposedEffect."""
    # Tokenize respecting quoted strings
    try:
        tokens = shlex.split(line)
    except ValueError:
        # Fallback for malformed quotes
        tokens = line.split()

    if not tokens:
        return None

    intent_type = tokens[0].lower()
    args = tokens[1:]

    parsers = {
        "spawn_object": _parse_spawn_object,
        "add_npc": _parse_add_npc,
        "offer_item": _parse_offer_item,
        "grant_currency": _parse_grant_currency,
        "transfer_item": _parse_transfer_item,
        "apply_damage": _parse_apply_damage,
        "request_roll": _parse_request_roll,
        "start_combat": _parse_start_combat,
        "remove_entity": _parse_remove_entity,
        "set_flag": _parse_set_flag,
    }

    parser = parsers.get(intent_type)
    if not parser:
        logger.warning("unknown_intent_type", intent_type=intent_type, line=line)
        return None

    return parser(args, line)


def _parse_spawn_object(args: list[str], line: str) -> Optional[ProposedEffect]:
    """spawn_object <id> "<name>" ["<description>"]"""
    if len(args) < 2:
        raise ValueError("spawn_object requires id and name")

    obj_id = args[0]
    obj_name = args[1]
    obj_desc = args[2] if len(args) > 2 else None

    return ProposedEffect(
        effect_type=EffectType.SPAWN_OBJECT,
        object_name=obj_name,
        object_description=obj_desc,
        object_properties={"id": obj_id},
    )


def _parse_add_npc(args: list[str], line: str) -> Optional[ProposedEffect]:
    """add_npc <id> "<name>" <disposition> ["<description>"]"""
    if len(args) < 2:
        raise ValueError("add_npc requires at least id and name")

    valid_dispositions = {"hostile", "unfriendly", "neutral", "friendly", "allied"}

    npc_id = args[0]
    npc_name = args[1]

    # Handle flexible argument order - disposition might be missing or description in its place
    disposition = "neutral"  # Default
    npc_desc = None

    if len(args) >= 3:
        # Check if args[2] is a valid disposition or a description
        if args[2].lower() in valid_dispositions:
            disposition = args[2].lower()
            npc_desc = args[3] if len(args) > 3 else None
        else:
            # args[2] is likely the description, disposition was omitted
            npc_desc = args[2]
            logger.warning(
                "add_npc_disposition_missing",
                line=line,
                assumed_disposition="neutral",
            )

    return ProposedEffect(
        effect_type=EffectType.ADD_NPC,
        npc_name=npc_name,
        npc_description=npc_desc,
        npc_disposition=disposition,
        source=f"npc:{npc_id}",
    )


def _parse_offer_item(args: list[str], line: str) -> Optional[ProposedEffect]:
    """offer_item <from>-><to> "<item>" [qty=N] [confirm]"""
    if len(args) < 2:
        raise ValueError("offer_item requires from->to and item")

    # Parse from->to
    transfer_spec = args[0]
    if "->" in transfer_spec:
        from_entity, to_entity = transfer_spec.split("->", 1)
    else:
        raise ValueError(f"Invalid transfer spec: {transfer_spec}")

    item_name = args[1]
    quantity = 1
    requires_confirm = True  # Default for offers

    for arg in args[2:]:
        if arg.startswith("qty="):
            quantity = int(arg[4:])
        elif arg.lower() == "confirm":
            requires_confirm = True
        elif arg.lower() == "noconfirm":
            requires_confirm = False

    return ProposedEffect(
        effect_type=EffectType.TRANSFER_ITEM,
        from_entity=from_entity,
        to_entity=to_entity,
        item_name=item_name,
        quantity=quantity,
        requires_confirmation=requires_confirm,
        confirmation_prompt=f"Accept {item_name}?",
    )


def _parse_grant_currency(args: list[str], line: str) -> Optional[ProposedEffect]:
    """grant_currency <target> <amount><denom> [confirm]"""
    if len(args) < 2:
        raise ValueError("grant_currency requires target and amount")

    target = args[0]
    amount_str = args[1]

    # Parse amount and denomination
    match = re.match(r"(\d+)(cp|sp|ep|gp|pp)", amount_str.lower())
    if not match:
        raise ValueError(f"Invalid currency format: {amount_str}")

    amount = int(match.group(1))
    denom = match.group(2)

    requires_confirm = "confirm" in [a.lower() for a in args[2:]]

    effect = ProposedEffect(
        effect_type=EffectType.GRANT_CURRENCY,
        target=target,
        requires_confirmation=requires_confirm,
    )

    # Set the appropriate currency field
    if denom == "cp":
        effect.copper = amount
    elif denom == "sp":
        effect.silver = amount
    elif denom == "ep":
        effect.electrum = amount
    elif denom == "gp":
        effect.gold = amount
    elif denom == "pp":
        effect.platinum = amount

    if requires_confirm:
        effect.confirmation_prompt = f"Accept {amount}{denom}?"

    return effect


def _parse_transfer_item(args: list[str], line: str) -> Optional[ProposedEffect]:
    """transfer_item <from>-><to> "<item>" [qty=N]"""
    if len(args) < 2:
        raise ValueError("transfer_item requires from->to and item")

    transfer_spec = args[0]
    if "->" in transfer_spec:
        from_entity, to_entity = transfer_spec.split("->", 1)
    else:
        raise ValueError(f"Invalid transfer spec: {transfer_spec}")

    item_name = args[1]
    quantity = 1

    for arg in args[2:]:
        if arg.startswith("qty="):
            quantity = int(arg[4:])

    return ProposedEffect(
        effect_type=EffectType.TRANSFER_ITEM,
        from_entity=from_entity,
        to_entity=to_entity,
        item_name=item_name,
        quantity=quantity,
        requires_confirmation=False,  # Direct transfers don't need confirmation
    )


def _parse_apply_damage(args: list[str], line: str) -> Optional[ProposedEffect]:
    """apply_damage <target> <amount> <type> "<reason>" """
    if len(args) < 4:
        raise ValueError("apply_damage requires target, amount, type, reason")

    target = args[0]
    amount = int(args[1])
    damage_type = args[2]
    reason = args[3]

    return ProposedEffect(
        effect_type=EffectType.APPLY_DAMAGE,
        target=target,
        amount=amount,
        damage_type=damage_type,
        reason=reason,
    )


def _parse_request_roll(args: list[str], line: str) -> Optional[ProposedEffect]:
    """request_roll <target> <type> <ability_or_skill> dc=<N> "<reason>" """
    if len(args) < 4:
        raise ValueError("request_roll requires target, type, ability/skill, dc")

    target = args[0]
    roll_type_raw = args[1].lower()
    ability_or_skill = args[2].lower()

    # Map to standard roll types
    roll_type_map = {
        "save": "saving_throw",
        "saving_throw": "saving_throw",
        "check": "ability_check",
        "ability_check": "ability_check",
        "skill": "skill_check",
        "skill_check": "skill_check",
    }
    roll_type = roll_type_map.get(roll_type_raw, "skill_check")

    # Find DC
    dc = None
    reason = None
    for arg in args[3:]:
        if arg.lower().startswith("dc="):
            dc = int(arg[3:])
        elif not arg.startswith("dc="):
            reason = arg

    if dc is None:
        raise ValueError("request_roll requires dc=N")

    # Determine if it's ability or skill
    abilities = {"strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma",
                 "str", "dex", "con", "int", "wis", "cha"}

    effect = ProposedEffect(
        effect_type=EffectType.REQUEST_ROLL,
        target=target,
        roll_type=roll_type,
        dc=dc,
        roll_reason=reason,
    )

    if ability_or_skill in abilities:
        effect.ability = ability_or_skill
    else:
        effect.skill = ability_or_skill

    return effect


def _parse_start_combat(args: list[str], line: str) -> Optional[ProposedEffect]:
    """start_combat "<reason>" """
    reason = args[0] if args else "Combat begins!"

    return ProposedEffect(
        effect_type=EffectType.START_COMBAT,
        reason=reason,
    )


def _parse_remove_entity(args: list[str], line: str) -> Optional[ProposedEffect]:
    """remove_entity <entity_id>"""
    if not args:
        raise ValueError("remove_entity requires entity_id")

    return ProposedEffect(
        effect_type=EffectType.REMOVE_ENTITY,
        target=args[0],
    )


def _parse_set_flag(args: list[str], line: str) -> Optional[ProposedEffect]:
    """set_flag <flag_name> <value>"""
    if len(args) < 2:
        raise ValueError("set_flag requires flag_name and value")

    flag_name = args[0]
    value_str = args[1].lower()

    # Parse value
    if value_str == "true":
        value = True
    elif value_str == "false":
        value = False
    elif value_str.isdigit():
        value = int(value_str)
    else:
        value = args[1]  # Keep as string

    return ProposedEffect(
        effect_type=EffectType.SET_FLAG,
        flag_name=flag_name,
        flag_value=value,
    )

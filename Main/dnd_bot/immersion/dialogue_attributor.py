"""LLM-based dialogue attribution: determines which entity speaks each quote block.

Given narrator prose and the list of entities in the scene, asks a cheap LLM
to return structured JSON mapping each quoted block to its speaker.
Replaces fragile regex-based attribution.
"""

import json
import re

import structlog

logger = structlog.get_logger()

_ATTRIBUTION_PROMPT = """You are a highly precise dialogue attribution engine. Your exact task is to map {num_quotes} quoted dialogue blocks to the characters who speak them.

INPUT CONTEXT:
Characters Present & Narrator Hints (Intended Turn Order):
{entities}

Total Quote Blocks Detected: {num_quotes}

CRITICAL RULES - STRICTLY ENFORCED:
1. THE "OFF-SCRIPT" RULE: The Narrator Hints are UNRELIABLE. A character listed as speaking might not actually have any dialogue in the prose. A character NOT listed might speak. The hints sometimes confuse "mentioned in narration" with "speaks dialogue." ALWAYS determine the speaker from the prose itself -- who is described speaking, what pronouns are used (he/she/they), what gender matches. The hints are a loose guide, nothing more.
2. THE MONOLOGUE RULE: A single character can speak multiple times in a row. If a character's speech is broken up by their own physical actions (e.g., "Line 1." He steps forward. "Line 2." He draws a sword. "Line 3."), they own ALL of those blocks.
3. SPLIT QUOTES: "Look out," he yelled, "it's a trap!" counts as TWO quote blocks. You must list the same speaker twice in a row: ["Speaker Name", "Speaker Name"].
4. EXACT MATCH: Output the character name exactly as it appears in the list. Output null only if it is completely impossible to deduce.
5. NO FORMATTING: Output ONLY a raw JSON array of exactly {num_quotes} string elements. No markdown blocks, no explanations, no conversational text.

PROSE TO ANALYZE:
{prose}

Output the JSON array of exactly {num_quotes} elements now:"""


async def attribute_dialogue(
    narrative: str,
    scene_entities: list,
    player_characters: list = None,
    proposed_effects: list = None,
) -> dict[int, str]:
    """Use the brain LLM to attribute each quote block to a speaker.

    Args:
        narrative: Full narrator prose.
        scene_entities: List of SceneEntity objects in the scene.
        player_characters: List of Character objects.

    Returns:
        Dict mapping 1-indexed quote number to entity name.
    """
    # Find all quote blocks
    quote_pattern = re.compile(r'[\u201c"]\s*(.+?)\s*[\u201d"]', re.DOTALL)
    quotes = [(m.start(), m.end(), m.group(1)) for m in quote_pattern.finditer(narrative)
              if len(m.group(1).strip()) > 2]

    if not quotes:
        return {}

    # Build entity list with full details so the LLM can reason about pronouns
    entity_names = []
    for e in scene_entities:
        if hasattr(e, 'entity_type') and e.entity_type.value == "npc":
            parts = [e.name]
            # Extract gender from description prefix if present
            desc = getattr(e, 'description', '') or ''
            if desc.startswith("[male]"):
                parts.append("(male)")
                desc = desc[6:].strip()
            elif desc.startswith("[female]"):
                parts.append("(female)")
                desc = desc[8:].strip()
            if desc:
                parts.append(f"- {desc[:100]}")
            disposition = getattr(e, 'disposition', None)
            if disposition:
                parts.append(f"[{disposition.value if hasattr(disposition, 'value') else disposition}]")
            entity_names.append(" ".join(parts))

    for char in (player_characters or []):
        parts = [f"{char.name} (player character)"]
        if char.race_index:
            parts.append(f"- {char.race_index} {char.class_index}")
        entity_names.append(" ".join(parts))

    # Add narrator hints about who speaks and when (rough order, not exact block mapping)
    from ..llm.effects import EffectType
    narrator_hints = []
    for effect in (proposed_effects or []):
        indices = getattr(effect, 'dialogue_indices', [])
        emotions = getattr(effect, 'dialogue_emotions', [])
        if indices or emotions:
            name = None
            if effect.effect_type == EffectType.ADD_NPC:
                name = effect.npc_name
            elif effect.effect_type == EffectType.REF_ENTITY:
                name = effect.ref_alias_used or effect.ref_entity_id
            if name:
                hint = f"{name} speaks"
                if indices:
                    hint += f" (roughly speeches {indices})"
                if emotions:
                    hint += f" with emotions: {', '.join(emotions)}"
                narrator_hints.append(hint)

    if narrator_hints:
        entity_names.append("\nNarrator's rough speaking order (may not match exact quote blocks):")
        for h in narrator_hints:
            entity_names.append(f"  - {h}")

    if not entity_names:
        return {}

    entities_str = "\n".join(f"- {name}" for name in entity_names)

    prompt = _ATTRIBUTION_PROMPT.format(
        entities=entities_str,
        prose=narrative[:3000],
        num_quotes=len(quotes),
    )

    try:
        from ..llm.client import get_llm_client
        client = get_llm_client()

        response = await client.chat(
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0,
            think=False,
        )

        content = response.content if hasattr(response, 'content') else str(response)
        content = content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
            content = content.strip()

        attributions = json.loads(content)

        if not isinstance(attributions, list):
            logger.warning("dialogue_attribution_invalid_format", content=content[:100])
            return {}

        # Build mapping: 1-indexed quote number -> entity name
        result = {}
        for i, speaker in enumerate(attributions):
            if speaker and isinstance(speaker, str) and speaker.lower() != "null":
                result[i + 1] = speaker

        logger.info(
            "dialogue_attributed",
            total_quotes=len(quotes),
            attributed=len(result),
            speakers=list(set(result.values())),
        )

        return result

    except Exception as e:
        logger.warning("dialogue_attribution_failed", error=str(e), exc_info=True)
        return {}

"""Narrative prose parser: splits narrator output into voice segments.

Parses quoted dialogue and narration from narrator prose. Uses regex
for quote detection and entity name matching for speaker attribution.
No LLM call needed -- cross-references with ref_entity tool calls
and scene registry for accurate speaker identification.
"""

import re
from typing import Optional

import structlog

from ..models.immersion import NarrativeSegment, SegmentType
from ..models.character import Character

logger = structlog.get_logger()

# Patterns for quoted dialogue (straight quotes, curly quotes, single quotes around dialogue)
# Match only double-quoted dialogue (straight or curly)
# Single quotes are too ambiguous (contractions like she's, don't, etc.)
_QUOTE_PATTERN = re.compile(
    r'[\u201c"]\s*(.+?)\s*[\u201d"]'    # "..." or \u201c...\u201d
    , re.DOTALL
)

# Pattern to find capitalized proper names (multi-word)
_PROPER_NAME_MULTI = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
# Pattern for single capitalized words (potential single-word names)
_PROPER_NAME_SINGLE = re.compile(r'\b([A-Z][a-z]{2,})\b')


async def parse_narrative_async(
    narrative: str,
    proposed_effects: list = None,
    scene_registry=None,
    player_characters: list[Character] = None,
) -> list[NarrativeSegment]:
    """Async version that uses LLM attribution when entities are available."""
    if not narrative or not narrative.strip():
        return []

    proposed_effects = proposed_effects or []
    player_characters = player_characters or []
    scene_entities = scene_registry.get_all() if scene_registry else []

    # Try LLM-based attribution first (most accurate)
    llm_attribution = {}
    if scene_entities:
        try:
            from .dialogue_attributor import attribute_dialogue
            llm_attribution = await attribute_dialogue(
                narrative, scene_entities, player_characters, proposed_effects,
            )
        except Exception as e:
            import structlog
            structlog.get_logger().debug("llm_attribution_skipped", error=str(e))

    return _parse_with_attribution(
        narrative, proposed_effects, scene_registry,
        player_characters, llm_attribution,
    )


def parse_narrative(
    narrative: str,
    proposed_effects: list = None,
    scene_registry=None,
    player_characters: list[Character] = None,
) -> list[NarrativeSegment]:
    """Synchronous version (falls back to regex/tool-call attribution)."""
    if not narrative or not narrative.strip():
        return []
    proposed_effects = proposed_effects or []
    player_characters = player_characters or []
    return _parse_with_attribution(
        narrative, proposed_effects, scene_registry, player_characters, {},
    )


def _parse_with_attribution(
    narrative: str,
    proposed_effects: list,
    scene_registry,
    player_characters: list[Character],
    llm_attribution: dict[int, str],
) -> list[NarrativeSegment]:
    """Core parsing logic with attribution from LLM, tool calls, or regex.

    Args:
        narrative: Full narrator prose text.
        proposed_effects: ProposedEffect list from narrator tool calls.
        scene_registry: SceneEntityRegistry for name matching.
        player_characters: List of player Characters in the session.
        llm_attribution: Dict of 1-indexed quote -> entity name from LLM.

    Returns:
        Ordered list of NarrativeSegment objects.
    """
    # Build dialogue attribution map from tool calls (fallback)
    dialogue_map = _build_dialogue_map(proposed_effects, scene_registry)

    # Find all quoted dialogue with their positions
    quotes = _find_quotes(narrative)

    if not quotes:
        # No dialogue -- entire text is narration
        return [NarrativeSegment(text=narrative.strip(), segment_type=SegmentType.NARRATION)]

    segments: list[NarrativeSegment] = []
    last_end = 0

    for quote_idx, (quote_start, quote_end, dialogue_text) in enumerate(quotes, start=1):
        # Add narration before this quote (if any)
        narration_text = narrative[last_end:quote_start].strip()
        if narration_text:
            segments.append(NarrativeSegment(
                text=narration_text,
                segment_type=SegmentType.NARRATION,
            ))

        # Speaker attribution: tool calls first, regex fallback
        speaker_name = None
        speaker_entity_id = None
        emotion = None

        # Attribution priority: LLM > tool calls > regex
        if quote_idx in llm_attribution:
            # LLM told us who speaks this block
            llm_speaker = llm_attribution[quote_idx]
            speaker_name = llm_speaker
            speaker_entity_id = None
            if scene_registry:
                entity = scene_registry.get_by_name(llm_speaker)
                if entity:
                    speaker_name = entity.name
                    speaker_entity_id = entity.npc_id or entity.id
            # Check if tool calls had emotion for this speaker
            if quote_idx in dialogue_map:
                _, _, emotion = dialogue_map[quote_idx]

        elif quote_idx in dialogue_map:
            speaker_name, speaker_entity_id, emotion = dialogue_map[quote_idx]

        else:
            # Regex fallback: check after then before
            context_after = narrative[quote_end:min(len(narrative), quote_end + 80)]
            speaker_name, speaker_entity_id = _identify_speaker_after(
                context_after, scene_registry, player_characters,
            )
            if not speaker_name:
                context_before = narrative[max(0, quote_start - 150):quote_start]
                speaker_name, speaker_entity_id = _identify_speaker(
                    context_before, {}, scene_registry, player_characters,
                )

        segments.append(NarrativeSegment(
            text=dialogue_text,
            segment_type=SegmentType.DIALOGUE,
            speaker_name=speaker_name,
            speaker_entity_id=speaker_entity_id,
            emotion=emotion,
        ))

        last_end = quote_end

    # Add trailing narration after the last quote
    trailing = narrative[last_end:].strip()
    if trailing:
        segments.append(NarrativeSegment(
            text=trailing,
            segment_type=SegmentType.NARRATION,
        ))

    # Post-process: carry forward + scene-aware attribution for unattributed dialogue.
    #
    # Pass 1: Carry forward -- short narration between quotes doesn't break speaker.
    # Pass 2: Scene-aware -- if still unattributed and we have scene entities,
    #          try to find the speaker from the narration context around the quote.
    #          Last resort: if only one entity ref'd by the narrator hasn't been
    #          assigned any quotes yet, it's probably them.

    # Build set of entities the narrator referenced (from proposed_effects)
    referenced_entity_ids = set()
    for effect in proposed_effects:
        if hasattr(effect, 'ref_entity_id') and effect.ref_entity_id:
            referenced_entity_ids.add(effect.ref_entity_id)
        if hasattr(effect, 'npc_name') and effect.npc_name:
            referenced_entity_ids.add(effect.npc_name.lower())

    # Pass 1: Carry forward
    last_speaker_name = None
    last_speaker_id = None
    last_was_dialogue = False
    for seg in segments:
        if seg.segment_type == SegmentType.DIALOGUE:
            if seg.speaker_name:
                last_speaker_name = seg.speaker_name
                last_speaker_id = seg.speaker_entity_id
                last_was_dialogue = True
            elif last_speaker_name and last_was_dialogue:
                seg.speaker_name = last_speaker_name
                seg.speaker_entity_id = last_speaker_id
                last_was_dialogue = True
            else:
                last_was_dialogue = True
        elif seg.segment_type == SegmentType.NARRATION:
            if len(seg.text) > 80:
                last_speaker_name = None
                last_speaker_id = None
            last_was_dialogue = False

    # Pass 2: Scene-aware fallback for still-unattributed dialogue
    if scene_registry:
        # Get all entities with voices (NPCs that can speak)
        all_entities = [e for e in (scene_registry.get_all() if scene_registry else [])
                       if e.entity_type.value == "npc"]

        # Find which entities already have quotes attributed
        attributed_ids = {seg.speaker_entity_id for seg in segments
                         if seg.segment_type == SegmentType.DIALOGUE and seg.speaker_entity_id}

        for seg in segments:
            if seg.segment_type == SegmentType.DIALOGUE and not seg.speaker_name:
                # Try regex on nearby narration for entity name
                seg_idx = segments.index(seg)

                # Check narration before and after this dialogue
                for offset in [-1, 1]:
                    check_idx = seg_idx + offset
                    if 0 <= check_idx < len(segments) and segments[check_idx].segment_type == SegmentType.NARRATION:
                        narr_text = segments[check_idx].text
                        for entity in all_entities:
                            if entity.name.lower() in narr_text.lower():
                                seg.speaker_name = entity.name
                                seg.speaker_entity_id = entity.npc_id or entity.id
                                break
                    if seg.speaker_name:
                        break

                # Last resort: if only one unattributed entity remains, assign it
                if not seg.speaker_name and all_entities:
                    unattributed = [e for e in all_entities
                                   if (e.npc_id or e.id) not in attributed_ids]
                    if len(unattributed) == 1:
                        seg.speaker_name = unattributed[0].name
                        seg.speaker_entity_id = unattributed[0].npc_id or unattributed[0].id

    logger.debug(
        "prose_parsed",
        total_segments=len(segments),
        dialogue_count=sum(1 for s in segments if s.segment_type == SegmentType.DIALOGUE),
        narration_count=sum(1 for s in segments if s.segment_type == SegmentType.NARRATION),
    )

    return segments


def _build_dialogue_map(
    proposed_effects: list,
    scene_registry,
) -> dict[int, tuple[str, str, Optional[str]]]:
    """Build a map of quote_index -> (speaker_name, entity_id, emotion) from tool call effects.

    The narrator declares dialogue_indices and dialogue_emotions in ref_entity
    and add_npc tool calls, telling us exactly which quotes belong to which
    entity and how they should be delivered.
    """
    from ..llm.effects import EffectType

    dialogue_map: dict[int, tuple[str, str, Optional[str]]] = {}

    for effect in proposed_effects:
        indices = getattr(effect, 'dialogue_indices', []) or []
        if not indices:
            continue

        emotions = getattr(effect, 'dialogue_emotions', []) or []

        name = None
        entity_id = None

        if effect.effect_type == EffectType.REF_ENTITY:
            entity_id = effect.ref_entity_id
            name = effect.ref_alias_used
            if scene_registry and entity_id:
                entity = scene_registry.get_by_name(entity_id)
                if entity:
                    name = entity.name
                    entity_id = entity.npc_id or entity.id
            if not name:
                name = entity_id

        elif effect.effect_type == EffectType.ADD_NPC:
            name = effect.npc_name
            entity_id = (effect.source or "").replace("npc:", "")

        if name:
            for i, idx in enumerate(indices):
                emotion = emotions[i] if i < len(emotions) else None
                dialogue_map[idx] = (name, entity_id, emotion)

    return dialogue_map


def _find_quotes(text: str) -> list[tuple[int, int, str]]:
    """Find all quoted dialogue in text. Returns (start, end, inner_text) tuples."""
    results = []
    for match in _QUOTE_PATTERN.finditer(text):
        dialogue = match.group(1)
        if dialogue and len(dialogue.strip()) > 2:  # Skip trivially short quotes
            results.append((match.start(), match.end(), dialogue.strip()))
    return results


def _extract_referenced_entities(proposed_effects: list) -> dict[str, str]:
    """Extract entity_id -> name mapping from ref_entity effects."""
    from ..llm.effects import EffectType

    entities = {}
    for effect in proposed_effects:
        if effect.effect_type == EffectType.REF_ENTITY and effect.ref_entity_id:
            # Use alias if available, otherwise the entity_id itself
            name = effect.ref_alias_used or effect.ref_entity_id.replace("-", " ").title()
            entities[effect.ref_entity_id] = name
        elif effect.effect_type == EffectType.ADD_NPC and effect.npc_name:
            source_id = (effect.source or "").replace("npc:", "")
            if source_id:
                entities[source_id] = effect.npc_name
    return entities


# Patterns for "she said" / "he rasps" style attribution AFTER a quote
_AFTER_QUOTE_PATTERN = re.compile(
    r'^\s*(?:,?\s*)?'  # optional comma/space after closing quote
    r'(?:(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+)?'  # optional proper name
    r'(?:she|he|they|the\s+\w+)\s+'  # pronoun or "the <noun>"
    r'(?:says?|said|rasps?|rasped|whispers?|whispered|shouts?|shouted|'
    r'gasps?|gasped|growls?|growled|calls?|called|murmurs?|murmured|'
    r'hisses?|hissed|barks?|barked|snaps?|snapped|pleads?|pleaded|'
    r'stammers?|stammered|cries?|cried|screams?|screamed|demands?|demanded)',
    re.IGNORECASE
)


def _identify_speaker_after(
    context_after: str,
    scene_registry,
    player_characters: list,
) -> tuple[Optional[str], Optional[str]]:
    """Try to identify speaker from text AFTER the quote ("she rasps", "Elaine says").

    This catches the common pattern: "dialogue," she said.
    """
    # Check for "Name said" pattern right after the quote
    # Name must start with uppercase (no IGNORECASE for the name part)
    name_after = re.match(
        r'^\s*,?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+'
        r'(?:says?|said|rasps?|rasped|whispers?|whispered|shouts?|shouted|'
        r'gasps?|gasped|growls?|growled|calls?|called|murmurs?|murmured|'
        r'hisses?|hissed|barks?|barked|snaps?|snapped|pleads?|pleaded|'
        r'stammers?|stammered|cries?|cried|screams?|screamed|demands?|demanded)',
        context_after,
    )
    _PRONOUNS = {"She", "He", "They", "It", "The", "Her", "His"}
    if name_after:
        candidate = name_after.group(1).strip()
        if candidate in _PRONOUNS:
            return None, None  # Pronoun, not a name -- let before-check handle it
        if scene_registry:
            entity = scene_registry.get_by_name(candidate)
            if entity:
                return entity.name, entity.npc_id or entity.id
        for char in (player_characters or []):
            if candidate.lower() in char.name.lower():
                return char.name, f"pc:{char.id}"
        return candidate, None

    # Check for pronoun + verb pattern and try to resolve from scene context
    # "she rasps" — look for the most recently mentioned female entity
    pronoun_match = re.match(
        r'^\s*,?\s*(?:she|he|they)\s+'
        r'(?:says?|said|rasps?|rasped|whispers?|whispered|shouts?|shouted|'
        r'gasps?|gasped|growls?|growled|murmurs?|murmured|'
        r'hisses?|hissed|barks?|barked|snaps?|snapped|pleads?|pleaded|'
        r'stammers?|stammered|cries?|cried|screams?|screamed|demands?|demanded)',
        context_after, re.IGNORECASE
    )
    if pronoun_match:
        # Can't resolve pronoun without more context -- return None
        # The carry-forward logic will handle this
        pass

    return None, None


def _identify_speaker(
    context_before: str,
    referenced_entities: dict[str, str],
    scene_registry,
    player_characters: list[Character],
) -> tuple[Optional[str], Optional[str]]:
    """Try to identify who is speaking based on context before a quote.

    Strategy: find ALL proper names in the context, take the LAST one
    (nearest to the quote), and match against known entities.

    Returns (speaker_name, speaker_entity_id) or (None, None).
    """
    # Find all multi-word proper names in context
    names = _PROPER_NAME_MULTI.findall(context_before)

    # Also find single-word capitalized names
    single_names = _PROPER_NAME_SINGLE.findall(context_before)

    # Build combined candidate list: single-word first, multi-word last
    # reversed() will try multi-word (more specific) before single-word
    all_candidates = single_names + list(names)

    if not all_candidates:
        return None, None

    # Try each name starting from the LAST (nearest to the quote)
    for candidate_name in reversed(all_candidates):
        candidate_name = candidate_name.strip()

        # Skip pronouns, articles, and common non-name words
        skip_words = {
            "The", "But", "And", "Then", "What", "Your", "You", "His", "Her", "Its",
            "She", "He", "They", "That", "This", "These", "Those", "Who", "Which",
            "Where", "When", "How", "Not", "All", "One", "Two", "Three",
            "Please", "Someone", "No", "Outside", "Inside", "Every", "Neither",
            "People", "Whatever", "Desperation", "Something",
        }
        if candidate_name.split()[0] in skip_words:
            continue

        # Try to match against scene registry
        if scene_registry:
            entity = scene_registry.get_by_name(candidate_name)
            if entity:
                return entity.name, entity.npc_id or entity.id

        # Try to match against player characters
        for char in player_characters:
            if candidate_name.lower() in char.name.lower() or char.name.lower() in candidate_name.lower():
                return char.name, f"pc:{char.id}"

        # Try to match against referenced entities from tool calls
        for entity_id, entity_name in referenced_entities.items():
            if (candidate_name.lower() in entity_name.lower()
                    or entity_name.lower() in candidate_name.lower()):
                return entity_name, entity_id

        # Found a proper name but couldn't match to any known entity -- skip it.
        # Unmatched names are unreliable (could be a place name, adjective, etc.)
        # Let carry-forward handle attribution instead.
        continue

    return None, None

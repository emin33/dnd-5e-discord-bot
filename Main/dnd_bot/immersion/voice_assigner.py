"""NPC voice auto-assignment from the voice catalog.

When a new NPC is created via the add_npc effect and has no voice_id,
this module picks a voice from the catalog based on the NPC's description
(gender/age keyword detection). No LLM call needed.
"""

import random
import re
from typing import Optional

import structlog

from ..data.repositories.immersion_repo import get_immersion_repo
from ..data.repositories.npc_repo import get_npc_repo

logger = structlog.get_logger()

# Keyword sets for gender/age detection from NPC descriptions
_FEMALE_KEYWORDS = {
    "she", "her", "woman", "female", "lady", "maiden", "girl", "queen",
    "princess", "duchess", "countess", "priestess", "sorceress", "witch",
    "matron", "mother", "sister", "daughter", "wife", "barmaid", "waitress",
}
_MALE_KEYWORDS = {
    "he", "him", "man", "male", "lord", "king", "prince", "duke",
    "count", "priest", "sorcerer", "wizard", "father", "brother",
    "son", "husband", "barman", "bartender",
}
_YOUNG_KEYWORDS = {
    "young", "boy", "girl", "child", "teen", "teenager", "youth",
    "apprentice", "lad", "lass", "kid",
}
_OLD_KEYWORDS = {
    "old", "aged", "elder", "elderly", "ancient", "wizened", "grizzled",
    "gray-haired", "white-haired", "wrinkled", "venerable",
}


def _detect_gender(description: str) -> str:
    """Detect gender from NPC description. Returns 'male', 'female', or 'neutral'."""
    # Check for explicit gender tag from narrator tool (e.g. "[male] A scarred barkeep")
    if description.startswith("[male]"):
        return "male"
    if description.startswith("[female]"):
        return "female"

    words = set(re.findall(r'\b\w+\b', description.lower()))

    female_hits = len(words & _FEMALE_KEYWORDS)
    male_hits = len(words & _MALE_KEYWORDS)

    if female_hits > male_hits:
        return "female"
    elif male_hits > female_hits:
        return "male"
    return "neutral"


def _detect_age(description: str) -> str:
    """Detect age category from NPC description. Returns 'young', 'mature', or 'old'."""
    words = set(re.findall(r'\b\w+\b', description.lower()))

    if words & _YOUNG_KEYWORDS:
        return "young"
    elif words & _OLD_KEYWORDS:
        return "old"
    return "mature"


async def assign_voice(
    npc_description: str,
    scene_registry=None,
    npc_id: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[str]:
    """Auto-assign a voice from the catalog to an NPC.

    Args:
        npc_description: The NPC's description text.
        scene_registry: SceneEntityRegistry to check for voice collisions.
        npc_id: If provided, persist the voice_id to the NPC DB record.
        provider: Filter catalog to this provider only (e.g. "fish", "elevenlabs").

    Returns:
        The assigned voice_id, or None if no voices available.
    """
    gender = _detect_gender(npc_description)
    age = _detect_age(npc_description)

    repo = await get_immersion_repo()
    candidates = await repo.get_available_voices(gender=gender, age=age, provider=provider)

    # Fall back to just gender match if no age match
    if not candidates:
        candidates = await repo.get_available_voices(gender=gender, provider=provider)

    # Fall back to any voice for this provider
    if not candidates:
        candidates = await repo.get_available_voices(provider=provider)

    # Last resort: any voice at all
    if not candidates:
        candidates = await repo.get_all_voices()

    if not candidates:
        logger.warning("no_voices_in_catalog")
        return None

    # Exclude voices already in use in the current scene
    if scene_registry:
        used_voices = set()
        for entity in scene_registry.get_all():
            if getattr(entity, 'voice_id', None):
                used_voices.add(entity.voice_id)

        available = [v for v in candidates if v.voice_id not in used_voices]
        if available:
            candidates = available
        # If all voices are in use, allow reuse (better than silence)

    chosen = random.choice(candidates)

    # Persist to NPC record if we have an ID
    if npc_id:
        try:
            npc_repo = await get_npc_repo()
            await npc_repo.update_voice_id(npc_id, chosen.voice_id)
        except Exception as e:
            logger.warning("voice_persist_failed", npc_id=npc_id, error=str(e))

    logger.info(
        "voice_assigned",
        npc_id=npc_id,
        voice_name=chosen.name,
        voice_id=chosen.voice_id,
        detected_gender=gender,
        detected_age=age,
    )

    return chosen.voice_id

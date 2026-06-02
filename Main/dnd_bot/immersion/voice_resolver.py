"""Voice resolver: maps NarrativeSegments to TTS voice IDs.

For narration segments, uses the guild's configured narrator voice.
For dialogue segments, uses character_tts_provider override if set,
otherwise reads provider from the voice catalog entry. Not hardcoded
to any single provider.
"""

from typing import Optional

import structlog

from ..models.immersion import GuildImmersionSettings, NarrativeSegment, SegmentType
from ..models.character import Character
from ..data.repositories.npc_repo import get_npc_repo
from .voice_assigner import assign_voice

logger = structlog.get_logger()


async def resolve_voices(
    segments: list[NarrativeSegment],
    scene_registry=None,
    guild_settings: Optional[GuildImmersionSettings] = None,
    player_characters: list[Character] = None,
) -> list[NarrativeSegment]:
    """Fill in voice_id and voice_provider for each segment.

    Provider routing:
    - Narration: guild_settings.narrator_tts_provider (default: kokoro)
    - Dialogue: guild_settings.character_tts_provider if set,
                otherwise the voice catalog entry's provider (default: elevenlabs)
    """
    player_characters = player_characters or []

    # Narrator voice defaults
    narrator_provider = "kokoro"
    narrator_voice = "af_heart"
    if guild_settings:
        narrator_provider = guild_settings.narrator_tts_provider
        narrator_voice = guild_settings.narrator_tts_voice

    # Character voice provider override (empty = use per-voice catalog provider)
    character_provider_override = ""
    if guild_settings:
        character_provider_override = guild_settings.character_tts_provider

    npc_repo = await get_npc_repo()

    for segment in segments:
        if segment.segment_type == SegmentType.NARRATION:
            segment.voice_provider = narrator_provider
            segment.voice_id = narrator_voice

        elif segment.segment_type == SegmentType.DIALOGUE:
            voice_id, provider = await _resolve_dialogue_voice(
                segment, scene_registry, player_characters, npc_repo,
                character_provider_override,
                narrator_provider, narrator_voice,
            )
            segment.voice_id = voice_id
            segment.voice_provider = provider

    return segments


async def _get_voice_provider(voice_id: str, override: str) -> str:
    """Determine the provider for a voice_id.

    If override is set, use it. Otherwise look up the catalog entry.
    """
    if override:
        return override

    # Look up catalog entry to get its provider
    try:
        from ..data.repositories.immersion_repo import get_immersion_repo
        repo = await get_immersion_repo()
        entry = await repo.get_voice_by_id(voice_id)
        if entry:
            return entry.provider
    except Exception as e:
        logger.warning("voice_catalog_lookup_failed", voice_id=voice_id, error=str(e))

    return "elevenlabs"  # Default fallback


async def _resolve_dialogue_voice(
    segment: NarrativeSegment,
    scene_registry,
    player_characters: list[Character],
    npc_repo,
    character_provider_override: str,
    fallback_provider: str,
    fallback_voice: str,
) -> tuple[str, str]:
    """Resolve voice for a dialogue segment.

    Returns (voice_id, provider).
    """
    entity_id = segment.speaker_entity_id

    # Check if speaker is a player character (by entity_id or name match)
    is_player = False
    if entity_id and entity_id.startswith("pc:"):
        is_player = True
    elif segment.speaker_name:
        for char in player_characters:
            if segment.speaker_name.lower() in char.name.lower() or char.name.lower() in segment.speaker_name.lower():
                is_player = True
                break

    if is_player:
        # Player character -- use their voice if set, otherwise narrator
        for char in player_characters:
            if char.voice_id and (
                (entity_id and entity_id == f"pc:{char.id}") or
                (segment.speaker_name and segment.speaker_name.lower() in char.name.lower())
            ):
                provider = await _get_voice_provider(char.voice_id, character_provider_override)
                return char.voice_id, provider
        # No voice set -- use narrator voice
        return fallback_voice, fallback_provider

    # Check scene registry for NPC voice
    if scene_registry and segment.speaker_name:
        entity = scene_registry.get_by_name(segment.speaker_name)
        if entity:
            # Check if entity already has a voice_id
            if getattr(entity, 'voice_id', None):
                provider = await _get_voice_provider(entity.voice_id, character_provider_override)
                return entity.voice_id, provider

            # Try to load from persistent NPC record
            if entity.npc_id:
                npc = await npc_repo.get_by_id(entity.npc_id)
                if npc and npc.voice_id:
                    entity.voice_id = npc.voice_id
                    provider = await _get_voice_provider(npc.voice_id, character_provider_override)
                    return npc.voice_id, provider

            # Auto-assign a voice for this NPC
            description = getattr(entity, 'description', '') or ''
            voice_id = await assign_voice(
                npc_description=description,
                scene_registry=scene_registry,
                npc_id=entity.npc_id,
                provider=character_provider_override or None,
            )
            if voice_id:
                entity.voice_id = voice_id
                provider = await _get_voice_provider(voice_id, character_provider_override)
                return voice_id, provider

    # Fallback to narrator voice
    return fallback_voice, fallback_provider

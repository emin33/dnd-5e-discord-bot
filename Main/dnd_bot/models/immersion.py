"""Immersion feature data models: TTS voice assignment, image generation settings."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ImageFrequency(str, Enum):
    """How often to auto-generate scene images."""
    EVERY = "every"                # Every narrator response
    SCENE_CHANGE = "scene_change"  # New location or new NPC appears
    ON_DEMAND = "on_demand"        # Only via /imagine command


class GuildImmersionSettings(BaseModel):
    """Per-guild immersion feature configuration."""

    guild_id: int
    tts_enabled: bool = False
    image_enabled: bool = False
    image_frequency: ImageFrequency = ImageFrequency.ON_DEMAND
    narrator_tts_provider: str = "kokoro"
    narrator_tts_voice: str = "af_heart"
    character_tts_provider: str = ""  # Override for character dialogue (empty = use voice catalog provider)


class VoiceCatalogEntry(BaseModel):
    """A pre-defined TTS voice for NPC assignment."""

    voice_id: str
    name: str
    provider: str = "elevenlabs"
    gender: str = "neutral"   # male, female, neutral
    age: str = "mature"       # young, mature, old
    style_tags: list[str] = Field(default_factory=list)


class SegmentType(str, Enum):
    """Type of narrative segment for TTS."""
    NARRATION = "narration"
    DIALOGUE = "dialogue"


class NarrativeSegment(BaseModel):
    """A segment of narrator prose, tagged for TTS voice routing.

    The prose parser splits full narration into an ordered list of these.
    The voice resolver then fills in voice_id for each segment.
    """

    text: str
    segment_type: SegmentType
    speaker_entity_id: Optional[str] = None  # Scene entity slug for dialogue
    speaker_name: Optional[str] = None       # Display name of speaker

    # Filled by voice resolver
    voice_id: Optional[str] = None           # Provider-specific voice ID
    voice_provider: Optional[str] = None     # "elevenlabs", "openai", etc.

    # Emotion hint for providers that support it (Fish Speech, future others)
    emotion: Optional[str] = None            # "angry", "whispering", "excited", etc.

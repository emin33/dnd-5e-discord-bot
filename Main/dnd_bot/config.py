"""Configuration management using pydantic-settings + YAML profiles."""

from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# LLM Profile (loaded from config/profiles.yaml)
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider role."""
    provider: str = "ollama"     # ollama | groq | anthropic | deepseek | gemini | openrouter
    model: str = ""
    fallback_to_ollama: bool = False
    context_size: int = 0        # Ollama num_ctx cap (0 = model default)

    # Narrator tool tier — only consulted for the narrator role.
    # See dnd_bot.llm.narrator_tools.NARRATOR_TOOL_TIERS for the available
    # tiers ("core" / "core_plus" / "full"). Default "core" is safest.
    # Brain/triage roles ignore this field; they don't use narrator tools.
    tools: str = "core"

@dataclass
class MemoryConfig:
    """Memory system configuration."""
    buffer_size: int = 20           # Total message capacity
    verbatim_size: int = 8          # Tier 1: full prose messages
    condensed_size: int = 12        # Tier 2: per-exchange summary slots
    compaction_threshold: int = 6   # Tier 3: overflow before batch compaction

@dataclass
class TTSConfig:
    """Configuration for text-to-speech provider."""
    provider: str = "browser"  # browser | riva | openai | elevenlabs | kokoro
    model: str = ""            # e.g. "tts-1" for OpenAI
    voice: str = ""            # Voice name/ID (provider-specific)

@dataclass
class ImmersionConfig:
    """Configuration for immersion features (TTS voices + image generation)."""
    tts_enabled: bool = False
    image_enabled: bool = False
    image_frequency: str = "on_demand"  # every, scene_change, on_demand
    narrator_tts_provider: str = "kokoro"
    narrator_tts_voice: str = "af_heart"
    character_tts_provider: str = ""  # empty = use voice catalog provider
    image_provider: str = "fal"  # fal, openai, local
    image_model: str = ""  # HuggingFace model ID for local, or fal model path. Empty = use settings default.
    image_steps: int = 0  # Inference steps (0 = use settings default)
    image_guidance: float = 0.0  # Guidance scale (0 = use settings default)

@dataclass
class LLMProfile:
    """A named LLM configuration profile.

    Narrator tier slots (all optional; unset → fall back to ``narrator``):
    - ``narrator_premium``: used for high-significance turns (scene changes,
      new NPC introductions, combat starts/ends).
    - ``narrator_opening``: used for the session opener (``_generate_opening``).
      Falls back to ``narrator_premium`` if set, else ``narrator``.
    """
    name: str = "default"
    narrator: ProviderConfig = field(default_factory=ProviderConfig)
    narrator_premium: Optional[ProviderConfig] = None
    narrator_opening: Optional[ProviderConfig] = None
    brain: ProviderConfig = field(default_factory=ProviderConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    immersion: ImmersionConfig = field(default_factory=ImmersionConfig)


def load_profile(profile_name: str) -> LLMProfile:
    """Load a named profile from config/profiles.yaml."""
    profiles_path = Path(__file__).parent.parent / "config" / "profiles.yaml"

    if not profiles_path.exists():
        raise FileNotFoundError(f"Profiles file not found: {profiles_path}")

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = yaml.safe_load(f)

    profile_name = profile_name.strip()
    if profile_name not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")

    data = profiles[profile_name]

    narrator_data = data.get("narrator", {})
    narrator_premium_data = data.get("narrator_premium")
    narrator_opening_data = data.get("narrator_opening")
    brain_data = data.get("brain", {})
    memory_data = data.get("memory", {})
    tts_data = data.get("tts", {})
    immersion_data = data.get("immersion", {})

    def _provider_config_or_none(d: Optional[dict[str, Any]]) -> Optional[ProviderConfig]:
        if not d:
            return None
        return ProviderConfig(
            provider=d.get("provider", "ollama"),
            model=d.get("model", ""),
            fallback_to_ollama=d.get("fallback_to_ollama", False),
            context_size=d.get("context_size", 0),
            tools=d.get("tools", "core"),
        )

    return LLMProfile(
        name=profile_name,
        narrator=ProviderConfig(
            provider=narrator_data.get("provider", "ollama"),
            model=narrator_data.get("model", ""),
            fallback_to_ollama=narrator_data.get("fallback_to_ollama", False),
            context_size=narrator_data.get("context_size", 0),
            tools=narrator_data.get("tools", "core"),
        ),
        narrator_premium=_provider_config_or_none(narrator_premium_data),
        narrator_opening=_provider_config_or_none(narrator_opening_data),
        brain=ProviderConfig(
            provider=brain_data.get("provider", "ollama"),
            model=brain_data.get("model", ""),
            fallback_to_ollama=brain_data.get("fallback_to_ollama", False),
            context_size=brain_data.get("context_size", 0),
        ),
        memory=MemoryConfig(
            buffer_size=memory_data.get("buffer_size", 20),
            verbatim_size=memory_data.get("verbatim_size", 8),
            condensed_size=memory_data.get("condensed_size", 12),
            compaction_threshold=memory_data.get("compaction_threshold", 6),
        ),
        tts=TTSConfig(
            provider=tts_data.get("provider", "browser"),
            model=tts_data.get("model", ""),
            voice=tts_data.get("voice", ""),
        ),
        immersion=ImmersionConfig(
            tts_enabled=immersion_data.get("tts_enabled", False),
            image_enabled=immersion_data.get("image_enabled", False),
            image_frequency=immersion_data.get("image_frequency", "on_demand"),
            narrator_tts_provider=immersion_data.get("narrator_tts_provider", "kokoro"),
            narrator_tts_voice=immersion_data.get("narrator_tts_voice", "af_heart"),
            character_tts_provider=immersion_data.get("character_tts_provider", ""),
            image_provider=immersion_data.get("image_provider", "fal"),
            image_model=immersion_data.get("image_model", ""),
            image_steps=immersion_data.get("image_steps", 0),
            image_guidance=immersion_data.get("image_guidance", 0.0),
        ),
    )


# Cached profile instance
_profile: Optional[LLMProfile] = None


def get_profile() -> LLMProfile:
    """Get the active LLM profile. Cached after first load."""
    global _profile
    if _profile is None:
        settings = get_settings()
        _profile = load_profile(settings.active_profile)
    return _profile


def set_profile(profile_name: str) -> LLMProfile:
    """Override the active profile (used by test harness)."""
    global _profile
    _profile = load_profile(profile_name)
    return _profile


def switch_profile(profile_name: str) -> LLMProfile:
    """Switch the active profile at runtime, resetting all cached LLM singletons.

    Safe to call while the bot is running. The next turn will use the new
    profile's narrator/brain/memory settings. In-progress turns finish
    with their existing clients.
    """
    profile = set_profile(profile_name)

    # Reset all cached LLM singletons so they recreate with the new profile.
    # Import inline to avoid circular imports at module load time.
    from .llm.client import _reset_clients
    _reset_clients()

    from .llm.brains.narrator import _reset_narrator
    _reset_narrator()

    from .llm.brains.adjudicator import _reset_adjudicator
    _reset_adjudicator()

    from .llm.orchestrator import _reset_orchestrator
    _reset_orchestrator()

    # Reset voice provider singletons
    try:
        from .voice.tts_factory import _reset_tts
        _reset_tts()
    except ImportError:
        pass

    import structlog
    logger = structlog.get_logger()
    logger.info(
        "profile_switched",
        profile=profile.name,
        narrator=f"{profile.narrator.provider}/{profile.narrator.model}",
        brain=f"{profile.brain.provider}/{profile.brain.model}",
        tts=profile.tts.provider,
    )

    return profile


def list_profiles() -> list[str]:
    """Return available profile names from config/profiles.yaml."""
    profiles_path = Path(__file__).parent.parent / "config" / "profiles.yaml"
    if not profiles_path.exists():
        return []
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = yaml.safe_load(f)
    return list(profiles.keys())


# =============================================================================
# Base Settings (env vars + .env file)
# =============================================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    LLM routing is now handled by profiles (config/profiles.yaml).
    API keys and non-LLM settings remain here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Deliberate (pins the pydantic-settings default): unknown keys in
        # .env fail fast at boot, so .env must track the fields below exactly.
        # Unrelated OS env vars are unaffected (only field-named vars are read).
        # Keep .env.example in sync; tests/unit/test_env_example.py enforces it.
        extra="forbid",
    )

    # Discord
    discord_bot_token: str

    # Active LLM profile (from config/profiles.yaml)
    active_profile: str = "production"

    # Ollama connection (shared across profiles that use ollama)
    ollama_host: str = "http://localhost:11434"

    # API keys (shared across profiles)
    groq_api_key: str = ""
    groq_max_retries: int = 3
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    deepseek_api_key: str = ""
    gemini_api_key: str = ""
    elevenlabs_api_key: str = ""

    # LLM Temperature
    narrator_temperature: float = 0.75
    rules_temperature: float = 0.1

    # LLM Timeout (seconds)
    llm_timeout: float = 120.0

    # Database
    database_path: str = "data/dnd_bot.db"

    # ChromaDB
    chroma_persist_path: str = "data/chroma"

    # SRD Data
    srd_data_path: str = "../5e-database/src/2014"

    # Logging
    log_level: str = "INFO"

    # Debug
    debug_log_llm_output: bool = False

    # Image generation (optional, for immersion features)
    image_provider: str = "fal"  # fal, openai, local
    fal_key: str = ""  # From https://fal.ai/dashboard/keys (env: FAL_KEY)
    fal_model: str = "fal-ai/flux/dev"  # fal-ai/flux/dev (~$0.025) or fal-ai/flux-2-pro (~$0.03)
    openai_image_api_key: str = ""  # Direct OpenAI key for DALL-E (not OpenRouter)
    local_image_model: str = "black-forest-labs/FLUX.1-dev"  # HuggingFace model ID
    local_image_steps: int = 20
    local_image_guidance: float = 3.5

    # Inworld TTS (narrator voice)
    inworld_api_key: str = ""  # Base64-encoded key from studio.inworld.ai
    inworld_tts_model: str = "inworld-tts-1.5-mini"  # tts-1.5-mini (fast) or tts-1.5-max (quality)
    inworld_tts_voice: str = "Sarah"

    # Fish Speech (local TTS alternative to Riva)
    fish_speech_url: str = "http://localhost:8080"
    fish_speech_instances: int = 1  # Number of parallel Fish Speech servers on sequential ports from fish_speech_url

    # Voice / Riva (optional, only needed for voice mode). LiveKit and NGC
    # credentials are read from the process environment by the voice stack
    # (voice/api.py, voice/transports/livekit_transport.py, docker-compose),
    # not through Settings.
    riva_asr_url: str = "localhost:50051"
    riva_tts_url: str = "localhost:50052"
    tts_voice: str = "Magpie-Multilingual.EN-US.Aria"

    @property
    def srd_path(self) -> Path:
        """Resolved path to SRD data directory."""
        return Path(__file__).parent.parent / self.srd_data_path

    @property
    def db_path(self) -> Path:
        """Resolved path to database file."""
        return Path(__file__).parent.parent / self.database_path

    @property
    def chroma_path(self) -> Path:
        """Resolved path to ChromaDB directory."""
        return Path(__file__).parent.parent / self.chroma_persist_path


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

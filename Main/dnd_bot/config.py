"""Configuration management using pydantic-settings + YAML profiles."""

from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# LLM Profile (loaded from config/profiles.yaml)
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider role."""
    provider: str = "ollama"     # ollama | groq | anthropic
    model: str = ""
    fallback_to_ollama: bool = False

@dataclass
class MemoryConfig:
    """Memory system configuration."""
    buffer_size: int = 20
    compaction_threshold: int = 6

@dataclass
class LLMProfile:
    """A named LLM configuration profile."""
    name: str = "default"
    narrator: ProviderConfig = field(default_factory=ProviderConfig)
    brain: ProviderConfig = field(default_factory=ProviderConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_profile(profile_name: str) -> LLMProfile:
    """Load a named profile from config/profiles.yaml."""
    profiles_path = Path(__file__).parent.parent / "config" / "profiles.yaml"

    if not profiles_path.exists():
        raise FileNotFoundError(f"Profiles file not found: {profiles_path}")

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = yaml.safe_load(f)

    if profile_name not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")

    data = profiles[profile_name]

    narrator_data = data.get("narrator", {})
    brain_data = data.get("brain", {})
    memory_data = data.get("memory", {})

    return LLMProfile(
        name=profile_name,
        narrator=ProviderConfig(
            provider=narrator_data.get("provider", "ollama"),
            model=narrator_data.get("model", ""),
            fallback_to_ollama=narrator_data.get("fallback_to_ollama", False),
        ),
        brain=ProviderConfig(
            provider=brain_data.get("provider", "ollama"),
            model=brain_data.get("model", ""),
            fallback_to_ollama=brain_data.get("fallback_to_ollama", False),
        ),
        memory=MemoryConfig(
            buffer_size=memory_data.get("buffer_size", 20),
            compaction_threshold=memory_data.get("compaction_threshold", 6),
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
    gemini_api_key: str = ""

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
    debug_show_intents: bool = False
    debug_log_llm_output: bool = False

    @property
    def database_url(self) -> str:
        """SQLite database URL."""
        return f"sqlite+aiosqlite:///{self.database_path}"

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

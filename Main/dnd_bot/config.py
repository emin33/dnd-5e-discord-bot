"""Configuration management using pydantic-settings."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Discord
    discord_bot_token: str

    # LLM Provider: "ollama" for local, "groq" for Groq cloud API
    llm_provider: str = "ollama"

    # Ollama (local)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"

    # Groq (cloud)
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3-32b"
    groq_max_retries: int = 3          # Retries before falling back to Ollama
    groq_fallback_to_ollama: bool = True  # Fall back to local Ollama on rate limit

    # LLM Temperature
    narrator_temperature: float = 0.75
    rules_temperature: float = 0.1

    # LLM Timeout (seconds) - max time to wait for a single LLM response
    # Set high enough for cold model loads (~90s for 30B models)
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
    debug_show_intents: bool = False  # If True, append INTENTS to Discord messages
    debug_log_llm_output: bool = False  # If True, write full LLM outputs to data/llm_debug.log

    # Gemini (used by test_eval.py only — not used in game runtime)
    gemini_api_key: str = ""

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

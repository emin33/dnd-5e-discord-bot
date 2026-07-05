"""TTS provider factory — mirrors the LLM client dispatch pattern.

Lazy singleton, factory dispatch on provider string, _reset for profile switching.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import structlog

logger = structlog.get_logger()

# Module-level singleton (same pattern as llm/client.py _client)
_tts_instance = None
_tts_lock: Optional[asyncio.Lock] = None


def _create_tts(provider: str, **kwargs):
    """Create a TTS provider instance by provider name.

    Args:
        provider: "riva" | "openai" | "elevenlabs" | "inworld" | "fish" | "kokoro" | "browser"
        **kwargs: Provider-specific settings (voice, model, api_key, etc.)
    """
    from ..config import get_settings
    settings = get_settings()

    if provider == "riva":
        from .tts import RivaTTS
        return RivaTTS(
            server_url=kwargs.get("server_url", settings.riva_tts_url),
            voice=kwargs.get("voice") or settings.tts_voice,
        )

    elif provider == "openai":
        from .tts_openai import OpenAITTS
        return OpenAITTS(
            model=kwargs.get("model", "tts-1"),
            voice=kwargs.get("voice", "onyx"),
            # OpenRouter doesn't proxy audio.speech; pass None so the OpenAI
            # client falls back to OPENAI_API_KEY env var.
            api_key=None,
        )

    elif provider == "elevenlabs":
        try:
            from .tts_elevenlabs import ElevenLabsTTS
            return ElevenLabsTTS(
                voice=kwargs.get("voice", ""),
                model=kwargs.get("model", ""),
                api_key=settings.elevenlabs_api_key or None,
            )
        except ImportError:
            logger.warning("elevenlabs_sdk_not_installed", hint="pip install elevenlabs")
            from .tts_browser import BrowserTTS
            return BrowserTTS()

    elif provider == "inworld":
        from .tts_inworld import InworldTTS
        return InworldTTS(
            model=kwargs.get("model") or settings.inworld_tts_model,
            voice=kwargs.get("voice") or settings.inworld_tts_voice,
            api_key=kwargs.get("api_key") or settings.inworld_api_key or None,
        )

    elif provider == "fish":
        from .tts_fish import FishSpeechTTS
        return FishSpeechTTS(
            server_url=kwargs.get("server_url", settings.fish_speech_url),
            voice=kwargs.get("voice", ""),
        )

    elif provider == "kokoro":
        from .tts_kokoro import KokoroTTS
        return KokoroTTS(
            voice=kwargs.get("voice", "af_heart"),
            speed=kwargs.get("speed", 1.0),
            device=kwargs.get("device", ""),  # "" = auto, "cpu" for ROCm
        )

    else:  # browser
        from .tts_browser import BrowserTTS
        return BrowserTTS()


def get_tts():
    """Get the TTS provider singleton. Lazy-initialized from active profile."""
    global _tts_instance
    if _tts_instance is None:
        from ..config import get_profile
        profile = get_profile()
        tts_cfg = profile.tts

        try:
            _tts_instance = _create_tts(
                provider=tts_cfg.provider,
                model=tts_cfg.model,
                voice=tts_cfg.voice,
            )
        except Exception as e:
            logger.error("tts_init_failed", provider=tts_cfg.provider, error=str(e), exc_info=True)
            from .tts_browser import BrowserTTS
            _tts_instance = BrowserTTS()

        logger.info("tts_provider_loaded", provider=type(_tts_instance).__name__)

    return _tts_instance


def _reset_tts() -> None:
    """Clear TTS singleton so it recreates from the active profile."""
    global _tts_instance
    _tts_instance = None


def get_tts_lock() -> asyncio.Lock:
    """Get or create the TTS serialization lock."""
    global _tts_lock
    if _tts_lock is None:
        _tts_lock = asyncio.Lock()
    return _tts_lock


def needs_lock(tts) -> bool:
    """Check if TTS provider requires serialized calls.

    True for any in-process model that isn't thread-safe under concurrent
    invocation: Riva gRPC stub, Kokoro (shared PyTorch `KPipeline` module).
    HTTP-based providers (OpenAI, ElevenLabs, Fish, Inworld) are fine
    concurrently — each request gets its own connection.
    """
    from .tts import RivaTTS
    if isinstance(tts, RivaTTS):
        return True
    try:
        from .tts_kokoro import KokoroTTS
        if isinstance(tts, KokoroTTS):
            return True
    except ImportError:
        # Kokoro is optional — if it can't be imported it's not in use.
        pass
    return False


def is_browser_tts(tts) -> bool:
    """Check if TTS is the browser no-op sentinel."""
    from .tts_browser import BrowserTTS
    return isinstance(tts, BrowserTTS)

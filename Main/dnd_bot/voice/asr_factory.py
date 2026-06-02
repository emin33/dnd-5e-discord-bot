"""ASR provider factory — mirrors the LLM client dispatch pattern.

Lazy singleton, factory dispatch on provider string, _reset for profile switching.
"""

from __future__ import annotations

from typing import Optional

import structlog

logger = structlog.get_logger()

# Module-level singleton
_asr_instance = None


def _create_asr(provider: str, **kwargs):
    """Create an ASR provider instance by provider name.

    Args:
        provider: "riva" | "openai" | "browser"
        **kwargs: Provider-specific settings (model, api_key, etc.)
    """
    from ..config import get_settings
    settings = get_settings()

    if provider == "riva":
        from .asr import RivaASR
        return RivaASR(
            server_url=kwargs.get("server_url", settings.riva_asr_url),
        )

    elif provider == "openai":
        from .asr_openai import OpenAIASR
        return OpenAIASR(
            model=kwargs.get("model", "whisper-1"),
            # OpenRouter doesn't proxy audio.transcriptions; pass None so the
            # OpenAI client falls back to OPENAI_API_KEY env var.
            api_key=None,
        )

    elif provider == "deepgram":
        try:
            from .asr_deepgram import DeepgramASR
            return DeepgramASR(
                model=kwargs.get("model", "nova-3"),
                api_key=settings.deepgram_api_key or None,
            )
        except ImportError:
            logger.warning("deepgram_sdk_not_installed", hint="pip install deepgram-sdk")
            from .asr_browser import BrowserASR
            return BrowserASR()

    else:  # browser
        from .asr_browser import BrowserASR
        return BrowserASR()


def get_asr():
    """Get the ASR provider singleton. Lazy-initialized from active profile."""
    global _asr_instance
    if _asr_instance is None:
        from ..config import get_profile
        profile = get_profile()
        asr_cfg = profile.asr

        try:
            _asr_instance = _create_asr(
                provider=asr_cfg.provider,
                model=asr_cfg.model,
            )
        except Exception as e:
            logger.error("asr_init_failed", provider=asr_cfg.provider, error=str(e))
            from .asr_browser import BrowserASR
            _asr_instance = BrowserASR()

        logger.info("asr_provider_loaded", provider=type(_asr_instance).__name__)

    return _asr_instance


def _reset_asr():
    """Clear ASR singleton so it recreates from the active profile."""
    global _asr_instance
    _asr_instance = None


def is_browser_asr(asr) -> bool:
    """Check if ASR is the browser no-op sentinel."""
    from .asr_browser import BrowserASR
    return isinstance(asr, BrowserASR)

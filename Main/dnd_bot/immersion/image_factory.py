"""Image provider factory: singleton dispatch for image generation providers.

Follows the same factory pattern as voice/tts_factory.py:
- Module-level singleton
- Provider string dispatch
- Reset on profile switch
"""

import threading
from typing import Optional, Protocol

import structlog

logger = structlog.get_logger()


class ImageProvider(Protocol):
    """Protocol for image generation providers."""

    async def generate(self, prompt: str, size: str = "1024x1024") -> bytes:
        """Generate an image from a text prompt. Returns PNG bytes."""
        ...


def _create_image_provider(provider: str, **kwargs) -> ImageProvider:
    """Create an image provider instance."""
    if provider == "fal":
        from .image_fal import FalImageProvider
        return FalImageProvider(
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", "fal-ai/flux/dev"),
        )
    elif provider == "openai":
        from .image_openai import OpenAIImageProvider
        return OpenAIImageProvider(
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", "dall-e-3"),
        )
    elif provider == "local":
        from .image_local import LocalImageProvider
        return LocalImageProvider(
            model=kwargs.get("model", "black-forest-labs/FLUX.1-dev"),
            dtype=kwargs.get("dtype", "float16"),
            steps=kwargs.get("steps", 20),
            guidance_scale=kwargs.get("guidance_scale", 3.5),
        )
    else:
        raise ValueError(f"Unknown image provider: {provider}")


# Module-level singleton
_provider_instance: Optional[ImageProvider] = None
_provider_name: Optional[str] = None
_provider_lock = threading.Lock()


def get_image_provider() -> ImageProvider:
    """Get the active image provider singleton. Profile takes precedence over settings."""
    global _provider_instance, _provider_name

    if _provider_instance is not None:
        return _provider_instance

    with _provider_lock:
        # Double-check after acquiring lock
        if _provider_instance is not None:
            return _provider_instance

        from ..config import get_settings, get_profile
        settings = get_settings()

        # Profile's immersion config takes precedence over settings
        try:
            profile = get_profile()
            immersion = profile.immersion
            provider = immersion.image_provider or settings.image_provider
        except Exception as e:
            logger.warning("image_profile_load_failed", error=str(e), exc_info=True)
            immersion = None
            provider = getattr(settings, 'image_provider', 'fal')
        kwargs = {}

        if provider == "fal":
            kwargs["api_key"] = settings.fal_key or None
            kwargs["model"] = (immersion.image_model if immersion and immersion.image_model else "") or settings.fal_model
        elif provider == "openai":
            kwargs["api_key"] = settings.openai_image_api_key or None
        elif provider == "local":
            kwargs["model"] = (immersion.image_model if immersion and immersion.image_model else "") or settings.local_image_model
            kwargs["steps"] = (immersion.image_steps if immersion and immersion.image_steps else 0) or settings.local_image_steps
            # Don't treat guidance=0.0 as "unset" — Flux Schnell legitimately uses 0.0.
            kwargs["guidance_scale"] = (
                immersion.image_guidance
                if (immersion and immersion.image_guidance is not None)
                else settings.local_image_guidance
            )

        _provider_instance = _create_image_provider(provider, **kwargs)
        _provider_name = provider

        logger.info("image_provider_initialized", provider=provider)
        return _provider_instance


def get_provider_name() -> Optional[str]:
    """Get the name of the active image provider."""
    return _provider_name


def _reset_image_provider():
    """Reset the image provider singleton (for profile switching)."""
    global _provider_instance, _provider_name
    _provider_instance = None
    _provider_name = None

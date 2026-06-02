"""OpenAI DALL-E image generation provider."""

import base64
from typing import Optional

import structlog

logger = structlog.get_logger()


class OpenAIImageProvider:
    """Generate images using OpenAI's DALL-E API."""

    def __init__(self, api_key: str = "", model: str = "dall-e-3"):
        self.api_key = api_key or None  # None lets OpenAI read from OPENAI_API_KEY env
        self.model = model
        self._client = None
        logger.info("openai_image_provider_initialized", model=model)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            # 60s timeout caps the executor thread block — without it, a hung
            # DALL-E request can hold a worker for up to 10 minutes (default).
            if self.api_key:
                self._client = OpenAI(api_key=self.api_key, timeout=60.0)
            else:
                self._client = OpenAI(timeout=60.0)
        return self._client

    async def generate(self, prompt: str, size: str = "1024x1024") -> bytes:
        """Generate an image from a prompt. Returns PNG bytes."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, prompt, size)

    def _generate_sync(self, prompt: str, size: str) -> bytes:
        """Synchronous image generation."""
        client = self._get_client()

        response = client.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
            response_format="b64_json",
        )

        b64_data = response.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)

        logger.info(
            "image_generated",
            provider="openai",
            model=self.model,
            size=size,
            bytes=len(image_bytes),
        )

        return image_bytes

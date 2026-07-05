"""FAL AI image generation provider (Flux models).

Uses FAL's REST API for fast, high-quality image generation.
Supports Flux 2 Pro, Flux 1 Dev, and other models.
~$0.03/image for 1024x1024.

Requires: FAL_KEY in environment
  Sign up at https://fal.ai/dashboard/keys
"""

from __future__ import annotations

import asyncio

import structlog

logger = structlog.get_logger()


class FalImageProvider:
    """FAL AI image generation via Flux models."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "fal-ai/flux/dev",
    ):
        self.model = model
        self._api_key = api_key
        if not self._api_key:
            import os
            self._api_key = os.environ.get("FAL_KEY", "")
        logger.info("fal_image_provider_initialized", model=model)

    async def generate(self, prompt: str, size: str = "1024x1024") -> bytes:
        """Generate an image from a prompt. Returns PNG bytes."""

        width, height = 1024, 1024
        if "x" in size:
            parts = size.split("x")
            width, height = int(parts[0]), int(parts[1])

        headers = {
            "Authorization": f"Key {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "num_images": 1,
        }

        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(
            None, self._submit_and_fetch, headers, payload
        )
        return image_bytes

    def _submit_and_fetch(self, headers: dict, payload: dict) -> bytes:
        """Submit generation request and fetch result. Retries once on timeout."""
        import requests

        result = None
        for attempt in range(2):
            try:
                response = requests.post(
                    f"https://fal.run/{self.model}",
                    json=payload,
                    headers=headers,
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                break
            except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                if attempt == 0:
                    logger.warning("fal_error_retrying", error=str(e), exc_info=True)
                    import time
                    time.sleep(3)
                    continue
                raise

        # Result contains images array with url
        images = result.get("images", [])
        if not images:
            raise RuntimeError("FAL returned no images")

        image_url = images[0].get("url")
        if not image_url:
            raise RuntimeError("FAL image has no URL")

        # Download the image
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()

        logger.info(
            "image_generated",
            provider="fal",
            model=self.model,
            bytes=len(img_response.content),
        )

        return img_response.content

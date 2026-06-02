"""Local image generation via Hugging Face diffusers.

Loads Flux or SDXL directly in-process -- no external server needed.
Just pip install diffusers and point at a model. Works like any other
local model in the pipeline.

Requires: pip install diffusers transformers accelerate
VRAM: ~12GB for Flux Dev fp8, ~6GB for SDXL
"""

from __future__ import annotations

import asyncio
import io
from typing import Optional

import structlog

logger = structlog.get_logger()

# Lazy-loaded pipeline (stays in VRAM once loaded)
_pipe = None
_loaded_model: Optional[str] = None


def _load_pipeline(model: str, dtype: str = "float16"):
    """Load a diffusers pipeline. Cached after first call."""
    global _pipe, _loaded_model

    if _pipe is not None and _loaded_model == model:
        return _pipe

    import torch
    from diffusers import AutoPipelineForText2Image

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    logger.info("loading_image_model", model=model, dtype=dtype)

    _pipe = AutoPipelineForText2Image.from_pretrained(
        model,
        torch_dtype=torch_dtype,
    )

    # Use CPU offload -- keeps model in RAM, moves components to GPU only when needed.
    # This allows Flux to coexist with other GPU models (Fish Speech, LLM brain).
    _pipe.enable_model_cpu_offload()

    _loaded_model = model
    logger.info("image_model_loaded", model=model)
    return _pipe


class LocalImageProvider:
    """Generate images using a local diffusers model (Flux, SDXL, etc.)."""

    def __init__(
        self,
        model: str = "black-forest-labs/FLUX.1-dev",
        dtype: str = "float16",
        steps: int = 15,
        guidance_scale: float = 3.5,
    ):
        self.model = model
        self.dtype = dtype
        self.steps = steps
        self.guidance_scale = guidance_scale
        logger.info(
            "local_image_provider_initialized",
            model=model,
            steps=steps,
        )

    async def generate(self, prompt: str, size: str = "1024x1024") -> bytes:
        """Generate an image from a prompt. Returns PNG bytes."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, prompt, size
        )

    def _generate_sync(self, prompt: str, size: str = "1024x1024") -> bytes:
        """Synchronous image generation."""
        width, height = 1024, 1024
        if "x" in size:
            parts = size.split("x")
            width, height = int(parts[0]), int(parts[1])

        pipe = _load_pipeline(self.model, self.dtype)

        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
        )

        image = result.images[0]

        # Convert PIL Image to PNG bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        logger.info(
            "image_generated",
            provider="local",
            model=self.model,
            size=f"{width}x{height}",
            bytes=buf.getbuffer().nbytes,
        )

        return buf.getvalue()

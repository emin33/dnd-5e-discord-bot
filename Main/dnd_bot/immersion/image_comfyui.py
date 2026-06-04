"""ComfyUI local image generation provider.

Connects to a local ComfyUI instance via its REST API for free
image generation. Supports both SDXL and Flux workflows, selected
via the checkpoint_type config.

Requires: ComfyUI running locally with a checkpoint downloaded.
  Flux Dev: ~12GB VRAM, download flux1-dev-fp8.safetensors
  SDXL: ~6GB VRAM, download sd_xl_base_1.0.safetensors
"""

import asyncio
import json
import uuid

import structlog

logger = structlog.get_logger()


# Flux Dev workflow -- no negative prompt, uses FluxGuidance instead of CFG
_FLUX_WORKFLOW = {
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"},
    },
    "5": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "10": {
        "class_type": "FluxGuidance",
        "inputs": {"guidance": 3.5, "conditioning": ["6", 0]},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["10", 0],
            "negative": ["6", 0],  # Flux ignores negative, but KSampler requires it
            "latent_image": ["5", 0],
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "dnd_scene", "images": ["8", 0]},
    },
}

# SDXL workflow -- standard CFG with negative prompt
_SDXL_WORKFLOW = {
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "blurry, low quality, deformed, ugly, text, watermark", "clip": ["4", 1]},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 25,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "dnd_scene", "images": ["8", 0]},
    },
}

_WORKFLOWS = {
    "flux": _FLUX_WORKFLOW,
    "sdxl": _SDXL_WORKFLOW,
}


class ComfyUIImageProvider:
    """Generate images via a local ComfyUI REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8188",
        checkpoint_type: str = "flux",
        checkpoint_name: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.checkpoint_type = checkpoint_type
        self.checkpoint_name = checkpoint_name  # Override the default ckpt filename
        logger.info(
            "comfyui_image_provider_initialized",
            url=self.base_url,
            type=checkpoint_type,
        )

    async def generate(self, prompt: str, size: str = "1024x1024") -> bytes:
        """Generate an image using ComfyUI. Returns PNG bytes."""
        try:
            import aiohttp
        except ImportError:
            raise RuntimeError("aiohttp required for ComfyUI provider: pip install aiohttp")

        width, height = 1024, 1024
        if "x" in size:
            parts = size.split("x")
            width, height = int(parts[0]), int(parts[1])

        # Select and configure workflow
        base_workflow = _WORKFLOWS.get(self.checkpoint_type, _FLUX_WORKFLOW)
        workflow = json.loads(json.dumps(base_workflow))

        # Set dimensions
        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height

        # Set prompt text
        workflow["6"]["inputs"]["text"] = prompt

        # Set random seed
        workflow["3"]["inputs"]["seed"] = uuid.uuid4().int % (2**32)

        # Override checkpoint name if configured
        if self.checkpoint_name:
            workflow["4"]["inputs"]["ckpt_name"] = self.checkpoint_name

        # Submit to ComfyUI
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow},
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"ComfyUI queue failed: {error}")
                result = await resp.json()
                prompt_id = result["prompt_id"]

            # Poll for completion
            for _ in range(180):  # 3 minute timeout (Flux can be slow)
                await asyncio.sleep(1)
                async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
                    if resp.status != 200:
                        continue
                    history = await resp.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        for node_id, node_output in outputs.items():
                            images = node_output.get("images", [])
                            if images:
                                image_info = images[0]
                                filename = image_info["filename"]
                                subfolder = image_info.get("subfolder", "")
                                params = {"filename": filename}
                                if subfolder:
                                    params["subfolder"] = subfolder
                                async with session.get(
                                    f"{self.base_url}/view",
                                    params=params,
                                ) as img_resp:
                                    if img_resp.status == 200:
                                        image_bytes = await img_resp.read()
                                        logger.info(
                                            "image_generated",
                                            provider="comfyui",
                                            type=self.checkpoint_type,
                                            size=f"{width}x{height}",
                                            bytes=len(image_bytes),
                                        )
                                        return image_bytes

            raise TimeoutError("ComfyUI image generation timed out")

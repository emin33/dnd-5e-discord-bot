"""OpenAI TTS provider.

Uses OpenAI's audio.speech API to synthesize text to speech.
Returns raw PCM int16 audio at 24kHz mono.

Requires OPENAI_API_KEY in environment (or openrouter_api_key in Settings
if using OpenAI-compatible endpoint).
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class OpenAITTS:
    """OpenAI TTS provider (tts-1 or tts-1-hd).

    Voices: alloy, echo, fable, onyx, nova, shimmer
    """

    def __init__(
        self,
        model: str = "tts-1",
        voice: str = "onyx",
        api_key: Optional[str] = None,
    ):
        from openai import OpenAI

        self.model = model or "tts-1"
        self.voice = voice or "onyx"
        self._client = OpenAI(api_key=api_key)
        logger.info("openai_tts_initialized", model=self.model, voice=self.voice)

    @property
    def sample_rate(self) -> int:
        return 24000

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to PCM int16 audio at 24kHz."""
        response = self._client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",  # Raw 24kHz 16-bit mono PCM
        )
        pcm_bytes = response.read()
        return np.frombuffer(pcm_bytes, dtype=np.int16)

    async def synthesize_async(self, text: str) -> np.ndarray:
        """Async wrapper — runs sync OpenAI call in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)

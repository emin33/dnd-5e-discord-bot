"""ElevenLabs TTS provider.

Uses the ElevenLabs SDK to synthesize text to speech.
Returns raw PCM int16 audio at 44.1kHz mono.

Requires: pip install elevenlabs
Requires: ELEVENLABS_API_KEY in environment
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class ElevenLabsTTS:
    """ElevenLabs TTS provider.

    Voices are referenced by voice_id (e.g. "pNInz6obpgDQGcFmaJgB" for Adam).
    Browse voices at https://elevenlabs.io/voice-library
    """

    def __init__(
        self,
        voice: str = "pNInz6obpgDQGcFmaJgB",
        model: str = "eleven_multilingual_v2",
        api_key: Optional[str] = None,
    ):
        from elevenlabs.client import ElevenLabs

        self.voice_id = voice
        self.model_id = model or "eleven_multilingual_v2"
        self._client = ElevenLabs(api_key=api_key)
        logger.info("elevenlabs_tts_initialized", voice=self.voice_id, model=self.model_id)

    @property
    def sample_rate(self) -> int:
        return 44100

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to PCM int16 audio at 44.1kHz."""
        audio_iter = self._client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model_id,
            output_format="pcm_44100",
        )
        # Collect all chunks
        chunks = b"".join(audio_iter)
        return np.frombuffer(chunks, dtype=np.int16)

    async def synthesize_async(self, text: str) -> np.ndarray:
        """Async wrapper — runs sync ElevenLabs call in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)

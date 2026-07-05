"""Riva TTS (Text-to-Speech) wrapper.

Wraps NVIDIA Riva Magpie NIM for text-to-speech synthesis.
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog

logger = structlog.get_logger()

# Audio constants
TTS_SAMPLE_RATE = 44100


class RivaTTS:
    """NVIDIA Riva Magpie TTS via gRPC.

    Synthesizes text to audio (int16 PCM arrays).
    """

    def __init__(
        self,
        server_url: str = "localhost:50052",
        voice: str = "Magpie-Multilingual.EN-US.Aria",
    ):
        import riva.client

        auth = riva.client.Auth(uri=server_url, use_ssl=False)
        self.service = riva.client.SpeechSynthesisService(auth)
        self._riva = riva.client
        self.voice = voice
        self.sample_rate = TTS_SAMPLE_RATE
        logger.info("riva_tts_initialized", server=server_url, voice=voice)

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to a single audio array.

        Returns:
            numpy int16 array of audio samples at 44.1kHz.
        """
        responses = self.service.synthesize_online(
            [text],
            voice_name=self.voice,
            language_code="en-US",
            sample_rate_hz=self.sample_rate,
            encoding=self._riva.AudioEncoding.LINEAR_PCM,
        )
        chunks = []
        for resp in responses:
            chunk = np.frombuffer(resp.audio, dtype=np.int16)
            if len(chunk) > 0:
                chunks.append(chunk)

        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.int16)

    async def synthesize_async(self, text: str) -> np.ndarray:
        """Async wrapper around synchronous synthesize."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)

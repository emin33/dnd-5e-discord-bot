"""OpenAI Whisper ASR provider.

Uses OpenAI's audio.transcriptions API for speech-to-text.
Requires OPENAI_API_KEY in environment.
"""

from __future__ import annotations

import io
import struct
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class OpenAIASR:
    """OpenAI Whisper ASR provider."""

    def __init__(
        self,
        model: str = "whisper-1",
        api_key: Optional[str] = None,
    ):
        from openai import OpenAI

        self.model = model or "whisper-1"
        self._client = OpenAI(api_key=api_key)
        logger.info("openai_asr_initialized", model=self.model)

    @property
    def sample_rate(self) -> int:
        return 16000

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe int16 PCM audio to text."""
        wav_bytes = self._pcm_to_wav(audio)
        return self._transcribe_wav(wav_bytes)

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe raw PCM bytes to text."""
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        return self.transcribe(audio)

    def _transcribe_wav(self, wav_bytes: bytes) -> str:
        """Send WAV bytes to OpenAI Whisper."""
        buf = io.BytesIO(wav_bytes)
        buf.name = "audio.wav"  # OpenAI requires a filename
        response = self._client.audio.transcriptions.create(
            model=self.model,
            file=buf,
            language="en",
        )
        return response.text.strip()

    def _pcm_to_wav(self, audio: np.ndarray) -> bytes:
        """Wrap int16 PCM in a WAV container."""
        audio_bytes = audio.tobytes()
        sample_rate = self.sample_rate
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio_bytes)

        buf = io.BytesIO()
        buf.write(b'RIFF')
        buf.write(struct.pack('<I', 36 + data_size))
        buf.write(b'WAVE')
        buf.write(b'fmt ')
        buf.write(struct.pack('<IHHIIHH', 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
        buf.write(b'data')
        buf.write(struct.pack('<I', data_size))
        buf.write(audio_bytes)
        return buf.getvalue()

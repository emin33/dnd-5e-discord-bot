"""Deepgram ASR provider.

Uses Deepgram's REST API for speech-to-text.
Requires: pip install deepgram-sdk
Requires: DEEPGRAM_API_KEY in environment
"""

from __future__ import annotations

import io
import struct
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class DeepgramASR:
    """Deepgram ASR provider (Nova-2 or Nova-3).

    Fast, accurate cloud transcription. No GPU needed.
    """

    def __init__(
        self,
        model: str = "nova-3",
        api_key: Optional[str] = None,
    ):
        from deepgram import DeepgramClient

        self.model = model or "nova-3"
        self._client = DeepgramClient(api_key=api_key)
        logger.info("deepgram_asr_initialized", model=self.model)

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
        """Send WAV bytes to Deepgram."""
        from deepgram import PrerecordedOptions

        payload = {"buffer": wav_bytes}
        options = PrerecordedOptions(
            model=self.model,
            language="en",
            smart_format=True,
            punctuate=True,
        )

        response = self._client.listen.rest.v("1").transcribe_file(payload, options)
        transcript = (
            response.results.channels[0].alternatives[0].transcript
        )
        return transcript.strip()

    def _pcm_to_wav(self, audio: np.ndarray) -> bytes:
        """Wrap int16 PCM in a WAV container."""
        audio_bytes = audio.tobytes()
        sr = self.sample_rate
        buf = io.BytesIO()
        buf.write(b'RIFF')
        buf.write(struct.pack('<I', 36 + len(audio_bytes)))
        buf.write(b'WAVEfmt ')
        buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
        buf.write(b'data')
        buf.write(struct.pack('<I', len(audio_bytes)))
        buf.write(audio_bytes)
        return buf.getvalue()

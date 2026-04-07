"""Riva ASR (Automatic Speech Recognition) wrapper.

Wraps NVIDIA Riva Parakeet NIM for speech-to-text conversion.
Supports both synchronous transcription of complete audio segments
and could be extended for streaming recognition.
"""

from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger()

# Audio constants
ASR_SAMPLE_RATE = 16000
ASR_CHANNELS = 1


class RivaASR:
    """NVIDIA Riva Parakeet ASR via gRPC.

    Transcribes audio segments (numpy int16 arrays) to text.
    Requires Riva ASR NIM container running on the configured port.
    """

    def __init__(self, server_url: str = "localhost:50051"):
        import riva.client

        auth = riva.client.Auth(uri=server_url, use_ssl=False)
        self.service = riva.client.ASRService(auth)
        self._riva = riva.client
        self.config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code="en-US",
            sample_rate_hertz=ASR_SAMPLE_RATE,
            audio_channel_count=ASR_CHANNELS,
            max_alternatives=1,
            enable_automatic_punctuation=True,
        )
        logger.info("riva_asr_initialized", server=server_url)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe an audio segment to text.

        Args:
            audio: numpy int16 array of audio samples at 16kHz mono.

        Returns:
            Transcribed text string.
        """
        response = self.service.offline_recognize(audio.tobytes(), self.config)
        parts = []
        for result in response.results:
            if result.alternatives:
                parts.append(result.alternatives[0].transcript)
        text = " ".join(parts).strip()

        if text:
            logger.debug("asr_transcription", text=text[:80])
        return text

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe raw audio bytes (int16 PCM at 16kHz).

        Convenience method when you already have raw bytes.
        """
        response = self.service.offline_recognize(audio_bytes, self.config)
        parts = []
        for result in response.results:
            if result.alternatives:
                parts.append(result.alternatives[0].transcript)
        return " ".join(parts).strip()

"""Inworld TTS-1.5 cloud provider.

Uses Inworld's REST API for high-quality narrator TTS.
Models: tts-1.5-mini (fast, ~120ms TTFA) and tts-1.5-max (quality, ~200ms TTFA).
Returns raw PCM int16 audio at 24kHz mono.

Requires: INWORLD_API_KEY in environment
  (Base64-encoded API key from https://studio.inworld.ai)
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()

_API_BASE = "https://api.inworld.ai"


class InworldTTS:
    """Inworld TTS-1.5 cloud provider."""

    def __init__(
        self,
        model: str = "inworld-tts-1.5-mini",
        voice: str = "Diego",
        api_key: Optional[str] = None,
    ):
        self.model = model or "inworld-tts-1.5-max"
        self.voice = voice or "Diego"

        self._api_key = api_key
        if not self._api_key:
            import os
            self._api_key = os.environ.get("INWORLD_API_KEY", "")

        logger.info(
            "inworld_tts_initialized",
            model=self.model,
            voice=self.voice,
            has_key=bool(self._api_key),
        )

    @property
    def sample_rate(self) -> int:
        return 48000

    def synthesize(self, text: str, emotion: Optional[str] = None) -> np.ndarray:
        """Synthesize text to PCM int16 audio at 48kHz."""
        import requests
        import subprocess

        # Replace curly/smart quotes and apostrophes with ASCII equivalents
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2014", ", ").replace("\u2013", ", ").replace("--", ", ")

        # Prepend emotion tag (Inworld supports [emotion] markup)
        if emotion:
            text = f"[{emotion}] {text}"

        payload = {
            "text": text,
            "voiceId": self.voice,
            "modelId": self.model,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {self._api_key}",
        }

        response = requests.post(
            f"{_API_BASE}/tts/v1/voice",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        audio_b64 = data.get("audioContent", "")
        if not audio_b64:
            raise RuntimeError("Inworld returned no audioContent")

        mp3_bytes = base64.b64decode(audio_b64)

        # Decode MP3 to PCM via ffmpeg at native sample rate (48kHz)
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "pipe:1"],
            input=mp3_bytes,
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg MP3 decode failed: {result.stderr[:200]}")

        return np.frombuffer(result.stdout, dtype=np.int16)

    async def synthesize_async(self, text: str) -> np.ndarray:
        """Async wrapper -- runs sync HTTP call in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)

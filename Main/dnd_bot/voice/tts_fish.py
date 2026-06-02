"""Fish Speech local TTS provider.

Connects to a self-hosted Fish Speech server via its REST API.
Supports zero-shot voice cloning from reference audio and emotion tags.

Emotion tags (prepend to text):
  v1.5/S1: (happy) (angry) (sad) (excited) (whispering) (scared) etc.
  S2/S2-Pro: [excited] [whisper] [angry] -- natural language, sub-word control

Intensity: (slightly sad), (very excited), (extremely angry)
Layering: (sad)(whispering) I miss you...
Natural: Includes laughter, pauses, sighs when contextually appropriate

Returns raw PCM int16 audio at 44.1kHz mono.

Requires: Fish Speech server running (Docker or local)
  docker compose --profile server up   # API only, port 8080
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class FishSpeechTTS:
    """Fish Speech local TTS provider.

    Voices are referenced by reference_id (a voice stored on the server)
    or by providing reference audio for zero-shot cloning.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        voice: str = "",
        format: str = "wav",
    ):
        self.server_url = server_url.rstrip("/")
        self.reference_id = voice  # Voice reference ID on the server
        self.format = format
        logger.info(
            "fish_speech_tts_initialized",
            server=self.server_url,
            reference_id=self.reference_id or "(default)",
        )

    @property
    def sample_rate(self) -> int:
        return 44100

    def synthesize(self, text: str, emotion: Optional[str] = None) -> np.ndarray:
        """Synthesize text to PCM int16 audio at 44.1kHz.

        Args:
            text: Text to synthesize.
            emotion: Optional emotion tag (e.g. "angry", "whispering", "excited").
                     Prepended as (emotion) for S1 or [emotion] for S2 models.
        """
        import requests

        # Prepend emotion tag if provided
        if emotion:
            text = f"({emotion}) {text}"

        payload = {
            "text": text,
            "format": self.format,
        }
        if self.reference_id:
            payload["reference_id"] = self.reference_id

        response = requests.post(
            f"{self.server_url}/v1/tts",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        audio_bytes = response.content

        # Parse WAV to extract raw PCM
        if self.format == "wav":
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
                audio = np.frombuffer(frames, dtype=np.int16)
                # Resample if not 44.1kHz
                if sr != 44100:
                    from ..immersion.tts_assembler import _resample
                    audio = _resample(audio, sr, 44100)
                return audio
        else:
            # Raw PCM assumed
            return np.frombuffer(audio_bytes, dtype=np.int16)

    async def synthesize_async(self, text: str) -> np.ndarray:
        """Async wrapper -- runs sync HTTP call in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)

    async def synthesize_stream(self, text: str):
        """Streaming synthesis via WebSocket (for real-time playback).

        Yields PCM int16 numpy chunks as they arrive.
        """
        try:
            import aiohttp

            payload = {
                "text": text,
                "format": "pcm",
            }
            if self.reference_id:
                payload["reference_id"] = self.reference_id

            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"{self.server_url}/v1/tts/stream"
                ) as ws:
                    import json
                    await ws.send_str(json.dumps(payload))

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            yield np.frombuffer(msg.data, dtype=np.int16)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.warning("fish_speech_stream_error", error=str(ws.exception()))
                            break

        except ImportError:
            logger.warning("aiohttp_not_installed_for_fish_speech_streaming")
            # Fallback to non-streaming
            audio = await self.synthesize_async(text)
            yield audio

"""Kokoro local TTS provider.

Ultra-lightweight (~82M params, ~400-500MB VRAM) open-source TTS model.
Uses the `kokoro` PyTorch package from hexgrad with espeak-ng for phonemes.

Returns raw PCM int16 audio at 24kHz mono.

Requires:
  pip install kokoro>=0.9.4 soundfile
  System: espeak-ng installed (apt install espeak-ng / brew install espeak)

Voices follow the pattern: {accent}{gender}_{name}
  af_ = American female, am_ = American male
  bf_ = British female, bm_ = British male
  e.g. af_heart, am_fenrir, bf_emma, bm_george

ROCm note:
  Kokoro has known MIOpen GEMM workspace failures on AMD GPUs
  (pytorch#150168) where GPU runs at CPU speed. The model is only 82M
  params so CPU is perfectly fast. Pass device="cpu" or set
  KOKORO_DEVICE=cpu to force CPU inference in ROCm environments.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class KokoroTTS:
    """Kokoro local TTS provider.

    Voices are built-in model voice presets identified by name
    (e.g. "af_heart", "bm_george"). No server required.
    """

    def __init__(self, voice: str = "af_heart", speed: float = 1.0, device: str = ""):
        import torch
        from kokoro import KPipeline

        self._voice = voice
        self._speed = speed

        # Device selection: explicit param > env var > auto-detect
        # On ROCm, Kokoro has MIOpen issues (pytorch#150168) so CPU is
        # recommended. The model is 82M params — CPU is fast enough.
        device = device or os.environ.get("KOKORO_DEVICE", "")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # Derive lang_code from voice prefix: a* -> American, b* -> British
        lang_code = "b" if voice.startswith("b") else "a"
        self._pipeline = KPipeline(lang_code=lang_code, device=device)

        logger.info(
            "kokoro_tts_initialized",
            voice=self._voice,
            speed=self._speed,
            lang_code=lang_code,
            device=self._device,
        )

    @property
    def sample_rate(self) -> int:
        return 24000

    def synthesize(self, text: str, emotion: Optional[str] = None) -> np.ndarray:
        """Synthesize text to PCM int16 audio at 24kHz.

        Args:
            text: Text to synthesize.
            emotion: Accepted for interface compatibility but ignored
                     (Kokoro does not support emotion tags).
        """
        import torch

        chunks: list[np.ndarray] = []
        for _graphemes, _phonemes, audio in self._pipeline(
            text, voice=self._voice, speed=self._speed
        ):
            if audio is None:
                continue
            # KPipeline yields float32 tensors/arrays in [-1.0, 1.0]
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            chunk_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            chunks.append(chunk_int16)

        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.int16)

    async def synthesize_async(self, text: str) -> np.ndarray:
        """Async wrapper -- runs sync inference in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)

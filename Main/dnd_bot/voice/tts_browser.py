"""Browser TTS - no-op sentinel provider.

When active, the /api/tts endpoint returns 503 and the browser's
built-in speechSynthesis API handles TTS on the client side.
"""

from __future__ import annotations

import numpy as np


class BrowserTTS:
    """No-op TTS provider. Signals that TTS is handled client-side."""

    @property
    def sample_rate(self) -> int:
        return 0

    def synthesize(self, text: str) -> np.ndarray:
        return np.array([], dtype=np.int16)

    async def synthesize_async(self, text: str) -> np.ndarray:
        return np.array([], dtype=np.int16)

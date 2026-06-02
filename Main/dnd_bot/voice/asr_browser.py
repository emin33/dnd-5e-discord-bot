"""Browser ASR - no-op sentinel provider.

When active, speech recognition is handled client-side by the
browser's Web Speech API (webkitSpeechRecognition).
"""

from __future__ import annotations

import numpy as np


class BrowserASR:
    """No-op ASR provider. Signals that ASR is handled client-side."""

    @property
    def sample_rate(self) -> int:
        return 0

    def transcribe(self, audio: np.ndarray) -> str:
        return ""

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        return ""

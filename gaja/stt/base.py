from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str
    duration_ms: float
    audio_s: float


class STT(Protocol):
    async def transcribe(self, pcm: np.ndarray, language: str | None = None) -> Transcript: ...

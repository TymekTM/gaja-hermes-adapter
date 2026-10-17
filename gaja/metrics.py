from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

TIMING_FIELDS = (
    "stt_ms",
    "router_ms",
    "tool_select_ms",
    "tool_ms",
    "llm_ttft_ms",
    "first_sentence_ms",
    "tts_first_ms",
    "e2e_first_audio_ms",
)


@dataclass
class TurnMetrics:
    turn: int
    # Input of the turn: "audio" (e2e_first_audio_ms counts from the VAD speech end), "text", or
    # "result" (a Hermes result being announced). route: local, tool, hermes (ack) or cancel.
    kind: str = "text"
    route: str | None = None
    tool: str | None = None
    stt_ms: float | None = None
    stt_wait_ms: float | None = None
    stt_majflt: int | None = None
    stt_audio_s: float | None = None
    router_ms: float | None = None
    tool_select_ms: float | None = None
    tool_ms: float | None = None
    llm_ttft_ms: float | None = None
    first_sentence_ms: float | None = None
    tts_first_ms: float | None = None
    e2e_first_audio_ms: float | None = None
    spec_ms: float | None = None
    committed_speculative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def cancelled_speculation(self) -> bool:
        return self.spec_ms is not None and not self.committed_speculative


class MetricsRecorder:
    def __init__(self, log_path: Path, window: int) -> None:
        self._path = log_path
        self._recent: deque[TurnMetrics] = deque(maxlen=window)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.counters: Counter[str] = Counter()
        self._file = self._path.open("a", encoding="utf-8", buffering=1)

    def count(self, name: str, n: int = 1) -> None:
        """Event counters (e.g. device wake_accepted / wake_rejected:no_phrase), served under /metrics."""
        self.counters[name] += n

    def record(self, session_id: str, metrics: TurnMetrics) -> None:
        if not metrics.cancelled_speculation:
            self._recent.append(metrics)
        line = {"ts": time.time(), "session": session_id, **metrics.to_dict()}
        if getattr(self, "_file", None) is None or self._file.closed:
            self._file = self._path.open("a", encoding="utf-8", buffering=1)
        self._file.write(json.dumps(line) + "\n")
        self._file.flush()

    def close(self) -> None:
        if getattr(self, "_file", None) is not None and not self._file.closed:
            self._file.close()

    def __del__(self) -> None:
        self.close()

    def summary(self) -> dict[str, Any]:
        """Pooled percentiles plus the same per turn kind (audio turns are the headline latency)."""
        result: dict[str, Any] = {"turns": len(self._recent), **_percentiles(list(self._recent))}
        result["by_kind"] = {
            kind: {"turns": len(group), **_percentiles(group)}
            for kind in sorted({m.kind for m in self._recent})
            if (group := [m for m in self._recent if m.kind == kind])
        }
        if self.counters:
            result["counters"] = dict(self.counters)
        return result


def _percentiles(turns: list[TurnMetrics]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in TIMING_FIELDS:
        values = [getattr(m, field) for m in turns if getattr(m, field) is not None]
        result[field] = (
            {"p50": round(float(np.percentile(values, 50)), 1), "p90": round(float(np.percentile(values, 90)), 1)}
            if values
            else None
        )
    return result

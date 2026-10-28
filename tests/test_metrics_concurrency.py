"""The /metrics summary must take one consistent snapshot while turns record in worker threads."""

from __future__ import annotations

import threading
from collections import deque

from gaja.metrics import MetricsRecorder, TurnMetrics


class BlockingDeque(deque):
    """Appends only when the test releases them, so a record() can be held in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def append(self, item) -> None:  # type: ignore[override]
        self.entered.set()
        self.release.wait(5)
        super().append(item)


def test_metrics_summary_takes_its_snapshot_under_the_record_lock(tmp_path) -> None:
    """/metrics iterates the recent deque while record() appends to it from a worker thread.

    A real concurrent append raises "deque mutated during iteration" (a 500 from /metrics).
    Hold a record() inside append and check that summary() cannot iterate until it finishes.
    """
    recorder = MetricsRecorder(tmp_path / "turns.jsonl", 50)
    blocking = BlockingDeque()
    recorder._recent = blocking  # type: ignore[assignment]  # hold record() in flight
    writer = threading.Thread(target=recorder.record, args=("s", TurnMetrics(turn=1)), daemon=True)
    writer.start()
    assert blocking.entered.wait(2.0), "record() never reached the append"

    result: dict = {}

    def read() -> None:
        result["summary"] = recorder.summary()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        reader.join(0.2)
        assert reader.is_alive(), "summary() read the deque while record() was still appending"
    finally:
        blocking.release.set()
        writer.join(2.0)
        reader.join(2.0)
    assert result["summary"]["turns"] == 1
    assert result["summary"]["by_kind"]["text"]["turns"] == 1

"""Event-loop lag monitor: a stalled loop shows up in the journal and in ``/api/diagnostics``.

The device audio path runs on the asyncio loop (uvicorn receive -> VAD -> wake handling), so anything that
blocks the loop for seconds -- page-in of a swapped-out heap, a full garbage collection that touches it, a sync
call that should have been offloaded -- delays a wake past the device's 3 s uplink guard. A ticker task sleeps
``INTERVAL_S`` and measures how late it woke; the lateness is the time the loop was not running tasks.

Cost: one timer per ``INTERVAL_S`` and one callback per garbage collection, both microseconds.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

INTERVAL_S = 0.1
WARN_S = 0.25  # a wake-up this late is logged (the device waits ~3 s before it gives up on the link)
LOG_EVERY_S = 10.0  # at most one WARNING per this long; the rest are counted into the next one
RECENT_S = 600.0  # window of ``worst_recent_ms``
KEEP_MIN_S = 0.05  # lags below this are not kept for the recent window
GC_NOTE_S = 0.05  # a collection this long is named in the warning


class LoopLagMonitor:
    def __init__(
        self,
        *,
        interval_s: float = INTERVAL_S,
        warn_s: float = WARN_S,
        log_every_s: float = LOG_EVERY_S,
        recent_s: float = RECENT_S,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._interval = interval_s
        self._warn = warn_s
        self._log_every = log_every_s
        self._recent_s = recent_s
        self._clock = clock
        self._task: asyncio.Task[None] | None = None
        self._recent: deque[tuple[float, float]] = deque()  # (clock time, lag s)
        self._worst_ever = 0.0
        self._stalls = 0
        self._last_stall: dict[str, Any] | None = None
        self._unreported = 0
        self._unreported_worst = 0.0
        self._last_logged = float("-inf")
        self._gc_started = 0.0
        self._gc_worst = 0.0  # longest collection since the last tick

    # ---- gc attribution ---------------------------------------------------------------------

    def _on_gc(self, phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            self._gc_started = self._clock()
        elif self._gc_started:
            self._gc_worst = max(self._gc_worst, self._clock() - self._gc_started)
            self._gc_started = 0.0

    # ---- lifecycle --------------------------------------------------------------------------

    def start(self) -> None:
        if self._task is not None:
            return
        gc.callbacks.append(self._on_gc)
        self._task = asyncio.create_task(self._run(), name="loop-lag-monitor")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            expected = self._clock() + self._interval
            await asyncio.sleep(self._interval)
            self.observe(self._clock() - expected)

    # ---- measurement ------------------------------------------------------------------------

    def observe(self, lag_s: float) -> None:
        """Record one wake-up that was ``lag_s`` late (public so tests can drive it without waiting)."""
        now = self._clock()
        gc_s, self._gc_worst = self._gc_worst, 0.0
        lag_s = max(0.0, lag_s)
        self._worst_ever = max(self._worst_ever, lag_s)
        if lag_s >= KEEP_MIN_S:
            self._recent.append((now, lag_s))
        while self._recent and now - self._recent[0][0] > self._recent_s:
            self._recent.popleft()
        if lag_s < self._warn:
            return
        self._stalls += 1
        self._last_stall = {
            "at": time.time(),
            "lag_ms": round(lag_s * 1000),
            "gc_ms": round(gc_s * 1000),
        }
        self._unreported += 1
        self._unreported_worst = max(self._unreported_worst, lag_s)
        if now - self._last_logged < self._log_every:
            return
        gc_note = f", a garbage collection took {gc_s * 1000:.0f} ms" if gc_s >= GC_NOTE_S else ""
        log.warning(
            "event loop blocked for %.0f ms (%d stall%s since the last report, worst %.0f ms%s)",
            lag_s * 1000,
            self._unreported,
            "" if self._unreported == 1 else "s",
            self._unreported_worst * 1000,
            gc_note,
        )
        self._last_logged = now
        self._unreported = 0
        self._unreported_worst = 0.0

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        recent = [lag for t, lag in self._recent if now - t <= self._recent_s]
        return {
            "running": self._task is not None and not self._task.done(),
            "warn_ms": round(self._warn * 1000),
            "worst_recent_ms": round(max(recent, default=0.0) * 1000),
            "recent_window_s": round(self._recent_s),
            "worst_ever_ms": round(self._worst_ever * 1000),
            "stalls": self._stalls,
            "last_stall": self._last_stall,
        }


def freeze_startup_heap() -> int:
    """Move everything alive now (modules, models, config) out of the garbage collector's reach.

    A full collection walks every tracked object; on a heap that has been swapped out it faults the whole
    thing back in on the loop thread (26 ms resident for the bare imports, seconds from swap). The startup
    objects live as long as the process, so there is nothing to collect in them. Returns the frozen count.
    """
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()

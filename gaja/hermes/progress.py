from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

IMMEDIATE_TYPES = frozenset({"tool_call", "permission_cancelled"})
TEXT_DELTA = "text_delta"


@dataclass(frozen=True)
class ProgressItem:
    type: str
    text: str


class ProgressCoalescer:
    """Merges Hermes progress so the client sees at most one update per interval, except immediate types."""

    def __init__(self, interval_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._interval = interval_s
        self._clock = clock
        self._last: float | None = None
        self._pending: list[ProgressItem] = []

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    def push(self, kind: str, text: str) -> list[ProgressItem]:
        if kind in IMMEDIATE_TYPES:
            return [*self.flush(), ProgressItem(kind, text)]
        if kind == TEXT_DELTA and self._pending and self._pending[-1].type == TEXT_DELTA:
            self._pending[-1] = ProgressItem(TEXT_DELTA, self._pending[-1].text + text)
        else:
            self._pending = [item for item in self._pending if item.type != kind]
            self._pending.append(ProgressItem(kind, text))
        now = self._clock()
        if self._last is None or now - self._last >= self._interval:
            return self.flush()
        return []

    def flush(self) -> list[ProgressItem]:
        if not self._pending:
            return []
        items, self._pending = self._pending, []
        self._last = self._clock()
        return items

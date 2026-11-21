"""When Gaja itself is using the LLM, so clients sharing llama-server can step aside.

Gaja marks a voice turn from its first sign (a device wake, a user turn starting) and every request of
its own LLM client. External traffic on the shared llama-server (gaja/api/llm_proxy.py) does not start
while Gaja is busy or within ``hold_s`` of its last activity: a turn is several LLM calls in a row
(router, reply, tool follow-up, priming the next turn) with short gaps between them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager


POLL_S = 0.05


class LLMActivity:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._inflight = 0
        self._last = float("-inf")

    def touch(self) -> None:
        """Gaja is about to need the LLM (a wake, a user turn starting)."""
        self._last = self._clock()

    @asynccontextmanager
    async def request(self) -> AsyncIterator[None]:
        """Wraps one of Gaja's own LLM requests."""
        self._inflight += 1
        self.touch()
        try:
            yield
        finally:
            self._inflight -= 1
            self.touch()

    def busy(self, hold_s: float) -> bool:
        return self._inflight > 0 or self._clock() - self._last < hold_s

    def idle_in(self, hold_s: float) -> float:
        """Seconds until Gaja counts as idle if nothing else happens (0 when it already is)."""
        if self._inflight > 0:
            return hold_s
        return max(0.0, hold_s - (self._clock() - self._last))

    async def wait_idle(self, hold_s: float, timeout_s: float) -> bool:
        """Waits until Gaja has been idle for ``hold_s``; False when ``timeout_s`` runs out first."""
        deadline = self._clock() + timeout_s
        while self.busy(hold_s):
            left = deadline - self._clock()
            if left <= 0:
                return False
            # Polled, not an Event: one module-wide instance outlives any single event loop.
            await asyncio.sleep(min(left, POLL_S))
        return True


ACTIVITY = LLMActivity()

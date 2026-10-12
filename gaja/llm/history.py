from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


class History:
    """Per-session conversation, oldest first, kept under a turn count and character budget.

    Trimming happens in big steps: once over budget, the oldest turns are dropped until the history
    is at most ``trim_to`` of the budget, cutting at a user turn. Between trims the message list
    only grows at the end, so llama-server can reuse the cached prompt prefix for every turn.
    """

    def __init__(self, system: str, max_turns: int, max_chars: int, *, trim_to: float = 0.5) -> None:
        self._system = system
        self._max_turns = max_turns
        self._max_chars = max_chars
        self._trim_to = trim_to
        self._turns: list[Turn] = []

    def set_system(self, system: str) -> None:
        self._system = system

    def add(self, role: str, content: str) -> None:
        if not content.strip():
            return
        self._turns = self._fit([*self._turns, Turn(role, content)])

    def extend(self, turns: Iterable[tuple[str, str]]) -> None:
        for role, content in turns:
            self.add(role, content)

    def context(self) -> list[dict[str, Any]]:
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self._system}, *self.context()]

    def messages_with(self, pending: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
        """The messages the history would hold after adding ``pending``, without adding them."""
        turns = list(self._turns)
        for role, content in pending:
            if content.strip():
                turns = self._fit([*turns, Turn(role, content)])
        return [{"role": "system", "content": self._system}, *({"role": t.role, "content": t.content} for t in turns)]

    def messages_after(self, role: str, content: str) -> list[dict[str, Any]]:
        return self.messages_with([(role, content)])

    def _over(self, turns: list[Turn], max_turns: int, max_chars: int) -> bool:
        return len(turns) > 2 * max_turns or sum(len(t.content) for t in turns) > max_chars

    def _fit(self, turns: list[Turn]) -> list[Turn]:
        if not self._over(turns, self._max_turns, self._max_chars):
            return turns
        keep_turns = max(1, int(self._max_turns * self._trim_to))
        keep_chars = int(self._max_chars * self._trim_to)
        while len(turns) > 1 and self._over(turns, keep_turns, keep_chars):
            turns = turns[1:]
            while len(turns) > 1 and turns[0].role != "user":
                turns = turns[1:]
        if self._over(turns, self._max_turns, self._max_chars) and turns:
            # A single turn over the whole budget: keep its tail.
            last = turns[-1]
            turns = [Turn(last.role, last.content[-self._max_chars :])]
        return turns

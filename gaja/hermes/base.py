from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gaja.household import Identity


@dataclass
class HermesEvent:
    kind: Literal["accepted", "progress", "result", "error"]
    text: str
    data: dict[str, Any] | None = None


@runtime_checkable
class HermesClient(Protocol):
    def submit(
        self,
        task: str,
        *,
        context: list[dict[str, Any]],
        session_id: str,
        language: str | None = None,
        profile: str = "default",
        identity: Identity | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[HermesEvent]: ...

    async def health(self) -> bool: ...

    def ready(self) -> bool: ...

    async def forget(self, session_id: str) -> None: ...

    async def aclose(self) -> None: ...


class StubHermes:
    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    async def submit(
        self,
        task: str,
        *,
        context: list[dict[str, Any]],
        session_id: str,
        language: str | None = None,
        profile: str = "default",
        identity: Identity | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[HermesEvent]:
        yield HermesEvent("accepted", "")
        await asyncio.sleep(self._delay_s)
        yield HermesEvent("result", "Hermes nie jest jeszcze podłączony, więc tego teraz nie zrobię.")

    async def health(self) -> bool:
        return True

    def ready(self) -> bool:
        return True

    async def forget(self, session_id: str) -> None:
        return None

    async def aclose(self) -> None:
        return None

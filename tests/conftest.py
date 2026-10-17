from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import gaja.config
from gaja.config import Settings
from gaja.hermes.base import HermesEvent
from gaja.llm.client import LLMEvent, LLMUnavailable
from gaja.llm.prompt import SystemPrompt
from gaja.llm.router import Router
from gaja.metrics import MetricsRecorder
from gaja.pipeline import Services
from gaja.stt.base import Transcript
from gaja.text.style import Style
from gaja.tts.base import EnginePool


class FakeSTT:
    engine = "fake"
    requested = "fake"
    device = "cpu"
    fallback_reason = None

    def __init__(self, text: str = "ile to dwa plus dwa", *, duration_ms: float = 3.0, wait_ms: float = 0.0) -> None:
        self.text = text
        self.duration_ms = duration_ms
        self.wait_ms = wait_ms
        self.last_hotwords: str | None = None
        self.last_prompt: str | None = None

    async def transcribe(
        self,
        pcm: np.ndarray,
        language: str | None = None,
        *,
        hotwords: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        self.last_hotwords = hotwords
        self.last_prompt = prompt
        return Transcript(
            text=self.text,
            language="pl",
            duration_ms=self.duration_ms,
            audio_s=len(pcm) / 16000,
            wait_ms=self.wait_ms,
        )


class FakeTTS:
    name = "fake"
    sample_rate = 24000
    languages = frozenset({"pl", "en"})

    async def synthesize(self, text: str, style: Style, language: str) -> np.ndarray:
        return np.full(2400, 0.1, dtype=np.float32)

    async def warm_up(self) -> None:
        return None


class FakeLLM:
    def __init__(
        self,
        replies: list[list[LLMEvent]] | None = None,
        *,
        down: bool = False,
        hang: bool = False,
        p_hermes: float = 0.0,
        p_tool: float = 0.0,
        tool_call: str = "",
    ) -> None:
        self._replies = list(replies or [])
        self._down = down
        self._hang = hang
        self._p_hermes = p_hermes
        self._p_tool = p_tool
        self._tool_call = tool_call
        self.generated: list[str] = []
        self.chat_primes: list[list[dict[str, Any]]] = []
        self.calls: list[tuple[list[dict[str, Any]], bool]] = []
        self.prompts: list[str] = []
        self.primed: list[str] = []
        self.closed = False

    async def complete(self, prompt: str, *, timeout_s: float, slot: int | None = None) -> dict[str, Any]:
        if self._down:
            raise LLMUnavailable("fake down")
        self.prompts.append(prompt)
        hermes, tool = self._p_hermes, self._p_tool
        local = 1.0 - hermes - tool
        top = []
        if hermes > 0:
            top.append({"token": "Hermes", "logprob": math.log(hermes)})
        if tool > 0:
            top.append({"token": " TOOL", "logprob": math.log(tool)})
        if local > 0:
            top.append({"token": "Lokalnie", "logprob": math.log(local)})
        return {"completion_probabilities": [{"top_logprobs": top}]}

    async def generate(
        self, prompt: str, *, n_predict: int, grammar: str, timeout_s: float, slot: int | None = None
    ) -> str:
        if self._down:
            raise LLMUnavailable("fake down")
        self.generated.append(prompt)
        return self._tool_call

    async def prime_chat(self, messages: list[dict[str, Any]]) -> None:
        if self._down:
            raise LLMUnavailable("fake down")
        self.chat_primes.append(messages)

    async def prime(self, prompt: str, *, timeout_s: float, slot: int | None = None) -> None:
        if self._down:
            raise LLMUnavailable("fake down")
        self.primed.append(prompt)

    async def stream(self, messages: list[dict[str, Any]], *, allow_tools: bool) -> AsyncIterator[LLMEvent]:
        self.calls.append((messages, allow_tools))
        if self._down:
            raise LLMUnavailable("fake down")
        events = self._replies.pop(0) if self._replies else []
        try:
            for event in events:
                yield event
            if self._hang:
                await asyncio.sleep(3600)
        finally:
            self.closed = True

    async def health(self) -> bool:
        return not self._down

    async def aclose(self) -> None:
        return None


class FakeHermes:
    def __init__(self, result: str = "Pogoda: dwanaście stopni.", error: str | None = None, *, hang: bool = False) -> None:
        self.result = result
        self.error = error
        self.hang = hang
        self.tasks: list[str] = []
        self.contexts: list[list[dict[str, Any]]] = []
        self.languages: list[str | None] = []
        self.forgotten: list[str] = []
        self.profiles: list[str] = []
        self.identities: list[Any] = []
        self.instructions: list[str | None] = []
        self.resets: list[str] = []
        self.new_topics: list[str] = []
        self.cancelled = False
        self.release = asyncio.Event() if hang else None

    async def submit(
        self,
        task: str,
        *,
        context: list[dict[str, Any]],
        session_id: str,
        language: str | None = None,
        profile: str = "default",
        identity: Any = None,
        instructions: str | None = None,
    ) -> AsyncIterator[HermesEvent]:
        self.tasks.append(task)
        self.profiles.append(profile)
        self.identities.append(identity)
        self.instructions.append(instructions)
        self.contexts.append(context)
        self.languages.append(language)
        yield HermesEvent("accepted", "")
        if self.hang:
            try:
                assert self.release is not None
                await self.release.wait()
            except (asyncio.CancelledError, GeneratorExit):
                self.cancelled = True
                raise
        if self.error is not None:
            yield HermesEvent("error", self.error)
            return
        yield HermesEvent("result", self.result)

    async def health(self) -> bool:
        return True

    def ready(self) -> bool:
        return True

    async def forget(self, session_id: str) -> None:
        self.forgotten.append(session_id)

    def reset(self, person_key: str) -> None:
        self.resets.append(person_key)

    def new_topic(self, session_id: str) -> None:
        self.new_topics.append(session_id)

    async def aclose(self) -> None:
        return None


def fake_vad() -> Any:
    def prob(window: np.ndarray) -> float:
        return 1.0 if float(np.abs(window).max()) > 0.05 else 0.0

    return prob


@pytest.fixture(autouse=True)
def _no_local_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # config.local.toml is this machine's deployment (e.g. [devices] enabled); tests see only config.toml.
    monkeypatch.setattr(gaja.config, "LOCAL_CONFIG_PATH", tmp_path / "no-config.local.toml")


@pytest.fixture
def make_services(tmp_path: Path):
    def factory(
        *,
        llm: FakeLLM | None = None,
        stt: FakeSTT | None = None,
        hermes: FakeHermes | None = None,
        router: bool = False,
        tools: Any = None,
        cfg: dict[str, Any] | None = None,
    ) -> Services:
        settings = Settings(
            metrics={"turns_log": tmp_path / "turns.jsonl", "window": 50},
            hermes={"stub_delay_s": 0.0, "backend": "stub"},
            tts={"engine_by_language": {"pl": "fake", "en": "fake"}},
            **(cfg or {}),
        )
        tts = FakeTTS()
        llm = llm or FakeLLM()
        return Services(
            cfg=settings,
            stt=stt or FakeSTT(),
            tts=EnginePool({"fake": lambda: tts}),
            llm=llm,
            prompt=SystemPrompt(text="sys", tools=[]),
            hermes=hermes or FakeHermes(),
            metrics=MetricsRecorder(tmp_path / "turns.jsonl", 50),
            new_vad=fake_vad,
            router=Router(
                llm,
                "router-sys",
                threshold=0.97,
                history_turns=2,
                timeout_s=2.0,
                tools=tools.schemas() if tools is not None else None,
            )
            if router
            else None,
            tools=tools,
        )

    return factory

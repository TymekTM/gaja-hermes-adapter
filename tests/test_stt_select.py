import asyncio

import numpy as np
import pytest

from gaja.stt import select
from gaja.stt.base import Transcript
from gaja.stt.select import SwitchingSTT


class FakeEngine:
    def __init__(self, label: str, *, fail_load: str | None = None, fail_infer: str | None = None) -> None:
        self.label = label
        self.fail_load = fail_load
        self.fail_infer = fail_infer
        self.calls = 0

    async def warm_up(self) -> None:
        if self.fail_load:
            raise RuntimeError(self.fail_load)

    async def transcribe(self, pcm: np.ndarray, language: str | None = None) -> Transcript:
        self.calls += 1
        if self.fail_infer:
            raise RuntimeError(self.fail_infer)
        return Transcript(text=self.label, language=language or "pl", duration_ms=1.0, audio_s=len(pcm) / 16000)


@pytest.fixture
def whistle(monkeypatch):
    built: list[FakeEngine] = []

    def make() -> FakeEngine:
        engine = FakeEngine("whistle")
        built.append(engine)
        return engine

    monkeypatch.setattr(select, "WhistleSTT", make)
    return built


PCM = np.zeros(16000, dtype=np.float32)


def test_preferred_engine_serves_when_it_loads(whistle) -> None:
    primary = FakeEngine("gpu")
    stt = SwitchingSTT("faster-whisper", lambda: primary)
    asyncio.run(stt.load())
    result = asyncio.run(stt.transcribe(PCM, "pl"))
    assert result.text == "gpu"
    assert (stt.engine, stt.device, stt.fallback_reason) == ("faster-whisper", "cuda", None)
    assert whistle == []


def test_parakeet_reports_its_own_name_on_the_cpu(whistle) -> None:
    stt = SwitchingSTT("parakeet", lambda: FakeEngine("parakeet"))
    asyncio.run(stt.load())
    assert asyncio.run(stt.transcribe(PCM, "pl")).text == "parakeet"
    assert (stt.engine, stt.device, stt.fallback_reason) == ("parakeet", "cpu", None)


def test_cuda_oom_at_load_falls_back_to_whistle_and_reports_it(whistle) -> None:
    stt = SwitchingSTT("faster-whisper", lambda: FakeEngine("gpu", fail_load="CUDA failed with error out of memory"))
    asyncio.run(stt.load())
    result = asyncio.run(stt.transcribe(PCM, "pl"))
    assert result.text == "whistle"
    assert (stt.engine, stt.device) == ("whistle", "cpu")
    assert stt.requested == "faster-whisper"
    assert "out of memory" in stt.fallback_reason


def test_missing_dependency_at_construction_falls_back(whistle) -> None:
    def broken() -> FakeEngine:
        raise ImportError("No module named 'faster_whisper'")

    stt = SwitchingSTT("faster-whisper", broken)
    asyncio.run(stt.load())
    assert stt.engine == "whistle"
    assert "faster_whisper" in stt.fallback_reason


class FlakyEngine(FakeEngine):
    """Fails the first ``failures`` calls (transcribe or warm_up), then works."""

    def __init__(self, failures: int) -> None:
        super().__init__("gpu")
        self.failures = failures
        self.warmups = 0

    async def warm_up(self) -> None:
        self.warmups += 1
        await self._maybe_fail()

    async def transcribe(self, pcm: np.ndarray, language: str | None = None) -> Transcript:
        self.calls += 1
        await self._maybe_fail()
        return Transcript(text="gpu", language=language or "pl", duration_ms=1.0, audio_s=len(pcm) / 16000)

    async def _maybe_fail(self) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("CUDA failed with error out of memory")


def test_transient_inference_error_is_retried_on_the_gpu(whistle) -> None:
    primary = FlakyEngine(failures=0)
    stt = SwitchingSTT("faster-whisper", lambda: primary, retry_delay_s=0)

    async def scenario():
        await stt.load()
        primary.failures = 1
        return await stt.transcribe(PCM, "pl")

    assert asyncio.run(scenario()).text == "gpu"
    assert primary.calls == 2
    assert not stt.degraded and stt.engine == "faster-whisper"
    assert whistle == []


def test_repeated_inference_failure_degrades_then_probe_recovers(whistle) -> None:
    primary = FlakyEngine(failures=0)
    stt = SwitchingSTT("faster-whisper", lambda: primary, retry_delay_s=0, cooldown_s=0.05)

    async def scenario():
        await stt.load()
        primary.failures = 2
        first = await stt.transcribe(PCM, "pl")
        degraded = (stt.degraded, stt.engine, stt.fallback_reason)
        second = await stt.transcribe(PCM, "pl")
        await asyncio.sleep(0.2)
        third = await stt.transcribe(PCM, "pl")
        await stt.aclose()
        return first, degraded, second, third

    first, degraded, second, third = asyncio.run(scenario())
    assert first.text == "whistle"
    assert degraded[0] is True and degraded[1] == "whistle" and "inference" in degraded[2]
    assert second.text == "whistle"
    assert third.text == "gpu"
    assert not stt.degraded and stt.fallback_reason is None


def test_failing_probe_backs_off(whistle) -> None:
    primary = FlakyEngine(failures=0)
    stt = SwitchingSTT("faster-whisper", lambda: primary, retry_delay_s=0, cooldown_s=0.02, max_cooldown_s=0.05)

    async def scenario():
        await stt.load()
        primary.failures = 4
        await stt.transcribe(PCM, "pl")
        await asyncio.sleep(0.3)
        await stt.aclose()

    asyncio.run(scenario())
    assert primary.warmups >= 3
    assert not stt.degraded


def test_whistle_requested_loads_only_whistle(whistle) -> None:
    stt = SwitchingSTT("whistle")
    asyncio.run(stt.load())
    assert len(whistle) == 1
    assert (stt.engine, stt.device, stt.fallback_reason) == ("whistle", "cpu", None)
    assert asyncio.run(stt.transcribe(PCM, None)).text == "whistle"

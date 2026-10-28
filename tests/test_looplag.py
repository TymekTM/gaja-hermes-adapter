"""Loop-lag monitor and the off-loop VAD build (docs/TODO.md: first wake after an idle)."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import threading
import time

import numpy as np
import pytest

from gaja.audio.vad import WINDOW, warm_vad
from gaja.looplag import LoopLagMonitor, freeze_startup_heap
from gaja.pipeline import Session
from tests.test_pipeline_close import RecordingOutbox


async def _max_lag(until: asyncio.Event, interval: float = 0.01) -> float:
    """Worst lateness of a short sleep while ``until`` is not set: what a loop-bound caller would feel."""
    worst = 0.0
    while not until.is_set():
        started = time.perf_counter()
        await asyncio.sleep(interval)
        worst = max(worst, time.perf_counter() - started - interval)
    return worst


@pytest.mark.asyncio
async def test_monitor_warns_when_the_loop_is_blocked(caplog: pytest.LogCaptureFixture) -> None:
    monitor = LoopLagMonitor(interval_s=0.01, warn_s=0.1, log_every_s=0.0)
    monitor.start()
    try:
        await asyncio.sleep(0.05)
        with caplog.at_level(logging.WARNING, logger="gaja.looplag"):
            time.sleep(0.3)  # a sync call on the loop
            await asyncio.sleep(0.05)
    finally:
        await monitor.stop()
    assert any("event loop blocked" in r.getMessage() for r in caplog.records)
    snap = monitor.snapshot()
    assert snap["worst_recent_ms"] >= 250
    assert snap["worst_ever_ms"] >= 250
    assert snap["stalls"] >= 1
    assert snap["last_stall"]["lag_ms"] >= 250


@pytest.mark.asyncio
async def test_monitor_is_quiet_on_a_healthy_loop(caplog: pytest.LogCaptureFixture) -> None:
    monitor = LoopLagMonitor(interval_s=0.01, warn_s=0.2, log_every_s=0.0)
    monitor.start()
    with caplog.at_level(logging.WARNING, logger="gaja.looplag"):
        await asyncio.sleep(0.3)
    await monitor.stop()
    assert not caplog.records
    assert monitor.snapshot()["stalls"] == 0
    assert monitor.snapshot()["last_stall"] is None


def test_monitor_rate_limits_the_warning_and_counts_the_rest(caplog: pytest.LogCaptureFixture) -> None:
    now = [100.0]
    monitor = LoopLagMonitor(warn_s=0.25, log_every_s=10.0, clock=lambda: now[0])
    with caplog.at_level(logging.WARNING, logger="gaja.looplag"):
        monitor.observe(0.4)
        now[0] += 1
        monitor.observe(0.9)  # inside the rate limit: counted, not logged
        assert len(caplog.records) == 1
        now[0] += 10
        monitor.observe(0.3)
    assert len(caplog.records) == 2
    assert "2 stalls since the last report, worst 900 ms" in caplog.records[1].getMessage()
    assert monitor.snapshot()["stalls"] == 3


def test_monitor_recent_window_forgets_old_stalls() -> None:
    now = [0.0]
    monitor = LoopLagMonitor(recent_s=60.0, clock=lambda: now[0])
    monitor.observe(0.5)
    now[0] += 61
    snap = monitor.snapshot()
    assert snap["worst_recent_ms"] == 0
    assert snap["worst_ever_ms"] == 500


def test_monitor_names_a_slow_garbage_collection(caplog: pytest.LogCaptureFixture) -> None:
    monitor = LoopLagMonitor(warn_s=0.1, log_every_s=0.0)
    monitor._on_gc("start", {})
    time.sleep(0.08)
    monitor._on_gc("stop", {})
    with caplog.at_level(logging.WARNING, logger="gaja.looplag"):
        monitor.observe(0.3)
    assert monitor.snapshot()["last_stall"]["gc_ms"] >= 70
    assert "garbage collection took" in caplog.records[0].getMessage()
    monitor.observe(0.3)  # the next tick starts from zero again
    assert monitor.snapshot()["last_stall"]["gc_ms"] == 0


@pytest.mark.asyncio
async def test_monitor_registers_and_removes_its_gc_callback() -> None:
    monitor = LoopLagMonitor()
    monitor.start()
    assert monitor._on_gc in gc.callbacks
    await monitor.stop()
    assert monitor._on_gc not in gc.callbacks


def test_freeze_startup_heap_moves_objects_out_of_the_collector() -> None:
    try:
        assert freeze_startup_heap() > 0
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()


@pytest.mark.asyncio
async def test_warm_vad_builds_and_runs_a_window_off_the_loop() -> None:
    loop_thread = threading.get_ident()
    seen: dict[str, object] = {}

    def new_vad():
        seen["build_thread"] = threading.get_ident()
        time.sleep(0.3)  # an onnxruntime session create

        def prob(window: np.ndarray) -> float:
            seen["window"] = window.shape
            seen["infer_thread"] = threading.get_ident()
            return 0.0

        return prob

    done = asyncio.Event()
    ticker = asyncio.create_task(_max_lag(done))
    ms = await warm_vad(new_vad)
    done.set()
    lag = await ticker
    assert ms >= 250
    assert seen["build_thread"] != loop_thread
    assert seen["infer_thread"] != loop_thread
    assert seen["window"] == (WINDOW,)
    assert lag < 0.1


@pytest.mark.asyncio
async def test_hello_builds_the_vad_off_the_loop(make_services) -> None:
    """The per-session VAD build (first use: the onnxruntime session) must not stall the loop at hello."""
    services = make_services()
    inner = services.new_vad

    def slow_new_vad():
        time.sleep(0.3)
        return inner()

    services.new_vad = slow_new_vad
    session = Session(services, RecordingOutbox())
    done = asyncio.Event()
    ticker = asyncio.create_task(_max_lag(done))
    await session.handle_text(json.dumps({"type": "hello", "lang": "pl"}))
    done.set()
    assert await ticker < 0.1
    await session.close()

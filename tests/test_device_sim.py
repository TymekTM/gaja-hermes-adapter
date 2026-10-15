"""clients/device_sim.py: helpers and an in-process run against the embedded device listener (stub STT/LLM/TTS)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from gaja.devices import listener as listener_module
from gaja.devices.listener import DeviceListener
from gaja.devices.registry import DeviceRegistry
from gaja.llm.client import TextDelta
from tests.conftest import FakeLLM
from tests.test_device_protocol import FAST, TOKEN, ScriptedSTT

_spec = importlib.util.spec_from_file_location("device_sim", Path(__file__).resolve().parent.parent / "clients" / "device_sim.py")
sim = importlib.util.module_from_spec(_spec)
sys.modules["device_sim"] = sim
_spec.loader.exec_module(sim)


@pytest.fixture(autouse=True)
def _no_auth_delay(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "AUTH_FAIL_DELAY_S", 0.0)


def write_speech(path: Path, seconds: float, rate: int = 16000) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    sf.write(path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate, subtype="PCM_16")
    return path


# ---- helpers -------------------------------------------------------------------------------------


def test_load_wav_resamples_and_downmixes_to_16k_mono(tmp_path) -> None:
    path = tmp_path / "a.wav"
    stereo = np.stack([np.sin(np.arange(48000) / 20), np.sin(np.arange(48000) / 20)], axis=1).astype(np.float32) * 0.5
    sf.write(path, stereo, 48000, subtype="PCM_16")
    samples = sim.load_wav_16k(str(path))
    assert samples.dtype == np.dtype("<i2")
    assert abs(len(samples) - 16000) <= 2


def test_frames_are_20ms_and_the_last_one_is_padded() -> None:
    frames = sim.split_frames(np.ones(600, dtype="<i2"))
    assert [len(f) for f in frames] == [640, 640]
    assert frames[1][-2:] == b"\x00\x00"


def test_preroll_is_the_audio_before_the_wake_instant() -> None:
    samples = np.arange(16000 * 3, dtype=np.int32).astype("<i2")
    preroll, live = sim.split_preroll(samples, 2.0, 500)
    assert len(preroll) == 8000 and len(live) == 16000
    assert preroll[-1] == samples[31999] and live[0] == samples[32000]
    short, _ = sim.split_preroll(samples, 0.1, 1500)
    assert len(short) == 1600


def test_playback_cancel_drops_what_was_not_played_yet() -> None:
    pb = sim.Playback()
    pb.rate = 16000
    pb.feed(bytes(3200 * 2), now=10.0)  # 0.2 s, plays 10.0 .. 10.2
    pb.feed(bytes(3200 * 2), now=10.0)  # queued behind it: 10.2 .. 10.4
    assert pb.playing(10.1)
    dropped = pb.cancel(10.1)
    assert dropped == pytest.approx(0.3)
    assert len(pb.pcm()) == 1600 * 2
    assert not pb.playing(10.1)


# ---- in-process runs -----------------------------------------------------------------------------


def run_sim(make_services, tmp_path, *, stt, llm=None, devices=None, **opts: Any):
    registry_path = tmp_path / "devices.toml"
    registry_path.write_text(f'[devices.sim-1]\ntoken = "{TOKEN}"\n', encoding="utf-8")
    registry_path.chmod(0o600)
    services = make_services(cfg={"devices": {**FAST, **(devices or {})}}, stt=stt, **({"llm": llm} if llm else {}))
    wav = write_speech(tmp_path / "in.wav", opts.pop("seconds", 2.0))
    out = tmp_path / "out.wav"

    async def scenario() -> Any:
        listener = DeviceListener(services, DeviceRegistry(registry_path), port=0)
        assert await listener.start()
        try:
            options = sim.SimOptions(url=f"ws://127.0.0.1:{listener.port}/device", token=opts.pop("token", TOKEN), wav=str(wav),
                                     wake_at=0.5, out=str(out), timeout=10.0, verbose=bool(__import__("os").environ.get("SIM_VERBOSE")), **opts)
            return await sim.DeviceSim(options).run()
        finally:
            await listener.stop()

    return asyncio.run(scenario()), out, services


def test_sim_full_turn_writes_audio_and_measures_latency(make_services, tmp_path) -> None:
    llm = FakeLLM([[TextDelta("Jest słonecznie. ")]])
    result, out, services = run_sim(make_services, tmp_path, stt=ScriptedSTT("hej gaja jaka pogoda"), llm=llm)
    assert result.exit_code == 0
    kinds = [e["type"] for e in result.events]
    assert "wake_accepted" in kinds and "follow_up" in kinds
    assert result.timings["wake_to_first_audio_ms"] > 0
    data, rate = sf.read(out, dtype="int16")
    assert rate == result.rate == 16000 and len(data) > 0
    transcript = next(e for e in result.events if e["type"] == "transcript")
    assert transcript["text"] == "jaka pogoda"


def test_sim_no_aec_offers_a_24k_only_speaker(make_services, tmp_path) -> None:
    llm = FakeLLM([[TextDelta("Jest słonecznie. ")]])
    result, out, _ = run_sim(make_services, tmp_path, stt=ScriptedSTT("hej gaja jaka pogoda"), llm=llm, aec=False, out_rates=(24000,))
    assert result.exit_code == 0 and result.rate == 24000
    assert sf.info(out).samplerate == 24000


def test_sim_wake_rejected_exits_3(make_services, tmp_path) -> None:
    result, out, services = run_sim(make_services, tmp_path, stt=ScriptedSTT("hej maja jaka pogoda"))
    assert result.exit_code == 3
    rejected = next(e for e in result.events if e["type"] == "wake_rejected")
    assert rejected["reason"] == "no_phrase"
    assert "wake_to_verdict_ms" in result.timings
    assert not out.exists()


def test_sim_bad_token_exits_2(make_services, tmp_path) -> None:
    result, _, _ = run_sim(make_services, tmp_path, stt=ScriptedSTT("x"), token="z" * 40)
    assert result.exit_code == 2


def test_sim_barge_in_cuts_the_reply_with_a_second_wake(make_services, tmp_path) -> None:
    llm = FakeLLM([[TextDelta("To jest bardzo długa odpowiedź. ")], [TextDelta("Dobrze. ")]])
    result, _, _ = run_sim(
        make_services, tmp_path, stt=ScriptedSTT("hej gaja opowiedz coś", "hej gaja stop"), llm=llm, barge_in_at=0.0,
        devices={"follow_up_ms": 300},
    )
    assert result.exit_code == 0
    assert sum(1 for e in result.events if e["type"] == "wake_accepted") == 2
    assert [e["text"] for e in result.events if e["type"] == "transcript"][-1] == "stop"


def test_sim_follow_up_wav_is_sent_without_a_wake_phrase(make_services, tmp_path) -> None:
    llm = FakeLLM([[TextDelta("Jest słonecznie. ")], [TextDelta("Dziękuję. ")]])
    follow = write_speech(tmp_path / "follow.wav", 1.0)
    result, _, services = run_sim(
        make_services, tmp_path, stt=ScriptedSTT("hej gaja jaka pogoda", "hej gaja jaka pogoda", "a jutro"), llm=llm, follow_up_wav=str(follow),
    )
    assert result.exit_code == 0
    assert [e["text"] for e in result.events if e["type"] == "transcript"] == ["jaka pogoda", "a jutro"]


# ---- record mode (protocol v1.10) ----------------------------------------------------------------


def test_sim_record_mode_streams_a_take_end_to_end(make_services, tmp_path) -> None:
    """The sim records through the admin API like the firmware will: start, frames, mark, stop, storage."""
    from gaja.admin.auth import write_credentials
    from gaja.devices.listener import DeviceListener
    from gaja.server import create_app
    from fastapi.testclient import TestClient

    password = "correct horse battery"
    creds = tmp_path / "admin.toml"
    write_credentials(creds, password)
    rec_dir = tmp_path / "rec"
    services = make_services(cfg={
        "devices": {**FAST},
        "wakeword": {"recordings_dir": str(rec_dir), "max_total_mb": 64},
        "admin": {"credentials_file": str(creds)},
    })
    services.stt = ScriptedSTT("hej gaja")  # any frame that leaked to the pipeline would be transcribed
    registry_path = tmp_path / "devices2.toml"
    registry_path.write_text(f'[devices.sim-1]\ntoken = "{TOKEN}"\n', encoding="utf-8")
    wav = write_speech(tmp_path / "take.wav", 2.0)

    async def scenario() -> Any:
        listener = DeviceListener(services, DeviceRegistry(registry_path), port=0)
        assert await listener.start()
        try:
            options = sim.SimOptions(url=f"ws://127.0.0.1:{listener.port}/device", token=TOKEN, wav=str(wav),
                                     record=True, mark_at=0.8, timeout=15.0,
                                     verbose=bool(__import__("os").environ.get("SIM_VERBOSE")))
            sim_task = asyncio.create_task(sim.DeviceSim(options).run())
            for _ in range(100):
                if listener.connected:
                    break
                await asyncio.sleep(0.05)
            with TestClient(create_app(services.cfg, services)) as api:
                assert (await asyncio.to_thread(
                    api.post, "/api/admin/login", json={"password": password})).status_code == 200
                started = await asyncio.to_thread(api.post, "/api/admin/record/takes", json={
                    "device_id": "sim-1", "session": "s1", "label": "wake", "prompt_id": "wake_normal", "max_ms": 3000,
                })
                assert started.status_code == 200, started.text
                assert started.json()["state"] == "recording"
                # stop after ~1.5 s of real-time audio; the sim answers record_end and finishes
                await asyncio.sleep(1.5)
                stopped = await asyncio.to_thread(api.post, "/api/admin/record/takes/sim-1/stop")
                assert stopped.status_code == 200, stopped.text
                result = await sim_task
            return result, api
        finally:
            await listener.stop()

    result, api = asyncio.run(scenario())
    assert result.exit_code == 0
    sent = [e["type"] for e in result.record_log]
    assert sent == ["record_started", "record_mark", "record_end"]
    assert next(e for e in result.record_log if e["type"] == "record_end")["reason"] == "stop"
    take = api.get("/api/admin/record/takes/sim-1").json()["take"]
    assert take["state"] == "done" and take["device_detected"] is True
    assert take["result"]["duration_s"] == pytest.approx(1.5, abs=0.4)
    wav_path = rec_dir / "s1/wake/wake_normal/000.wav"
    assert wav_path.is_file()
    meta = [json.loads(line) for line in (rec_dir / "s1/meta.jsonl").read_text().splitlines()]
    assert meta[-1]["source"] == "device" and meta[-1]["device_detected"] is True
    assert meta[-1]["device_id"] == "sim-1"
    # nothing leaked into the conversation pipeline while the take recorded
    assert services.stt.calls == []

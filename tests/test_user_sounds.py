"""User sounds directory (WAV/MP3/OGG/FLAC), decode caps, chunked sound sending, and long /say texts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI

from gaja.api.devices import SOUNDS_DIR, available_sounds, create_devices_router, load_sound
from gaja.devices.registry import DeviceInfo
from gaja.pipeline import Session
from tests.test_device_protocol import FAST, HELLO, TOKEN
from tests.test_speaker_mcp import FakeListenerWithSession, RecordingOutbox

RATE = 16000


def _tone_pcm(seconds: float = 0.3, amplitude: float = 8000.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(RATE * seconds), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype("<i2")


def _write_sound(path: Path, amplitude: float = 8000.0, seconds: float = 0.3, fmt: str | None = None) -> None:
    sf.write(path, _tone_pcm(seconds, amplitude), RATE, format=fmt)


def _make_app(user_dir: Path, listener: Any = None) -> FastAPI:
    app = FastAPI()
    app.include_router(create_devices_router(lambda origin: origin is None, user_sounds_dir=user_dir))
    app.state.device_listener = listener
    return app


# ---- loader: user dir, precedence, formats --------------------------------------------------------


def test_user_dir_sounds_are_listed_and_win_on_name_clash(tmp_path: Path) -> None:
    _write_sound(tmp_path / "bell-2.ogg", fmt="OGG")
    _write_sound(tmp_path / "chime.flac", fmt="FLAC")

    names = sorted(available_sounds(SOUNDS_DIR, tmp_path))
    assert "bell-2" in names and "chime" in names

    clash = available_sounds(SOUNDS_DIR, tmp_path)["chime"]
    assert clash == tmp_path / "chime.flac"  # the user file wins over the built-in assets/sounds WAV

    builtin = load_sound("chime", SOUNDS_DIR)
    mine = load_sound("chime", SOUNDS_DIR, tmp_path)
    assert mine is not None and builtin is not None
    assert mine[0] != builtin[0]  # different file actually loaded


def test_ogg_and_flac_user_files_decode_to_mono_pcm16(tmp_path: Path) -> None:
    _write_sound(tmp_path / "bell.ogg", fmt="OGG")
    _write_sound(tmp_path / "bell.flac", fmt="FLAC")
    for path in (tmp_path / "bell.ogg", tmp_path / "bell.flac"):
        loaded = load_sound(path.stem, SOUNDS_DIR, tmp_path)
        assert loaded is not None
        pcm, rate = loaded
        assert rate == RATE
        arr = np.frombuffer(pcm, dtype="<i2")
        assert abs(arr.astype(np.int32).mean()) < 100  # whole periods of a sine: near-zero-mean decode
        peak = float(np.max(np.abs(arr)))
        assert 6500 < peak <= 8200  # amplitude roughly preserved (Vorbis is lossy, FLAC exact)


def test_mp3_decode_path_mixes_stereo_down_to_mono(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """This libsndfile (1.2.2) cannot write MP3, so the decode path is unit-tested with sf faked."""
    (tmp_path / "alarm.mp3").write_bytes(b"whatever; sf.read is faked below")
    stereo = np.zeros((4410, 2), dtype="<i2")
    stereo[:, 0] = 10000
    stereo[:, 1] = -5000

    class _Info:
        frames = 4410
        samplerate = 44100

    monkeypatch.setattr("gaja.api.devices.sf.info", lambda path: _Info())
    monkeypatch.setattr("gaja.api.devices.sf.read", lambda path, dtype, always_2d: (stereo, 44100))

    loaded = load_sound("alarm", SOUNDS_DIR, tmp_path)
    assert loaded is not None
    pcm, rate = loaded
    assert rate == 44100
    arr = np.frombuffer(pcm, dtype="<i2")
    assert arr.shape == (4410,)
    assert np.all(arr == 2500)  # (10000 + -5000) / 2, mixed down to mono


def test_user_dir_names_stay_allow_listed(tmp_path: Path) -> None:
    # A controlled built-in dir: the real assets/sounds may hold whatever sounds this machine has,
    # and the allow-listing decision must be about names, not about the checkout's contents.
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    _write_sound(builtin / "chime.wav")
    user = tmp_path / "user"
    user.mkdir()
    _write_sound(user / "good-name.wav")
    _write_sound(user / "Bad Name.wav")  # space: not in [a-z0-9][a-z0-9_-]*
    _write_sound(user / "UPPER.WAV")
    (user / ".hidden.wav").write_bytes(b"")
    (user / "note.txt").write_bytes(b"not a sound")
    (user / "subdir").mkdir()

    names = set(available_sounds(builtin, user))
    assert names == {"chime", "good-name"}


@pytest.mark.parametrize(
    "bad_name",
    ["", "unknown", "../etc/passwd", "/etc/passwd", "chime.wav", "subdir/secret", ".."],
)
def test_load_sound_user_dir_rejects_unknown_or_traversal_without_touching_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_name: str
) -> None:
    _write_sound(tmp_path / "good-name.wav")
    touched: list[str] = []
    monkeypatch.setattr("gaja.api.devices.sf.info", lambda f: touched.append(str(f)))
    monkeypatch.setattr("gaja.api.devices.sf.read", lambda f, dtype, always_2d: touched.append(str(f)))
    assert load_sound(bad_name, SOUNDS_DIR, tmp_path) is None
    assert touched == []


# ---- decode caps -----------------------------------------------------------------------------------


def test_too_long_sound_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_sound(tmp_path / "long-clip.ogg", seconds=1.0, fmt="OGG")
    monkeypatch.setattr("gaja.api.devices.MAX_SOUND_SECONDS", 0.5)
    with pytest.raises(ValueError, match="too long"):
        load_sound("long-clip", SOUNDS_DIR, tmp_path)


def test_too_large_sound_file_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_sound(tmp_path / "big.ogg", fmt="OGG")
    monkeypatch.setattr("gaja.api.devices.MAX_SOUND_FILE_BYTES", 8)
    with pytest.raises(ValueError, match="too large"):
        load_sound("big", SOUNDS_DIR, tmp_path)


def test_garbage_sound_file_rejected_with_clear_error(tmp_path: Path) -> None:
    (tmp_path / "junk.wav").write_bytes(b"this is not audio" * 8)
    with pytest.raises(ValueError, match="could not be decoded"):
        load_sound("junk", SOUNDS_DIR, tmp_path)


# ---- HTTP API: user sounds -------------------------------------------------------------------------


def test_sounds_endpoint_lists_user_sounds_and_picks_up_new_files_without_restart(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(_make_app(tmp_path))
    before = client.get("/api/devices/sounds").json()["sounds"]
    assert "ding" not in before

    _write_sound(tmp_path / "ding.ogg", fmt="OGG")
    after = client.get("/api/devices/sounds").json()["sounds"]
    assert "ding" in after
    assert {"alert", "chime", "success"}.issubset(set(after))


@pytest.mark.parametrize("bad_sound", ["../etc/passwd", "../../config.toml", "chime.wav", "   "])
def test_play_endpoint_rejects_traversal_and_bad_names_with_400(tmp_path: Path, bad_sound: str) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(_make_app(tmp_path))
    res = client.post("/api/devices/atom-pokoj/play", json={"sound": bad_sound})
    assert res.status_code == 400
    assert "unknown sound" in res.json()["detail"]


def test_play_endpoint_returns_400_with_clear_error_for_over_long_user_sound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    _write_sound(tmp_path / "long-clip.ogg", seconds=1.0, fmt="OGG")
    monkeypatch.setattr("gaja.api.devices.MAX_SOUND_SECONDS", 0.5)
    client = TestClient(_make_app(tmp_path))
    res = client.post("/api/devices/atom-pokoj/play", json={"sound": "long-clip"})
    assert res.status_code == 400
    assert "too long" in res.json()["detail"]


@pytest.mark.asyncio
async def test_play_endpoint_plays_user_ogg_and_user_name_wins_over_builtin(make_services, tmp_path: Path) -> None:
    """End to end: a user OGG is decoded and streamed instead of the built-in chime.wav."""
    services = make_services(cfg={"devices": {**FAST}})
    raw_out = RecordingOutbox()
    session = Session(services, raw_out, device=DeviceInfo("atom-pokoj"))
    await session.handle_text(json.dumps({**HELLO, "device_id": "atom-pokoj", "token": TOKEN}))
    raw_out.sent.clear()

    # A user "chime" at a distinctive amplitude: the played bytes must be this file, not the built-in.
    _write_sound(tmp_path / "chime.ogg", amplitude=8000.0, fmt="OGG")
    listener = FakeListenerWithSession("atom-pokoj", session)
    app = _make_app(tmp_path, listener)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/devices/atom-pokoj/play", json={"sound": "chime"})
    assert res.status_code == 200
    assert res.json() == {"ok": True, "sound": "chime"}

    json_msgs = [m for m in raw_out.sent if isinstance(m, dict)]
    byte_chunks = [m for m in raw_out.sent if isinstance(m, bytes)]
    audio_start = next(m for m in json_msgs if m["type"] == "audio_start")
    assert audio_start["sample_rate"] == RATE  # 16 kHz source plays at the device's 16 kHz untouched

    expected, _ = sf.read(str(tmp_path / "chime.ogg"), dtype="int16", always_2d=True)
    assert b"".join(byte_chunks) == expected.reshape(-1).tobytes()

    await session.close()


# ---- chunked sending -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_second_sound_is_streamed_in_multiple_slices(make_services, monkeypatch: pytest.MonkeyPatch) -> None:
    """A web client (raw outbox) must not get a multi-second clip as one giant binary frame."""
    monkeypatch.setattr("gaja.pipeline.SOUND_SLICE_S", 1.0)
    services = make_services(cfg={"devices": {**FAST}})
    raw_out = RecordingOutbox()
    session = Session(services, raw_out)  # no device: the raw outbox shows the slices as sent

    pcm = (_tone_pcm(5.0)).astype("<i2").tobytes()  # 5 s at 16 kHz
    await session.play_sound(pcm, RATE, name="long")

    json_msgs = [m for m in raw_out.sent if isinstance(m, dict)]
    byte_chunks = [m for m in raw_out.sent if isinstance(m, bytes)]
    assert len(byte_chunks) == 5  # 5 s / 1 s slices, each a separate frame
    assert all(len(c) == RATE * 2 for c in byte_chunks)
    assert b"".join(byte_chunks) == pcm  # the audio survives slicing untouched

    types = [m["type"] for m in json_msgs]
    assert types[0] == "state"  # speaking
    start_idx = raw_out.sent.index(next(m for m in json_msgs if m["type"] == "audio_start"))
    end_idx = raw_out.sent.index(next(m for m in json_msgs if m["type"] == "audio_end"))
    first_bytes = next(i for i, m in enumerate(raw_out.sent) if isinstance(m, bytes))
    last_bytes = max(i for i, m in enumerate(raw_out.sent) if isinstance(m, bytes))
    assert start_idx < first_bytes <= last_bytes < end_idx
    assert raw_out.sent[start_idx]["sample_rate"] == RATE

    await session.close()


@pytest.mark.asyncio
async def test_multi_second_sound_reaches_device_as_many_frames(make_services) -> None:
    """On a device session the slices go through the paced outbox: many small frames, protocol intact."""
    services = make_services(cfg={"devices": {**FAST}})
    raw_out = RecordingOutbox()
    session = Session(services, raw_out, device=DeviceInfo("atom-pokoj"))
    await session.handle_text(json.dumps({**HELLO, "device_id": "atom-pokoj", "token": TOKEN}))
    raw_out.sent.clear()

    out_rate = session._out.rate  # negotiated device rate (ResamplingOutbox)
    assert out_rate == 16000
    t = np.linspace(0, 5.0, int(out_rate * 5.0), endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 8000).astype("<i2").tobytes()
    await session.play_sound(pcm, out_rate, name="long")

    byte_chunks = [m for m in raw_out.sent if isinstance(m, bytes)]
    assert len(byte_chunks) > 1, "a 5 s sound must not arrive as one binary frame"
    assert b"".join(byte_chunks) == pcm  # identity resample: bytes preserved

    json_msgs = [m for m in raw_out.sent if isinstance(m, dict)]
    start_idx = raw_out.sent.index(next(m for m in json_msgs if m["type"] == "audio_start"))
    end_idx = raw_out.sent.index(next(m for m in json_msgs if m["type"] == "audio_end"))
    first_bytes = next(i for i, m in enumerate(raw_out.sent) if isinstance(m, bytes))
    last_bytes = max(i for i, m in enumerate(raw_out.sent) if isinstance(m, bytes))
    assert start_idx < first_bytes <= last_bytes < end_idx

    await session.close()


# ---- long announcements ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_say_endpoint_accepts_4000_chars_and_rejects_4001(make_services, tmp_path: Path) -> None:
    services = make_services(cfg={"devices": {**FAST}})
    raw_out = RecordingOutbox()
    session = Session(services, raw_out, device=DeviceInfo("atom-pokoj"))
    await session.handle_text(json.dumps({**HELLO, "device_id": "atom-pokoj", "token": TOKEN}))
    raw_out.sent.clear()

    listener = FakeListenerWithSession("atom-pokoj", session)
    from fastapi.testclient import TestClient

    client = TestClient(_make_app(tmp_path, listener))

    ok = client.post("/api/devices/atom-pokoj/say", json={"text": "a" * 4000})
    assert ok.status_code == 200
    assert ok.json() == {"ok": True}

    too_long = client.post("/api/devices/atom-pokoj/say", json={"text": "a" * 4001})
    assert too_long.status_code == 422

    await session.close()

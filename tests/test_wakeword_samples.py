from __future__ import annotations

import io
import json
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gaja.wakeword_samples import create_router

ALLOWED = "https://ttm-server.taila5df9a.ts.net:8443"


def make_wav(seconds: float = 1.0, *, rate: int = 16000, channels: int = 1, width: int = 2, amp: float = 0.3) -> bytes:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pcm = (amp * 32767 * np.sin(2 * np.pi * 440 * t)).astype("<i2")
    if channels == 2:
        pcm = np.repeat(pcm, 2)
    data = pcm.tobytes() if width == 2 else (pcm >> 8).astype("i1").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(data)
    return buf.getvalue()


@pytest.fixture
def rec_dir(tmp_path: Path) -> Path:
    return tmp_path / "rec"


@pytest.fixture
def client(rec_dir: Path) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(rec_dir, lambda o: o is None or o == ALLOWED, max_total_mb=1))
    return TestClient(app)


def post(client, wav, *, session="s1", label="wake", prompt_id="wake_normal", take=0, headers=None, **extra):
    params = {"session": session, "label": label, "prompt_id": prompt_id, "take": take, **extra}
    return client.post("/api/wakeword/samples", params=params, content=wav, headers={"Content-Type": "audio/wav", **(headers or {})})


def test_valid_upload_is_stored_with_meta(client, rec_dir):
    r = post(client, make_wav(1.5), mic="Test Mic", headers={"User-Agent": "pytest-ua"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1 and body["next_take"] == 1 and body["total"] == 1
    assert body["duration_s"] == pytest.approx(1.5, abs=0.01)
    assert body["peak"] == pytest.approx(0.3, abs=0.01)
    assert (rec_dir / "s1/wake/wake_normal/000.wav").is_file()
    meta = [json.loads(line) for line in (rec_dir / "s1/meta.jsonl").read_text().splitlines()]
    assert len(meta) == 1
    assert meta[0]["label"] == "wake" and meta[0]["prompt_id"] == "wake_normal" and meta[0]["take"] == 0
    assert meta[0]["mic"] == "Test Mic" and meta[0]["user_agent"] == "pytest-ua"
    assert {"ts", "duration_s", "peak", "rms"} <= set(meta[0])


def test_duplicate_take_conflicts(client):
    assert post(client, make_wav()).status_code == 200
    assert post(client, make_wav()).status_code == 409


@pytest.mark.parametrize(
    "wav",
    [
        make_wav(rate=44100),
        make_wav(channels=2),
        make_wav(width=1),
        b"RIFF" + b"\x00" * 100,
        b"not a wav at all, just text" * 10,
        make_wav(0.05),
    ],
    ids=["44k", "stereo", "8bit", "bad-riff", "text", "too-short"],
)
def test_wrong_format_rejected(client, rec_dir, wav):
    r = post(client, wav)
    assert r.status_code in (415, 422), r.text
    assert not list(rec_dir.rglob("*.wav"))


def test_oversize_and_content_type_rejected(client):
    assert post(client, make_wav(1.0) + b"\x00" * (4 * 1024 * 1024)).status_code == 413
    r = client.post("/api/wakeword/samples", params={"session": "s", "label": "wake", "prompt_id": "p", "take": 0}, content=make_wav(), headers={"Content-Type": "text/plain"})
    assert r.status_code == 415


@pytest.mark.parametrize(
    "overrides",
    [
        {"session": "../evil"},
        {"session": "a/b"},
        {"session": "UPPER"},
        {"prompt_id": "../../x"},
        {"prompt_id": "a b"},
        {"prompt_id": ""},
        {"label": "other"},
        {"label": "../wake"},
        {"take": -1},
    ],
)
def test_bad_path_components_rejected(client, rec_dir, overrides):
    r = post(client, make_wav(), **overrides)
    assert r.status_code in (400, 422), r.text
    assert not rec_dir.exists() or not list(rec_dir.rglob("*"))


def test_delete_path_traversal_rejected(client, rec_dir):
    post(client, make_wav())
    secret = rec_dir.parent / "secret.wav"
    secret.write_bytes(b"x")
    for url in (
        "/api/wakeword/samples/..%2F/wake/wake_normal/0",
        "/api/wakeword/samples/s1/wake/%2e%2e/0",
        "/api/wakeword/samples/s1/wake/wake_normal/abc",
        "/api/wakeword/samples/s1/../../secret/0",
    ):
        assert client.delete(url).status_code in (400, 404, 422), url
    assert secret.exists()
    assert (rec_dir / "s1/wake/wake_normal/000.wav").exists()


def test_origin_rejected_for_all_methods(client, rec_dir):
    bad = {"Origin": "https://evil.example"}
    assert post(client, make_wav(), headers=bad).status_code == 403
    assert not rec_dir.exists() or not list(rec_dir.rglob("*.wav"))
    assert post(client, make_wav(), headers={"Origin": ALLOWED}).status_code == 200
    assert client.delete("/api/wakeword/samples/s1/wake/wake_normal/0", headers=bad).status_code == 403
    assert client.get("/api/wakeword/progress", params={"session": "s1"}, headers=bad).status_code == 403
    assert client.get("/api/wakeword/export.zip", headers=bad).status_code == 403
    assert client.get("/api/wakeword/sessions", headers=bad).status_code == 403
    assert (rec_dir / "s1/wake/wake_normal/000.wav").exists()
    # no Origin header (curl, scripts) is allowed, like the WebSocket
    assert client.get("/api/wakeword/sessions").status_code == 200


def test_progress_counts_and_resume(client):
    for take in range(3):
        assert post(client, make_wav(1.0), take=take).status_code == 200
    assert post(client, make_wav(2.0), label="hard_negative", prompt_id="hn_hej_maja").status_code == 200
    assert post(client, make_wav(3.0), label="ambient", prompt_id="amb_tv").status_code == 200
    assert post(client, make_wav(1.0), session="other").status_code == 200
    data = client.get("/api/wakeword/progress", params={"session": "s1"}).json()
    assert data["total"] == 5
    assert data["labels"]["wake"]["wake_normal"]["count"] == 3
    assert data["labels"]["wake"]["wake_normal"]["next_take"] == 3
    assert data["labels"]["wake"]["wake_normal"]["duration_s"] == pytest.approx(3.0, abs=0.05)
    assert data["labels"]["hard_negative"]["hn_hej_maja"]["count"] == 1
    assert data["labels"]["ambient"]["amb_tv"]["duration_s"] == pytest.approx(3.0, abs=0.05)
    assert client.get("/api/wakeword/progress", params={"session": "nope"}).json() == {"session": "nope", "total": 0, "labels": {}}
    assert client.get("/api/wakeword/progress", params={"session": "../x"}).status_code == 400
    sessions = client.get("/api/wakeword/sessions").json()["sessions"]
    assert {s["session"] for s in sessions} == {"s1", "other"}
    assert next(s for s in sessions if s["session"] == "s1")["total"] == 5


def test_delete_last_take(client, rec_dir):
    post(client, make_wav(), take=0)
    post(client, make_wav(), take=1)
    r = client.delete("/api/wakeword/samples/s1/wake/wake_normal/1")
    assert r.status_code == 200 and r.json()["count"] == 1 and r.json()["next_take"] == 1
    assert not (rec_dir / "s1/wake/wake_normal/001.wav").exists()
    assert client.delete("/api/wakeword/samples/s1/wake/wake_normal/1").status_code == 404
    last = json.loads((rec_dir / "s1/meta.jsonl").read_text().splitlines()[-1])
    assert last["deleted"] is True and last["take"] == 1


def test_export_zip_contains_tree_and_meta(client):
    wav = make_wav(1.0)
    post(client, wav, take=0)
    post(client, make_wav(0.5), label="hard_negative", prompt_id="hn_okej")
    post(client, make_wav(0.5), session="s2")
    r = client.get("/api/wakeword/export.zip")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert zf.testzip() is None
        names = set(zf.namelist())
        assert names == {
            "wakeword/s1/wake/wake_normal/000.wav",
            "wakeword/s1/hard_negative/hn_okej/000.wav",
            "wakeword/s1/meta.jsonl",
            "wakeword/s2/wake/wake_normal/000.wav",
            "wakeword/s2/meta.jsonl",
        }
        assert zf.read("wakeword/s1/wake/wake_normal/000.wav") == wav
    only = client.get("/api/wakeword/export.zip", params={"session": "s2"})
    with zipfile.ZipFile(io.BytesIO(only.content)) as zf:
        assert all(n.startswith("wakeword/s2/") for n in zf.namelist())
    # an empty tree still yields a valid (empty) archive
    assert zipfile.ZipFile(io.BytesIO(client.get("/api/wakeword/export.zip", params={"session": "empty"}).content)).namelist() == []


def test_total_size_cap(client):
    big = make_wav(60.0)  # ~1.9 MB each, cap in this fixture is 1 MB
    assert post(client, big).status_code == 507


def test_server_registers_router_and_config(tmp_path, make_services):
    from gaja.config import Settings
    from gaja.server import create_app

    assert Settings().wakeword.recordings_dir == Path("data/wakeword/recordings")
    services = make_services(cfg={"wakeword": {"recordings_dir": str(tmp_path / "r")}})
    url = {"session": "s", "label": "wake", "prompt_id": "p"}
    with TestClient(create_app(services.cfg, services)) as client:
        ok = client.post(
            "/api/wakeword/samples", params={**url, "take": 0}, content=make_wav(),
            headers={"Content-Type": "audio/wav", "Origin": "http://127.0.0.1:8790"},
        )
        assert ok.status_code == 200, ok.text
        bad = client.post(
            "/api/wakeword/samples", params={**url, "take": 1}, content=make_wav(),
            headers={"Content-Type": "audio/wav", "Origin": "https://evil.example"},
        )
        assert bad.status_code == 403
    assert (tmp_path / "r/s/wake/p/000.wav").is_file()

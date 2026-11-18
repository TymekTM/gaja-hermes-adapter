"""HTTP API /api/pc (protocol v1.14): bearer auth, Origin refusal, list, call, 404/400/504."""

from __future__ import annotations

import asyncio
import json
import stat
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gaja.api.pc import create_pc_router, load_or_create_api_token
from gaja.config import PcConfig, Settings
from gaja.devices.audio import PcCaps
from gaja.devices.pc import PcRelay


class FakePc:
    """A registered PC device whose pc_calls are answered by test code (same loop as the API)."""

    def __init__(self, relay: PcRelay, device_id: str, tools: list[str] | None = None, *, hang: bool = False) -> None:
        self.relay = relay
        self.device_id = device_id
        self.hang = hang
        self.calls: list[dict[str, Any]] = []
        caps = PcCaps(os="windows", host="TYMEK-PC", tools=tools if tools is not None else ["shell", "screenshot", "clipboard"])
        relay.register(self, device_id, caps, self._send)

    async def _send(self, payload: dict[str, Any]) -> None:
        self.calls.append(payload)
        if self.hang:
            return  # never answers: the server-side timeout must fire
        self.answer({"ok": True, "text": f"did {payload['tool']}", "data": {"echo": payload["args"]}})

    def answer(self, result: dict[str, Any]) -> None:
        call_id = self.calls[-1]["id"]
        self.relay.on_result(self.device_id, {"type": "pc_result", "id": call_id, **result})


class FakeServices:
    def __init__(self, relay: PcRelay, cfg: PcConfig) -> None:
        self.pc_relay = relay
        self.cfg = Settings(pc=cfg)


def make_app(tmp_path, *, relay: PcRelay | None = None, cfg: PcConfig | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(create_pc_router())
    app.state.services = FakeServices(relay or PcRelay(), cfg or PcConfig(api_token_file=tmp_path / "pc_api.token"))
    return app


@pytest.fixture
def token_file(tmp_path) -> Any:
    return tmp_path / "pc_api.token"


@pytest.fixture
def auth_header(token_file) -> dict[str, str]:
    token = load_or_create_api_token(token_file)
    return {"Authorization": f"Bearer {token}"}


# ---- token file ------------------------------------------------------------------------------------


def test_token_file_created_once_with_modes(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "pc_api.token"
    token = load_or_create_api_token(path)
    assert len(token) == 64 and int(token, 16) >= 0  # 32 bytes hex
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    # A second read returns the same token and does not rewrite the file.
    before = path.stat().st_mtime_ns
    assert load_or_create_api_token(path) == token
    assert path.stat().st_mtime_ns == before


# ---- auth ------------------------------------------------------------------------------------------


def test_missing_token_is_401(tmp_path) -> None:
    client = TestClient(make_app(tmp_path))
    assert client.get("/api/pc").status_code == 401
    assert client.post("/api/pc/pc-1/call", json={"tool": "shell"}).status_code == 401


def test_wrong_token_is_401(tmp_path, token_file) -> None:
    load_or_create_api_token(token_file)
    client = TestClient(make_app(tmp_path))
    assert client.get("/api/pc", headers={"Authorization": "Bearer " + "0" * 64}).status_code == 401
    assert client.get("/api/pc", headers={"Authorization": token_file.name}).status_code == 401


def test_any_origin_header_is_403_even_with_a_valid_token(tmp_path, auth_header) -> None:
    client = TestClient(make_app(tmp_path))
    for origin in ("https://evil.example", "https://localhost:8790", "null"):
        res = client.get("/api/pc", headers={**auth_header, "Origin": origin})
        assert res.status_code == 403, origin
    # without Origin the same request works
    assert client.get("/api/pc", headers=auth_header).status_code == 200


def test_good_token_is_200(tmp_path, auth_header) -> None:
    client = TestClient(make_app(tmp_path))
    res = client.get("/api/pc", headers=auth_header)
    assert res.status_code == 200
    assert res.json() == {"devices": []}


def test_pc_disabled_means_no_token_works(tmp_path, token_file) -> None:
    """With [pc] disabled there is no valid token: /api/pc answers 401 for everything."""
    load_or_create_api_token(token_file)
    cfg = PcConfig(enabled=False, api_token_file=token_file)
    client = TestClient(make_app(tmp_path, cfg=cfg))
    token = token_file.read_text().strip()
    assert client.get("/api/pc", headers={"Authorization": f"Bearer {token}"}).status_code == 401


# ---- GET /api/pc -----------------------------------------------------------------------------------


def test_list_shows_connected_pcs(tmp_path, auth_header) -> None:
    relay = PcRelay()
    FakePc(relay, "pc-tymek")
    client = TestClient(make_app(tmp_path, relay=relay))
    res = client.get("/api/pc", headers=auth_header)
    assert res.status_code == 200
    devices = res.json()["devices"]
    assert len(devices) == 1
    assert devices[0]["id"] == "pc-tymek"
    assert devices[0]["host"] == "TYMEK-PC"
    assert devices[0]["os"] == "windows"
    assert "shell" in devices[0]["tools"]
    assert devices[0]["connected_s"] >= 0.0


# ---- POST /api/pc/{id}/call ------------------------------------------------------------------------


def test_call_happy_path_returns_the_result_payload(tmp_path, auth_header) -> None:
    relay = PcRelay()
    pc = FakePc(relay, "pc-tymek")
    client = TestClient(make_app(tmp_path, relay=relay))
    res = client.post(
        "/api/pc/pc-tymek/call",
        headers=auth_header,
        json={"tool": "shell", "args": {"command": "dir"}, "timeout_s": 5},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True and body["text"] == "did shell"
    assert body["data"] == {"echo": {"command": "dir"}}
    assert "image" not in body
    assert pc.calls[0]["type"] == "pc_call"
    assert pc.calls[0]["timeout_s"] == 5
    assert pc.calls[0]["id"] and len(pc.calls[0]["id"]) == 12


def test_call_unknown_device_is_404(tmp_path, auth_header) -> None:
    client = TestClient(make_app(tmp_path))
    res = client.post("/api/pc/pc-nope/call", headers=auth_header, json={"tool": "shell"})
    assert res.status_code == 404
    assert res.json()["ok"] is False
    assert "not connected" in res.json()["error"]


def test_call_tool_not_advertised_is_400(tmp_path, auth_header) -> None:
    relay = PcRelay()
    FakePc(relay, "pc-tymek", tools=["clipboard"])
    client = TestClient(make_app(tmp_path, relay=relay))
    res = client.post("/api/pc/pc-tymek/call", headers=auth_header, json={"tool": "shell"})
    assert res.status_code == 400
    assert "does not advertise" in res.json()["error"]


def test_call_timeout_is_504_with_json_body(tmp_path, auth_header, monkeypatch: pytest.MonkeyPatch) -> None:
    import gaja.api.pc as _  # noqa: F401  (the router reads GRACE_S from gaja.devices.pc at call time)

    import gaja.devices.pc as pc_module

    monkeypatch.setattr(pc_module, "GRACE_S", 0.2)
    relay = PcRelay()
    FakePc(relay, "pc-tymek", hang=True)  # takes the call, never answers
    client = TestClient(make_app(tmp_path, relay=relay))
    t0 = time.monotonic()
    res = client.post("/api/pc/pc-tymek/call", headers=auth_header, json={"tool": "shell", "timeout_s": 1})
    assert res.status_code == 504
    assert res.json() == {"ok": False, "error": "timeout"}
    assert time.monotonic() - t0 < 5.0


def test_call_defaults_and_caps_the_timeout(tmp_path, token_file, auth_header) -> None:
    relay = PcRelay()
    pc = FakePc(relay, "pc-tymek")
    cfg = PcConfig(api_token_file=token_file, default_timeout_s=7, max_timeout_s=9)
    client = TestClient(make_app(tmp_path, relay=relay, cfg=cfg))
    assert client.post("/api/pc/pc-tymek/call", headers=auth_header, json={"tool": "shell"}).status_code == 200
    assert pc.calls[0]["timeout_s"] == 7
    # 300 is protocol-max but the config cap wins.
    assert client.post("/api/pc/pc-tymek/call", headers=auth_header, json={"tool": "shell", "timeout_s": 300}).status_code == 200
    assert pc.calls[1]["timeout_s"] == 9
    # out of protocol range: validation error
    assert client.post("/api/pc/pc-tymek/call", headers=auth_header, json={"tool": "shell", "timeout_s": 0}).status_code == 422
    assert client.post("/api/pc/pc-tymek/call", headers=auth_header, json={"tool": "shell", "timeout_s": 301}).status_code == 422


def test_call_passes_an_image_through(tmp_path, auth_header) -> None:
    relay = PcRelay()

    class ShotPc(FakePc):
        async def _send(self, payload: dict[str, Any]) -> None:
            self.calls.append(payload)
            self.relay.on_result(
                self.device_id,
                {
                    "type": "pc_result",
                    "id": payload["id"],
                    "ok": True,
                    "text": "1920x1080 monitor 0",
                    "image": {"mime": "image/png", "b64": "AAAA", "width": 1920, "height": 1080},
                },
            )

    ShotPc(relay, "pc-tymek")
    client = TestClient(make_app(tmp_path, relay=relay))
    res = client.post("/api/pc/pc-tymek/call", headers=auth_header, json={"tool": "screenshot"})
    body = res.json()
    assert body["image"]["mime"] == "image/png"
    assert body["image"]["width"] == 1920


def test_call_body_validation(tmp_path, auth_header) -> None:
    relay = PcRelay()
    FakePc(relay, "pc-tymek")
    client = TestClient(make_app(tmp_path, relay=relay))
    ok = client.post("/api/pc/pc-tymek/call", headers=auth_header, json={"tool": "shell"})
    assert ok.status_code == 200  # args and timeout_s are optional
    missing_tool = client.post("/api/pc/pc-tymek/call", headers=auth_header, json={})
    assert missing_tool.status_code == 422

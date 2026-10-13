"""Admin panel backend: password/session auth, settings snapshot and config.local.toml edits."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gaja import config as config_module
from gaja.admin import auth as auth_module
from gaja.admin import settings_store
from gaja.admin.api import create_admin_router
from gaja.admin.auth import COOKIE, AdminAuth, write_credentials
from gaja.config import ROOT, Settings

ORIGIN = "http://localhost:8790"


@pytest.fixture
def cfg_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    base = tmp_path / "config.toml"
    base.write_text((ROOT / "config.toml").read_text())
    local = tmp_path / "config.local.toml"
    local.write_text('# my overrides\n[llm]\ntemperature = 0.5 # warmer\n')
    monkeypatch.setattr(config_module, "CONFIG_PATH", base)
    monkeypatch.setattr(config_module, "LOCAL_CONFIG_PATH", local)
    return base, local


@pytest.fixture
def client(tmp_path: Path, cfg_paths: tuple[Path, Path]) -> TestClient:
    creds = tmp_path / "admin.toml"
    write_credentials(creds, "correct horse battery")
    app = FastAPI()
    app.include_router(create_admin_router(AdminAuth(creds), lambda o: o is None or o == ORIGIN))

    class Services:
        cfg = Settings()

    app.state.services = Services()
    return TestClient(app, headers={"origin": ORIGIN})


def test_credentials_file_is_private(tmp_path: Path) -> None:
    creds = tmp_path / "admin.toml"
    write_credentials(creds, "secret password")
    assert creds.stat().st_mode & 0o777 == 0o600
    assert "secret password" not in creds.read_text()


def test_login_required_and_cookie_session(client: TestClient) -> None:
    assert client.get("/api/admin/config").status_code == 401
    assert client.get("/api/admin/session").json() == {"logged_in": False, "configured": True}
    assert client.post("/api/admin/login", json={"password": "wrong"}).status_code == 401
    res = client.post("/api/admin/login", json={"password": "correct horse battery"})
    assert res.status_code == 200
    assert "httponly" in res.headers["set-cookie"].lower()
    assert client.get("/api/admin/session").json()["logged_in"] is True
    assert client.get("/api/admin/config").status_code == 200
    client.post("/api/admin/logout")
    assert client.get("/api/admin/config").status_code == 401


def test_foreign_origin_rejected(client: TestClient) -> None:
    res = client.post("/api/admin/login", json={"password": "correct horse battery"}, headers={"origin": "https://evil.example"})
    assert res.status_code == 403


def test_password_change_invalidates_sessions(tmp_path: Path) -> None:
    creds = tmp_path / "admin.toml"
    write_credentials(creds, "first password")
    auth = AdminAuth(creds)
    token = auth.login("first password")
    assert auth.valid(token)
    write_credentials(creds, "second password")
    assert not auth.valid(token)
    assert not auth.valid(token[:-2] + "xx")


def test_lockout_after_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_module, "MAX_FAILURES", 3)
    creds = tmp_path / "admin.toml"
    write_credentials(creds, "right password")
    auth = AdminAuth(creds)
    for _ in range(3):
        assert auth.login("nope") is None
    assert auth.login("right password") is None


def test_unconfigured_refuses_login(tmp_path: Path) -> None:
    auth = AdminAuth(tmp_path / "missing.toml")
    assert not auth.configured
    assert auth.login("anything") is None
    assert not auth.valid("1.2.3")


def test_describe_config_reads_comments() -> None:
    text = "# Section help\n[llm]\n# Line one\n# line two\ntemperature = 0.3\n\n# orphan\n\nmodel = 'x'\n"
    help_ = settings_store.describe_config(text)
    assert help_["llm"] == "Section help"
    assert help_["llm.temperature"] == "Line one line two"
    assert "llm.model" not in help_


def test_snapshot_reports_values(cfg_paths: tuple[Path, Path]) -> None:
    snap = settings_store.snapshot(Settings())
    by_path = {f["path"]: f for f in snap["fields"]}
    temp = by_path["llm.temperature"]
    assert temp["value"] == 0.5 and temp["default"] == 0.3 and temp["overridden"]
    assert by_path["llm.tool_strategy"]["kind"] == "choice"
    assert by_path["llm.tool_strategy"]["choices"] == ["native", "pythonic"]
    assert by_path["vad.smart_turn_model_path"]["optional"] is True
    assert by_path["stt.faster_whisper.model"]["section"] == "stt.faster_whisper"
    assert by_path["admin.credentials_file"]["read_only"]
    assert by_path["server.port"]["danger"]
    assert "Slots are 4096 tokens" in by_path["llm.history_max_turns"]["help"]


def test_patch_keeps_comments_and_removes_overrides(cfg_paths: tuple[Path, Path]) -> None:
    _, local = cfg_paths
    settings_store.apply_patch({"vad.min_silence_ms": 600, "stt.faster_whisper.beam_size": 2})
    text = local.read_text()
    assert "# my overrides" in text and "# warmer" in text
    data = tomllib.loads(text)
    assert data["vad"]["min_silence_ms"] == 600
    assert data["stt"]["faster_whisper"]["beam_size"] == 2
    assert Settings().vad.min_silence_ms == 600
    settings_store.apply_patch({"vad.min_silence_ms": None, "stt.faster_whisper.beam_size": None})
    data = tomllib.loads(local.read_text())
    assert "vad" not in data and "stt" not in data
    assert data["llm"]["temperature"] == 0.5


@pytest.mark.parametrize(
    "patch",
    [{"llm.tool_strategy": "bogus"}, {"vad.min_silence_ms": "abc"}, {"nope.key": 1}, {"admin.enabled": False}],
)
def test_patch_rejects_invalid(cfg_paths: tuple[Path, Path], patch: dict) -> None:
    _, local = cfg_paths
    before = local.read_text()
    with pytest.raises(ValueError):
        settings_store.apply_patch(patch)
    assert local.read_text() == before


def test_config_api_round_trip(client: TestClient) -> None:
    client.post("/api/admin/login", json={"password": "correct horse battery"})
    res = client.put("/api/admin/config", json={"values": {"router.threshold": 0.9}})
    assert res.status_code == 200, res.text
    snap = client.get("/api/admin/config").json()
    field = next(f for f in snap["fields"] if f["path"] == "router.threshold")
    assert field["value"] == 0.9 and field["restart_pending"]
    assert snap["restart_pending"] is True
    assert client.put("/api/admin/config", json={"values": {"router.threshold": "x"}}).status_code == 400

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
from gaja.household import HouseholdStore

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

    hermes_home = tmp_path / "hermes"
    (hermes_home / "profiles" / "serzant").mkdir(parents=True)
    (hermes_home / "profiles" / "serzant" / ".env").write_text("API_SERVER_KEY=very-secret\n")
    (hermes_home / "profiles" / "dom").mkdir(parents=True)
    (hermes_home / "profiles" / "default").mkdir(parents=True)

    class Services:
        cfg = Settings(hermes={"gateway": {"hermes_home": str(hermes_home)}})
        household = HouseholdStore(tmp_path / "household.json")
        sessions: set = set()

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


def test_lockout_is_scoped_to_the_client_address(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ten bad logins from one address must not lock everyone out: the owner (another address) has to
    # stay able to log in while the attacker's address is refused.
    monkeypatch.setattr(auth_module, "MAX_FAILURES", 3)
    attacker = TestClient(client.app, headers={"origin": ORIGIN}, client=("203.0.113.7", 4444))
    owner = TestClient(client.app, headers={"origin": ORIGIN}, client=("192.168.0.10", 4445))
    for _ in range(3):
        assert attacker.post("/api/admin/login", json={"password": "nope"}).status_code == 401
    res = attacker.post("/api/admin/login", json={"password": "correct horse battery"})
    assert res.status_code == 429, res.text  # attacker's address is locked out...
    res = owner.post("/api/admin/login", json={"password": "correct horse battery"})
    assert res.status_code == 200, res.text  # ...the owner's is not


def test_failure_tracking_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Many distinct addresses must not grow the failure dict without limit.
    monkeypatch.setattr(auth_module, "MAX_FAILURES", 3)
    monkeypatch.setattr(auth_module, "MAX_TRACKED_CLIENTS", 4)
    creds = tmp_path / "admin.toml"
    write_credentials(creds, "right password")
    auth = AdminAuth(creds)
    for i in range(200):
        auth.record_failure(f"10.0.{i // 256}.{i % 256}")
    assert len(auth._failures) <= auth_module.MAX_TRACKED_CLIENTS


def test_unconfigured_refuses_login(tmp_path: Path) -> None:
    auth = AdminAuth(tmp_path / "missing.toml")
    assert not auth.configured
    assert auth.login("anything") is None
    assert not auth.valid("1.2.3")


def test_non_ascii_cookie_is_unauthorized_not_an_error(client: TestClient) -> None:
    # Raw header bytes are legal on the wire (latin-1) but crashed hmac.compare_digest with a
    # TypeError -> 500. A junk cookie is just "not logged in".
    for raw in (b"gaja_admin=\xe4\xf6.abc.def", b"gaja_admin=abc.\xff\xfe", b"gaja_admin=\xc4\x99"):
        res = client.get("/api/admin/config", headers=[(b"origin", ORIGIN.encode()), (b"cookie", raw)])
        assert res.status_code == 401, (raw, res.status_code, res.text)
        res = client.get("/api/admin/session", headers=[(b"origin", ORIGIN.encode()), (b"cookie", raw)])
        assert res.status_code == 200, (raw, res.status_code, res.text)
        assert res.json() == {"logged_in": False, "configured": True}


def test_valid_rejects_a_non_ascii_token(tmp_path: Path) -> None:
    creds = tmp_path / "admin.toml"
    write_credentials(creds, "pw")
    auth = AdminAuth(creds)
    assert not auth.valid("abc.def.żółć")  # non-ASCII must land in the signature part


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
    assert by_path["home.write_allowlist"]["read_only"]
    assert by_path["home.aliases"]["read_only"]
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
    [
        {"llm.tool_strategy": "bogus"},
        {"vad.min_silence_ms": "abc"},
        {"nope.key": 1},
        {"admin.enabled": False},
        {"home.write_allowlist": ["climate.*"]},
        {"home.aliases": {"piec": "climate.sypialnia"}},
    ],
)
def test_patch_rejects_invalid(cfg_paths: tuple[Path, Path], patch: dict) -> None:
    _, local = cfg_paths
    before = local.read_text()
    with pytest.raises(ValueError):
        settings_store.apply_patch(patch)
    assert local.read_text() == before


@pytest.mark.parametrize(
    "patch",
    [
        {"server.port": True},  # JSON bool for an int field: pydantic coerces to 1 and the raw
        # `port = true` was written to config.local.toml before the type check
        {"vad.smart_turn": "true"},  # JSON string for a bool field
        {"llm.temperature": "0.9"},  # JSON string for a float field
        {"llm.model": 5},  # number for a string field
        {"llm.temperature": float("nan")},  # TOML can store nan/inf; no setting means either
        {"llm.temperature": float("inf")},
    ],
)
def test_patch_rejects_wrong_json_type(cfg_paths: tuple[Path, Path], patch: dict) -> None:
    _, local = cfg_paths
    before = local.read_text()
    with pytest.raises(ValueError, match="expects"):
        settings_store.apply_patch(patch)
    assert local.read_text() == before


def test_patch_still_accepts_the_right_types(cfg_paths: tuple[Path, Path]) -> None:
    settings_store.apply_patch({"vad.smart_turn": True, "llm.temperature": 0.9})
    data = tomllib.loads(cfg_paths[1].read_text())
    assert data["vad"]["smart_turn"] is True and data["llm"]["temperature"] == 0.9


def test_home_safety_settings_are_read_only_in_admin_panel(cfg_paths: tuple[Path, Path]) -> None:
    snap = settings_store.snapshot(Settings())
    by_path = {f["path"]: f for f in snap["fields"]}
    assert by_path["home.write_allowlist"]["read_only"] is True
    assert by_path["home.aliases"]["read_only"] is True
    # NEW-6: the household file itself (and the data it feeds) is panel-read-only.
    assert by_path["home.household_file"]["read_only"] is True
    assert by_path["home.spoken_forms"]["read_only"] is True
    assert by_path["home.boiler_id_substrings"]["read_only"] is True
    with pytest.raises(ValueError, match="read-only"):
        settings_store.apply_patch({"home.write_allowlist": ["climate.salon"]})
    with pytest.raises(ValueError, match="read-only"):
        settings_store.apply_patch({"home.aliases": {"piec": "climate.boiler_heating"}})
    with pytest.raises(ValueError, match="read-only"):
        settings_store.apply_patch({"home.household_file": "data/elsewhere.toml"})


def test_config_api_round_trip(client: TestClient) -> None:
    client.post("/api/admin/login", json={"password": "correct horse battery"})
    res = client.put("/api/admin/config", json={"values": {"router.threshold": 0.9}})
    assert res.status_code == 200, res.text
    snap = client.get("/api/admin/config").json()
    field = next(f for f in snap["fields"] if f["path"] == "router.threshold")
    assert field["value"] == 0.9 and field["restart_pending"]
    assert snap["restart_pending"] is True
    assert client.put("/api/admin/config", json={"values": {"router.threshold": "x"}}).status_code == 400


def test_household_api(client: TestClient) -> None:
    assert client.get("/api/admin/household").status_code == 401
    client.post("/api/admin/login", json={"password": "correct horse battery"})
    res = client.get("/api/admin/household")
    assert res.status_code == 200
    body = res.json()
    assert body["hermes_profiles"] == ["default", "dom", "serzant"]
    assert body["profile_keys"] == {"default": False, "dom": False, "serzant": True}
    assert "very-secret" not in res.text

    doc = body["household"]
    doc["profiles"].append({"id": "serzant", "label": "Sierżant", "description": "lista zadań"})
    doc["people"] = [{"id": "tymek", "name": "Tymek", "default_profile": "default", "profiles": ["serzant"]}]
    res = client.put("/api/admin/household", json=doc)
    assert res.status_code == 200, res.text
    assert client.get("/api/admin/household").json()["household"]["people"][0]["name"] == "Tymek"

    doc["people"][0]["profiles"] = ["nope"]
    assert client.put("/api/admin/household", json=doc).status_code == 400


def test_a_parallel_burst_gets_at_most_max_failures_guesses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The route checks the lockout, then runs the ~50 ms scrypt check in a worker thread: without
    # counting guesses still in flight (and a lock), a parallel burst of 200 got ~72 guesses checked.
    import concurrent.futures

    monkeypatch.setattr(auth_module, "MAX_FAILURES", 5)
    creds = tmp_path / "admin.toml"
    write_credentials(creds, "right password")
    auth = AdminAuth(creds)
    checked = 0
    real_check = auth_module.check_password

    def counting_check(password: str, encoded: str) -> bool:
        nonlocal checked
        checked += 1
        return real_check(password, encoded)

    monkeypatch.setattr(auth_module, "check_password", counting_check)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(lambda _: auth.login("nope", "203.0.113.7"), range(64)))
    assert checked <= 5
    assert auth.locked_out("203.0.113.7")


def test_other_addresses_cannot_evict_a_locked_out_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_module, "MAX_FAILURES", 3)
    monkeypatch.setattr(auth_module, "MAX_TRACKED_CLIENTS", 4)
    creds = tmp_path / "admin.toml"
    write_credentials(creds, "right password")
    auth = AdminAuth(creds)
    for _ in range(3):
        auth.record_failure("203.0.113.7")
    for i in range(50):
        auth.record_failure(f"10.0.0.{i}")
        assert auth.locked_out("203.0.113.7")
    assert auth.login("right password", "203.0.113.7") is None
    assert len(auth._failures) <= auth_module.MAX_TRACKED_CLIENTS


def test_concurrent_failures_from_many_addresses_never_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import concurrent.futures

    monkeypatch.setattr(auth_module, "MAX_TRACKED_CLIENTS", 8)
    auth = AdminAuth(tmp_path / "admin.toml")

    def hammer(i: int) -> None:
        auth.record_failure(f"10.0.{i % 7}.{i % 50}")
        auth.locked_out(f"10.0.{i % 5}.{i % 40}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(hammer, range(4000)))  # raises here if a worker raised
    assert len(auth._failures) <= 8


@pytest.mark.parametrize(
    ("peer", "key"),
    [
        ("::ffff:198.51.100.9", "198.51.100.9"),
        ("198.51.100.9", "198.51.100.9"),
        ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
        ("2001:db8:1:2:bbbb::7", "2001:db8:1:2::/64"),
        ("?", "?"),
    ],
)
def test_lockout_buckets_normalise_address_forms(peer: str, key: str) -> None:
    assert auth_module.client_key(peer) == key

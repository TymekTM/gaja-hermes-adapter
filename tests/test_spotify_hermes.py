"""Hermes-brokered Spotify tokens (token_source = "hermes"): subprocess path via fake brokers.

The fake brokers are real scripts run by a real interpreter (sys.executable), so the whole
asyncio.create_subprocess_exec path (env, HERMES_HOME, timeout, single-flight, cooldown) is
exercised without ever touching the real ~/.hermes or calling Spotify. One integration test
runs the REAL broker with the REAL Hermes interpreter against a tmp HERMES_HOME holding a fake
non-expiring auth.json (no network); skipped when `hermes` is not installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

import gaja.tools.spotify_hermes as spotify_hermes
from gaja.config import Settings, SpotifyConfig
from gaja.household import HouseholdStore
from gaja.tools.base import current_session_context
from gaja.tools.spotify_hermes import (
    broker_script_path,
    hermes_access_token,
    hermes_home_for,
    hermes_spotify_ready,
    profile_for_current_turn,
    reset_hermes_spotify_caches,
    resolve_hermes_interpreter,
    run_hermes_broker,
    set_spotify_household,
    usable_hermes_interpreter,
)
from gaja.tools.spotify_tool import is_spotify_ready, reset_spotify_caches, spotify

FAKE_BROKER_SRC = '''
import datetime, json, os, pathlib, sys, time

home = pathlib.Path(os.environ["HERMES_HOME"]).name
with pathlib.Path(os.environ["FAKE_BROKER_COUNT"]).open("a", encoding="utf-8") as fh:
    fh.write(("F" if "--force-refresh" in sys.argv[1:] else "") + home + "\\n")
if os.environ.get("FAKE_BROKER_SLOW_PROBE") and home.startswith("gaja-hermes-probe-"):
    time.sleep(0.8)  # F7: hold the interpreter probe open mid-flight
mode = os.environ.get("FAKE_BROKER_MODE", "")
if mode == "write_pid_sleep":  # F9: identify the child, then outlive a bare cancellation
    pathlib.Path(os.environ["FAKE_BROKER_PID_FILE"]).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(30)
if mode == "garbage":  # F6: stdout that breaks the JSON contract
    print("not json at all")
    sys.exit(0)
if mode == "stderr_secret":  # F3: a secret on stderr, nothing on stdout
    sys.stderr.write("refresh failed for refresh_token=RT-SECRET-2\\n")
    sys.exit(0)
if mode == "grandchild":  # F4: a grandchild inherits the stdout/stderr pipes
    import subprocess as _sp
    _sp.Popen(["sleep", "12"])
    time.sleep(60)
if mode == "term_ignoring_helper":  # R2-2: the leader dies on SIGTERM, its helper ignores it
    import subprocess as _sp
    helper = _sp.Popen(["sh", "-c", "trap '' TERM; sleep 30"])
    pathlib.Path(os.environ["FAKE_BROKER_PID_FILE"]).write_text(str(helper.pid), encoding="utf-8")
    time.sleep(60)
if mode == "sigterm_trap":  # F5: the broker must get a SIGTERM before any SIGKILL
    import signal as _sig
    def _on_term(*_a):
        pathlib.Path(os.environ["FAKE_BROKER_SIGTERM_MARKER"]).write_text("terminated", encoding="utf-8")
        sys.exit(0)
    _sig.signal(_sig.SIGTERM, _on_term)
    time.sleep(60)
spec = json.loads(pathlib.Path(os.environ["FAKE_BROKER_ROUTER"]).read_text(encoding="utf-8")).get(home)
if spec is None:
    print(json.dumps({"error": "spotify_auth_missing", "relogin": True}))
    sys.exit(0)
if spec.get("delay"):
    time.sleep(float(spec["delay"]))
payload = dict(spec["payload"])
if "expires_in_s" in payload:  # expiry computed at spawn time (like the real broker)
    payload["expires_at"] = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=payload.pop("expires_in_s"))
    ).isoformat()
print(json.dumps(payload))
'''


class FakeBroker:
    """Router-driven fake broker: per-Hermes-home responses plus a spawn counter."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.router_path = tmp_path / "broker_router.json"
        self.count_path = tmp_path / "broker_spawns.txt"
        script = tmp_path / "fake_broker.py"
        script.write_text(FAKE_BROKER_SRC, encoding="utf-8")
        self.script = script
        self.router_path.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("FAKE_BROKER_ROUTER", str(self.router_path))
        monkeypatch.setenv("FAKE_BROKER_COUNT", str(self.count_path))
        monkeypatch.setattr(spotify_hermes, "broker_script_path", lambda: script)

    def configure(self, home_name: str, payload: dict[str, Any] | None, *, delay: float = 0.0) -> None:
        router = json.loads(self.router_path.read_text(encoding="utf-8") or "{}")
        if payload is None:
            router.pop(home_name, None)
        else:
            router[home_name] = {"payload": payload, "delay": delay}
        self.router_path.write_text(json.dumps(router), encoding="utf-8")

    def spawns(self) -> list[str]:
        try:
            return self.count_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

    def count(self) -> int:
        return len(self.spawns())


def _ok_payload(access_token: str, *, expires_in_s: float = 3600.0) -> dict[str, Any]:
    return {
        "access_token": access_token,
        "expires_in_s": expires_in_s,  # the fake broker stamps expires_at at spawn time
        "scope": "user-read-playback-state",
        "base_url": "https://api.spotify.com/v1",
        # Must be dropped by the broker contract; the real broker never emits it, but a
        # compromised/buggy one must not get it into Gaja either.
        "refresh_token": "SECRET_REFRESH_MUST_NOT_LEAK",
    }


def _hermes_settings(tmp_path: Path, **extra: Any) -> Settings:
    return Settings(
        spotify={
            "enabled": True,
            "token_source": "hermes",
            "hermes_python": sys.executable,
            "hermes_home": str(tmp_path / "hermes-home"),
            "timeout_s": 3.0,
            **extra,
        }
    )


def _cfg(settings: Settings) -> SpotifyConfig:
    return settings.spotify


@pytest.fixture(autouse=True)
def _clean_hermes_state() -> Any:
    reset_spotify_caches()
    set_spotify_household(None)
    yield
    set_spotify_household(None)
    reset_spotify_caches()


@pytest.fixture()
def broker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeBroker:
    return FakeBroker(tmp_path, monkeypatch)


def _write_household(tmp_path: Path) -> HouseholdStore:
    data = {
        "people": [
            {"id": "ola", "name": "Ola", "default_profile": "ola"},
            {"id": "marek", "name": "Marek", "default_profile": "marek"},
        ],
        "profiles": [
            {"id": "default", "label": "Gaja"},
            {"id": "ola", "label": "Ola"},
            {"id": "marek", "label": "Marek"},
        ],
        "general_profile": "default",
    }
    path = tmp_path / "household.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return HouseholdStore(path)


# -- helpers -------------------------------------------------------------------------


def test_hermes_home_for_profiles(tmp_path: Path) -> None:
    cfg = SpotifyConfig(hermes_home=str(tmp_path / "hermes-home"))
    assert hermes_home_for(cfg, "default") == tmp_path / "hermes-home"
    assert hermes_home_for(cfg, "ola") == tmp_path / "hermes-home" / "profiles" / "ola"
    default_cfg = SpotifyConfig()
    assert hermes_home_for(default_cfg, "default") == Path.home() / ".hermes"


def test_resolve_hermes_interpreter_config_wins(tmp_path: Path) -> None:
    assert resolve_hermes_interpreter(sys.executable) == sys.executable
    missing = tmp_path / "nope" / "python"
    assert resolve_hermes_interpreter(str(missing)) is None
    if shutil.which("hermes"):
        found = resolve_hermes_interpreter("")
        assert found is not None and Path(found).is_file()


@pytest.mark.asyncio
async def test_usable_interpreter_explicit_config_is_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A configured interpreter is used as-is (never probed, never replaced).
    cfg = SpotifyConfig(hermes_python=sys.executable)
    assert await usable_hermes_interpreter(cfg) == sys.executable


@pytest.mark.asyncio
async def test_spawn_failure_is_not_a_usable_interpreter(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F6: a candidate that cannot even start counted as "usable", so the pipx fallback was
    # never tried and the dead choice was cached for the process lifetime.
    missing = tmp_path / "missing-hermes-python"
    monkeypatch.setattr(spotify_hermes, "resolve_hermes_interpreter", lambda configured="": str(missing))
    monkeypatch.setattr(spotify_hermes, "_pipx_venv_interpreter", lambda: sys.executable)
    cfg = SpotifyConfig(token_source="hermes", hermes_home=str(tmp_path / "hermes-home"))
    assert await usable_hermes_interpreter(cfg) == sys.executable


@pytest.mark.asyncio
async def test_probe_bad_output_is_not_a_usable_interpreter(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F6: an interpreter whose probe run breaks the JSON contract is unusable — there is no
    # fallback candidate left here, so hermes mode must answer None (not sys.executable).
    monkeypatch.setenv("FAKE_BROKER_MODE", "garbage")
    monkeypatch.setattr(spotify_hermes, "resolve_hermes_interpreter", lambda configured="": sys.executable)
    monkeypatch.setattr(spotify_hermes, "_pipx_venv_interpreter", lambda: None)
    cfg = SpotifyConfig(token_source="hermes", hermes_home=str(tmp_path / "hermes-home"))
    assert await usable_hermes_interpreter(cfg) is None


@pytest.mark.asyncio
async def test_shebang_env_is_skipped_for_a_real_python(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F10: the first shebang token can be /usr/bin/env (or a shell) — running the broker
    # under it hands the broker path and Gaja's whole env to an arbitrary launcher. Only a
    # candidate whose basename starts with "python" is probed; others are skipped.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hermes_sh = fake_bin / "hermes"
    hermes_sh.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    hermes_sh.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    # env needs an executable target: give the fake broker a real interpreter shebang.
    broker.script.write_text(
        f"#!{sys.executable}\n" + broker.script.read_text(encoding="utf-8"), encoding="utf-8"
    )
    broker.script.chmod(0o755)
    monkeypatch.setattr(spotify_hermes, "_pipx_venv_interpreter", lambda: sys.executable)
    cfg = SpotifyConfig(token_source="hermes", hermes_home=str(tmp_path / "hermes-home"))
    assert await usable_hermes_interpreter(cfg) == sys.executable


@pytest.mark.asyncio
async def test_concurrent_profiles_share_the_interpreter_probe(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F7: _hermes_interpreter_resolved was set before the probes finished, so a second
    # profile's flight arriving mid-probe read None, took the 60 s cooldown and failed.
    monkeypatch.setenv("FAKE_BROKER_SLOW_PROBE", "1")
    monkeypatch.setattr(spotify_hermes, "resolve_hermes_interpreter", lambda configured="": sys.executable)
    monkeypatch.setattr(spotify_hermes, "_pipx_venv_interpreter", lambda: None)
    cfg = SpotifyConfig(token_source="hermes", hermes_home=str(tmp_path / "hermes-home"))
    broker.configure("hermes-home", _ok_payload("tok-house"))
    broker.configure("ola", _ok_payload("tok-ola"))

    default_flight = asyncio.create_task(hermes_access_token(cfg))
    for _ in range(200):  # let the default flight reach the (slow) interpreter probe
        if broker.count() >= 1:
            break
        await asyncio.sleep(0.02)
    ola_token = await hermes_access_token(cfg, "ola")
    assert ola_token is not None and ola_token.access_token == "tok-ola"
    assert not any("ola" in k for k in spotify_hermes._hermes_failed_until)
    default_token = await default_flight
    assert default_token is not None and default_token.access_token == "tok-house"


# -- cache / single-flight / cooldown / timeout ---------------------------------------


@pytest.mark.asyncio
async def test_hermes_cache_hit_spawns_once(tmp_path: Path, broker: FakeBroker) -> None:
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-a"))
    first = await hermes_access_token(_cfg(settings))
    second = await hermes_access_token(_cfg(settings))
    assert first is not None and second is not None
    assert first.access_token == "tok-a" == second.access_token
    assert broker.count() == 1


@pytest.mark.asyncio
async def test_hermes_expired_token_respawns(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(spotify_hermes, "HERMES_TOKEN_EXPIRY_SKEW_S", 0.0)
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-old", expires_in_s=0.6))
    first = await hermes_access_token(_cfg(settings))
    assert first is not None and first.access_token == "tok-old"
    await asyncio.sleep(0.7)  # token lapses (skew 0)
    broker.configure("hermes-home", _ok_payload("tok-new"))
    second = await hermes_access_token(_cfg(settings))
    assert second is not None and second.access_token == "tok-new"
    assert broker.count() == 2


@pytest.mark.asyncio
async def test_hermes_single_flight_one_spawn(
    tmp_path: Path, broker: FakeBroker
) -> None:
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-sf"), delay=0.4)
    tokens = await asyncio.gather(*(hermes_access_token(_cfg(settings)) for _ in range(3)))
    assert all(t is not None and t.access_token == "tok-sf" for t in tokens)
    assert broker.count() == 1


@pytest.mark.asyncio
async def test_hermes_failure_cooldown_blocks_respawn(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(spotify_hermes, "HERMES_BROKER_FAILURE_COOLDOWN_S", 0.05)
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", {"error": "spotify_refresh_failed", "relogin": True})
    assert await hermes_access_token(_cfg(settings)) is None
    assert await hermes_access_token(_cfg(settings)) is None  # cooldown: no respawn
    assert broker.count() == 1
    await asyncio.sleep(0.08)
    assert await hermes_access_token(_cfg(settings)) is None  # respawned, failed again
    assert broker.count() == 2


@pytest.mark.asyncio
async def test_broker_no_output_logs_a_fixed_string_only(
    tmp_path: Path,
    broker: FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # F3: with the fd guard the broker's stderr can hold anything Hermes printed (the live
    # probe caught a refresh_token line there); the last line must never reach Gaja's log.
    monkeypatch.setenv("FAKE_BROKER_MODE", "stderr_secret")
    settings = _hermes_settings(tmp_path)
    with caplog.at_level(logging.WARNING, logger="gaja.tools.spotify_hermes"):
        assert await hermes_access_token(_cfg(settings)) is None
    assert "broker produced no output" in caplog.text
    assert "RT-SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_broker_timeout_kills_the_process_group(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F4: proc.kill() reaps only the direct child; a grandchild holding the pipes keeps
    # communicate() waiting for the grandchild's full lifetime (probe: a 1 s timeout took
    # 40 s). The broker runs in its own session and its whole group is killed; every wait
    # on the timeout path stays bounded.
    monkeypatch.setenv("FAKE_BROKER_MODE", "grandchild")
    monkeypatch.setattr(spotify_hermes, "HERMES_BROKER_TIMEOUT_S", 0.5)
    settings = _hermes_settings(tmp_path)
    started = time.monotonic()
    assert await hermes_access_token(_cfg(settings)) is None
    elapsed = time.monotonic() - started
    assert elapsed < 8.0, f"broker timeout path waited {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_broker_timeout_sends_sigterm_before_sigkill(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F5: a SIGKILL at timeout can land mid-refresh — after Spotify rotated the refresh token
    # but before Hermes persisted the new one, the login would be lost. The broker gets a
    # SIGTERM and a short grace to shut down before the group is SIGKILLed.
    marker = tmp_path / "sigterm.marker"
    monkeypatch.setenv("FAKE_BROKER_MODE", "sigterm_trap")
    monkeypatch.setenv("FAKE_BROKER_SIGTERM_MARKER", str(marker))
    monkeypatch.setattr(spotify_hermes, "HERMES_BROKER_TIMEOUT_S", 0.5)
    settings = _hermes_settings(tmp_path)
    assert await hermes_access_token(_cfg(settings)) is None
    assert marker.exists(), "broker was SIGKILLed without a SIGTERM grace"


@pytest.mark.asyncio
async def test_timeout_kills_a_helper_that_ignores_sigterm(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # R2-2: the SIGKILL used to depend on the group leader still running; a helper that ignores
    # SIGTERM outlived a leader that exited on it.
    pid_file = tmp_path / "helper.pid"
    monkeypatch.setenv("FAKE_BROKER_MODE", "term_ignoring_helper")
    monkeypatch.setenv("FAKE_BROKER_PID_FILE", str(pid_file))
    monkeypatch.setattr(spotify_hermes, "HERMES_BROKER_TIMEOUT_S", 0.5)
    settings = _hermes_settings(tmp_path)
    assert await hermes_access_token(_cfg(settings)) is None
    helper = int(pid_file.read_text(encoding="utf-8").strip())
    try:
        await _await_pid_gone(helper)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(helper, signal.SIGKILL)


@pytest.mark.asyncio
async def test_live_token_serves_while_a_failed_rewarm_cools_down(tmp_path: Path, broker: FakeBroker) -> None:
    # R2-1: a failed pre-expiry re-warm set the cooldown, and the cooldown check ran before the
    # live-token return, so every turn lost Spotify while the old token was still valid.
    settings = _hermes_settings(tmp_path)
    cfg = _cfg(settings)
    key = str(spotify_hermes.hermes_home_for(cfg, "default"))
    spotify_hermes._hermes_tokens[key] = spotify_hermes.HermesSpotifyToken(
        "AT-live", time.time() + 300, "s", "https://api.spotify.com/v1"
    )
    spotify_hermes._hermes_failed_until[key] = time.monotonic() + 45
    token = await hermes_access_token(cfg)
    assert token is not None and token.access_token == "AT-live"
    assert broker.count() == 0


@pytest.mark.asyncio
async def test_cancelled_interpreter_probe_leaves_the_cache_unresolved(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # R2-3: a probe cancelled by a cache reset marked the cache resolved with no interpreter on
    # its way out, so later callers got None for the process lifetime.
    monkeypatch.setenv("FAKE_BROKER_SLOW_PROBE", "1")
    monkeypatch.setattr(spotify_hermes, "resolve_hermes_interpreter", lambda configured="": sys.executable)
    monkeypatch.setattr(spotify_hermes, "_pipx_venv_interpreter", lambda: None)
    cfg = SpotifyConfig(token_source="hermes", hermes_home=str(tmp_path / "hermes-home"))
    task = asyncio.create_task(usable_hermes_interpreter(cfg))
    for _ in range(200):  # let the probe reach its (slow) broker run
        if broker.count() >= 1:
            break
        await asyncio.sleep(0.02)
    reset_hermes_spotify_caches()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert spotify_hermes._hermes_interpreter_resolved is False


@pytest.mark.asyncio
async def test_forced_refresh_waits_out_the_cooldown(tmp_path: Path, broker: FakeBroker) -> None:
    # F8: the 401 path always forces, so a revoked token must not mean one forced spawn (and
    # one refresh rotation) per request — forced runs wait out the failure cooldown too.
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", {"error": "spotify_refresh_failed", "relogin": True})
    assert await hermes_access_token(_cfg(settings), force_refresh=True) is None  # forced run fails
    assert await hermes_access_token(_cfg(settings), force_refresh=True) is None  # cooldown: no spawn
    assert broker.count() == 1


async def _await_pid_gone(pid: int, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"process {pid} still alive after {timeout_s}s")


@pytest.mark.asyncio
async def test_cancelled_flight_kills_the_broker_child(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F9: cancelling the maintenance task cancels only its own await — the shielded flight
    # kept running and its broker child survived the loop as an orphan. Cancelling a flight
    # must stop the child it spawned.
    pid_file = tmp_path / "broker.pid"
    monkeypatch.setenv("FAKE_BROKER_MODE", "write_pid_sleep")
    monkeypatch.setenv("FAKE_BROKER_PID_FILE", str(pid_file))
    settings = _hermes_settings(tmp_path)
    task = asyncio.create_task(hermes_access_token(_cfg(settings)))
    for _ in range(200):
        if pid_file.exists():
            break
        await asyncio.sleep(0.02)
    assert pid_file.exists(), "broker child never started"
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    reset_hermes_spotify_caches()  # cancels the in-flight flight (what shutdown used to do)
    await asyncio.gather(task, return_exceptions=True)
    await _await_pid_gone(pid)


@pytest.mark.asyncio
async def test_shutdown_hermes_spotify_reaps_in_flight_children(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F9: the server's shutdown entry point cancels in-flight runs AND waits for their
    # children to be stopped, so no broker process outlives the event loop.
    pid_file = tmp_path / "broker.pid"
    monkeypatch.setenv("FAKE_BROKER_MODE", "write_pid_sleep")
    monkeypatch.setenv("FAKE_BROKER_PID_FILE", str(pid_file))
    settings = _hermes_settings(tmp_path)
    task = asyncio.create_task(hermes_access_token(_cfg(settings)))
    for _ in range(200):
        if pid_file.exists():
            break
        await asyncio.sleep(0.02)
    assert pid_file.exists(), "broker child never started"
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    await asyncio.gather(task, spotify_hermes.shutdown_hermes_spotify(), return_exceptions=True)
    await _await_pid_gone(pid)


@pytest.mark.asyncio
async def test_broker_base_url_is_validated(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F11a: the bearer token is only ever sent where the broker says — a compromised or buggy
    # broker must not be able to aim it at another host, or at plain http.
    monkeypatch.setattr(spotify_hermes, "HERMES_BROKER_FAILURE_COOLDOWN_S", 0.0)
    settings = _hermes_settings(tmp_path)
    for bad in ("https://attacker.example/v1", "http://api.spotify.com/v1",
                "https://api.spotify.com.evil.example/v1", ""):
        payload = _ok_payload("tok-exfil")
        payload["base_url"] = bad
        broker.configure("hermes-home", payload)
        token = await hermes_access_token(_cfg(settings))
        assert token is None, f"base_url {bad!r} must not yield a token"


@pytest.mark.asyncio
async def test_maintenance_loop_survives_an_unexpected_exception(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F11d: an exception escaping a maintenance pass (e.g. an OSError from the probe's
    # mkdtemp) used to end the loop silently — no rewarm, no readiness, forever.
    monkeypatch.setattr(spotify_hermes, "HERMES_MAINTENANCE_POLL_S", 0.05)
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-m"))
    real_access = spotify_hermes.hermes_access_token
    calls = 0

    async def flaky(cfg: Any, profile_id: str = spotify_hermes.DEFAULT_PROFILE, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("unexpected")
        return await real_access(cfg, profile_id, **kwargs)

    monkeypatch.setattr(spotify_hermes, "hermes_access_token", flaky)
    task = asyncio.create_task(spotify_hermes.hermes_spotify_maintenance(settings))
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not hermes_spotify_ready(_cfg(settings)):
            assert not task.done(), "maintenance loop died on an unexpected exception"
            await asyncio.sleep(0.02)
        assert hermes_spotify_ready(_cfg(settings)), "loop never warmed the cache"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_hermes_timeout_returns_none(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(spotify_hermes, "HERMES_BROKER_TIMEOUT_S", 0.5)
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-slow"), delay=5.0)
    assert await hermes_access_token(_cfg(settings)) is None
    assert await hermes_access_token(_cfg(settings)) is None  # cooldown after timeout
    assert broker.count() == 1
    assert is_spotify_ready(settings) is False


@pytest.mark.asyncio
async def test_hermes_auth_missing_is_not_ready_and_no_fallback(
    tmp_path: Path, broker: FakeBroker
) -> None:
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-house"))  # default profile logged in
    broker.configure("ola", None)  # Ola's profile has no Spotify login
    assert await hermes_access_token(_cfg(settings)) is not None
    assert await hermes_access_token(_cfg(settings), "ola") is None  # never the house token
    assert await hermes_access_token(_cfg(settings), "ola") is None  # still suppressed
    assert broker.spawns().count("ola") == 1  # missing verdict: one probe, no hammering
    assert hermes_spotify_ready(_cfg(settings)) is True  # default profile still serves


@pytest.mark.asyncio
async def test_hermes_readiness_never_spawns(tmp_path: Path, broker: FakeBroker) -> None:
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-ready"))
    assert is_spotify_ready(settings) is False  # cold cache, and no spawn for that answer
    assert broker.count() == 0
    assert await hermes_access_token(_cfg(settings)) is not None
    for _ in range(5):
        assert is_spotify_ready(settings) is True
    assert broker.count() == 1


@pytest.mark.asyncio
async def test_hermes_maintenance_rewarms_before_expiry(
    tmp_path: Path, broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(spotify_hermes, "HERMES_TOKEN_EXPIRY_SKEW_S", 0.0)
    monkeypatch.setattr(spotify_hermes, "HERMES_MAINTENANCE_POLL_S", 0.1)
    monkeypatch.setattr(spotify_hermes, "HERMES_WARM_BEFORE_EXPIRY_S", 0.5)
    monkeypatch.setattr(spotify_hermes, "HERMES_BROKER_TIMEOUT_S", 5.0)
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-m", expires_in_s=0.8))
    task = asyncio.create_task(spotify_hermes.hermes_spotify_maintenance(settings))
    lapsed: list[float] = []
    try:
        # Wait for the startup warm to land, then watch continuously: without the pre-expiry
        # rewarm the token would lapse for up to one poll interval after each expiry.
        for _ in range(200):
            if hermes_spotify_ready(_cfg(settings)):
                break
            await asyncio.sleep(0.02)
        assert hermes_spotify_ready(_cfg(settings)), "startup warm never landed"
        deadline = time.monotonic() + 1.8
        while time.monotonic() < deadline:
            if not hermes_spotify_ready(_cfg(settings)):
                lapsed.append(time.monotonic())
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not lapsed, f"readiness lapsed {len(lapsed)} samples without a rewarm"
    assert broker.count() >= 2  # startup warm + at least one pre-expiry rewarm


# -- per person ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spotify_uses_person_profile_token(
    tmp_path: Path, broker: FakeBroker
) -> None:
    settings = _hermes_settings(tmp_path)
    set_spotify_household(_write_household(tmp_path))
    broker.configure("hermes-home", _ok_payload("tok-house"))
    broker.configure("ola", _ok_payload("tok-ola"))
    broker.configure("marek", _ok_payload("tok-marek"))

    seen_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json={"is_playing": False, "item": {"name": "X"}})

    async def _ask(person: str | None) -> None:
        seen_auth.clear()
        token = current_session_context.set({"person": person, "session_id": "s1"})
        try:
            await spotify(
                "now_playing",
                lang="pl",
                settings=settings,
                transport=httpx.MockTransport(handler),
            )
        finally:
            current_session_context.reset(token)

    await _ask("ola")
    assert seen_auth == ["Bearer tok-ola"]
    await _ask("marek")
    assert seen_auth == ["Bearer tok-marek"]
    await _ask(None)
    assert seen_auth == ["Bearer tok-house"]
    assert set(broker.spawns()) == {"hermes-home", "ola", "marek"}


@pytest.mark.asyncio
async def test_profile_for_current_turn_uses_household(tmp_path: Path) -> None:
    set_spotify_household(_write_household(tmp_path))
    token = current_session_context.set({"person": "marek"})
    try:
        assert profile_for_current_turn() == "marek"
    finally:
        current_session_context.reset(token)
    token = current_session_context.set({"person": "ghost"})
    try:
        assert profile_for_current_turn() == "default"  # unknown person -> default home
    finally:
        current_session_context.reset(token)
    assert profile_for_current_turn() == "default"


def test_wanted_profiles_cover_people(tmp_path: Path) -> None:
    set_spotify_household(_write_household(tmp_path))
    assert spotify_hermes.wanted_profiles() == ["default", "ola", "marek"]


# -- tool behaviour --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spotify_hermes_broker_failure_hands_off(
    tmp_path: Path, broker: FakeBroker
) -> None:
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", {"error": "spotify_refresh_failed", "relogin": True})
    result = await spotify("now_playing", lang="pl", settings=settings)
    assert result.error == "hermes_spotify_unavailable"
    assert result.handoff is True
    assert result.speak == ""


@pytest.mark.asyncio
async def test_spotify_hermes_401_forces_broker_refresh(
    tmp_path: Path, broker: FakeBroker
) -> None:
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-stale"))
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("Authorization", ""))
        if len(calls) == 1:
            return httpx.Response(401, json={"error": {"message": "token expired"}})
        return httpx.Response(200, json={"is_playing": False, "item": {"name": "X"}})

    token = current_session_context.set({"person": None})
    try:
        result = await spotify(
            "now_playing",
            lang="pl",
            settings=settings,
            transport=httpx.MockTransport(handler),
        )
    finally:
        current_session_context.reset(token)

    assert result.error is None
    assert calls[0] == "Bearer tok-stale"
    assert len(calls) == 2
    # The retry went back to the broker with --force-refresh (new forced spawn).
    assert "Fhermes-home" in broker.spawns()


@pytest.mark.asyncio
async def test_refresh_token_never_reaches_gaja(
    tmp_path: Path, broker: FakeBroker, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _hermes_settings(tmp_path)
    broker.configure("hermes-home", _ok_payload("tok-secret-free"))

    with caplog.at_level(logging.DEBUG, logger="gaja.tools.spotify_hermes"):
        token = await hermes_access_token(_cfg(settings))
    assert token is not None
    assert token.access_token == "tok-secret-free"
    # Structurally impossible to carry a refresh token...
    assert set(asdict(token)) == {"access_token", "expires_at", "scope", "base_url"}
    # ...and it never reached any log line or the ToolResult data either.
    assert "SECRET_REFRESH_MUST_NOT_LEAK" not in caplog.text

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"is_playing": False, "item": {"name": "X"}})

    result = await spotify(
        "now_playing",
        lang="pl",
        settings=settings,
        transport=httpx.MockTransport(handler),
    )
    assert "SECRET_REFRESH_MUST_NOT_LEAK" not in json.dumps(result.to_dict())
    assert "access_token" not in json.dumps(result.to_dict())


def test_parse_expires_at_variants() -> None:
    iso_z = "2026-10-27T12:00:00Z"
    assert spotify_hermes._parse_expires_at(iso_z) == datetime(
        2026, 10, 27, 12, 0, tzinfo=timezone.utc
    ).timestamp()
    assert spotify_hermes._parse_expires_at(1759272000) == 1759272000.0
    assert spotify_hermes._parse_expires_at("1759272000") == 1759272000.0
    assert spotify_hermes._parse_expires_at("") is None
    assert spotify_hermes._parse_expires_at("not-a-time") is None
    assert spotify_hermes._parse_expires_at(True) is None


# -- integration: the real broker, real interpreter, tmp HERMES_HOME --------------------


def _fake_auth_store(path: Path, *, expires_in_h: float = 4.0) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    state = {
        "client_id": "integration-test-client",
        "access_token": "FAKE_INTEGRATION_ACCESS",
        "refresh_token": "FAKE_INTEGRATION_REFRESH_NEVER_LEAK",
        "expires_at": (now + timedelta(hours=expires_in_h)).isoformat(),
        "obtained_at": now.isoformat(),
        "expires_in": int(expires_in_h * 3600),
        "granted_scope": "user-read-playback-state",
        "scope": "user-read-playback-state",
        "api_base_url": "https://api.spotify.com/v1",
        "token_type": "Bearer",
        "auth_type": "oauth_pkce",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "providers": {"spotify": state}}), encoding="utf-8")
    return {"access": state["access_token"], "refresh": state["refresh_token"]}


@pytest.mark.skipif(shutil.which("hermes") is None, reason="hermes CLI not installed")
@pytest.mark.asyncio
async def test_real_broker_against_tmp_hermes_home(tmp_path: Path) -> None:
    reset_spotify_caches()
    home = tmp_path / "hermes-home"
    secrets_fake = _fake_auth_store(home / "auth.json")
    sha_before = hashlib.sha256((home / "auth.json").read_bytes()).hexdigest()

    python = await usable_hermes_interpreter(SpotifyConfig())
    assert python is not None, "no interpreter could load the hermes spotify plugin"
    assert python != sys.executable  # the Hermes interpreter, not Gaja's venv

    # Raw subprocess: the stdout contract (exactly one JSON line, only the four safe keys).
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["HERMES_HOME"] = str(home)
    proc = await asyncio.create_subprocess_exec(
        python,
        str(broker_script_path()),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
    assert proc.returncode == 0
    lines = stdout_b.decode("utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert set(payload) == {"access_token", "expires_at", "scope", "base_url"}
    assert payload["access_token"] == secrets_fake["access"]
    assert secrets_fake["refresh"] not in stdout_b.decode("utf-8", "replace")

    # Through the Gaja client: token object, cache, readiness.
    token = await run_hermes_broker(python, broker_script_path(), home)
    assert token is not None
    assert token.access_token == secrets_fake["access"]
    assert token.base_url == "https://api.spotify.com/v1"
    assert token.expires_at > time.time() + 3600  # non-expiring fake: no refresh happened
    assert hashlib.sha256((home / "auth.json").read_bytes()).hexdigest() == sha_before


# A plugin whose import needs a repo-root module the editable-install finder does not map
# (hermes_dir_acl ships beside hermes_cli in the Hermes repo root, not inside hermes_cli).
REPO_ROOT_PLUGIN_SRC = '''\
from hermes_dir_acl import make_user_dir


def resolve_runtime_credentials(force_refresh=False, refresh_if_expiring=True, refresh_skew_seconds=0.0):
    assert make_user_dir is not None
    return {
        "access_token": "REPO-ROOT-OK",
        "expires_at": "2099-01-01T00:00:00Z",
        "scope": "user-read-playback-state",
        "base_url": "https://api.spotify.com/v1",
        "refresh_token": "NEVER-SHOWN-OUTSIDE-THE-BROKER",
    }
'''


@pytest.mark.skipif(shutil.which("hermes") is None, reason="hermes CLI not installed")
@pytest.mark.asyncio
async def test_real_broker_maps_hermes_repo_root_for_plugin_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # L1: under the pipx venv interpreter hermes_cli resolves to the editable checkout, but a
    # top-level repo-root module added after `pip install -e` (hermes_dir_acl) is not mapped by
    # the finder — every real profile refresh died with plugin_import_failed until the broker
    # puts hermes_cli's parent directory on sys.path like Hermes's own launcher does.
    reset_spotify_caches()
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "spotify_auth.py").write_text(REPO_ROOT_PLUGIN_SRC, encoding="utf-8")
    monkeypatch.setenv("GAJA_HERMES_SPOTIFY_PLUGIN_DIR", str(plugin_dir))
    home = tmp_path / "empty-home"
    home.mkdir()

    python = await usable_hermes_interpreter(SpotifyConfig())
    assert python is not None, "no interpreter could load the hermes spotify plugin"
    token = await run_hermes_broker(python, broker_script_path(), home)
    assert token is not None
    assert token.access_token == "REPO-ROOT-OK"
    assert token.base_url == "https://api.spotify.com/v1"


def _broker_child_env(home: Path) -> dict[str, str]:
    """Clean env for a raw broker child: HERMES_HOME set, pytest/buffering markers gone."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTEST_CURRENT_TEST", "PYTHONUNBUFFERED")
    }
    env["HERMES_HOME"] = str(home)
    return env


@pytest.mark.asyncio
async def test_real_broker_plugin_prints_never_reach_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F2: the fd-level guard sends plugin print()s to fd 2, but Python buffers sys.stdout —
    # without a flush while fd 1 still points at fd 2 the buffered noise follows the restored
    # fd 1 and lands on real stdout, breaking the "exactly one JSON line" contract.
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "spotify_auth.py").write_text(
        "def resolve_runtime_credentials(force_refresh=False, refresh_if_expiring=True,\n"
        "                              refresh_skew_seconds=0.0):\n"
        '    print("PLUGIN-NOISE refresh_token=RT-SECRET-2")\n'
        '    return {"access_token": "AT-OK", "expires_at": "2099-01-01T00:00:00Z",\n'
        '            "scope": "s", "base_url": "https://api.spotify.com/v1",\n'
        '            "refresh_token": "RT-SECRET-2"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("GAJA_HERMES_SPOTIFY_PLUGIN_DIR", str(plugin_dir))
    home = tmp_path / "home"
    home.mkdir()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(broker_script_path()),
        env=_broker_child_env(home),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=60)
    out = stdout_b.decode("utf-8", "replace")
    lines = out.strip().splitlines()
    assert len(lines) == 1, f"broker stdout must be exactly one JSON line, got {lines!r}"
    payload = json.loads(lines[0])
    assert payload["access_token"] == "AT-OK"
    assert "PLUGIN-NOISE" not in out
    assert "RT-SECRET-2" not in out
    assert "PLUGIN-NOISE" in stderr_b.decode("utf-8", "replace")  # noise went to stderr


@pytest.mark.skipif(shutil.which("hermes") is None, reason="hermes CLI not installed")
@pytest.mark.asyncio
async def test_real_broker_auth_missing_for_empty_home(tmp_path: Path) -> None:
    reset_spotify_caches()
    home = tmp_path / "empty-home"
    home.mkdir()
    python = await usable_hermes_interpreter(SpotifyConfig())
    assert python is not None
    token = await run_hermes_broker(python, broker_script_path(), home)
    assert token is None

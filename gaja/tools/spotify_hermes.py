"""Borrow a short-lived Spotify access token from Hermes (token_source = "hermes").

Why a broker subprocess: Spotify rotates refresh tokens, so two independent refreshers of one
login invalidate each other. Hermes stays the only refresher — ``scripts/hermes_spotify_token.py``
runs under Hermes's interpreter and calls Hermes's own ``spotify_auth.resolve_runtime_credentials``
(which refreshes under ``hermes_cli.auth._auth_store_lock()``). Gaja receives only the four safe
fields; the refresh token never enters this process, its logs or its data directory, and the
access token lives in memory only until ``expires_at - 60 s``.

Rules implemented here:
- one subprocess per need (single-flight per Hermes home; concurrent turns await the same flight),
- a broker failure means "not ready" and the turn hands off to Hermes; respawns are cooled down
  for 60 s so every turn does not spawn a process,
- readiness (:func:`hermes_spotify_ready`) is cache-only — never spawns,
- a background maintenance task (started by the server) warms the cache at startup and shortly
  before expiry,
- the turn's person selects their Hermes profile (HERMES_HOME) and Hermes resolves that profile
  exactly as its own processes would — including Hermes's per-provider fallback to the household
  root store when the profile has no login of its own; Gaja itself never substitutes another
  profile's cached token, and ``spotify_auth_missing`` (no login in the profile or the root)
  simply means that person is not ready.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gaja.config import ROOT, Settings, SpotifyConfig
from gaja.tools.base import current_session_context

log = logging.getLogger(__name__)

DEFAULT_PROFILE = "default"

# A cached token is used until expires_at - 60 s, mirroring the file mode's skew.
HERMES_TOKEN_EXPIRY_SKEW_S = 60.0
# Upper bound for one broker run (interpreter boot + plugin import + a refresh under Hermes's lock).
HERMES_BROKER_TIMEOUT_S = 25.0
# Bound for draining the broker's pipes after its process group was killed.
HERMES_BROKER_KILL_WAIT_S = 5.0
# Grace between SIGTERM and SIGKILL on the timeout path: a hard kill can land mid-refresh,
# after Spotify rotated the refresh token but before Hermes persisted the new one.
HERMES_BROKER_TERM_GRACE_S = 2.0
# After a broker failure, do not respawn for this long (every turn must not spawn a process).
HERMES_BROKER_FAILURE_COOLDOWN_S = 60.0
# A profile that answered spotify_auth_missing is re-checked at most this often.
HERMES_AUTH_MISSING_RECHECK_S = 900.0
# Rewarm a cached token this long before it lapses.
HERMES_WARM_BEFORE_EXPIRY_S = 120.0
# Maintenance loop cadence (also how quickly a newly added person gets warmed).
HERMES_MAINTENANCE_POLL_S = 30.0


@dataclass(frozen=True)
class HermesSpotifyToken:
    """Only the four safe fields the broker emits — there is no field for a refresh token."""

    access_token: str
    expires_at: float  # epoch seconds (parsed from Hermes's ISO string or epoch number)
    scope: str
    base_url: str


# In-memory state, keyed by the Hermes home path (one login per Hermes profile).
_hermes_tokens: dict[str, HermesSpotifyToken] = {}
_hermes_failed_until: dict[str, float] = {}  # broker failure cooldown ends (monotonic)
_hermes_missing_until: dict[str, float] = {}  # spotify_auth_missing re-check (monotonic)
_hermes_flights: dict[str, asyncio.Task[HermesSpotifyToken | None]] = {}
_hermes_interpreter: str | None = None  # probe cache for the broker interpreter
_hermes_interpreter_resolved = False
_hermes_interpreter_task: asyncio.Task[str | None] | None = None  # one probe flight, shared

# Household store hook (set by gaja.server at startup); maps a turn's person to their profile.
_spotify_household: Any = None


def set_spotify_household(store: Any) -> None:
    """Give the tool the HouseholdStore (person -> Hermes profile; None clears it)."""
    global _spotify_household
    _spotify_household = store


def reset_hermes_spotify_caches() -> None:
    """Clear every in-memory broker cache/flight (unit-test isolation)."""
    for task in list(_hermes_flights.values()):
        task.cancel()
    _hermes_flights.clear()
    _hermes_tokens.clear()
    _hermes_failed_until.clear()
    _hermes_missing_until.clear()
    global _hermes_interpreter, _hermes_interpreter_resolved, _hermes_interpreter_task
    if _hermes_interpreter_task is not None:
        _hermes_interpreter_task.cancel()
        _hermes_interpreter_task = None
    _hermes_interpreter = None
    _hermes_interpreter_resolved = False


def broker_script_path() -> Path:
    """The broker script (runs under the Hermes interpreter, never Gaja's venv)."""
    return ROOT / "scripts" / "hermes_spotify_token.py"


def hermes_home_for(cfg: SpotifyConfig, profile_id: str = DEFAULT_PROFILE) -> Path:
    """HERMES_HOME for a profile: the root itself for "default", <root>/profiles/P otherwise."""
    base = Path(cfg.hermes_home).expanduser() if cfg.hermes_home.strip() else Path.home() / ".hermes"
    profile = (profile_id or DEFAULT_PROFILE).strip()
    return base if profile == DEFAULT_PROFILE else base / "profiles" / profile


def _shebang_interpreter(executable: str) -> str | None:
    """Interpreter named by an executable's shebang line (never a version, always resolved)."""
    path = shutil.which(executable)
    if path is None:
        return None
    try:
        first = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    parts = first[2:].split()
    return parts[0] if parts else None


def resolve_hermes_interpreter(configured: str = "") -> str | None:
    """Configured interpreter if set (must exist), else the shebang of `hermes` on PATH."""
    if configured.strip():
        candidate = Path(os.path.expandvars(configured)).expanduser()
        if candidate.is_file():
            return str(candidate)
        log.warning("[spotify] hermes_python=%s does not exist; hermes mode unusable", configured)
        return None
    return _shebang_interpreter("hermes")


def _pipx_venv_interpreter() -> str | None:
    """Interpreter of the pipx-managed hermes-agent venv (the conventional location).

    Some installs keep a stale `hermes` on PATH while the running services use a pipx venv
    (e.g. an editable checkout); the probe in :func:`usable_hermes_interpreter` decides.
    """
    pipx_home = Path(os.environ.get("PIPX_HOME", str(Path.home() / ".local/share/pipx")))
    candidate = pipx_home / "venvs" / "hermes-agent" / "bin" / "python"
    return str(candidate) if candidate.is_file() else None


def _broker_env(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    # The child must never look like a pytest run (hermes_cli's seat belt keys off this marker).
    env.pop("PYTEST_CURRENT_TEST", None)
    return env


def _kill_broker_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Signal the broker's whole process group (start_new_session=True makes pgid == pid)."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, sig)


async def _stop_broker_group(proc: asyncio.subprocess.Process) -> None:
    """Stop the broker's process group, bounded: SIGTERM, a short grace, then SIGKILL.

    SIGTERM first so a mid-refresh Hermes can still persist a rotated refresh token before it
    dies (F5); the group kill catches helpers that inherited the pipes (F4), and both waits are
    bounded so the timeout path always returns.
    """
    _kill_broker_group(proc, signal.SIGTERM)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=HERMES_BROKER_TERM_GRACE_S)
    # SIGKILL the group even when the leader exited on SIGTERM: a helper that ignores SIGTERM
    # may still hold the pipes. Killing an empty group is a no-op.
    _kill_broker_group(proc, signal.SIGKILL)
    if proc.returncode is None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=HERMES_BROKER_KILL_WAIT_S)


def _valid_spotify_base_url(url: str) -> bool:
    """The broker's base_url must point at https://api.spotify.com — the bearer token is only
    ever sent there (F11a: a compromised or buggy broker must not be able to aim it elsewhere)."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    return parsed.scheme == "https" and (parsed.hostname or "").lower() == "api.spotify.com"


async def _run_broker(
    python: str,
    script: Path,
    home: Path,
    *,
    timeout_s: float | None = None,
    force_refresh: bool = False,
    probe: bool = False,
) -> tuple[HermesSpotifyToken | None, str | None]:
    """One broker run -> (token, error_code); exactly one of the two is set.

    ``probe`` marks the interpreter probe's empty throwaway home, where a missing login is the
    expected answer and not worth a log line about a "profile".
    """
    timeout = HERMES_BROKER_TIMEOUT_S if timeout_s is None else timeout_s
    argv = [python, str(script)]
    if force_refresh:
        argv.append("--force-refresh")
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=_broker_env(home),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group: a grandchild cannot outlive a kill
        )
    except OSError as exc:
        log.warning("[spotify] hermes broker could not start %s (%s)", python, type(exc).__name__)
        return None, "broker_spawn_failed"
    try:
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        # Stop the whole group: the direct child may have spawned helpers that inherited the
        # pipes, and an unbounded communicate() would wait for their lifetime (F4). SIGTERM
        # first so a mid-refresh broker can persist its state (F5); the drain stays bounded.
        await _stop_broker_group(proc)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.communicate(), timeout=HERMES_BROKER_KILL_WAIT_S)
        log.warning("[spotify] hermes broker timed out after %.1fs (home=%s)", timeout, home.name)
        return None, "broker_timeout"
    except asyncio.CancelledError:
        # The flight was cancelled (shutdown / cache reset): its child must not survive as
        # an orphan (F9). Bounded, and the cancellation is re-raised after the group is dead.
        await _stop_broker_group(proc)
        raise
    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    if not stdout:
        # A fixed string only: with the fd guard the broker's stderr can hold anything Hermes
        # or the plugin printed (the live probe caught a refresh_token line there).
        log.warning("[spotify] hermes broker produced no output (home=%s)", home.name)
        return None, "broker_no_output"
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        log.warning("[spotify] hermes broker output was not JSON (home=%s)", home.name)
        return None, "broker_bad_output"
    if not isinstance(payload, dict):
        log.warning("[spotify] hermes broker output was not a JSON object (home=%s)", home.name)
        return None, "broker_bad_output"
    if "error" in payload:
        code = str(payload.get("error") or "broker_error")
        if code == "spotify_auth_missing":
            if not probe:
                log.info(
                    "[spotify] hermes profile %s has no Spotify login (its own or the household root)",
                    home.name,
                )
        else:
            log.warning("[spotify] hermes broker failed (home=%s): %s", home.name, code)
        return None, code
    access = str(payload.get("access_token") or "").strip()
    expires_at = _parse_expires_at(payload.get("expires_at"))
    base_url = str(payload.get("base_url") or "").rstrip("/")
    if not access or expires_at is None or not _valid_spotify_base_url(base_url):
        log.warning(
            "[spotify] hermes broker output missing access_token/expires_at or has a bad base_url (home=%s)",
            home.name,
        )
        return None, "broker_bad_output"
    return (
        HermesSpotifyToken(
            access_token=access,
            expires_at=expires_at,
            scope=str(payload.get("scope") or ""),
            base_url=base_url,
        ),
        None,
    )


async def run_hermes_broker(
    python: str,
    script: Path,
    home: Path,
    *,
    timeout_s: float | None = None,
    force_refresh: bool = False,
) -> HermesSpotifyToken | None:
    """One broker run, token only (the integration test and diagnostics use this)."""
    token, _ = await _run_broker(python, script, home, timeout_s=timeout_s, force_refresh=force_refresh)
    return token


def _parse_expires_at(value: Any) -> float | None:
    """Epoch seconds from Hermes's ISO timestamp (Z or offset) or an epoch number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:  # Hermes writes tz-aware UTC; naive strings get local time
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.timestamp()


async def _probe_interpreter(python: str, script: Path) -> bool:
    """True when the broker under this interpreter can load the plugin + hermes_cli.

    Probed against an empty throwaway HERMES_HOME (never a real profile): the expected
    answer is the clean ``spotify_auth_missing`` line, which still proves the whole
    plugin -> hermes_cli -> auth-store chain imports and runs.
    """
    probe_home = Path(tempfile.mkdtemp(prefix="gaja-hermes-probe-"))
    try:
        _, code = await _run_broker(python, script, probe_home, timeout_s=HERMES_BROKER_TIMEOUT_S, probe=True)
        # Usable means the whole chain really ran: a token, or the clean spotify_auth_missing
        # an empty home must answer. Anything else — spawn failure, timeout, no output, bad
        # output, plugin_import_failed, broker_error — disqualifies the candidate (F6).
        return code is None or code == "spotify_auth_missing"
    finally:
        shutil.rmtree(probe_home, ignore_errors=True)


async def _resolve_hermes_interpreter(cfg: SpotifyConfig) -> str | None:
    """Resolve and probe the candidates once; fills the process-lifetime cache."""
    global _hermes_interpreter, _hermes_interpreter_resolved
    try:
        if cfg.hermes_python.strip():
            _hermes_interpreter = resolve_hermes_interpreter(cfg.hermes_python)
            return _hermes_interpreter

        candidates: list[str] = []
        shebang = resolve_hermes_interpreter("")
        if shebang:
            candidates.append(shebang)
        pipx = _pipx_venv_interpreter()
        if pipx and pipx not in candidates:
            candidates.append(pipx)
        script = broker_script_path()
        for candidate in candidates:
            # The shebang's first token can be /usr/bin/env or a shell — never run the broker
            # under those with the broker path and Gaja's full env (F10).
            if not Path(candidate).name.startswith("python"):
                log.info("[spotify] interpreter %s is not a python interpreter; skipping", candidate)
                continue
            if await _probe_interpreter(candidate, script):
                _hermes_interpreter = candidate
                return candidate
            log.info("[spotify] interpreter %s cannot load the hermes spotify plugin; trying next", candidate)
        if candidates:
            log.warning(
                "[spotify] no usable hermes interpreter for the spotify broker; set [spotify] hermes_python"
            )
        _hermes_interpreter = None
        return None
    except Exception as exc:  # noqa: BLE001 - the shared flight must never raise into many awaiters
        log.warning("[spotify] hermes interpreter probe failed (%s)", type(exc).__name__)
        _hermes_interpreter = None
        return None
    finally:
        # A cancelled probe (cache reset, shutdown) leaves the cache unresolved for the next caller.
        current = asyncio.current_task()
        if current is None or not current.cancelling():
            _hermes_interpreter_resolved = True


async def usable_hermes_interpreter(cfg: SpotifyConfig) -> str | None:
    """Interpreter the broker should run under, probed once per process.

    Order: ``hermes_python`` (explicit, unprobed — authoritative), then the shebang of
    ``hermes`` on PATH, then the pipx hermes-agent venv (some installs keep a stale
    ``hermes`` on PATH while production runs the editable venv). A shebang/pipx candidate
    only counts when its basename starts with ``python`` (F10). Concurrent callers share
    one probe flight (shielded, so a cancelled turn never cancels the probe others await):
    the first caller's cfg decides, later callers await the same task instead of reading a
    half-resolved None (F7).
    """
    global _hermes_interpreter_task
    if _hermes_interpreter_resolved:
        return _hermes_interpreter
    if _hermes_interpreter_task is None:
        _hermes_interpreter_task = asyncio.create_task(_resolve_hermes_interpreter(cfg))
    return await asyncio.shield(_hermes_interpreter_task)


def _live_token(key: str) -> HermesSpotifyToken | None:
    token = _hermes_tokens.get(key)
    if token is None:
        return None
    if time.time() < token.expires_at - HERMES_TOKEN_EXPIRY_SKEW_S:
        return token
    _hermes_tokens.pop(key, None)
    return None


def _in_cooldown(key: str) -> bool:
    until = _hermes_failed_until.get(key)
    if until is not None and time.monotonic() < until:
        return True
    _hermes_failed_until.pop(key, None)
    return False


async def hermes_access_token(
    cfg: SpotifyConfig,
    profile_id: str = DEFAULT_PROFILE,
    *,
    force_refresh: bool = False,
) -> HermesSpotifyToken | None:
    """A live access token for the profile's Hermes home, or None when unavailable.

    Single-flight per home: concurrent turns await one broker run (shielded, so a cancelled
    turn never cancels the token others are waiting for). Failures cool down for
    HERMES_BROKER_FAILURE_COOLDOWN_S; ``spotify_auth_missing`` (no login in the profile or the
    household root store — Hermes itself falls back to the root per provider) additionally
    suppresses spawns for HERMES_AUTH_MISSING_RECHECK_S. Gaja never substitutes another
    profile's cached token.
    """
    key = str(hermes_home_for(cfg, profile_id))
    # A live token keeps serving while a failed re-warm cools down. A forced run still waits out
    # the cooldown: the 401 path always forces, and a revoked token must not mean one spawn (and
    # one refresh rotation) per request (F8).
    if not force_refresh:
        token = _live_token(key)
        if token is not None:
            return token
    if _in_cooldown(key):
        return None
    missing = _hermes_missing_until.get(key)
    if missing is not None and time.monotonic() < missing:
        return None
    _hermes_missing_until.pop(key, None)

    flight = _hermes_flights.get(key)
    if flight is None:
        flight = asyncio.create_task(_obtain_token(cfg, profile_id, key, force_refresh=force_refresh))
        _hermes_flights[key] = flight

        def _done(task: asyncio.Task[HermesSpotifyToken | None], key: str = key) -> None:
            _hermes_flights.pop(key, None)
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                log.warning("[spotify] hermes broker flight failed: %s", type(exc).__name__)

        flight.add_done_callback(_done)
    return await asyncio.shield(flight)


async def _obtain_token(
    cfg: SpotifyConfig,
    profile_id: str,
    key: str,
    *,
    force_refresh: bool,
) -> HermesSpotifyToken | None:
    python = await usable_hermes_interpreter(cfg)
    if python is None:
        _hermes_failed_until[key] = time.monotonic() + HERMES_BROKER_FAILURE_COOLDOWN_S
        return None
    token, code = await _run_broker(
        python,
        broker_script_path(),
        hermes_home_for(cfg, profile_id),
        force_refresh=force_refresh,
    )
    if token is None:
        _hermes_failed_until[key] = time.monotonic() + HERMES_BROKER_FAILURE_COOLDOWN_S
        if code == "spotify_auth_missing":
            # Not a failure of the broker: that profile has no login of its own and none in the
            # household root store. Re-check rarely.
            _hermes_missing_until[key] = time.monotonic() + HERMES_AUTH_MISSING_RECHECK_S
        return None
    _hermes_failed_until.pop(key, None)
    _hermes_missing_until.pop(key, None)
    _hermes_tokens[key] = token
    log.info(
        "[spotify] borrowed hermes access token (profile %s, expires in %.0f s)",
        profile_id,
        max(0.0, token.expires_at - time.time()),
    )
    return token


async def shutdown_hermes_spotify() -> None:
    """Stop the broker machinery at server shutdown: cancel in-flight runs (their children
    are stopped, not orphaned) and await the cleanup, so no broker survives the loop (F9)."""
    tasks: list[asyncio.Task[Any]] = list(_hermes_flights.values())
    if _hermes_interpreter_task is not None:
        tasks.append(_hermes_interpreter_task)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def hermes_spotify_ready(cfg: SpotifyConfig) -> bool:
    """Cache-only readiness for the per-turn router gate: never spawns a process."""
    if not cfg.enabled:
        return False
    return any(_live_token(key) is not None for key in list(_hermes_tokens))


def profile_for_current_turn() -> str:
    """Hermes profile of the turn's identified person (their default profile), else "default"."""
    ctx = current_session_context.get()
    person_id = str((ctx or {}).get("person") or "").strip()
    if person_id and _spotify_household is not None:
        person = _spotify_household.data.person(person_id)
        if person is not None:
            return person.default_profile or DEFAULT_PROFILE
    return DEFAULT_PROFILE


def wanted_profiles() -> list[str]:
    """Profiles to keep warm: the household default plus every person's default profile."""
    profiles = [DEFAULT_PROFILE]
    if _spotify_household is not None:
        for person in _spotify_household.data.people:
            if person.default_profile and person.default_profile not in profiles:
                profiles.append(person.default_profile)
    return profiles


async def hermes_spotify_maintenance(settings: Settings) -> None:
    """Background task: warm the cache at startup and shortly before expiry (never in a turn).

    One pass per poll interval over the wanted profiles (the first pass is the startup warm:
    a cold cache warms unforced). An unexpected exception escaping a pass is logged and the
    loop backs off to the next poll instead of dying silently (F11d).
    """
    cfg = settings.spotify
    while True:
        try:
            for profile in wanted_profiles():
                key = str(hermes_home_for(cfg, profile))
                token = _live_token(key)
                if token is not None and token.expires_at - time.time() > HERMES_WARM_BEFORE_EXPIRY_S:
                    continue  # plenty of life left
                missing = _hermes_missing_until.get(key)
                if missing is not None and time.monotonic() < missing:
                    continue
                if _in_cooldown(key):
                    continue
                # Due: force past Gaja's own cache so Hermes refreshes (its code, its lock);
                # a still-live old token keeps serving if this run fails.
                await hermes_access_token(cfg, profile, force_refresh=token is not None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the maintenance loop must survive this
            log.warning(
                "[spotify] hermes broker maintenance pass failed (%s); backing off to the next poll",
                type(exc).__name__,
            )
        await asyncio.sleep(HERMES_MAINTENANCE_POLL_S)

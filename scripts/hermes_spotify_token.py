#!/usr/bin/env python3
"""Borrow Hermes's Spotify access token for one Gaja turn (run under the HERMES interpreter).

    <hermes-python> scripts/hermes_spotify_token.py [--force-refresh]

Gaja never refreshes Spotify tokens itself: Spotify rotates refresh tokens, so two independent
refreshers of one login invalidate each other. This broker runs under Hermes's own interpreter
(the shebang of `hermes`), calls the spotify plugin's ``spotify_auth.resolve_runtime_credentials``
— which refreshes only under ``hermes_cli.auth._auth_store_lock()`` — and prints exactly ONE JSON
line to stdout:

    {"access_token": "...", "expires_at": "...", "scope": "...", "base_url": "..."}

or, on failure:

    {"error": "<code>", "relogin": true|false}

The refresh token (also present in the dict Hermes returns) is dropped here and must never reach
stdout, stderr or the exit status. The Hermes profile is selected by ``HERMES_HOME`` (default
``~/.hermes``, a named profile ``p`` is ``~/.hermes/profiles/p``), exactly like every other
Hermes process. Stdlib-only on purpose: this file must import in Hermes's venv, not Gaja's.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

# The only keys Gaja may ever see. Everything else in the credentials dict Hermes returns
# (refresh_token, client_id, redirect_uri, ...) stays in this process and dies with it.
OUTPUT_KEYS = ("access_token", "expires_at", "scope", "base_url")

# Hermes loads directory plugins as "hermes_plugins.<slug>" with __path__ = [plugin_dir]
# (hermes_cli/plugins_loader.py); the plugin's modules import each other with relative imports
# (``from .spotify_auth import ...``), so the package context has to exist for them.
_NS_PARENT = "hermes_plugins"
_PKG = "hermes_plugins.spotify"


def _plugin_dir() -> Path:
    """The spotify plugin directory: explicit override, then $HERMES_HOME, then ~/.hermes."""
    override = os.environ.get("GAJA_HERMES_SPOTIFY_PLUGIN_DIR", "").strip()
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    candidates: list[Path] = []
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        candidates.append(Path(os.path.expandvars(env_home)).expanduser())
    candidates.append(Path.home() / ".hermes")
    for base in candidates:
        candidate = base / "plugins" / "spotify"
        if (candidate / "spotify_auth.py").is_file():
            return candidate
    return candidates[0] / "plugins" / "spotify"  # missing -> classified as plugin_import_failed


def _ensure_hermes_repo_root_on_path() -> None:
    """Map the Hermes repo root onto sys.path the way Hermes's own launcher does.

    Top-level companions of ``hermes_cli`` (``hermes_constants``, ``hermes_dir_acl``, ...) import
    from the repo root; Hermes's launcher inserts that root at ``sys.path[0]``. An editable
    install's static finder only maps the modules that existed at ``pip install -e`` time, so a
    module added later (``hermes_dir_acl``) is invisible to the plugin — the import blows up in
    the middle of a refresh — until the repo root itself is on sys.path.
    """
    try:
        import hermes_cli  # noqa: F401 - imported for its location only
    except ImportError:
        return  # without hermes_cli the plugin import fails with its own classified error
    root = Path(hermes_cli.__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _import_spotify_auth() -> types.ModuleType:
    """Load the plugin's ``spotify_auth`` module the way Hermes's plugin loader does."""
    _ensure_hermes_repo_root_on_path()
    plugin_dir = _plugin_dir()
    if _NS_PARENT not in sys.modules:  # namespace parent, as hermes_cli/plugins_loader.py builds it
        ns_pkg = types.ModuleType(_NS_PARENT)
        ns_pkg.__path__ = []  # type: ignore[attr-defined]
        ns_pkg.__package__ = _NS_PARENT
        sys.modules[_NS_PARENT] = ns_pkg
    if _PKG not in sys.modules:  # package shell without exec'ing __init__.py: no tool registration
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(plugin_dir)]
        pkg.__package__ = _PKG
        sys.modules[_PKG] = pkg
    module_name = f"{_PKG}.spotify_auth"
    spec = importlib.util.spec_from_file_location(module_name, plugin_dir / "spotify_auth.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {plugin_dir / 'spotify_auth.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _auth_error_class() -> type[Exception] | None:
    try:
        from hermes_cli.auth_constants import AuthError  # type: ignore[import-not-found]

        return AuthError  # type: ignore[return-value]
    except Exception:
        return None


def _classify(exc: BaseException) -> tuple[str, bool]:
    """(error code, relogin flag) for an exception; never includes the exception's message."""
    auth_error = _auth_error_class()
    if auth_error is not None and isinstance(exc, auth_error):
        code = getattr(exc, "code", None) or "spotify_auth_error"
        return str(code), bool(getattr(exc, "relogin_required", False))
    if isinstance(exc, ImportError):
        return "plugin_import_failed", False
    return "broker_error", False


def _resolve(force_refresh: bool) -> dict[str, object]:
    """Run the plugin's credential resolution and map any failure to a one-code error dict."""
    try:
        auth = _import_spotify_auth()
        creds = auth.resolve_runtime_credentials(force_refresh=force_refresh)  # type: ignore[attr-defined]
        return {key: creds.get(key) for key in OUTPUT_KEYS}
    except Exception as exc:  # noqa: BLE001 - the error line is the contract
        code, relogin = _classify(exc)
        # Exception type only: a message could carry state (paths, URLs) we promised not to leak.
        sys.stderr.write(f"hermes_spotify_token: {code} ({type(exc).__name__})\n")
        return {"error": code, "relogin": relogin}


def main(argv: list[str]) -> int:
    force_refresh = "--force-refresh" in argv
    # fd-level stdout guard: hermes_cli or the plugin could print (banners, warnings) on import or
    # during refresh; whatever they write to fd 1 goes to stderr, so our JSON line is the ONLY
    # stdout content the caller ever parses.
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        payload = _resolve(force_refresh)
    except BaseException:  # noqa: BLE001 - never let a traceback with state reach stderr
        payload = {"error": "broker_error", "relogin": False}
    finally:
        # While fd 1 still points at fd 2, push any plugin print()s out of Python's sys.stdout
        # buffer: without this flush they would follow the restored fd 1 and land on real stdout
        # ahead of our JSON line, breaking the one-line contract (F2).
        try:
            sys.stdout.flush()
        except OSError:
            pass  # a dead pipe must not stop the JSON line from being attempted
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Admin panel authentication: one password (scrypt hash) and HMAC-signed session cookies.

The credentials file (TOML, mode 600, outside the repo) holds `password_hash` and `session_key`.
Changing the password rotates the session key, which logs out every open session.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
import tomllib
from pathlib import Path

COOKIE = "gaja_admin"
_SCRYPT = {"n": 2**15, "r": 8, "p": 1}
# Failed logins: after MAX_FAILURES within FAILURE_WINDOW_S every login is refused until the window passes.
MAX_FAILURES = 10
FAILURE_WINDOW_S = 600.0


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, maxmem=2**26, dklen=32, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${_b64(salt)}${_b64(digest)}"


def check_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = encoded.split("$")
        if scheme != "scrypt":
            return False
        got = hashlib.scrypt(
            password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p), maxmem=2**26, dklen=32
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, _unb64(digest))


def write_credentials(path: Path, password: str) -> None:
    """Store a new password hash and a fresh session key (mode 600)."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f'password_hash = "{hash_password(password)}"\nsession_key = "{secrets.token_hex(32)}"\n'
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
    os.replace(tmp, path)


class AdminAuth:
    def __init__(self, credentials_file: Path, session_hours: float = 12.0) -> None:
        self.path = credentials_file.expanduser()
        self.session_s = session_hours * 3600
        self._failures: list[float] = []

    def _load(self) -> dict[str, str] | None:
        # Read on every use, so a password set with the script works without a restart.
        try:
            data = tomllib.loads(self.path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            return None
        if not data.get("password_hash") or not data.get("session_key"):
            return None
        return data

    @property
    def configured(self) -> bool:
        return self._load() is not None

    def locked_out(self) -> bool:
        now = time.monotonic()
        self._failures = [t for t in self._failures if now - t < FAILURE_WINDOW_S]
        return len(self._failures) >= MAX_FAILURES

    def login(self, password: str) -> str | None:
        """Session token for a correct password, else None (and the failure counts toward the lockout)."""
        creds = self._load()
        if creds is None or self.locked_out():
            return None
        if not check_password(password, creds["password_hash"]):
            self._failures.append(time.monotonic())
            return None
        self._failures.clear()
        expires = int(time.time() + self.session_s)
        payload = f"{expires}.{secrets.token_urlsafe(12)}"
        return f"{payload}.{self._sign(creds, payload)}"

    def valid(self, token: str | None) -> bool:
        if not token:
            return False
        creds = self._load()
        if creds is None:
            return False
        payload, _, sig = token.rpartition(".")
        if not payload or not hmac.compare_digest(sig, self._sign(creds, payload)):
            return False
        try:
            return int(payload.split(".", 1)[0]) > time.time()
        except ValueError:
            return False

    @staticmethod
    def _sign(creds: dict[str, str], payload: str) -> str:
        return _b64(hmac.new(bytes.fromhex(creds["session_key"]), payload.encode(), hashlib.sha256).digest())

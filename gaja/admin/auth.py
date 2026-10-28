"""Admin panel authentication: one password (scrypt hash) and HMAC-signed session cookies.

The credentials file (TOML, mode 600, outside the repo) holds `password_hash` and `session_key`.
Changing the password rotates the session key, which logs out every open session.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import os
import secrets
import threading
import time
import tomllib
from pathlib import Path

COOKIE = "gaja_admin"
_SCRYPT = {"n": 2**15, "r": 8, "p": 1}
# Failed logins: after MAX_FAILURES within FAILURE_WINDOW_S every login from that address is
# refused until the window passes.
MAX_FAILURES = 10
FAILURE_WINDOW_S = 600.0
# Hard cap on tracked addresses, so a flood from spoofed/varied sources cannot grow the dict
# without limit. The coldest address that is not locked out is evicted first; a locked-out
# address is never evicted (that would unlock it), and when every slot holds one, an unknown
# address counts as locked out until a slot frees up.
MAX_TRACKED_CLIENTS = 256


def client_key(client: str) -> str:
    """Lockout bucket of a peer address: an IPv4-mapped IPv6 address is its IPv4 address, and
    other IPv6 addresses share their /64 (one host usually owns the whole prefix)."""
    try:
        addr = ipaddress.ip_address(client)
    except ValueError:
        return client
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network(f"{addr}/64", strict=False))
    return str(addr)


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
        self._failures: dict[str, list[float]] = {}
        self._inflight: dict[str, int] = {}  # guesses past the lockout check, still in scrypt
        self._lock = threading.Lock()

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

    def _recent_failures(self, client: str) -> list[float]:
        """Failures of one bucket within the window; drops its expired entries. Caller holds _lock."""
        now = time.monotonic()
        recent = [t for t in self._failures.get(client, ()) if now - t < FAILURE_WINDOW_S]
        if recent:
            self._failures[client] = recent
        else:
            self._failures.pop(client, None)
        return recent

    def _over_budget(self, client: str) -> bool:
        """Caller holds _lock. Failures plus guesses still being checked count, so a parallel
        burst cannot get more than MAX_FAILURES guesses past the scrypt check."""
        if len(self._recent_failures(client)) + self._inflight.get(client, 0) >= MAX_FAILURES:
            return True
        return client not in self._failures and not self._make_room()

    def _make_room(self) -> bool:
        """Free a slot for a new bucket. Caller holds _lock. False when every slot is locked out."""
        while len(self._failures) >= MAX_TRACKED_CLIENTS:
            candidates = [c for c in list(self._failures) if len(self._recent_failures(c)) < MAX_FAILURES]
            if len(self._failures) < MAX_TRACKED_CLIENTS:
                break  # expired entries were pruned while checking
            if not candidates:
                return False
            self._failures.pop(min(candidates, key=lambda c: self._failures[c][-1]))
        return True

    def locked_out(self, client: str = "local") -> bool:
        with self._lock:
            return self._over_budget(client_key(client))

    def record_failure(self, client: str = "local") -> None:
        key = client_key(client)
        with self._lock:
            self._recent_failures(key)
            if key in self._failures or self._make_room():
                self._failures.setdefault(key, []).append(time.monotonic())

    def login(self, password: str, client: str = "local") -> str | None:
        """Session token for a correct password, else None (and the failure counts toward the
        lockout of that client address). Thread-safe: the route runs it in a worker thread."""
        creds = self._load()
        if creds is None:
            return None
        key = client_key(client)
        with self._lock:
            if self._over_budget(key):
                return None
            self._inflight[key] = self._inflight.get(key, 0) + 1
        try:
            ok = check_password(password, creds["password_hash"])
        finally:
            with self._lock:
                left = self._inflight.pop(key) - 1
                if left:
                    self._inflight[key] = left
        if not ok:
            self.record_failure(client)
            return None
        with self._lock:
            self._failures.pop(key, None)
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
        # compare_digest raises TypeError on non-ASCII str (a crafted cookie) -> compare bytes, so a
        # junk cookie is just "not logged in" (401), not a 500.
        if not payload or not hmac.compare_digest(sig.encode(), self._sign(creds, payload).encode()):
            return False
        try:
            return int(payload.split(".", 1)[0]) > time.time()
        except ValueError:
            return False

    @staticmethod
    def _sign(creds: dict[str, str], payload: str) -> str:
        return _b64(hmac.new(bytes.fromhex(creds["session_key"]), payload.encode(), hashlib.sha256).digest())

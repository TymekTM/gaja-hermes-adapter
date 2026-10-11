"""Per-device tokens, read from a TOML file kept outside the repository.

    [devices.atom-salon]
    token = "..."          # from scripts/device_token.py
    name = "Salon"          # optional
    enabled = true          # optional, false refuses the device

The file is re-read when its mtime changes, so adding a device needs no restart. Tokens are compared with
``hmac.compare_digest`` and are never logged.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import secrets
import tomllib
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEVICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MIN_TOKEN_LEN = 16
# Compared against when the device id is unknown, so a miss takes as long as a wrong token.
_DECOY = secrets.token_hex(32)


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    name: str = ""


def generate_token() -> str:
    return secrets.token_urlsafe(32)


class DeviceRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()
        self._mtime: float | None = None
        self._entries: dict[str, tuple[str, DeviceInfo]] = {}

    @property
    def path(self) -> Path:
        return self._path

    def _reload(self) -> None:
        try:
            stat = self._path.stat()
        except OSError:
            if self._mtime is not None or self._entries:
                log.warning("device token file %s is not readable; no device can connect", self._path)
            self._mtime, self._entries = None, {}
            return
        if stat.st_mtime_ns == self._mtime:
            return
        if stat.st_mode & 0o077:
            log.warning("device token file %s is accessible by group/others (mode %o); chmod 600", self._path, stat.st_mode & 0o777)
        try:
            data = tomllib.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log.error("device token file %s is invalid (%s); keeping the previous entries", self._path, type(exc).__name__)
            return
        entries: dict[str, tuple[str, DeviceInfo]] = {}
        for device_id, row in (data.get("devices") or {}).items():
            token = row.get("token") if isinstance(row, dict) else None
            if not DEVICE_ID.match(device_id) or not isinstance(token, str) or len(token) < MIN_TOKEN_LEN:
                log.warning("device %r skipped: bad id or token shorter than %d characters", device_id, MIN_TOKEN_LEN)
                continue
            if row.get("enabled", True) is not True:
                continue
            entries[device_id] = (token, DeviceInfo(device_id, str(row.get("name", ""))))
        self._mtime, self._entries = stat.st_mtime_ns, entries
        log.info("device registry: %d device(s) from %s", len(entries), self._path)

    def authenticate(self, device_id: object, token: object) -> DeviceInfo | None:
        self._reload()
        if not isinstance(device_id, str) or not isinstance(token, str):
            return None
        entry = self._entries.get(device_id)
        expected = entry[0] if entry is not None else _DECOY
        ok = hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))
        return entry[1] if ok and entry is not None else None

    def count(self) -> int:
        self._reload()
        return len(self._entries)


def append_device(path: Path, device_id: str, token: str, name: str = "") -> None:
    """Add a ``[devices.<id>]`` table (file created with mode 600). Used by scripts/device_token.py."""
    if not DEVICE_ID.match(device_id):
        raise ValueError("device id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if device_id in (existing.get("devices") or {}):
        raise ValueError(f"device {device_id!r} already exists in {path}")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(f'\n[devices."{device_id}"]\ntoken = "{token}"\n')
        if name:
            handle.write(f"name = {json.dumps(name, ensure_ascii=False)}\n")

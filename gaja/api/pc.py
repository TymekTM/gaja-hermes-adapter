"""PC tool HTTP API (device protocol v1.14, docs/DESKTOP.md): GET /api/pc, POST /api/pc/{id}/call.

Mounted on the main app. This is a remote-code-execution endpoint reachable through ``tailscale
serve``, so the browser-origin check is not enough: every request needs a bearer token (the
content of ``[pc] api_token_file``, created on first start, mode 600, parent 700) compared with
``hmac.compare_digest``, and ANY request carrying an ``Origin`` header is refused — browsers
never get to call it.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gaja.config import PcConfig, resolve

log = logging.getLogger(__name__)

# Protocol: timeout_s is 1..300 per pc_call.
MIN_TIMEOUT_S = 1
MAX_TIMEOUT_S = 300


def load_or_create_api_token(path: Path) -> str:
    """The bearer token for /api/pc: read it, or create it on first start (32 random bytes hex).

    File mode 600, parent directory 700. Called on first use (app start or first request), never
    logged.
    """
    path = path.expanduser()
    if not path.is_absolute():
        path = resolve(path)
    token = ""
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    if token:
        return token
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    token = secrets.token_hex(32)
    # Write a private temp file and link it into place: link() fails if another process (the MCP,
    # a second request) created the token first, and then everyone uses the winner's token. An
    # empty or loose-mode file left behind is replaced by a fresh mode-600 one.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
        try:
            os.link(tmp, path)
        except FileExistsError:
            existing = path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
            os.replace(tmp, path)  # the existing file was empty: take its place
    finally:
        tmp.unlink(missing_ok=True)
    log.info("created PC API token file %s (mode 600)", path)
    return token


def _pc_cfg(request: Request) -> PcConfig:
    services = getattr(request.app.state, "services", None)
    cfg = getattr(services, "cfg", None) if services is not None else None
    pc = getattr(cfg, "pc", None)
    if not isinstance(pc, PcConfig):
        return PcConfig()
    return pc


def _relay(request: Request) -> Any:
    services = getattr(request.app.state, "services", None)
    return getattr(services, "pc_relay", None) if services is not None else None


def check_pc_auth(request: Request) -> None:
    """Bearer-token auth. An Origin header is refused first: browsers never get to call /api/pc."""
    if request.headers.get("origin") is not None:
        raise HTTPException(403, "browser requests are not allowed on the PC API")
    cfg = _pc_cfg(request)
    expected = load_or_create_api_token(cfg.api_token_file) if cfg.enabled else ""
    auth = request.headers.get("authorization") or ""
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not expected or not token or not hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(401, "unauthorized")


class PcCallRequest(BaseModel):
    tool: str = Field(min_length=1, max_length=32)
    args: dict[str, Any] = Field(default_factory=dict)
    timeout_s: int | None = Field(default=None, ge=MIN_TIMEOUT_S, le=MAX_TIMEOUT_S)


def _error(status: int, error: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"ok": False, "error": error})


def create_pc_router() -> APIRouter:
    router = APIRouter(prefix="/api/pc", dependencies=[Depends(check_pc_auth)])

    @router.get("")
    async def list_pcs(request: Request) -> dict[str, Any]:
        """Connected PC devices (id, host, os, tools, connected_s)."""
        relay = _relay(request)
        return {"devices": relay.devices() if relay is not None else []}

    @router.post("/{device_id}/call")
    async def call_tool(device_id: str, body: PcCallRequest, request: Request) -> Any:
        """Run one PC tool on a connected device and wait for its result."""
        relay = _relay(request)
        entry = relay.get(device_id) if relay is not None else None
        if entry is None:
            return _error(404, f"device {device_id!r} is not connected")
        if body.tool not in entry.tools:
            return _error(400, f"device {device_id!r} does not advertise tool {body.tool!r}")
        cfg = _pc_cfg(request)
        timeout_s = min(body.timeout_s or cfg.default_timeout_s, cfg.max_timeout_s, MAX_TIMEOUT_S)
        timeout_s = max(timeout_s, MIN_TIMEOUT_S)
        result = await relay.call(device_id, body.tool, body.args, timeout_s)
        if result.error == "timeout":
            return _error(504, "timeout")
        if result.error == "disconnected":
            return _error(404, "disconnected")
        return result.payload()

    return router

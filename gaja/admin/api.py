"""Admin panel HTTP API under /api/admin (the page itself is web/admin.html).

Every route except login and session status needs a valid admin cookie and an allowed Origin.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from gaja.admin import settings_store
from gaja.admin.auth import COOKIE, AdminAuth
from gaja.hermes.gateway import list_profiles, profile_env_path, read_env_key
from gaja.household import Household

log = logging.getLogger(__name__)

BENCH_LOCK = "/tmp/gaja-bench.lock"
RESTART_LOG = "/tmp/gaja-adapter-restart.log"
UNIT = "gaja-adapter.service"
STARTED_AT = time.time()


class LoginRequest(BaseModel):
    password: str


class PatchRequest(BaseModel):
    values: dict[str, Any]


def _secure(request: Request) -> bool:
    origin = request.headers.get("origin") or ""
    return request.url.scheme == "https" or origin.startswith("https://")


def create_admin_router(auth: AdminAuth, origin_ok: Callable[[str | None], bool]) -> APIRouter:
    def check_origin(request: Request) -> None:
        if not origin_ok(request.headers.get("origin")):
            raise HTTPException(403, "origin not allowed")

    def require_admin(request: Request) -> None:
        check_origin(request)
        if not auth.valid(request.cookies.get(COOKIE)):
            raise HTTPException(401, "login required")

    public = APIRouter(prefix="/api/admin", dependencies=[Depends(check_origin)])
    router = APIRouter(prefix="/api/admin", dependencies=[Depends(require_admin)])

    @public.get("/session")
    async def session(request: Request) -> dict[str, Any]:
        return {"logged_in": auth.valid(request.cookies.get(COOKIE)), "configured": auth.configured}

    @public.post("/login")
    async def login(req: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
        if not auth.configured:
            raise HTTPException(503, "no admin password set; run scripts/admin_password.py on the server")
        if auth.locked_out():
            raise HTTPException(429, "too many failed logins, try again in a few minutes")
        # scrypt takes ~50 ms; keep it off the event loop.
        token = await asyncio.to_thread(auth.login, req.password)
        if token is None:
            log.warning("admin login failed from %s", request.client.host if request.client else "?")
            raise HTTPException(401, "wrong password")
        response.set_cookie(
            COOKIE, token, max_age=int(auth.session_s), httponly=True, samesite="strict",
            secure=_secure(request), path="/",
        )
        return {"ok": True}

    @public.post("/logout")
    async def logout(response: Response) -> dict[str, Any]:
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @router.get("/config")
    async def get_config(request: Request) -> dict[str, Any]:
        snap = await asyncio.to_thread(settings_store.snapshot, request.app.state.services.cfg)
        snap["restart_pending"] = any(f["restart_pending"] for f in snap["fields"])
        return snap

    @router.put("/config")
    async def put_config(req: PatchRequest) -> dict[str, Any]:
        try:
            changed = await asyncio.to_thread(settings_store.apply_patch, req.values)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        log.info("admin changed settings: %s", ", ".join(changed))
        return {"changed": changed}

    @router.get("/household")
    async def get_household(request: Request) -> dict[str, Any]:
        services = request.app.state.services
        store = services.household
        if store is None:
            raise HTTPException(503, "household is not loaded")
        gateway = services.cfg.hermes.gateway
        home = Path(gateway.hermes_home)
        ids = sorted({*list_profiles(home), *(p.id for p in store.data.profiles)})
        # Only whether each profile has a key; the keys themselves never leave the server.
        keys = {pid: read_env_key(profile_env_path(home, pid), gateway.key_var) is not None for pid in ids}
        sessions = [
            {"id": s.id, "device_id": s.device_id, "person": s.identity.person.id if s.identity.person else None,
             "source": s.identity.source}
            for s in list(services.sessions)
        ]
        return {
            "household": store.data.model_dump(),
            "hermes_profiles": ids,
            "profile_keys": keys,
            "gateway": {"enabled": gateway.enabled, "url": gateway.url, "key_var": gateway.key_var},
            "sessions": sessions,
        }

    @router.put("/household")
    async def put_household(body: dict[str, Any], request: Request) -> dict[str, Any]:
        store = request.app.state.services.household
        if store is None:
            raise HTTPException(503, "household is not loaded")
        try:
            data = await asyncio.to_thread(store.replace, Household.model_validate(body))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        log.info("admin changed the household (%d people, %d profiles)", len(data.people), len(data.profiles))
        return {"household": data.model_dump()}

    @router.get("/status")
    async def status() -> dict[str, Any]:
        return {"started_at": STARTED_AT, "systemd": _under_systemd()}

    @router.post("/restart")
    async def restart() -> dict[str, Any]:
        if not _under_systemd():
            raise HTTPException(409, f"the adapter is not running as {UNIT}; restart it by hand")
        _queue_restart("admin panel")
        return {"ok": True}

    combined = APIRouter()
    combined.include_router(public)
    combined.include_router(router)
    return combined


def _under_systemd() -> bool:
    return bool(os.environ.get("INVOCATION_ID"))


def _queue_restart(reason: str) -> None:
    """Ask systemd to restart this unit (it outlives us); refused while a benchmark holds the lock."""
    with open(BENCH_LOCK, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HTTPException(409, "a benchmark holds the restart lock; try again later") from None
        with open(RESTART_LOG, "a") as fh:
            fh.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} restarting: {reason}\n")
        subprocess.run(["systemctl", "--user", "--no-block", "restart", UNIT], check=True, timeout=10)

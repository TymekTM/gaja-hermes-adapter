"""Device take recording API (protocol v1.10): start/stop/status of recordings through a device mic.

Every endpoint needs a valid admin cookie and an allowed Origin, exactly like the admin panel API:
this can turn on a microphone in someone's home, so it is never anonymous. The device side of the
protocol lives in `gaja/devices/recording.py`; finished takes are stored by the same `SampleStore`
as browser uploads, so undo/progress/export work unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from gaja.admin.auth import COOKIE, AdminAuth
from gaja.devices import recording as recording_mod
from gaja.devices.recording import DeviceRecording
from gaja.pipeline import Session
from gaja.wakeword_samples import SampleStore, _label, _name

if TYPE_CHECKING:
    from gaja.devices.controller import DeviceController

log = logging.getLogger(__name__)

# The last take per device, so a status/stop poll still answers after the device disconnected
# (the take was then finalised as partial by the connection's cleanup).
_LAST: dict[str, DeviceRecording] = {}

_ERROR_CODES = {"busy": 409, "unsupported": 409, "mic": 503, "not_started": 504}


class StartRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    session: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=32)
    prompt_id: str = Field(min_length=1, max_length=64)
    max_ms: int | None = Field(default=None, ge=1000, le=120000)


def _device_session(request: Request, device_id: str) -> Session:
    services = getattr(request.app.state, "services", None)
    for session in list(getattr(services, "sessions", ()) or ()):
        if session.is_device and session.device_id == device_id:
            return session
    raise HTTPException(404, f"device {device_id!r} is not connected")


def _gone(device_id: str) -> HTTPException:
    return HTTPException(503, f"device {device_id!r} went away")


def _refuse_unless_idle(controller: DeviceController, device_id: str) -> None:
    if not controller.caps.record:
        raise HTTPException(409, f"device {device_id!r} does not support recording (caps.record)")
    if controller.recording is not None and controller.recording.active:
        raise HTTPException(409, f"a recording is already in progress on {device_id!r}")
    if controller.busy:
        # Refused without contacting the device: it must be idle (listening, no turn in progress).
        raise HTTPException(409, f"device {device_id!r} is busy (a turn is in progress)")


def _find_recording(request: Request, device_id: str) -> tuple[Session | None, DeviceRecording | None]:
    """The last recording on the device's connection (live or finished), else the one before a reconnect."""
    try:
        session = _device_session(request, device_id)
    except HTTPException:
        session = None
    recording = session._device.recording if session is not None and session._device is not None else None
    if recording is None:
        recording = _LAST.get(device_id)
    return session, recording


def create_device_record_router(auth: AdminAuth, store: SampleStore, origin_ok: Callable[[str | None], bool]) -> APIRouter:
    def require_admin(request: Request) -> None:
        if not origin_ok(request.headers.get("origin")):
            raise HTTPException(403, "origin not allowed")
        if not auth.valid(request.cookies.get(COOKIE)):
            raise HTTPException(401, "login required")

    router = APIRouter(prefix="/api/admin/record", dependencies=[Depends(require_admin)])

    @router.get("/devices")
    async def record_devices(request: Request) -> dict[str, Any]:
        services = getattr(request.app.state, "services", None)
        out: list[dict[str, Any]] = []
        for session in sorted(
            (s for s in list(getattr(services, "sessions", ()) or ()) if s.is_device),
            key=lambda s: s.device_id or "",
        ):
            controller = session._device
            if controller is None:
                continue
            rec = controller.recording
            out.append(
                {
                    "id": controller.info.id,
                    "name": controller.info.name or controller.info.id,
                    "fw": controller.fw,
                    "caps_record": bool(controller.caps.record),
                    "busy": controller.busy,
                    "recording": rec is not None and rec.active,
                    "take": rec.status() if rec is not None and rec.active else None,
                }
            )
        return {"devices": out}

    @router.post("/takes")
    async def start_take(body: StartRequest, request: Request) -> dict[str, Any]:
        _name(body.session, "session"), _label(body.label), _name(body.prompt_id, "prompt_id")
        session = _device_session(request, body.device_id)
        controller = session._device
        assert controller is not None
        _refuse_unless_idle(controller, body.device_id)
        try:
            await controller.prepare_for_recording()  # close an open follow-up mic window: back to idle
        except ConnectionError:
            raise _gone(body.device_id) from None
        # prepare_for_recording() yielded to the event loop: a second request or a wake may have got in.
        # Nothing below awaits before the recording is registered in begin(), so this check is final.
        _refuse_unless_idle(controller, body.device_id)
        take = store.next_take(body.session, body.label, body.prompt_id)
        try:
            recording = await recording_mod.begin(
                controller, store=store, session=body.session, label=body.label,
                prompt_id=body.prompt_id, take=take, max_ms=body.max_ms or recording_mod.DEFAULT_MAX_MS,
            )
        except ConnectionError:
            raise _gone(body.device_id) from None
        _LAST[body.device_id] = recording
        if recording.state == "error":
            if recording.error == "not_started":
                raise HTTPException(504, f"device {body.device_id!r} did not start the take (no record_started in time)")
            raise HTTPException(_ERROR_CODES.get(recording.error, 502), f"device refused: {recording.error}")
        return {"ok": True, **recording.status()}

    @router.get("/takes/{device_id}")
    async def take_status(device_id: str, request: Request) -> dict[str, Any]:
        _, recording = _find_recording(request, device_id)
        return {"device_id": device_id, "take": recording.status() if recording is not None else None}

    @router.post("/takes/{device_id}/stop")
    async def stop_take(device_id: str, request: Request) -> dict[str, Any]:
        session, recording = _find_recording(request, device_id)
        if recording is None:
            raise HTTPException(404, f"no recording on device {device_id!r}")
        if recording.active and session is not None and session._device is not None:
            try:
                await session._device.record_stop()
            except ConnectionError:
                raise _gone(device_id) from None
            try:
                await asyncio.wait_for(recording.ended.wait(), recording_mod.STOP_ACK_S)
            except TimeoutError:
                log.warning("device %s did not answer record_stop; keeping the take as is", device_id)
        return {"ok": True, **recording.status()}

    return router

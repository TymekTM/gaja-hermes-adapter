"""Device list HTTP API: GET /api/devices.

Returns configured and connected voice devices with status, last seen, and last wake times.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

log = logging.getLogger(__name__)

# In-memory tracking for device events: device_id -> {"last_seen": ts, "last_wake": ts}
_DEVICE_TIMESTAMPS: dict[str, dict[str, float]] = {}


def record_device_activity(device_id: str, is_wake: bool = False) -> None:
    now = time.time()
    entry = _DEVICE_TIMESTAMPS.setdefault(device_id, {"last_seen": now, "last_wake": 0.0})
    entry["last_seen"] = now
    if is_wake:
        entry["last_wake"] = now


def create_devices_router(
    origin_ok: Callable[[str | None], bool],
) -> APIRouter:
    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not origin_ok(origin):
            log.warning("rejected devices request from origin %r", origin)
            raise HTTPException(403, "origin not allowed")

    router = APIRouter(prefix="/api/devices", dependencies=[Depends(check_origin)])

    @router.get("")
    async def list_devices(request: Request) -> dict[str, Any]:
        app = request.app
        listener = getattr(app.state, "device_listener", None)
        connected_ids = set(listener.connected) if listener else set()

        registry = getattr(listener, "registry", None)
        if registry is None:
            # Check if services config has devices enabled and tokens_file
            services = getattr(app.state, "services", None)
            if services is not None and getattr(services.cfg, "devices", None):
                from gaja.devices.registry import DeviceRegistry

                registry = DeviceRegistry(services.cfg.devices.tokens_file)

        devices: list[dict[str, Any]] = []
        if registry is not None:
            registry._reload()
            for dev_id, (_tok, info) in sorted(registry._entries.items()):
                online = dev_id in connected_ids
                activity = _DEVICE_TIMESTAMPS.get(dev_id, {})
                last_seen = activity.get("last_seen")
                if online and not last_seen:
                    last_seen = time.time()
                devices.append({
                    "id": info.id,
                    "name": info.name or info.id,
                    "online": online,
                    "last_seen": last_seen,
                    "last_wake": activity.get("last_wake"),
                })

        return {"devices": devices}

    return router

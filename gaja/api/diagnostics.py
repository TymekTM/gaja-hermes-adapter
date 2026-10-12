"""Diagnostics API: GET /api/diagnostics.

Returns GPU status (cached nvidia-smi query via subprocess), per-turn stage latencies,
and recent device wake verdicts.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

log = logging.getLogger(__name__)

_GPU_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}
_GPU_CACHE_TTL_S = 4.0


def _run_nvidia_smi() -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            parts = [p.strip() for p in proc.stdout.decode("utf-8", errors="replace").strip().split(",")]
            if len(parts) >= 7:
                return {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mb": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_free_mb": float(parts[4]),
                    "utilization_pct": float(parts[5]),
                    "temperature_c": float(parts[6]),
                }
    except Exception as exc:
        log.debug("nvidia-smi query unavailable: %s", exc)
    return None


async def _query_gpu() -> dict[str, Any] | None:
    now = time.monotonic()
    if now - _GPU_CACHE["ts"] < _GPU_CACHE_TTL_S:
        return _GPU_CACHE["data"]

    data = await asyncio.to_thread(_run_nvidia_smi)
    _GPU_CACHE["ts"] = now
    _GPU_CACHE["data"] = data
    return data


def create_diagnostics_router(
    origin_ok: Callable[[str | None], bool],
) -> APIRouter:
    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not origin_ok(origin):
            log.warning("rejected diagnostics request from origin %r", origin)
            raise HTTPException(403, "origin not allowed")

    router = APIRouter(prefix="/api/diagnostics", dependencies=[Depends(check_origin)])

    @router.get("")
    async def get_diagnostics(request: Request) -> dict[str, Any]:
        gpu = await _query_gpu()

        services = getattr(request.app.state, "services", None)
        stages: dict[str, Any] = {}
        if services is not None and getattr(services, "metrics", None) is not None:
            stages = services.metrics.summary()

        wakes: list[dict[str, Any]] = []
        try:
            from gaja.devices.controller import get_wake_log

            wakes = get_wake_log()
        except ImportError:
            wakes = []

        return {
            "gpu": gpu,
            "stages": stages,
            "wakes": wakes,
        }

    return router

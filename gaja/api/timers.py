from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request

from gaja.timers import TimerManager, get_timer_manager


def create_timers_router(
    manager: TimerManager | Callable[[], TimerManager],
    is_origin_allowed: Callable[[str | None], bool],
) -> APIRouter:
    router = APIRouter(prefix="/api/timers", tags=["timers"])

    def _get_manager() -> TimerManager:
        mgr = manager() if callable(manager) else manager
        return mgr if mgr is not None else get_timer_manager()

    def _check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not is_origin_allowed(origin):
            raise HTTPException(status_code=403, detail="Origin not allowed")

    @router.get("")
    async def list_timers(request: Request) -> dict[str, Any]:
        _check_origin(request)
        return {"timers": _get_manager().list_timers()}

    @router.delete("/{timer_id}")
    async def delete_timer(timer_id: str, request: Request) -> dict[str, Any]:
        _check_origin(request)
        _get_manager().delete_timer(timer_id)
        return {"ok": True}

    return router

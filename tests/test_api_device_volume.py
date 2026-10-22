"""GET/PUT /api/devices/{id}/volume: per-device speaker level over HTTP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from gaja.api.devices import create_devices_router
from gaja.devices.registry import DeviceInfo, append_device
from gaja.devices.volume import DEFAULT_LEVEL, STEP_FLOOR, VolumeStore, set_volume_store
from gaja.pipeline import Session
from tests.test_device_protocol import FAST, HELLO, TOKEN
from tests.test_device_volume_gain import RecordingOutbox


class VolumeListener:
    def __init__(self, registry: Any) -> None:
        self.registry = registry


def _make_app(
    tmp_path: Path,
    services: Any = None,
    listener: Any = None,
    store: VolumeStore | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(create_devices_router(lambda origin: origin is None))
    if services is not None:
        app.state.services = services
    app.state.device_listener = listener
    if store is not None:
        set_volume_store(store)
    return app


def _write_tokens(tmp_path: Path) -> Path:
    tokens = tmp_path / "devices.toml"
    append_device(tokens, "atom-pokoj", TOKEN, name="Pokój")
    return tokens


# ---- defaults --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_volume_defaults_to_100_for_an_untouched_device(tmp_path: Path) -> None:
    store = VolumeStore(tmp_path / "volumes.json")
    app = _make_app(tmp_path, store=store)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
        res = await ac.get("/api/devices/atom-pokoj/volume")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["level"] == DEFAULT_LEVEL
    assert body["gain"] == 1.0
    assert body["muted"] is False
    assert body["step"] and body["floor"] == STEP_FLOOR


# ---- PUT: validation, persistence, live sessions ---------------------------------------------------


@pytest.mark.asyncio
async def test_put_level_persists_and_updates_the_live_session(make_services, tmp_path: Path) -> None:
    store = VolumeStore(tmp_path / "volumes.json")
    services = make_services(cfg={"devices": {**FAST}})
    services.volume_store = store
    session = Session(services, RecordingOutbox(), device=DeviceInfo("atom-pokoj"))

    app = _make_app(tmp_path, services=services, store=store)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
        res = await ac.put("/api/devices/atom-pokoj/volume", json={"level": 40})

    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True and body["level"] == 40 and body["muted"] is False
    assert 0.0 < body["gain"] < 0.3
    # The live device session plays everything after this at the new gain.
    assert session._out.gain == pytest.approx(body["gain"])
    # And a fresh store reads the same level back from disk.
    assert VolumeStore(tmp_path / "volumes.json").level("atom-pokoj") == 40
    await session.close()


@pytest.mark.asyncio
async def test_put_delta_is_relative_and_respects_the_floor(tmp_path: Path) -> None:
    store = VolumeStore(tmp_path / "volumes.json")
    app = _make_app(tmp_path, store=store)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
        res = await ac.put("/api/devices/atom-pokoj/volume", json={"delta": -12})
        assert res.status_code == 200
        assert res.json()["level"] == DEFAULT_LEVEL - 12
        # Repeated "ciszej" can never go under the floor ...
        for _ in range(20):
            res = await ac.put("/api/devices/atom-pokoj/volume", json={"delta": -12})
        assert res.json()["level"] == STEP_FLOOR
        assert res.json()["muted"] is False
        # ... but an explicit level 0 (wycisz) may.
        res = await ac.put("/api/devices/atom-pokoj/volume", json={"level": 0})
        assert res.status_code == 200
        assert res.json()["muted"] is True and res.json()["gain"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},  # neither level nor delta
        {"level": 40, "delta": 12},  # both
        {"level": 101},  # out of range
        {"level": -1},
        {"delta": 1000},
        {"level": "loud"},
    ],
)
async def test_put_rejects_invalid_bodies_with_4xx(tmp_path: Path, body: dict[str, Any]) -> None:
    app = _make_app(tmp_path, store=VolumeStore(tmp_path / "volumes.json"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
        res = await ac.put("/api/devices/atom-pokoj/volume", json=body)
    assert 400 <= res.status_code < 500


@pytest.mark.asyncio
async def test_volume_endpoints_404_for_a_device_unknown_to_the_registry(make_services, tmp_path: Path) -> None:
    tokens = _write_tokens(tmp_path)
    services = make_services(cfg={"devices": {**FAST, "tokens_file": tokens}})
    services.volume_store = VolumeStore(tmp_path / "volumes.json")
    app = _make_app(tmp_path, services=services)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
        assert (await ac.get("/api/devices/atom-nieznany/volume")).status_code == 404
        assert (await ac.put("/api/devices/atom-nieznany/volume", json={"level": 30})).status_code == 404
        # The known one is fine (offline devices keep their persisted level).
        res = await ac.put("/api/devices/atom-pokoj/volume", json={"level": 30})
        assert res.status_code == 200


@pytest.mark.asyncio
async def test_volume_endpoints_inherit_the_origin_rule(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(create_devices_router(lambda origin: origin is None))
    app.state.services = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
        res = await ac.get(
            "/api/devices/atom-pokoj/volume", headers={"Origin": "https://evil.example"}
        )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_put_without_services_uses_the_module_store(tmp_path: Path) -> None:
    store = VolumeStore(tmp_path / "volumes.json")
    set_volume_store(store)
    app = _make_app(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
            res = await ac.put("/api/devices/atom-pokoj/volume", json={"level": 55})
        assert res.status_code == 200
        assert store.level("atom-pokoj") == 55
    finally:
        set_volume_store(None)

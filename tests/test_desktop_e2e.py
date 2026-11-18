"""End-to-end, in-process (protocol v1.14, docs/DESKTOP.md): the REAL device listener, PC relay
and /api/pc HTTP API serve the REAL gaja_desktop client code (desktop/, imported via sys.path).

Everything real except the hardware: capture/playback/wake are fakes (headless), the tray is not
started, and the screenshot tool's mss grab is faked (the JPEG encoding is the real one). No
dependencies are added to the root project — the test skips when the client's libraries
(websockets, numpy, platformdirs, pillow) are absent from this environment.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gaja.devices.listener import DeviceListener
from gaja.devices.registry import DeviceRegistry
from tests.conftest import FakeSTT
from tests.test_device_protocol import FAST, TOKEN

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = ROOT / "desktop"


# ---- the real desktop client, imported from desktop/ (skipped when its deps are missing) ----


def _desktop_modules() -> Any:
    for dep in ("websockets", "numpy", "platformdirs", "PIL"):
        pytest.importorskip(dep, reason=f"gaja_desktop dependency {dep!r} not installed in the root venv")
    if not (DESKTOP / "gaja_desktop" / "connection.py").is_file():
        pytest.skip("desktop/ client not checked out")
    sys.path.insert(0, str(DESKTOP))
    try:
        connection = importlib.import_module("gaja_desktop.connection")
        screenshot = importlib.import_module("gaja_desktop.tools.screenshot")
        shell = importlib.import_module("gaja_desktop.tools.shell")
        config = importlib.import_module("gaja_desktop.config")
    finally:
        sys.path.remove(str(DESKTOP))
    return type("Desktop", (), {"connection": connection, "screenshot": screenshot,
                                "shell": shell, "config": config})


class FakeCapture:
    """No microphone: no frames, ever (the wake word and the mic stream stay silent)."""

    def __init__(self, spec: str, callback: Any) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakePlayback:
    """No speakers: recorded reply audio, `busy()` always False (drained at once)."""

    def __init__(self, specs: list[str], rate: int) -> None:
        self.rate = rate
        self.written: list[bytes] = []

    def start(self) -> None:
        pass

    def play(self, pcm: bytes) -> None:
        self.written.append(pcm)

    def busy(self) -> bool:
        return False

    def cancel(self) -> None:
        self.written.clear()

    def close(self) -> None:
        pass


class NeverWake:
    """A wake-word detector that never fires (PTT is the mic path in these tests)."""

    score = 0.0

    def process(self, frame: bytes) -> bool:
        return False

    def reset(self) -> None:
        pass


class CountingSTT(FakeSTT):
    """Counts every transcribe call: a PTT wake must trigger none (no wake verification)."""

    def __init__(self) -> None:
        super().__init__("hej gaja")
        self.calls = 0

    async def transcribe(self, pcm, language=None, **kwargs):  # type: ignore[override]
        self.calls += 1
        return await super().transcribe(pcm, language, **kwargs)


# ---- shared plumbing (mirrors tests/test_pc_relay.py) --------------------------------------


def _make_listener(make_services, tmp_path: Path) -> tuple[DeviceListener, Any]:
    registry_path = tmp_path / "devices.toml"
    registry_path.write_text(f'[devices."pc-tymek"]\ntoken = "{TOKEN}"\n', encoding="utf-8")
    registry_path.chmod(0o600)
    services = make_services(cfg={"devices": {**FAST}, "pc": {}}, stt=CountingSTT())  # verify_wake stays True
    return DeviceListener(services, DeviceRegistry(registry_path), port=0), services


def _make_http_app(services: Any) -> Any:
    from fastapi import FastAPI

    from gaja.api.pc import create_pc_router

    app = FastAPI()
    app.include_router(create_pc_router())
    app.state.services = services
    return app


def _auth_headers(services: Any) -> dict[str, str]:
    from gaja.api.pc import load_or_create_api_token

    token = load_or_create_api_token(services.cfg.pc.api_token_file)
    return {"Authorization": f"Bearer {token}"}


def _make_client(desktop: Any, url: str) -> Any:
    cfg = desktop.config.DesktopConfig(url=url, device_id="pc-tymek", token=TOKEN, lang="pl")
    return desktop.connection.DeviceClient(
        cfg,
        tools=[desktop.shell.ShellTool(), desktop.screenshot.ScreenshotTool()],
        capture_factory=lambda spec, cb: FakeCapture(spec, cb),
        playback_factory=lambda specs, rate: FakePlayback(specs, rate),
        wake_factory=lambda: NeverWake(),
    )


async def _run_until_ready(client: Any, timeout: float = 5.0) -> asyncio.Task:
    task = asyncio.create_task(client.run())
    await asyncio.wait_for(client._ready_event.wait(), timeout)
    assert client.state == "idle"
    assert client.saw_pc is True  # ready.device.pc: the server accepted caps.pc
    return task


# ---- tests ----------------------------------------------------------------------------------


def test_real_client_shell_screenshot_and_unadvertised_over_the_api(
    make_services, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desktop = _desktop_modules()
    listener, services = _make_listener(make_services, tmp_path)
    import httpx

    async def scenario() -> None:
        from PIL import Image

        assert await listener.start()
        try:
            app = _make_http_app(services)
            client = _make_client(desktop, f"ws://127.0.0.1:{listener.port}/device")
            # The screenshot backend is faked; the encoding and the wire path are real.
            monitors = [{"index": 0, "left": -1920, "top": 0, "width": 3840, "height": 1080},
                        {"index": 1, "left": -1920, "top": 0, "width": 1920, "height": 1080},
                        {"index": 2, "left": 0, "top": 0, "width": 1920, "height": 1080}]

            def fake_grab(monitor: int) -> tuple[Any, list[dict[str, Any]]]:
                return Image.new("RGB", (monitors[monitor]["width"], monitors[monitor]["height"])), monitors

            monkeypatch.setattr(desktop.screenshot, "grab_monitor", fake_grab)
            task = await _run_until_ready(client)
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                             base_url="http://testserver") as http:
                    # The listener's registry entry is live behind the real API.
                    listed = await http.get("/api/pc", headers=_auth_headers(services))
                    assert listed.status_code == 200
                    devices = listed.json()["devices"]
                    assert [d["id"] for d in devices] == ["pc-tymek"]
                    assert set(devices[0]["tools"]) >= {"shell", "screenshot"}

                    # shell: a real `sh` subprocess on the client, answer relayed back.
                    shell = await http.post("/api/pc/pc-tymek/call", headers=_auth_headers(services),
                                            json={"tool": "shell",
                                                  "args": {"command": "echo zażółć", "shell": "sh"},
                                                  "timeout_s": 15})
                    assert shell.status_code == 200, shell.text
                    body = shell.json()
                    assert body["ok"] is True
                    assert "zażółć" in body["text"]
                    assert body["data"]["exit_code"] == 0

                    # screenshot: faked mss grab, real JPEG encode, image survives the relay.
                    shot = await http.post("/api/pc/pc-tymek/call", headers=_auth_headers(services),
                                           json={"tool": "screenshot", "timeout_s": 15})
                    assert shot.status_code == 200, shot.text
                    shot_body = shot.json()
                    assert shot_body["ok"] is True
                    image = shot_body["image"]
                    assert image["mime"] == "image/jpeg" and image["width"] == 1600
                    assert base64.b64decode(image["b64"])[:3] == b"\xff\xd8\xff"  # JPEG magic
                    assert "img_x" in shot_body["text"]  # the image->mouse mapping (docs/DESKTOP.md)
                    assert shot_body["data"]["monitors"][0]["left"] == -1920
                    assert len(json.dumps(shot_body).encode()) < 1 << 20  # under the listener's ws_max_size

                    # a tool the client never advertised: 400, before any pc_call is sent.
                    nope = await http.post("/api/pc/pc-tymek/call", headers=_auth_headers(services),
                                           json={"tool": "clipboard", "args": {"action": "get"}})
                    assert nope.status_code == 400
                    assert "does not advertise" in nope.json()["error"]
            finally:
                client.stop()
                await asyncio.wait_for(task, 5)
        finally:
            await listener.stop()

    asyncio.run(scenario())


def test_real_client_push_to_talk_is_accepted_without_verification(make_services, tmp_path) -> None:
    desktop = _desktop_modules()
    listener, services = _make_listener(make_services, tmp_path)

    async def scenario() -> None:
        assert await listener.start()
        try:
            client = _make_client(desktop, f"ws://127.0.0.1:{listener.port}/device")
            task = await _run_until_ready(client)
            seen: list[str] = []
            real_on_event = client._on_event

            def recording_on_event(evt: dict[str, Any]) -> None:
                seen.append(str(evt.get("type")))
                real_on_event(evt)

            client._on_event = recording_on_event
            try:
                await asyncio.wait_for(client.push_to_talk(), 5)  # the PTT hotkey path, verbatim
                for _ in range(100):
                    if "wake_accepted" in seen:
                        break
                    await asyncio.sleep(0.02)
                assert "wake_accepted" in seen
                assert "wake_rejected" not in seen
                assert client.state == "listening"
                assert services.stt.calls == 0  # the press is the intent: no STT verification ran
            finally:
                client.stop()
                await asyncio.wait_for(task, 5)
        finally:
            await listener.stop()

    asyncio.run(scenario())

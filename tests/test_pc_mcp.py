"""gaja.mcp_pc tools against a mocked /api/pc HTTP API (pattern of tests/test_speaker_mcp.py)."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
from typing import Any

import httpx
import pytest

from gaja import mcp_pc
from gaja.devices.audio import PcCaps

# A valid 1x1 PNG.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8//8/AwAF/gL+py2ltQAAAABJRU5ErkJggg=="
)


def make_transport(calls: list[tuple[str, str, dict[str, Any] | None]], replies: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        assert request.headers.get("authorization") == "Bearer test-token", "the bearer token must be sent"
        assert request.headers.get("origin") is None
        for path, reply in replies.items():
            if request.url.path == path:
                status = reply.get("_status", 200)
                payload = {k: v for k, v in reply.items() if not k.startswith("_")}
                return httpx.Response(status, json=payload)
        return httpx.Response(404, json={"ok": False, "error": "no such route"})

    return httpx.MockTransport(handler)


@pytest.fixture
def mock_api(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a MockTransport for every mcp_pc httpx call.

    Usage: ``calls = mock_api({"/api/pc": {...}, "/api/pc/pc-a/call": {...}})`` — returns the
    recorded (method, path, body) triples. A reply key ``_status`` overrides the HTTP status.
    """

    def install(replies: dict[str, Any]) -> list[tuple[str, str, dict[str, Any] | None]]:
        calls: list[tuple[str, str, dict[str, Any] | None]] = []
        monkeypatch.setattr(mcp_pc, "_load_token", lambda: "test-token")
        real_async_client = httpx.AsyncClient
        transport = make_transport(calls, replies)

        def transport_factory(**kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return real_async_client(**kwargs)

        monkeypatch.setattr(mcp_pc.httpx, "AsyncClient", transport_factory)
        return calls

    return install


def _devices_reply(*ids: str) -> dict[str, Any]:
    return {
        "/api/pc": {
            "devices": [
                {"id": i, "host": "H", "os": "windows", "tools": ["shell", "screenshot", "clipboard"], "connected_s": 1.0}
                for i in ids
            ]
        }
    }


# ---- pc_list ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pc_list(mock_api) -> None:
    calls = mock_api(_devices_reply("pc-a"))
    res = await mcp_pc.pc_list()
    assert res["devices"][0]["id"] == "pc-a"
    assert calls == [("GET", "/api/pc", None)]


# ---- auto-pick of the single connected PC -----------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_pick_uses_the_only_connected_pc(mock_api) -> None:
    calls = mock_api({**_devices_reply("pc-tymek"), "/api/pc/pc-tymek/call": {"ok": True, "text": "hi"}})
    res = await mcp_pc.pc_shell("echo hi")
    assert res["ok"] is True and res["text"] == "hi"
    assert calls[0][1] == "/api/pc"  # the device list for the auto-pick
    assert calls[1][1] == "/api/pc/pc-tymek/call"
    body = calls[1][2]
    assert body["tool"] == "shell" and body["args"]["command"] == "echo hi" and body["timeout_s"] == 60


@pytest.mark.asyncio
async def test_auto_pick_with_no_pc_connected_lists_an_error(mock_api) -> None:
    calls = mock_api(_devices_reply())
    res = await mcp_pc.pc_shell("echo hi")
    assert res["ok"] is False and "no PC is connected" in res["error"]
    assert calls[-1][1] == "/api/pc"  # it asked for the device list, nothing else


@pytest.mark.asyncio
async def test_auto_pick_with_two_pcs_asks_for_an_explicit_device_id(mock_api) -> None:
    mock_api(_devices_reply("pc-a", "pc-b"))
    res = await mcp_pc.pc_shell("echo hi")
    assert res["ok"] is False
    assert "more than one PC" in res["error"] and "pc-a" in res["error"] and "pc-b" in res["error"]


# ---- the tools -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pc_shell_sends_optional_args(mock_api) -> None:
    calls = mock_api({**_devices_reply("pc-a"), "/api/pc/pc-a/call": {"ok": True, "text": "out"}})
    res = await mcp_pc.pc_shell("dir", shell="cmd", cwd="C:\\", timeout_s=15, device_id="pc-a")
    assert res["ok"] is True
    body = calls[-1][2]
    assert body["tool"] == "shell"
    assert body["args"] == {"command": "dir", "shell": "cmd", "cwd": "C:\\"}
    assert body["timeout_s"] == 15


@pytest.mark.asyncio
async def test_pc_open_type_keys_mouse_clipboard_build_the_right_args(mock_api) -> None:
    calls = mock_api({**_devices_reply("pc-a"), "/api/pc/pc-a/call": {"ok": True, "text": "done"}})
    assert (await mcp_pc.pc_open("https://example.com", device_id="pc-a"))["ok"] is True
    assert (await mcp_pc.pc_type("hello world", device_id="pc-a"))["ok"] is True
    assert (await mcp_pc.pc_keys(combo="win+r", device_id="pc-a"))["ok"] is True
    assert (await mcp_pc.pc_keys(sequence=["ctrl+a", "delete"], device_id="pc-a"))["ok"] is True
    assert (await mcp_pc.pc_mouse("click", x=10, y=20, device_id="pc-a"))["ok"] is True
    assert (await mcp_pc.pc_clipboard("get", device_id="pc-a"))["ok"] is True
    assert (await mcp_pc.pc_clipboard("set", text="nowy", device_id="pc-a"))["ok"] is True
    sent = [body for _, path, body in calls if path.endswith("/call")]
    assert [b["tool"] for b in sent] == ["open", "type", "keys", "keys", "mouse", "clipboard", "clipboard"]
    assert sent[0]["args"] == {"target": "https://example.com"}
    assert sent[1]["args"] == {"text": "hello world"}
    assert sent[2]["args"] == {"combo": "win+r"}
    assert sent[3]["args"] == {"sequence": ["ctrl+a", "delete"]}
    assert sent[4]["args"] == {"action": "click", "x": 10, "y": 20}
    assert sent[5]["args"] == {"action": "get"}
    assert sent[6]["args"] == {"action": "set", "text": "nowy"}


@pytest.mark.asyncio
async def test_pc_screenshot_returns_an_image_plus_text(mock_api) -> None:
    from mcp.server.fastmcp import Image

    calls = mock_api(
        {
            **_devices_reply("pc-a"),
            "/api/pc/pc-a/call": {
                "ok": True,
                "text": "1920x1080, monitor 0",
                "image": {"mime": "image/png", "b64": base64.b64encode(TINY_PNG).decode(), "width": 1920, "height": 1080},
            },
        }
    )
    res = await mcp_pc.pc_screenshot(monitor=0, device_id="pc-a")
    assert isinstance(res, list) and len(res) == 2
    image, text = res
    assert isinstance(image, Image)
    assert image.data == TINY_PNG
    assert "1920x1080" in str(text)
    assert calls[-1][2]["args"] == {"monitor": 0, "max_width": 1600}


@pytest.mark.asyncio
async def test_pc_screenshot_without_an_image_is_an_error(mock_api) -> None:
    mock_api({**_devices_reply("pc-a"), "/api/pc/pc-a/call": {"ok": True, "text": "nope"}})
    res = await mcp_pc.pc_screenshot(device_id="pc-a")
    assert isinstance(res, dict) and res["ok"] is False


# ---- errors ----------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_error_payloads_are_returned_as_errors(mock_api) -> None:
    mock_api({**_devices_reply("pc-a"), "/api/pc/pc-a/call": {"_status": 504, "ok": False, "error": "timeout"}})
    res = await mcp_pc.pc_shell("sleep 5", device_id="pc-a")
    assert res == {"ok": False, "status": 504, "error": "timeout"}


@pytest.mark.asyncio
async def test_unauthorized_gets_a_hint_not_the_raw_detail(mock_api) -> None:
    mock_api({"/api/pc": {"_status": 401, "detail": "unauthorized"}})
    res = await mcp_pc.pc_list()
    assert res["ok"] is False and "token" in res["error"]


@pytest.mark.asyncio
async def test_adapter_unreachable_is_an_error_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExplodingClient:
        def __init__(self, **kwargs: Any) -> None:
            raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(mcp_pc, "_load_token", lambda: "test-token")
    monkeypatch.setattr(mcp_pc.httpx, "AsyncClient", ExplodingClient)
    res = await mcp_pc.pc_list()
    assert res["ok"] is False and "adapter request failed" in res["error"]


@pytest.mark.asyncio
async def test_input_validation_short_circuits_without_http(mock_api) -> None:
    calls = mock_api(_devices_reply("pc-a"))
    assert (await mcp_pc.pc_shell("", device_id="pc-a"))["ok"] is False
    assert (await mcp_pc.pc_shell("x", timeout_s=0, device_id="pc-a"))["ok"] is False
    assert (await mcp_pc.pc_shell("x", timeout_s=301, device_id="pc-a"))["ok"] is False
    assert (await mcp_pc.pc_shell("x", device_id="bad id!"))["ok"] is False
    assert (await mcp_pc.pc_type("", device_id="pc-a"))["ok"] is False
    assert (await mcp_pc.pc_type("x" * 10001, device_id="pc-a"))["ok"] is False
    assert (await mcp_pc.pc_keys(combo="", sequence=None, device_id="pc-a"))["ok"] is False
    assert (await mcp_pc.pc_keys(combo="a", sequence=["b"], device_id="pc-a"))["ok"] is False
    assert (await mcp_pc.pc_mouse("drag", device_id="pc-a"))["ok"] is False
    assert (await mcp_pc.pc_clipboard("clear", device_id="pc-a"))["ok"] is False
    assert (await mcp_pc.pc_clipboard("set", device_id="pc-a"))["ok"] is False
    assert calls == []  # no request left the process


# ---- budgets and registration ----------------------------------------------------------------------


def test_http_budget_is_timeout_plus_ten_capped_by_a_turn() -> None:
    assert mcp_pc._budget(60) == min(70.0, mcp_pc.turn_timeout_cap_s())
    assert mcp_pc._budget(10_000) == mcp_pc.turn_timeout_cap_s()


def test_every_tool_takes_an_optional_device_id() -> None:
    tools = [mcp_pc.pc_shell, mcp_pc.pc_open, mcp_pc.pc_screenshot, mcp_pc.pc_type,
             mcp_pc.pc_keys, mcp_pc.pc_mouse, mcp_pc.pc_clipboard]
    for tool in tools:
        param = inspect.signature(tool).parameters.get("device_id")
        assert param is not None and param.default == "", tool.__name__


def test_mcp_server_is_named_gaja_pc_and_registers_the_tools() -> None:
    tools = mcp_pc.mcp._tool_manager.list_tools()
    names = sorted(t.name for t in tools)
    assert names == [
        "pc_clipboard", "pc_keys", "pc_list", "pc_mouse", "pc_open",
        "pc_screenshot", "pc_shell", "pc_type",
    ]


def test_pccaps_tools_tuple_matches_the_protocol_surface() -> None:
    caps = PcCaps(os="windows", host="h", tools=["shell", "open", "screenshot", "type", "keys", "mouse", "clipboard"])
    assert set(caps.tools) == {"shell", "open", "screenshot", "type", "keys", "mouse", "clipboard"}

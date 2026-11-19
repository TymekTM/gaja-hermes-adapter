"""Stdio MCP server exposing PC tools (device protocol v1.14, docs/DESKTOP.md) to Hermes.

Runnable as::

    python -m gaja.mcp_pc

Talks to the local Gaja adapter HTTP API on loopback (default ``http://127.0.0.1:8790``,
``GAJA_API_URL``) with the ``/api/pc`` bearer token (``[pc] api_token_file``, created by the
adapter, never logged) and no ``Origin`` header. The PC executes every call without asking.

Per-call HTTP budget is ``timeout_s + 10 s`` capped below one Hermes turn (the same
``turn_timeout_cap_s`` cap as the speaker MCP): a tool call cannot outlive its turn.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Annotated, Any

import httpx
from mcp.server.fastmcp import FastMCP, Image
from pydantic import Field

from gaja.config import Settings
from gaja.mcp_speaker import get_api_url, turn_timeout_cap_s

# Slack over the protocol timeout_s the adapter already enforces (its own grace is 5 s).
HTTP_SLACK_S = 10.0
DEFAULT_TIMEOUT_S = 60.0
MAX_TIMEOUT_S = 300
# Listing PCs is an in-memory read on the adapter: a hung adapter must not eat a tool call's budget.
LIST_TIMEOUT_S = 5.0
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

mcp = FastMCP("gaja_pc")


def _api_token_file() -> Path:
    try:
        return Settings().pc.api_token_file.expanduser()
    except Exception:
        return Path("~/.config/gaja/pc_api.token").expanduser()


def _load_token() -> str:
    """The /api/pc bearer token (read-only here; the adapter creates the file on first start)."""
    try:
        return _api_token_file().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


async def _request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    token = _load_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(base_url=get_api_url(), timeout=timeout_s or DEFAULT_TIMEOUT_S, headers=headers) as http:
            resp = await http.request(method, path, json=json_body)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"adapter request failed: {exc}"}

    try:
        payload = resp.json()
    except Exception:
        payload = {"detail": resp.text}

    if resp.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
        error = payload.get("error") if isinstance(payload, dict) and payload.get("error") else detail
        if resp.status_code == 401:
            error = "adapter rejected the PC API token (check [pc] api_token_file and GAJA_API_URL)"
        return {"ok": False, "status": resp.status_code, "error": str(error or f"HTTP {resp.status_code}")}

    return payload if isinstance(payload, dict) else {"ok": True, "data": payload}


def _budget(timeout_s: float) -> float:
    return min(timeout_s + HTTP_SLACK_S, turn_timeout_cap_s())


async def _pick_device(explicit: str) -> tuple[str, dict[str, Any] | None]:
    """Resolve the target PC: the explicit id, or the only connected one.

    Returns ``(device_id, error)``; exactly one of them is None/empty.
    """
    dev = (explicit or "").strip()
    if dev:
        if not _DEVICE_ID_RE.match(dev):
            return "", {"ok": False, "error": "invalid device_id"}
        return dev, None
    res = await _request("GET", "/api/pc", timeout_s=min(LIST_TIMEOUT_S, turn_timeout_cap_s()))
    devices = res.get("devices") if isinstance(res, dict) else None
    if not isinstance(devices, list):
        return "", {"ok": False, "error": str(res.get("error") or "cannot list connected PCs")}
    ids = [str(d.get("id")) for d in devices if isinstance(d, dict) and d.get("id")]
    if len(ids) == 1:
        return ids[0], None
    if not ids:
        return "", {"ok": False, "error": "no PC is connected"}
    return "", {"ok": False, "error": "more than one PC is connected; pass device_id: " + ", ".join(sorted(ids))}


async def _pc_call(
    tool: str,
    args: dict[str, Any],
    timeout_s: float,
    device_id: str = "",
) -> dict[str, Any]:
    if not 1 <= timeout_s <= MAX_TIMEOUT_S:
        return {"ok": False, "error": f"timeout_s must be between 1 and {MAX_TIMEOUT_S}"}
    dev, err = await _pick_device(device_id)
    if err is not None:
        return err
    return await _request(
        "POST",
        f"/api/pc/{dev}/call",
        json_body={"tool": tool, "args": args, "timeout_s": int(timeout_s)},
        timeout_s=_budget(timeout_s),
    )


# ---- tools -----------------------------------------------------------------------------------------


@mcp.tool()
async def pc_list() -> dict[str, Any]:
    """List connected PCs (fields: id, host, os, tools, connected_s) that PC tools can act on."""
    return await _request("GET", "/api/pc", timeout_s=min(LIST_TIMEOUT_S, turn_timeout_cap_s()))


@mcp.tool()
async def pc_shell(
    command: str,
    shell: Annotated[str, Field(description='"powershell" | "cmd" | "sh" (default: powershell on Windows, sh elsewhere)')] = "",
    cwd: Annotated[str, Field(description="working directory for the command")] = "",
    timeout_s: Annotated[int, Field(ge=1, le=MAX_TIMEOUT_S)] = DEFAULT_TIMEOUT_S,
    device_id: str = "",
) -> dict[str, Any]:
    """Run a shell command on the user's PC and return combined stdout+stderr and exit_code.

    Runs without asking the user. A non-zero exit is still ok:true (the command ran);
    check data.exit_code. Long output is truncated by the client. Every tool call must fit one
    Hermes turn: keep timeout_s modest.
    """
    cmd = (command or "").strip()
    if not cmd:
        return {"ok": False, "error": "command must not be empty"}
    args: dict[str, Any] = {"command": cmd}
    if shell:
        args["shell"] = shell
    if cwd:
        args["cwd"] = cwd
    return await _pc_call("shell", args, timeout_s, device_id)


@mcp.tool()
async def pc_open(
    target: str,
    args: Annotated[list[str] | None, Field(description="extra arguments, for programs")] = None,
    device_id: str = "",
) -> dict[str, Any]:
    """Open a URL, file/folder or program on the user's PC (like double-clicking it)."""
    what = (target or "").strip()
    if not what:
        return {"ok": False, "error": "target must not be empty"}
    payload: dict[str, Any] = {"target": what}
    if args:
        payload["args"] = [str(a) for a in args]
    return await _pc_call("open", payload, DEFAULT_TIMEOUT_S, device_id)


# structured_output=False: the image goes back as MCP image content. With structured output on,
# FastMCP tries to serialize the Image object into JSON and every screenshot fails.
@mcp.tool(structured_output=False)
async def pc_screenshot(
    monitor: Annotated[int, Field(ge=0, description="0 = all monitors combined, 1.. = one monitor")] = 0,
    max_width: Annotated[int, Field(ge=200, le=4096)] = 1600,
    timeout_s: Annotated[int, Field(ge=1, le=MAX_TIMEOUT_S)] = DEFAULT_TIMEOUT_S,
    device_id: str = "",
) -> list[Any] | dict[str, Any]:
    """Take a screenshot of the user's PC screens and return it as an image plus a size/monitor list."""
    res = await _pc_call("screenshot", {"monitor": int(monitor), "max_width": int(max_width)}, timeout_s, device_id)
    if isinstance(res, dict) and res.get("ok") is True:
        image = res.get("image")
        if not isinstance(image, dict) or not image.get("b64"):
            return {"ok": False, "error": "the PC returned no image"}
        try:
            data = base64.b64decode(image["b64"])
        except Exception:
            return {"ok": False, "error": "the PC returned an undecodable image"}
        fmt = str(image.get("mime", "image/png")).rsplit("/", 1)[-1] or "png"
        return [Image(data=data, format=fmt), str(res.get("text") or "screenshot")]
    return res


@mcp.tool()
async def pc_type(text: str, device_id: str = "") -> dict[str, Any]:
    """Type text (at most 10000 characters) on the user's PC as if on the keyboard."""
    cleaned = text or ""
    if not cleaned.strip():
        return {"ok": False, "error": "text must not be empty"}
    if len(cleaned) > 10000:
        return {"ok": False, "error": "text must be at most 10000 characters"}
    return await _pc_call("type", {"text": cleaned}, DEFAULT_TIMEOUT_S, device_id)


@mcp.tool()
async def pc_keys(
    combo: Annotated[str, Field(description='one key combo like "ctrl+shift+esc", "win+r", "enter"')] = "",
    sequence: Annotated[list[str] | None, Field(description="combos pressed in order")] = None,
    device_id: str = "",
) -> dict[str, Any]:
    """Press a key combination (or a sequence of them) on the user's PC."""
    have_combo = bool((combo or "").strip())
    have_seq = bool(sequence)
    if have_combo == have_seq:
        return {"ok": False, "error": "pass exactly one of combo or sequence"}
    args: dict[str, Any] = {"combo": (combo or "").strip()} if have_combo else {"sequence": [str(c) for c in sequence or []]}
    return await _pc_call("keys", args, DEFAULT_TIMEOUT_S, device_id)


@mcp.tool()
async def pc_mouse(
    action: Annotated[str, Field(description='"move" | "click" | "double_click" | "right_click" | "scroll"')],
    x: Annotated[int | None, Field(ge=-100000, le=100000)] = None,
    y: Annotated[int | None, Field(ge=-100000, le=100000)] = None,
    amount: Annotated[int | None, Field(ge=-1000, le=1000, description="scroll clicks, positive = up")] = None,
    device_id: str = "",
) -> dict[str, Any]:
    """Move the mouse or click/scroll at absolute virtual-screen pixels (the screenshot-0 space)."""
    act = (action or "").strip()
    if act not in ("move", "click", "double_click", "right_click", "scroll"):
        return {"ok": False, "error": "action must be move, click, double_click, right_click or scroll"}
    args: dict[str, Any] = {"action": act}
    if x is not None:
        args["x"] = int(x)
    if y is not None:
        args["y"] = int(y)
    if amount is not None:
        args["amount"] = int(amount)
    return await _pc_call("mouse", args, DEFAULT_TIMEOUT_S, device_id)


@mcp.tool()
async def pc_clipboard(
    action: Annotated[str, Field(description='"get" or "set"')] = "get",
    text: Annotated[str, Field(description="new clipboard content (for set)")] = "",
    device_id: str = "",
) -> dict[str, Any]:
    """Read (get) or replace (set) the clipboard content of the user's PC."""
    act = (action or "get").strip()
    if act not in ("get", "set"):
        return {"ok": False, "error": "action must be get or set"}
    if act == "set" and not text:
        return {"ok": False, "error": "text is required for set"}
    args: dict[str, Any] = {"action": act}
    if act == "set":
        args["text"] = text
    return await _pc_call("clipboard", args, DEFAULT_TIMEOUT_S, device_id)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

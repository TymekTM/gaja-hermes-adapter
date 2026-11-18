"""Tool behaviour on this Linux box: shell really runs, timeouts kill, output is capped,
arg validation answers bad_request, and the screenshot JPEG fits the wire limit."""

import base64
import os
import time
from pathlib import Path

import pyperclip
import pytest

from gaja_desktop.tools import TEXT_LIMIT, ToolResult, create_tools, truncate_middle
from gaja_desktop.tools.clipboard import ClipboardTool
from gaja_desktop.tools.keys import KeysTool, parse_combo
from gaja_desktop.tools.mouse import MouseTool
from gaja_desktop.tools.screenshot import B64_LIMIT, ScreenshotTool, encode_jpeg_under_limit
from gaja_desktop.tools.shell import ShellTool
from gaja_desktop.tools.type import TypeTool


# ---- shell ------------------------------------------------------------------------------


def test_shell_runs_and_combines_output() -> None:
    result = ShellTool().run({"command": "echo hello; echo oops >&2; exit 3"}, timeout_s=10)
    assert result.ok is True  # a non-zero exit is still ok:true
    assert result.text == "hello\noops\n"
    assert result.data["exit_code"] == 3


def test_shell_default_is_sh_on_linux() -> None:
    result = ShellTool().run({"command": "printf %s hello"}, timeout_s=10)
    assert result.ok and result.text == "hello"


def test_shell_cwd() -> None:
    result = ShellTool().run({"command": "pwd", "cwd": str(Path(__file__).parent)}, timeout_s=10)
    assert result.text.strip() == str(Path(__file__).parent)
    bad = ShellTool().run({"command": "pwd", "cwd": "/definitely/not/here"}, timeout_s=10)
    assert not bad.ok and bad.error.startswith("bad_request:")


def test_shell_rejects_unknown_shell_and_missing_command() -> None:
    bad = ShellTool().run({"command": "x", "shell": "fish"}, timeout_s=5)
    assert not bad.ok and bad.error.startswith("bad_request:")
    bad2 = ShellTool().run({"shell": "sh"}, timeout_s=5)
    assert not bad2.ok and bad2.error.startswith("bad_request:")
    cmd = ShellTool().run({"command": "x", "shell": "cmd"}, timeout_s=5)
    assert not cmd.ok and "not available" in cmd.error  # cmd.exe does not exist here


def test_shell_timeout_kills_the_process_tree() -> None:
    marker = "gaja-kill-test-8f2c"
    started = time.monotonic()
    result = ShellTool().run({"command": f"sleep 30 # {marker}"}, timeout_s=1.0)
    elapsed = time.monotonic() - started
    assert not result.ok and result.error == "timeout"
    assert elapsed < 5.0, "the client must enforce the timeout itself"
    # the whole group died: no process with the marker survives
    survivors = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if marker.encode() in cmdline:
            survivors.append(pid)
    assert survivors == [], f"timeout must kill the process tree, still running: {survivors}"


def test_shell_output_truncated_to_protocol_limit() -> None:
    result = ShellTool().run({"command": "head -c 200000 /dev/zero | tr '\\0' 'a'"}, timeout_s=15)
    assert result.ok
    assert len(result.text) <= TEXT_LIMIT
    assert "truncated" in result.text


# ---- truncation helper -------------------------------------------------------------------


def test_truncate_middle() -> None:
    text = "A" * 40_000 + "MIDDLE" + "B" * 40_000
    out = truncate_middle(text, 1000)
    assert len(out) <= 1000
    assert out.startswith("A")
    assert out.endswith("B")
    assert "[truncated" in out
    assert truncate_middle("short", 1000) == "short"


def test_bad_request_shape() -> None:
    r = ToolResult.bad_request("nope")
    assert r.ok is False and r.error == "bad_request: nope"


# ---- tools that need an input backend: validation only here ------------------------------


def test_type_validation() -> None:
    tool = TypeTool()
    bad = tool.run({"text": "x" * 10_001}, timeout_s=5)
    assert not bad.ok and bad.error.startswith("bad_request:")
    empty = tool.run({}, timeout_s=5)
    assert not empty.ok and empty.error.startswith("bad_request:")
    ok_shape = tool.run({"text": "hi"}, timeout_s=5)
    # no display on this box: either it typed (X available) or it says why; never bad_request
    if not ok_shape.ok:
        assert ok_shape.error == "no keyboard backend on this machine"


def test_keys_parsing_and_validation() -> None:
    combo = parse_combo("ctrl+shift+esc")
    assert [getattr(k, "name", None) for k in combo] == ["ctrl", "shift", "esc"]  # modifiers first
    assert parse_combo("win+r") is not None
    assert parse_combo("enter") is not None
    assert parse_combo("notakey") is None
    tool = KeysTool()
    both = tool.run({"combo": "enter", "sequence": ["enter"]}, timeout_s=5)
    assert not both.ok and both.error.startswith("bad_request:")
    unknown = tool.run({"combo": "ctrl+bogus"}, timeout_s=5)
    assert not unknown.ok and unknown.error.startswith("bad_request:")
    empty = tool.run({}, timeout_s=5)
    assert not empty.ok and empty.error.startswith("bad_request:")


def test_mouse_validation() -> None:
    tool = MouseTool()
    bad_action = tool.run({"action": "teleport", "x": 1, "y": 1}, timeout_s=5)
    assert not bad_action.ok and bad_action.error.startswith("bad_request:")
    missing_xy = tool.run({"action": "click"}, timeout_s=5)
    assert not missing_xy.ok and missing_xy.error.startswith("bad_request:")
    bad_amount = tool.run({"action": "scroll", "amount": 1000}, timeout_s=5)
    assert not bad_amount.ok and bad_amount.error.startswith("bad_request:")


# ---- screenshot: the encoder is testable without a display -------------------------------


def test_screenshot_encoder_respects_wire_limit() -> None:
    from PIL import Image

    # noisy 4000x2500: worst case for JPEG
    img = Image.effect_noise((4000, 2500), 60).convert("RGB")
    encoded = encode_jpeg_under_limit(img, max_width=1600)
    assert encoded is not None
    b64, width, height = encoded
    assert len(b64) <= B64_LIMIT
    assert width == 1600 and 0 < height <= 2500
    raw = base64.b64decode(b64)
    assert raw[:3] == b"\xff\xd8\xff"  # JPEG magic
    # an all-black 1080p frame must stay far under the limit (sanity: not everything is downscaled)
    small = encode_jpeg_under_limit(Image.new("RGB", (1920, 1080)), max_width=1600)
    assert small is not None and len(small[0]) < 100 * 1024


def test_screenshot_arg_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = ScreenshotTool()
    assert tool.run({"monitor": "x"}, timeout_s=5).error.startswith("bad_request:")
    assert tool.run({"max_width": 1}, timeout_s=5).error.startswith("bad_request:")
    monkeypatch.setattr(tool, "available", lambda: True)
    out_of_range = tool.run({"monitor": 99}, timeout_s=5)
    assert not out_of_range.ok  # bad_request or a clean failure, but never an exception


# ---- clipboard with monkeypatched pyperclip -----------------------------------------------


def test_clipboard_get_set(monkeypatch: pytest.MonkeyPatch) -> None:
    store = {"clip": "hello"}
    monkeypatch.setattr(pyperclip, "paste", lambda: store["clip"])
    monkeypatch.setattr(pyperclip, "copy", lambda t: store.__setitem__("clip", t))
    tool = ClipboardTool()
    got = tool.run({"action": "get"}, timeout_s=5)
    assert got.ok and got.text == "hello"
    put = tool.run({"action": "set", "text": "new value"}, timeout_s=5)
    assert put.ok and store["clip"] == "new value"
    default = tool.run({}, timeout_s=5)  # default action is get
    assert default.ok and default.text == "new value"
    bad = tool.run({"action": "set"}, timeout_s=5)
    assert not bad.ok and bad.error.startswith("bad_request:")
    weird = tool.run({"action": "teleport"}, timeout_s=5)
    assert not weird.ok and weird.error.startswith("bad_request:")


def test_clipboard_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str:
        raise pyperclip.PyperclipException("no clipboard backend")

    monkeypatch.setattr(pyperclip, "paste", boom)
    tool = ClipboardTool()
    assert tool.available() is False
    monkeypatch.setattr(tool, "available", lambda: True)  # force-enable to check the runtime path
    got = tool.run({"action": "get"}, timeout_s=5)
    assert not got.ok and "clipboard unavailable" in got.error


# ---- registry ---------------------------------------------------------------------------


def test_create_tools_filters_enabled_and_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ScreenshotTool, "available", lambda self: False)
    tools = create_tools(["shell", "screenshot", "nonexistent"])
    names = [t.name for t in tools]
    assert "shell" in names  # really runs on this box
    assert "screenshot" not in names  # backend missing -> not advertised
    assert "nonexistent" not in names
    assert names == [n for n in names if n in ("shell", "open", "screenshot", "type", "keys", "mouse", "clipboard")]

"""Tool behaviour on this Linux box: shell really runs, timeouts kill, output is capped,
arg validation answers bad_request, and the screenshot JPEG fits the wire limit."""

import base64
import os
import time
from pathlib import Path

import pyperclip
import pytest

import gaja_desktop.tools.shell as shell_module
from gaja_desktop.tools import TEXT_LIMIT, ToolResult, create_tools, popen, truncate_middle
from gaja_desktop.tools.clipboard import ClipboardTool
from gaja_desktop.tools.keys import SPECIAL_KEYS, KeysTool, parse_combo
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


# ---- Windows command-line construction (platform monkeypatched) ---------------------------


@pytest.fixture
def fake_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_module.platform, "system", lambda: "Windows")


def test_build_argv_cmd_is_a_verbatim_command_line_string(fake_windows) -> None:
    """cmd.exe must get one string: a list would be list2cmdline-escaped and its quotes mangled."""
    line = shell_module.build_argv(r'dir "C:\Program Files"', "cmd")
    assert isinstance(line, str)
    assert line == r'cmd.exe /d /s /c "chcp 65001 >nul & dir "C:\Program Files""'
    # no backslash-escaped quotes (what subprocess.list2cmdline would have produced):
    assert '\\"' not in line


def test_build_argv_cmd_special_characters_survive(fake_windows) -> None:
    line = shell_module.build_argv('echo "a & b" | findstr "x>y"', "cmd")
    assert isinstance(line, str)
    assert line.startswith('cmd.exe /d /s /c "')
    assert line.endswith('"')
    assert 'echo "a & b" | findstr "x>y"' in line  # the command sits between, byte for byte


def test_build_argv_cmd_unavailable_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_module.platform, "system", lambda: "Linux")
    assert shell_module.build_argv("dir", "cmd") is None


def test_build_argv_powershell_keeps_the_utf8_prefix(fake_windows) -> None:
    argv = shell_module.build_argv('Write-Output "zażółć"', "powershell")
    assert argv == ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                    '[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; '
                    'Write-Output "zażółć"']


def test_build_argv_sh_is_a_list() -> None:
    assert shell_module.build_argv("echo zażółć", "sh") == ["sh", "-c", "echo zażółć"]


def test_popen_splits_a_string_argv_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive: a string argv (a Windows cmd line) never reaches popen on POSIX, but if one
    does it must run as words rather than as a program named 'echo hello'."""
    import subprocess

    started: list[Any] = []

    class FakePopen:
        def __init__(self, argv, **kwargs) -> None:
            started.append(argv)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    from gaja_desktop.tools import popen

    popen(["sh", "-c", "true"])
    popen("echo hello > /tmp/x")
    assert started[0] == ["sh", "-c", "true"]
    assert started[1] == ["echo", "hello", ">", "/tmp/x"]



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


def test_keys_special_names_exist_on_pynput() -> None:
    """Every advertised special key must be a real pynput Key member (altgr used to miss)."""
    from pynput.keyboard import Key

    for name in set(SPECIAL_KEYS.values()):
        assert getattr(Key, name, None) is not None, name


def test_keys_altgr_resolves_and_sorts_as_a_modifier() -> None:
    from pynput.keyboard import Key

    combo = parse_combo("altgr+e")
    assert combo is not None
    assert [getattr(k, "name", None) for k in combo] == ["alt_gr", None]  # modifier first
    assert combo[0] is Key.alt_gr
    assert combo[1].vk == 0x45  # 'E' — a virtual key, not a unicode packet
    # order given backwards still presses the modifier first
    assert parse_combo("e+altgr")[0] is Key.alt_gr


def test_keys_letters_and_digits_in_combos_become_virtual_keys() -> None:
    """Ctrl+c must press VK 0x43, not a KEYEVENTF_UNICODE 'c' (which Ctrl would not combine with)."""
    for combo, vk in (("ctrl+c", 0x43), ("ctrl+C", 0x43), ("win+r", 0x52),
                      ("ctrl+shift+3", 0x33), ("alt+0", 0x30), ("ctrl+alt+delete", None)):
        keys = parse_combo(combo)
        assert keys is not None, combo
        if vk is None:
            continue  # ctrl+alt+delete ends in a Key, checked below
        final = keys[-1]
        assert getattr(final, "vk", None) == vk, combo
        assert getattr(final, "char", None) is None or final.char.islower(), combo
    assert parse_combo("ctrl+alt+delete")[-1].name == "delete"


def test_keys_plain_key_without_modifier_stays_a_char() -> None:
    keys = parse_combo("a")
    assert keys is not None and keys[0].char == "a" and keys[0].vk is None


def test_keys_plain_non_ascii_without_modifier_stays_a_char() -> None:
    keys = parse_combo("ż")
    assert keys is not None and keys[0].char == "ż"


def test_mouse_validation() -> None:
    tool = MouseTool()
    bad_action = tool.run({"action": "teleport", "x": 1, "y": 1}, timeout_s=5)
    assert not bad_action.ok and bad_action.error.startswith("bad_request:")
    missing_xy = tool.run({"action": "click"}, timeout_s=5)
    assert not missing_xy.ok and missing_xy.error.startswith("bad_request:")
    bad_amount = tool.run({"action": "scroll", "amount": 1000}, timeout_s=5)
    assert not bad_amount.ok and bad_amount.error.startswith("bad_request:")


def test_mouse_accepts_negative_virtual_screen_coordinates() -> None:
    """A monitor left of/above the primary gives negative virtual-screen coordinates."""
    tool = MouseTool()
    bad = tool.run({"action": "move", "x": -200_000, "y": 0}, timeout_s=5)
    assert not bad.ok and bad.error.startswith("bad_request:")
    moved = tool.run({"action": "move", "x": -1920, "y": -200}, timeout_s=5)
    # no input backend on a headless box: the validation must pass; only the backend is missing
    assert moved.error != "bad_request: 'x'/'y' must be integers" and (
        moved.ok or moved.error == "no mouse backend on this machine")


def test_screenshot_text_maps_image_pixels_to_mouse_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The result text names the virtual-screen rect and the img->mouse formula (docs/DESKTOP.md)."""
    from PIL import Image

    import gaja_desktop.tools.screenshot as shot_module

    # Two 1920x1080 monitors side by side, the right one primary: the left one sits at (-1920, 0).
    monitors = [
        {"index": 0, "left": -1920, "top": 0, "width": 3840, "height": 1080},
        {"index": 1, "left": -1920, "top": 0, "width": 1920, "height": 1080},
        {"index": 2, "left": 0, "top": 0, "width": 1920, "height": 1080},
    ]

    def fake_grab(monitor: int):
        return Image.new("RGB", (monitors[monitor]["width"], monitors[monitor]["height"])), monitors

    monkeypatch.setattr(shot_module, "grab_monitor", fake_grab)
    result = ScreenshotTool().run({"monitor": 0, "max_width": 1600}, timeout_s=10)
    assert result.ok, result.error
    assert "(-1920,0)" in result.text
    assert "-1920 + img_x*3840/1600" in result.text
    assert "0 + img_y*1080/" in result.text
    assert result.data["monitors"][0]["left"] == -1920
    assert result.image["width"] == 1600


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


def test_create_tools_survives_a_tool_module_that_cannot_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing backend library must drop that one tool, not kill the client (module contract)."""
    import sys

    # A None entry in sys.modules makes importing that module raise ImportError.
    monkeypatch.setitem(sys.modules, "gaja_desktop.tools.screenshot", None)
    monkeypatch.setitem(sys.modules, "gaja_desktop.tools.clipboard", None)
    tools = create_tools(["shell", "screenshot", "clipboard"])
    names = [t.name for t in tools]
    assert "shell" in names
    assert "screenshot" not in names and "clipboard" not in names


def test_open_tool_spawns_helpers_without_a_console_window(monkeypatch: pytest.MonkeyPatch) -> None:
    import gaja_desktop.tools.open as open_module
    from gaja_desktop.tools.open import OpenTool

    started: list[tuple[list[str], int]] = []

    class FakePopen:
        def __init__(self, argv, **kwargs) -> None:
            started.append((argv, kwargs.get("creationflags", 0)))

    monkeypatch.setattr(open_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(open_module, "which", lambda p: "/usr/bin/" + p)
    monkeypatch.setattr(open_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(open_module, "_NO_WINDOW", 0x08000000)  # the real win32 CREATE_NO_WINDOW
    result = OpenTool().run({"target": "notepad", "args": ["x.txt"]}, timeout_s=5)
    assert result.ok and result.text.startswith("opened program")
    argv, flags = started[0]
    assert argv == ["/usr/bin/notepad", "x.txt"]
    assert flags & 0x08000000  # no console flash from the windowed exe

"""PC tools the agent can call through the adapter (`pc_call` -> `pc_result`, protocol v1.14).

One module per tool. Every tool:

- validates its args and answers ``ok:false, error:"bad_request: …"`` for junk;
- runs blocking work in its calling thread (the connection runs tools in worker threads, so a slow
  tool never stalls the audio loop or other calls);
- caps its text output (the protocol allows 65536 chars, middle-truncated).

A tool whose backend is missing on the machine (no X display, no clipboard helper, …) reports
``available() == False`` and is simply not advertised in the hello.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, ClassVar

#: Protocol limit for `pc_result.text` (docs/DESKTOP.md).
TEXT_LIMIT = 65_536
#: Read cap for subprocess output, well above TEXT_LIMIT, so a runaway child cannot eat RAM.
MAX_CAPTURE_BYTES = 256 * 1024


def truncate_middle(text: str, limit: int = TEXT_LIMIT) -> str:
    """Keep head and tail, drop the middle: ``\\n…[truncated N chars]…\\n`` marks what is gone."""
    if len(text) <= limit:
        return text
    marker = "\n…[truncated {} chars]…\n"
    keep = max(0, limit - 64)  # room for the longest possible marker
    head = text[: keep // 2]
    tail = text[-(keep - keep // 2):]
    dropped = len(text) - len(head) - len(tail)
    return head + marker.format(dropped) + tail


@dataclass
class ToolResult:
    """The payload of one `pc_result` (before JSON encoding)."""

    ok: bool
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    image: dict[str, Any] | None = None

    @classmethod
    def bad_request(cls, detail: str) -> "ToolResult":
        return cls(ok=False, error=f"bad_request: {detail}")


class Tool:
    """Base class: `run()` blocks (worker thread), returns a ToolResult, never raises."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def available(self) -> bool:
        """Whether this machine can run the tool at all (gates the hello advertisement)."""
        return True

    def run(self, args: dict[str, Any], timeout_s: float) -> ToolResult:
        raise NotImplementedError

    # ---- arg helpers ------------------------------------------------------------------

    @staticmethod
    def _str(args: dict[str, Any], key: str, *, required: bool = True, max_len: int = 0) -> str | None:
        value = args.get(key)
        if not isinstance(value, str) or (required and not value):
            return None
        if max_len and len(value) > max_len:
            return None
        return value

    @staticmethod
    def _int(args: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int | None:
        value = args.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = int(value)
        return value if lo <= value <= hi else None


def create_tools(enabled: list[str]) -> list[Tool]:
    """Instantiate every known tool, keep those enabled in the config AND runnable on this machine."""
    from .clipboard import ClipboardTool  # noqa: PLC0415
    from .keys import KeysTool  # noqa: PLC0415
    from .mouse import MouseTool  # noqa: PLC0415
    from .open import OpenTool  # noqa: PLC0415
    from .screenshot import ScreenshotTool  # noqa: PLC0415
    from .shell import ShellTool  # noqa: PLC0415
    from .type import TypeTool  # noqa: PLC0415

    classes = [ShellTool, OpenTool, ScreenshotTool, TypeTool, KeysTool, MouseTool, ClipboardTool]
    tools: list[Tool] = []
    for cls in classes:
        if cls.name not in enabled:
            continue
        try:
            tool = cls()
        except Exception:  # noqa: BLE001 - construction probing a missing backend must not kill us
            continue
        try:
            ok = tool.available()
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            tools.append(tool)
    return tools


# ---- subprocess plumbing shared by shell/open ------------------------------------------

#: The packaged client is a windowed exe: without this every powershell/cmd/taskkill child would
#: flash its own console window on the user's screen.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process and (best effort) its children, portably."""
    if proc.poll() is not None:
        return
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=5, creationflags=_NO_WINDOW)
        else:
            import os
            import signal

            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError):
        try:
            proc.kill()
        except OSError:
            pass


def popen(argv: list[str] | str, *, cwd: str | None = None) -> subprocess.Popen:
    """Start a child in its own process group so a timeout can kill the whole tree.

    A string `argv` is only produced for Windows shell command lines (see ``build_argv``): there
    it reaches ``CreateProcess`` verbatim. On POSIX it is defensively split into words.
    """
    if platform.system() == "Windows":
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=cwd,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | _NO_WINDOW)
    if isinstance(argv, str):
        import shlex

        argv = shlex.split(argv)
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=cwd, start_new_session=True)


def which(program: str) -> str | None:
    return shutil.which(program)

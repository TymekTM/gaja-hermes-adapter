"""`shell`: run a command (powershell/cmd on Windows, sh elsewhere), combined output, exit code.

A non-zero exit is still `ok:true` — the process ran. A timeout kills the whole process tree and
answers `ok:false, error:"timeout"` (with whatever output was captured so far).
"""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from typing import Any

from . import MAX_CAPTURE_BYTES, Tool, ToolResult, kill_tree, popen, truncate_middle

#: Cap the captured output before the protocol truncation, so `yes` cannot fill RAM.
_READ_CAP = MAX_CAPTURE_BYTES


def _default_shell() -> str:
    return "powershell" if platform.system() == "Windows" else "sh"


def build_argv(command: str, shell: str) -> list[str] | str | None:
    """The command line for one shell choice; None if that shell is not available here.

    ``cmd`` returns a *string*: on Windows ``Popen`` passes a string to ``CreateProcess`` as the
    command line verbatim, so the quotes inside ``command`` survive. An argv list would be joined
    with ``list2cmdline``, which backslash-escapes the inner quotes (``dir \\"C:\\Program Files\\"``)
    — cmd.exe does not understand that escaping and mangles the command.
    """
    if shell == "powershell":
        exe = "powershell" if platform.system() == "Windows" else None
        # Windows PowerShell 5.1 writes piped output in the OEM code page (Polish letters arrive
        # as mojibake); force UTF-8, which run_subprocess decodes.
        utf8 = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        return None if exe is None else [exe, "-NoProfile", "-NonInteractive", "-Command", utf8 + command]
    if shell == "cmd":
        if platform.system() != "Windows":
            return None
        # /d skips AutoRun scripts; /s strips exactly the first and last quote below and leaves
        # everything between them verbatim — the whole command, inner quotes included.
        return f'cmd.exe /d /s /c "chcp 65001 >nul & {command}"'  # UTF-8 output, as for powershell
    if shell == "sh":
        return ["sh", "-c", command]
    return None


def run_subprocess(argv: list[str] | str, timeout_s: float, cwd: str | None = None) -> tuple[str, int | None, bool]:
    """Run argv, capture combined stdout+stderr (capped), return (output, exit_code, timed_out).

    `argv` is an argument list, except for Windows shell command lines (``cmd.exe /d /s /c "..."``)
    which must be passed as a string so their quotes are not re-escaped (see ``build_argv``).
    A reader thread stops after the read cap and closes the pipe (the child dies on the next write
    or on the timeout kill); the whole process group is killed on timeout.
    """
    proc = popen(argv, cwd=cwd)
    buf = bytearray()
    reader = threading.Thread(target=_pump, args=(proc.stdout, buf), daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=max(0.1, timeout_s))
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    reader.join(timeout=2.0)
    return buf.decode("utf-8", errors="replace"), proc.returncode, timed_out


def _pump(pipe: Any, buf: bytearray) -> None:
    try:
        while len(buf) < _READ_CAP:
            chunk = pipe.read(8192)
            if not chunk:
                break
            buf.extend(chunk)
    except (OSError, ValueError):
        pass
    finally:
        try:
            pipe.close()  # a capped-out child fails on its next write instead of growing RAM
        except OSError:
            pass


class ShellTool(Tool):
    name = "shell"
    description = "Run a shell command and return stdout+stderr and the exit code"

    def run(self, args: dict[str, Any], timeout_s: float) -> ToolResult:
        command = self._str(args, "command")
        if command is None:
            return ToolResult.bad_request("shell needs 'command' (string)")
        shell = self._str(args, "shell", required=False) or _default_shell()
        cwd = self._str(args, "cwd", required=False)
        if cwd is not None:
            from pathlib import Path

            if not Path(cwd).is_dir():
                return ToolResult.bad_request(f"cwd does not exist: {cwd}")
        argv = build_argv(command, shell)
        if argv is None:
            return ToolResult.bad_request(f"shell {shell!r} is not available on {platform.system()}")
        started = time.monotonic()
        output, exit_code, timed_out = run_subprocess(argv, timeout_s, cwd=cwd)
        if timed_out:
            return ToolResult(ok=False, text=truncate_middle(output), error="timeout",
                              data={"exit_code": exit_code, "shell": shell,
                                    "elapsed_ms": int((time.monotonic() - started) * 1000)})
        return ToolResult(ok=True, text=truncate_middle(output),
                          data={"exit_code": exit_code, "shell": shell,
                                "elapsed_ms": int((time.monotonic() - started) * 1000)})

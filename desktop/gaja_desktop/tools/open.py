"""`open`: launch a URL, file/folder, or program (with optional arguments)."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any
from urllib.parse import urlparse

from . import Tool, ToolResult, which


def _looks_like_url(target: str) -> bool:
    try:
        parts = urlparse(target)
    except ValueError:
        return False
    return parts.scheme in ("http", "https", "file") and bool(parts.netloc or parts.scheme == "file")


class OpenTool(Tool):
    name = "open"
    description = "Open a URL, file, folder, or program"

    def available(self) -> bool:
        if platform.system() == "Windows":
            return True  # os.startfile is always there
        return which("xdg-open") is not None

    def run(self, args: dict[str, Any], timeout_s: float) -> ToolResult:
        target = self._str(args, "target")
        if target is None:
            return ToolResult.bad_request("open needs 'target' (url, path or program)")
        extra = args.get("args")
        if extra is not None and (
            not isinstance(extra, list) or not all(isinstance(a, str) for a in extra)
        ):
            return ToolResult.bad_request("'args' must be a list of strings")
        extra = extra or []
        try:
            if _looks_like_url(target):
                self._open_url(target)
            elif extra:
                # Program with arguments: search PATH like a shell would.
                program = which(target) or target
                subprocess.Popen([program, *extra], stdin=subprocess.DEVNULL)
            elif platform.system() == "Windows":
                os.startfile(target)  # noqa: S606 - the tool's purpose
            else:
                subprocess.Popen(["xdg-open", target], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            return ToolResult(ok=False, error=f"not found: {exc.filename or target}")
        except OSError as exc:
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        what = "program" if extra else ("url" if _looks_like_url(target) else "path")
        return ToolResult(ok=True, text=f"opened {what}: {target}" + (f" ({len(extra)} args)" if extra else ""))

    @staticmethod
    def _open_url(target: str) -> None:
        if platform.system() == "Windows":
            os.startfile(target)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", target], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

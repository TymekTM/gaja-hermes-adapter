"""`clipboard`: read or set the clipboard (pyperclip; on Linux needs xclip/xsel/wl-clipboard)."""

from __future__ import annotations

from typing import Any

import pyperclip

from . import Tool, ToolResult

MAX_CLIPBOARD_CHARS = 100_000  # guard against someone stuffing the agent with megabytes


class ClipboardTool(Tool):
    name = "clipboard"
    description = "Read or set the clipboard"

    def available(self) -> bool:
        try:
            pyperclip.paste()  # raises PyperclipException when no backend exists
            return True
        except pyperclip.PyperclipException:
            return False
        except Exception:  # noqa: BLE001 - odd backend errors still mean "cannot use it"
            return False

    def run(self, args: dict[str, Any], timeout_s: float) -> ToolResult:
        action = self._str(args, "action", required=False) or "get"
        if action not in ("get", "set"):
            return ToolResult.bad_request("'action' must be 'get' or 'set'")
        if action == "get":
            try:
                text = pyperclip.paste()
            except pyperclip.PyperclipException as exc:
                return ToolResult(ok=False, error=f"clipboard unavailable: {exc}")
            if not isinstance(text, str):
                return ToolResult(ok=False, error="clipboard holds non-text data")
            return ToolResult(ok=True, text=text[:MAX_CLIPBOARD_CHARS],
                              data={"truncated": len(text) > MAX_CLIPBOARD_CHARS})
        text = self._str(args, "text")
        if text is None:
            return ToolResult.bad_request("clipboard 'set' needs 'text'")
        try:
            pyperclip.copy(text)
        except pyperclip.PyperclipException as exc:
            return ToolResult(ok=False, error=f"clipboard unavailable: {exc}")
        return ToolResult(ok=True, text=f"clipboard set ({len(text)} characters)")

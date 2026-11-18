"""`type`: type text through the keyboard (pynput); the agent's dictation path."""

from __future__ import annotations

import threading
from typing import Any

from . import Tool, ToolResult

MAX_TYPE_CHARS = 10_000  # protocol limit (docs/DESKTOP.md)
_lock = threading.Lock()  # keyboard/mouse tools share the one input stream: serialise them


class InputControllerMixin:
    """Lazily constructed pynput controllers, shared by type/keys/mouse; None when unavailable."""

    _keyboard: Any = None
    _mouse: Any = None

    def keyboard(self) -> Any | None:
        if self._keyboard is None:
            try:
                from pynput.keyboard import Controller  # noqa: PLC0415

                self._keyboard = Controller()
            except Exception:  # noqa: BLE001 - no display / no backend
                return None
        return self._keyboard

    def mouse(self) -> Any | None:
        if self._mouse is None:
            try:
                from pynput.mouse import Controller  # noqa: PLC0415

                self._mouse = Controller()
            except Exception:  # noqa: BLE001
                return None
        return self._mouse


class TypeTool(InputControllerMixin, Tool):
    name = "type"
    description = "Type text as if on the keyboard"

    def available(self) -> bool:
        try:
            from pynput.keyboard import Controller  # noqa: PLC0415

            Controller()
            return True
        except Exception:  # noqa: BLE001
            return False

    def run(self, args: dict[str, Any], timeout_s: float) -> ToolResult:
        text = self._str(args, "text", max_len=MAX_TYPE_CHARS)
        if text is None:
            return ToolResult.bad_request(f"'type' needs 'text' (1..{MAX_TYPE_CHARS} chars)")
        kb = self.keyboard()
        if kb is None:
            return ToolResult(ok=False, error="no keyboard backend on this machine")
        with _lock:
            kb.type(text)
        return ToolResult(ok=True, text=f"typed {len(text)} characters")

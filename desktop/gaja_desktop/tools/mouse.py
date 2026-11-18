"""`mouse`: move, click, double-click, right-click, scroll — absolute virtual-screen pixel space."""

from __future__ import annotations

import time
from typing import Any

from . import Tool, ToolResult
from .type import InputControllerMixin, _lock

ACTIONS = ("move", "click", "double_click", "right_click", "scroll")

#: Absolute virtual-screen pixel bounds: the origin is (0,0) at the primary monitor's top-left,
#: so a monitor left of/above it has negative coordinates (Windows SM_XVIRTUALSCREEN < 0). The
#: bound covers any real multi-monitor setup while still catching junk.
MOUSE_COORD_MIN = -100_000
MOUSE_COORD_MAX = 100_000


class MouseTool(InputControllerMixin, Tool):
    name = "mouse"
    description = "Move the cursor, click and scroll"

    def available(self) -> bool:
        try:
            from pynput.mouse import Controller  # noqa: PLC0415

            Controller()
            return True
        except Exception:  # noqa: BLE001
            return False

    def run(self, args: dict[str, Any], timeout_s: float) -> ToolResult:
        action = self._str(args, "action", required=False) or "click"
        if action not in ACTIONS:
            return ToolResult.bad_request(f"'action' must be one of {', '.join(ACTIONS)}")

        def coord(key: str) -> int | None | str:  # None = absent, "bad" = present but invalid
            value = args.get(key)
            if value is None:
                return None
            checked = self._int(args, key, 0, MOUSE_COORD_MIN, MOUSE_COORD_MAX)
            return "bad" if checked is None else checked

        x, y = coord("x"), coord("y")
        amount = self._int(args, "amount", 3, -100, 100)
        if amount is None:
            return ToolResult.bad_request("'amount' must be an integer (-100..100)")
        if "bad" in (x, y):
            return ToolResult.bad_request(
                f"'x'/'y' must be integers {MOUSE_COORD_MIN}..{MOUSE_COORD_MAX} "
                "(absolute virtual-screen pixels; the origin is negative on setups with a "
                "monitor left of/above the primary — see the screenshot result's rect)")
        if (action == "move" or action.endswith("click")) and (x is None or y is None):
            return ToolResult.bad_request(f"'{action}' needs absolute 'x' and 'y'")
        mouse = self.mouse()
        if mouse is None:
            return ToolResult(ok=False, error="no mouse backend on this machine")
        with _lock:
            if action == "move":
                mouse.position = (x, y)
            elif action == "scroll":
                mouse.scroll(0, amount)
            elif action == "click":
                mouse.position = (x, y)
                time.sleep(0.01)
                mouse.click(_left(), 1)
            elif action == "double_click":
                mouse.position = (x, y)
                time.sleep(0.01)
                mouse.click(_left(), 2)
            elif action == "right_click":
                mouse.position = (x, y)
                time.sleep(0.01)
                mouse.click(_right(), 1)
            time.sleep(0.01)  # let the OS settle so the reported position is the final one
            position = mouse.position
        return ToolResult(ok=True, text=f"cursor at ({position[0]}, {position[1]})",
                          data={"x": int(position[0]), "y": int(position[1])})


def _left() -> Any:
    from pynput.mouse import Button  # noqa: PLC0415

    return Button.left


def _right() -> Any:
    from pynput.mouse import Button  # noqa: PLC0415

    return Button.right

"""Global push-to-talk hotkey (pynput). Default `ctrl+alt+g`; fires `client.push_to_talk()`.

The listener runs in its own daemon thread; the callback is invoked on it and must marshal into the
asyncio loop itself (the client does that with `call_soon_threadsafe`). On a machine without an
input backend (headless Linux) the hotkey simply does not start — PTT stays off, everything else
keeps working.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

#: config word -> pynput key name (GlobalHotKeys `<...>` tokens)
_MODIFIERS = {"ctrl": "ctrl", "control": "ctrl", "alt": "alt", "altgr": "alt_gr", "shift": "shift",
              "win": "cmd", "windows": "cmd", "cmd": "cmd", "super": "cmd", "meta": "cmd"}
_SPECIALS = {"esc": "esc", "escape": "esc", "enter": "enter", "return": "enter", "tab": "tab",
             "space": "space", "backspace": "backspace", "delete": "delete", "del": "delete",
             "insert": "insert", "home": "home", "end": "end", "up": "up", "down": "down",
             "left": "left", "right": "right", "page_up": "page_up", "pageup": "page_up",
             "page_down": "page_down", "pagedown": "page_down"}


def hotkey_binding(combo: str) -> str | None:
    """\"ctrl+alt+g\" -> pynput's `<ctrl>+<alt>+g`; None when the combo cannot be parsed."""
    tokens = [t for t in combo.strip().lower().split("+") if t]
    if not tokens:
        return None
    out: list[str] = []
    for token in tokens:
        if token in _MODIFIERS:
            out.append(f"<{_MODIFIERS[token]}>")
        elif token in _SPECIALS:
            out.append(f"<{_SPECIALS[token]}>")
        elif len(token) == 1 and (token.isalnum() or token in ".,-[];'`/="):
            out.append(token)
        elif len(token) == 2 and token[0] == "f" and token[1].isdigit():
            out.append(f"<{token}>")
        elif len(token) == 3 and token[0] == "f" and token[1:].isdigit():
            out.append(f"<{token}>")
        else:
            return None
    return "+".join(out)


class Hotkey:
    """One global hotkey; `start()` never raises, failures surface as `error` (and no PTT)."""

    def __init__(self, combo: str, callback: Callable[[], None]) -> None:
        self.combo = combo
        self.callback = callback
        self.error: str | None = None
        self._listener: object | None = None
        self._binding = hotkey_binding(combo)

    @property
    def available(self) -> bool:
        return self._binding is not None

    def start(self) -> bool:
        if self._binding is None:
            self.error = f"cannot parse hotkey {self.combo!r}"
            return False
        try:
            from pynput import keyboard  # noqa: PLC0415 (lazy; needs a display on Linux)

            self._listener = keyboard.GlobalHotKeys({self._binding: self.callback})
            self._listener.start()  # type: ignore[attr-defined]
            log.info("push-to-talk hotkey active: %s (%s)", self.combo, self._binding)
            return True
        except Exception as exc:  # noqa: BLE001 - headless boxes have no input backend
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("push-to-talk hotkey unavailable: %s", self.error)
            return False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            self._listener = None

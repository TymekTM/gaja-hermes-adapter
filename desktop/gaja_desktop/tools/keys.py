"""`keys`: press a key combo ("ctrl+shift+esc", "win+r", "enter") or a sequence of combos."""

from __future__ import annotations

import time
from typing import Any

from pynput.keyboard import Key, KeyCode

from .type import InputControllerMixin, _lock
from . import Tool, ToolResult

#: Names the agent may use; everything else must be a single character.
#: Values are pynput ``Key`` member names (``alt_gr``, ``print_screen``, …), which exist on the
#: win32, xorg and darwin backends alike.
SPECIAL_KEYS: dict[str, str] = {
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "altgr": "alt_gr", "alt_gr": "alt_gr",
    "shift": "shift",
    "win": "cmd", "windows": "cmd", "cmd": "cmd", "meta": "cmd", "super": "cmd",
    "esc": "esc", "escape": "esc", "enter": "enter", "return": "enter", "tab": "tab",
    "space": "space", "backspace": "backspace", "delete": "delete", "del": "delete",
    "insert": "insert", "home": "home", "end": "end", "page_up": "page_up", "pageup": "page_up",
    "page_down": "page_down", "pagedown": "page_down", "up": "up", "down": "down",
    "left": "left", "right": "right", "caps_lock": "caps_lock", "print_screen": "print_screen",
    "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4", "f5": "f5", "f6": "f6",
    "f7": "f7", "f8": "f8", "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
}

#: pynput ``Key`` member names of the modifiers (sort order: modifiers first, the final key last).
_MODIFIERS = {"ctrl", "alt", "alt_gr", "shift", "cmd"}

#: Virtual-key codes for a combo's plain keys: VK 0x41..0x5A (A-Z), 0x30..0x39 (0-9).
_VK_LETTERS = {chr(ord("a") + i): 0x41 + i for i in range(26)}
_VK_DIGITS = {str(d): 0x30 + d for d in range(10)}


def resolve_key(token: str) -> Any | None:
    """One combo element: a special name or a single character."""
    name = SPECIAL_KEYS.get(token.strip().lower())
    if name is not None:
        return getattr(Key, name, None)
    if len(token) == 1:
        return KeyCode.from_char(token)
    return None


def _as_virtual_key(key: Any, token: str) -> Any:
    """A plain letter/digit of a modifier combo becomes a virtual-key press (see parse_combo)."""
    stripped = token.strip()
    vk = _VK_LETTERS.get(stripped.lower(), _VK_DIGITS.get(stripped))
    if getattr(key, "name", None) is None and vk is not None:  # a KeyCode, not a Key enum member
        return KeyCode.from_vk(vk)
    return key


def parse_combo(combo: str) -> list[Any] | None:
    """\"ctrl+shift+esc\" -> [ctrl, shift, esc]; None when anything in it is unresolvable.

    A single letter/digit in a combo *with modifiers* is pressed as its virtual key
    (``KeyCode.from_vk``), not as the character: the win32 Controller sends a plain char that its
    layout can only produce with Shift/AltGr as a KEYEVENTF_UNICODE packet, and a Ctrl+<unicode
    packet> does not register as a shortcut in any application. A virtual key always does.
    """
    tokens = [t for t in combo.split("+") if t]
    if not tokens:
        return None
    keys = [resolve_key(t) for t in tokens]
    if any(k is None for k in keys):
        return None
    if any(getattr(k, "name", "") in _MODIFIERS for k in keys):
        keys = [_as_virtual_key(k, t) for t, k in zip(tokens, keys)]
    # modifiers first, the final key last — stable press order
    keys.sort(key=lambda k: (not any(getattr(k, "name", "") == m for m in _MODIFIERS),))
    return keys


class KeysTool(InputControllerMixin, Tool):
    name = "keys"
    description = "Press a key combo or a sequence of combos"

    def available(self) -> bool:
        try:
            from pynput.keyboard import Controller  # noqa: PLC0415

            Controller()
            return True
        except Exception:  # noqa: BLE001
            return False

    def run(self, args: dict[str, Any], timeout_s: float) -> ToolResult:
        combo = self._str(args, "combo", required=False)
        sequence = args.get("sequence")
        if combo is None and sequence is None:
            return ToolResult.bad_request("'keys' needs 'combo' or 'sequence'")
        if combo is not None and sequence is not None:
            return ToolResult.bad_request("give either 'combo' or 'sequence', not both")
        combos: list[str]
        if combo is not None:
            combos = [combo]
        else:
            if not isinstance(sequence, list) or not sequence or \
                    not all(isinstance(c, str) and c for c in sequence):
                return ToolResult.bad_request("'sequence' must be a non-empty list of combo strings")
            if len(sequence) > 64:
                return ToolResult.bad_request("'sequence' is limited to 64 combos")
            combos = sequence
        parsed: list[list[Any]] = []
        for c in combos:
            keys = parse_combo(c)
            if keys is None:
                return ToolResult.bad_request(f"cannot resolve combo {c!r}")
            parsed.append(keys)
        kb = self.keyboard()
        if kb is None:
            return ToolResult(ok=False, error="no keyboard backend on this machine")
        with _lock:
            for keys in parsed:
                _press(kb, keys)
                if len(parsed) > 1:
                    time.sleep(0.05)
        pressed = " + ".join("+".join(str(getattr(k, "name", None) or getattr(k, "char", "?"))
                                      for k in keys) for keys in parsed)
        return ToolResult(ok=True, text=f"pressed {pressed}" +
                          (f" ({len(parsed)} combos)" if len(parsed) > 1 else ""))


def _press(kb: Any, keys: list[Any]) -> None:
    for key in keys[:-1]:
        kb.press(key)
    kb.press(keys[-1])
    kb.release(keys[-1])
    for key in reversed(keys[:-1]):
        kb.release(key)

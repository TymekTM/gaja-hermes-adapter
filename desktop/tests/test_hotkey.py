"""Hotkey parsing is testable without a display; the pynput listener itself needs one."""

from gaja_desktop.hotkey import Hotkey, hotkey_binding


def test_binding_parsing() -> None:
    assert hotkey_binding("ctrl+alt+g") == "<ctrl>+<alt>+g"
    assert hotkey_binding("ctrl+shift+esc") == "<ctrl>+<shift>+<esc>"
    assert hotkey_binding("win+r") == "<cmd>+r"
    assert hotkey_binding("ctrl+alt+f9") == "<ctrl>+<alt>+<f9>"
    assert hotkey_binding("altgr+a") == "<alt_gr>+a"
    assert hotkey_binding("enter") == "<enter>"
    assert hotkey_binding("") is None
    assert hotkey_binding("ctrl+notakey") is None
    assert hotkey_binding("++") is None


def test_hotkey_reports_unparseable_combo() -> None:
    hotkey = Hotkey("ctrl+notakey", lambda: None)
    assert hotkey.available is False
    assert hotkey.start() is False
    assert "cannot parse" in (hotkey.error or "")

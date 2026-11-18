"""caps.ptt / caps.pc parsing and validation (protocol v1.14, docs/DESKTOP.md)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gaja.devices.audio import PC_TOOLS, DeviceCaps, PcCaps


def test_defaults_ptt_off_pc_none() -> None:
    caps = DeviceCaps()
    assert caps.ptt is False
    assert caps.pc is None


def test_pc_caps_full_valid_hello() -> None:
    caps = DeviceCaps.model_validate(
        {
            "ptt": True,
            "pc": {
                "os": "windows",
                "host": "TYMEK-PC",
                "tools": ["shell", "open", "screenshot", "type", "keys", "mouse", "clipboard"],
            },
        }
    )
    assert caps.ptt is True
    assert caps.pc is not None
    assert caps.pc.os == "windows"
    assert caps.pc.host == "TYMEK-PC"
    assert tuple(caps.pc.tools) == PC_TOOLS


@pytest.mark.parametrize("os_name", ["windows", "linux", "macos"])
def test_pc_caps_os_enum(os_name: str) -> None:
    caps = PcCaps.model_validate({"os": os_name, "host": "h", "tools": []})
    assert caps.os == os_name


@pytest.mark.parametrize("bad_os", ["Windows", "android", "", "win", None, 3])
def test_pc_caps_os_outside_enum_is_rejected(bad_os: object) -> None:
    with pytest.raises(ValidationError):
        PcCaps.model_validate({"os": bad_os, "host": "h", "tools": []})


def test_pc_caps_host_longer_than_64_chars_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PcCaps.model_validate({"os": "linux", "host": "x" * 65, "tools": []})
    assert PcCaps.model_validate({"os": "linux", "host": "x" * 64, "tools": []}).host == "x" * 64


def test_pc_caps_unknown_tool_names_are_dropped_not_rejected() -> None:
    caps = PcCaps.model_validate({"os": "linux", "host": "h", "tools": ["shell", "reboot", "shell", 7, "open"]})
    assert caps.tools == ["shell", "open"]  # order kept, duplicates, unknown names and non-strings dropped


def test_pc_caps_empty_tools_list_is_allowed() -> None:
    """A device may accept caps.pc but advertise nothing (its tray switch is off): no tool is callable."""
    caps = PcCaps.model_validate({"os": "macos", "host": "h", "tools": ["everything-else"]})
    assert caps.tools == []


def test_many_unknown_tools_from_a_newer_client_do_not_reject_the_hello() -> None:
    names = ["shell", "screenshot", "shell"] + [f"future{i}" for i in range(30)]
    caps = DeviceCaps.model_validate({"pc": {"os": "windows", "host": "h", "tools": names}})
    assert caps.pc is not None and caps.pc.tools == ["shell", "screenshot"]

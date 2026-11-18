"""Config load/save round trip (no real user config is ever touched: every test uses a tmp path)."""

from dataclasses import fields

import pytest

from gaja_desktop.config import ALL_TOOLS, DesktopConfig, load_config, save_config


def test_defaults() -> None:
    cfg = DesktopConfig()
    assert cfg.url == "ws://127.0.0.1:8792/device"
    assert cfg.device_id == "pc-desktop"
    assert cfg.token == ""
    assert cfg.outputs == [""]
    assert cfg.out_rates[0] == 48000  # 48000 offered first (docs/DESKTOP.md hello example)
    assert cfg.wake_enabled and cfg.preroll_ms == 1500
    assert cfg.hotkey == "ctrl+alt+g"
    assert sorted(cfg.tools) == sorted(ALL_TOOLS)


def test_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "cfg" / "desktop.toml"
    cfg = DesktopConfig()
    cfg.url = "wss://gaja.example.com/device"
    cfg.device_id = "pc-tymek"
    cfg.token = "secret-token"
    cfg.input = "loopback:Speakers"
    cfg.outputs = ["CABLE Input", "Speakers"]
    cfg.wake_enabled = False
    cfg.preroll_ms = 800
    cfg.hotkey = "ctrl+shift+f9"
    cfg.tools = ["shell", "screenshot"]
    save_config(cfg, path)
    assert path.exists()
    loaded = load_config(path)
    for f in fields(DesktopConfig):
        assert getattr(loaded, f.name) == getattr(cfg, f.name), f.name


def test_missing_file_gives_defaults(tmp_path: pytest.TempPathFactory) -> None:
    loaded = load_config(tmp_path / "does_not_exist.toml")
    assert loaded == DesktopConfig().sanitised()


def test_broken_file_gives_defaults(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "desktop.toml"
    path.write_text("not [ valid toml", encoding="utf-8")
    assert load_config(path) == DesktopConfig().sanitised()


def test_unknown_keys_and_partial_file(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "desktop.toml"
    path.write_text(
        '[device]\nurl = "ws://x/device"\nsurprise = 1\n[tools]\nenabled = ["shell", 42, "nope"]\n',
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.url == "ws://x/device"
    assert cfg.device_id == "pc-desktop"  # default kept
    assert cfg.tools == ["shell"]  # non-string and unknown entries dropped


def test_sanitised_fixes_garbage() -> None:
    cfg = DesktopConfig()
    cfg.tools = ["shell", "bogus"]
    cfg.outputs = []
    cfg.out_rates = [0, 99999, 24000]
    cfg.preroll_ms = 99_999
    cfg.device_id = ""
    fixed = cfg.sanitised()
    assert fixed.tools == ["shell"]
    assert fixed.outputs == [""]
    assert fixed.out_rates == [24000]
    assert fixed.preroll_ms == 5000
    assert fixed.device_id == "pc-desktop"

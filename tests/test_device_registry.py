from __future__ import annotations

import os
import stat

import pytest

from gaja.devices.registry import DeviceRegistry, append_device, generate_token

TOKEN = "t" * 32


def write(path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o600)


def test_missing_wrong_and_unknown_tokens_are_refused(tmp_path) -> None:
    path = tmp_path / "devices.toml"
    write(path, f'[devices.atom-1]\ntoken = "{TOKEN}"\nname = "Salon"\n')
    registry = DeviceRegistry(path)
    info = registry.authenticate("atom-1", TOKEN)
    assert info is not None and info.id == "atom-1" and info.name == "Salon"
    assert registry.authenticate("atom-1", None) is None
    assert registry.authenticate("atom-1", "") is None
    assert registry.authenticate("atom-1", "x" * 32) is None
    assert registry.authenticate("atom-1", TOKEN + "x") is None
    assert registry.authenticate("atom-2", TOKEN) is None
    assert registry.authenticate(None, TOKEN) is None
    assert registry.authenticate("atom-1", 12345) is None


def test_missing_file_means_nobody_connects(tmp_path) -> None:
    registry = DeviceRegistry(tmp_path / "nope.toml")
    assert registry.authenticate("atom-1", TOKEN) is None
    assert registry.count() == 0


def test_disabled_and_weak_entries_are_skipped(tmp_path) -> None:
    path = tmp_path / "devices.toml"
    write(
        path,
        f'[devices.off]\ntoken = "{TOKEN}"\nenabled = false\n'
        '[devices.weak]\ntoken = "short"\n'
        f'[devices."bad id!"]\ntoken = "{TOKEN}"\n'
        f'[devices.ok]\ntoken = "{TOKEN}"\n',
    )
    registry = DeviceRegistry(path)
    assert registry.count() == 1
    assert registry.authenticate("ok", TOKEN) is not None
    assert registry.authenticate("off", TOKEN) is None


def test_file_is_reloaded_when_it_changes(tmp_path) -> None:
    path = tmp_path / "devices.toml"
    write(path, f'[devices.a]\ntoken = "{TOKEN}"\n')
    registry = DeviceRegistry(path)
    assert registry.authenticate("b", "y" * 32) is None
    write(path, f'[devices.a]\ntoken = "{TOKEN}"\n[devices.b]\ntoken = "{"y" * 32}"\n')
    os.utime(path, ns=(1, 2_000_000_000))
    assert registry.authenticate("b", "y" * 32) is not None


def test_broken_file_keeps_previous_entries(tmp_path) -> None:
    path = tmp_path / "devices.toml"
    write(path, f'[devices.a]\ntoken = "{TOKEN}"\n')
    registry = DeviceRegistry(path)
    assert registry.authenticate("a", TOKEN) is not None
    write(path, "this is [not toml")
    os.utime(path, ns=(1, 3_000_000_000))
    assert registry.authenticate("a", TOKEN) is not None


def test_append_device_creates_a_private_file_and_refuses_duplicates(tmp_path) -> None:
    path = tmp_path / "sub" / "devices.toml"
    token = generate_token()
    append_device(path, "atom-1", token, "Salon \"A\"")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert DeviceRegistry(path).authenticate("atom-1", token) is not None
    with pytest.raises(ValueError):
        append_device(path, "atom-1", generate_token())
    with pytest.raises(ValueError):
        append_device(path, "bad id", generate_token())
    assert len(generate_token()) >= 32

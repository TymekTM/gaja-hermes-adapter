"""Host-side checks of the firmware 0.3.8 button barge-in and the pinned mesh node (source inspection).

gaja_device.cpp cannot be compiled on the host (ESP-IDF/ESPHome toolchain), so — like
test_firmware_037.py — these tests pin the properties the 0.3.8 changes must keep by reading the
sources: the button only barges in while PREPARE_PLAY/PLAYING, tells the server
interrupt{reason:"button"} BEFORE cancelling locally, returns to idle through cancel_playback_
(dropping a queued follow-up), and the wifi config pins the good mesh node ahead of a bssid-less
fallback (a hard pin would take the device offline when that node is down).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE_CPP = ROOT / "firmware" / "esphome" / "components" / "gaja_device" / "gaja_device.cpp"
DEVICE_H = ROOT / "firmware" / "esphome" / "components" / "gaja_device" / "gaja_device.h"
YAML = ROOT / "firmware" / "esphome" / "atom-voices3r-gaja.yaml"


def test_firmware_version_is_038_everywhere() -> None:
    """fw_version_ (hello `fw`) and the yaml firmware_version move together."""
    header = DEVICE_H.read_text(encoding="utf-8")
    yaml = YAML.read_text(encoding="utf-8")
    assert 'fw_version_{"gaja-esphome/0.3.8"}' in header
    assert 'firmware_version: "gaja-esphome/0.3.8"' in yaml
    # The yaml comment chain must still explain 0.3.7 (the chain is extended, not replaced).
    assert "0.3.7: the duplex speaker plays a held partial 20 ms chunk" in yaml


def test_button_interrupt_only_in_playback_and_tells_the_server_first() -> None:
    """request_button_interrupt: PREPARE_PLAY/PLAYING only, connected+ready guard, interrupt
    {reason:button} sent before cancel_playback_ (which must be the route back to idle)."""
    text = DEVICE_CPP.read_text(encoding="utf-8")
    fn = text[text.index("void GajaDevice::request_button_interrupt()") : text.index("}  // namespace")]
    assert "GajaState::PREPARE_PLAY" in fn and "GajaState::PLAYING" in fn
    assert re.search(r"state != GajaState::PREPARE_PLAY && state != GajaState::PLAYING", fn)
    # The same connected/ready guard a wake barge-in takes (the teardown path owns the state).
    assert "ws_connected_.load()" in fn and "server_ready_.load()" in fn
    assert 'root["type"] = "interrupt";' in fn and 'root["reason"] = "button";' in fn
    # Tell the server before the local cancel: it must stop streaming the rest of the turn.
    assert fn.index("send_text_") < fn.index("cancel_playback_(")
    assert 'cancel_playback_("button")' in fn  # -> IDLE, wake word back on, follow_up dropped


def test_cancel_playback_drops_the_queued_follow_up() -> None:
    """No follow-up window after a button cancel: cancel_playback_ clears the flag enter_idle_
    would otherwise act on."""
    text = DEVICE_CPP.read_text(encoding="utf-8")
    fn = text[text.index("void GajaDevice::cancel_playback_(") : text.index("void GajaDevice::resume_live_streaming_")]
    assert "follow_up_received_ = false;" in fn and "enter_idle_(false)" in fn


def test_is_playing_covers_exactly_the_playback_states() -> None:
    """The yaml branch condition: PREPARE_PLAY or PLAYING -- nothing else barges in, nothing else
    is excluded from the idle beep."""
    header = DEVICE_H.read_text(encoding="utf-8")
    fn = header[header.index("bool is_playing() const") :]
    fn = fn[: fn.index("}")]
    assert "PREPARE_PLAY" in fn and "PLAYING" in fn
    assert "LINGER" not in fn  # LINGER is post-playback: an idle-ish press, no interrupt


def test_yaml_button_routes_playing_to_the_interrupt_and_idle_to_the_beep() -> None:
    yaml = YAML.read_text(encoding="utf-8")
    button = yaml[yaml.index("binary_sensor:") : yaml.index("\nscript:")]
    assert "is_playing()" in button and "request_button_interrupt()" in button
    assert "id: play_sound" in button  # else branch: the manual beep (idle only, as before)
    # The interrupt branch must come first: after a cancel the device IS idle, and the beep
    # would play exactly when the user asked for silence.
    assert button.index("request_button_interrupt()") < button.index("id: play_sound")


def test_yaml_pins_the_good_mesh_node_with_a_bssid_less_fallback() -> None:
    """networks: pinned bssid (the clean -39 dBm node) first, then the same ssid without a bssid
    as the fallback; power_save_mode/post_connect_roaming/ap stay as they were."""
    yaml = YAML.read_text(encoding="utf-8")
    wifi = yaml[yaml.index("\nwifi:\n") : yaml.index("\nweb_server:\n")]
    assert 'bssid: "D8:32:14:47:E7:E2"' in wifi
    assert "priority: 1" in wifi and "priority: 0" in wifi
    networks = wifi[wifi.index("networks:") :]
    assert networks.count("ssid: !secret wifi_ssid") == 2  # the fallback has no bssid
    assert networks.count("bssid:") == 1
    assert "power_save_mode: none" in wifi and "post_connect_roaming: false" in wifi
    assert "\n  ap:" in wifi  # the recovery hotspot stays

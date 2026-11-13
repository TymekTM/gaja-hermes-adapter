"""Host-side checks of firmware 0.3.10 Wi-Fi event journal integration (source inspection)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE_CPP = ROOT / "firmware" / "esphome" / "components" / "gaja_device" / "gaja_device.cpp"
DEVICE_H = ROOT / "firmware" / "esphome" / "components" / "gaja_device" / "gaja_device.h"
YAML = ROOT / "firmware" / "esphome" / "atom-voices3r-gaja.yaml"


def test_firmware_0310_version_and_changelog_chain() -> None:
    header = DEVICE_H.read_text(encoding="utf-8")
    yaml = YAML.read_text(encoding="utf-8")
    in_header = re.search(r'fw_version_\{"gaja-esphome/(\d+)\.(\d+)\.(\d+)"\}', header)
    in_yaml = re.search(r'firmware_version: "gaja-esphome/(\d+)\.(\d+)\.(\d+)"', yaml)
    assert in_header and in_yaml
    assert in_header.groups() == in_yaml.groups()
    assert tuple(map(int, in_yaml.groups())) >= (0, 3, 10)
    assert "0.3.10:" in yaml and "0.3.9:" in yaml and "0.3.8:" in yaml


def test_rtc_storage_and_hello_capability() -> None:
    cpp = DEVICE_CPP.read_text(encoding="utf-8")
    assert re.search(r"static\s+RTC_NOINIT_ATTR\s+WifiJournalStorage<GajaDevice::WIFI_JOURNAL_CAPACITY>", cpp)
    hello_fn = cpp[cpp.index("void GajaDevice::send_hello_()") : cpp.index("bool GajaDevice::send_text_(")]
    assert 'caps["wifi_journal"] = true;' in hello_fn


def test_wifi_event_handler_is_non_allocating_and_synchronized() -> None:
    cpp = DEVICE_CPP.read_text(encoding="utf-8")
    fn = cpp[cpp.index("void GajaDevice::wifi_event_handler_(") : cpp.index("void GajaDevice::flush_wifi_events_(")]
    assert "ESP_LOG" not in fn
    assert "build_json" not in fn
    assert "std::string" not in fn
    assert "portENTER_CRITICAL" in fn and "portEXIT_CRITICAL" in fn
    assert "WIFI_EVENT_STA_DISCONNECTED" in fn and "WIFI_EVENT_STA_CONNECTED" in fn
    assert "IP_EVENT_STA_GOT_IP" in fn and "IP_EVENT_STA_LOST_IP" in fn


def test_flush_wifi_events_respects_idle_writability_and_delivery_ack() -> None:
    cpp = DEVICE_CPP.read_text(encoding="utf-8")
    fn = cpp[cpp.index("void GajaDevice::flush_wifi_events_(") : cpp.index("void GajaDevice::log_link_(")]
    assert "GajaState::IDLE" in fn
    assert "LOG_FLUSH_IDLE_MS" in fn
    assert "ws_connected_.load()" in fn and "server_ready_.load()" in fn and "ws_writable_()" in fn
    assert "esp_websocket_client_send_text" in fn
    assert "mark_delivered(snap)" in fn
    assert fn.index("esp_websocket_client_send_text") < fn.index("mark_delivered(snap)")

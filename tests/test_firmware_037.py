"""Host-side checks of the firmware 0.3.7 fixes (source inspection).

The fixes live in gaja_device.cpp / gaja_speaker.cpp, which need the ESP-IDF/ESPHome toolchain
and cannot be compiled on the host (unlike the pure headers the other test_firmware_*.py build).
These tests pin the properties the fixes must keep, the way test_device_aec.py pins the v1.11
frame-tag contract: a partial speaker chunk is held at most ONE iteration, a listen_end during
playback drops the queued follow-up, FOLLOWUP has an absolute hard cap, and the blocked-uplink
log no longer prints "blocked 0 ms" spam with useless errno/SO_SNDBUF fields.

The 0.3.6 incident they answer: Crab Rave (18 453 734 B at 48 kHz, 614 B past the last whole
1920 B chunk) stalled the speaker 8 s (tail cut by the stall watchdog); the follow-up window
then streamed 31.5 MB of mic audio for ~30 min with the wake word off.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE_CPP = ROOT / "firmware" / "esphome" / "components" / "gaja_device" / "gaja_device.cpp"
SPEAKER_CPP = ROOT / "firmware" / "esphome" / "components" / "gaja_device" / "speaker" / "gaja_speaker.cpp"
DEVICE_H = ROOT / "firmware" / "esphome" / "components" / "gaja_device" / "gaja_device.h"
YAML = ROOT / "firmware" / "esphome" / "atom-voices3r-gaja.yaml"


def test_speaker_holds_a_partial_chunk_at_most_one_iteration() -> None:
    """tx_task_ must not hold a partial 20 ms chunk until finish() (the 0.3.6 deadlock)."""
    text = SPEAKER_CPP.read_text(encoding="utf-8")
    task = text[text.index("void GajaDuplexSpeaker::tx_task_") :]
    hold = re.search(r"if \(n < DuplexBus::TX_CHUNK_BYTES(.*?)\{\n(.*?)\n    \}", task, re.S)
    assert hold, "the partial-chunk hold branch must stay in tx_task_"
    condition, body = hold.group(1), hold.group(2)
    # The hold needs the one-iteration flag, or the chunk sits until finish() and the reply stalls.
    assert "held_partial" in condition and "COMMAND_STOP_GRACEFUL" in condition
    assert re.search(r"held_partial = true;", body)
    # And the flag resets when the chunk is played (or the ring empties / a stop drops the queue).
    assert len(re.findall(r"held_partial = false;", task)) >= 3
    # The hold must NOT consume the ring: the chunk plays (zero-padded by write_chunk_) afterwards.
    assert "tx_ring_.advance" not in body


def test_followup_window_has_a_hard_cap() -> None:
    """FOLLOWUP must not outlive FOLLOWUP_HARD_CAP_MS measured from state_entered_ms_."""
    text = DEVICE_CPP.read_text(encoding="utf-8")
    assert re.search(r"FOLLOWUP_HARD_CAP_MS = 30000;", text)
    case = text[text.index("case GajaState::FOLLOWUP: {") :]
    case = case[: case.index("case GajaState::WAIT_REPLY:")]  # the loop()'s FOLLOWUP block
    assert "FOLLOWUP_HARD_CAP_MS" in case
    assert re.search(r"elapsed_ms\(now, this->state_entered_ms_\) > FOLLOWUP_HARD_CAP_MS", case)
    assert "close_mic_stream_()" in case and "enter_idle_()" in case  # the cap closes the window


def test_listen_end_during_playback_drops_the_queued_follow_up() -> None:
    """A listen_end in PLAYING/PREPARE_PLAY closes the follow-up window nobody else will close."""
    text = DEVICE_CPP.read_text(encoding="utf-8")
    handler = text[text.index("void GajaDevice::on_listen_end_()") : text.index("void GajaDevice::on_follow_up_")]
    assert "case GajaState::PLAYING:" in handler and "case GajaState::PREPARE_PLAY:" in handler
    play = handler[handler.index("case GajaState::PLAYING:") :]
    play = play[: play.index("default:")]
    assert re.search(r"follow_up_received_ = false;", play)
    assert "ESP_LOGI" in play  # INFO, not the debug "ignored" line of the default branch
    # finish_reply_ is the only reader of the flag; enter_idle_/cancel/reconnect clear it too.
    assert "this->follow_up_received_" not in re.sub(
        r"follow_up_received_ = (true|false)", "", text
    ).replace("if (this->follow_up_received_)", "")


def test_blocked_uplink_log_is_quiet_and_honest() -> None:
    """log_send_blocked_: first line at >= 500 ms, rate-limited across blocks, no errno/SO_SNDBUF."""
    text = DEVICE_CPP.read_text(encoding="utf-8")
    assert re.search(r"SEND_BLOCKED_FIRST_LOG_MS = 500;", text)
    fn = text[text.index("void GajaDevice::log_send_blocked_") : text.index("void GajaDevice::close_mic_stream_")]
    for gone in ("SO_SNDBUF", "so_error", "sndbuf", "errno"):
        assert gone not in fn, gone  # measured useless on lwIP (SO_SNDBUF rc -1, stale errno)
    # The rate-limit anchor must survive a successful send, or every new block logs at 0 ms.
    assert "this->send_blocked_log_ms_ = 0;" not in text
    # Nothing is printed at block start any more: send_frame_ arms the timestamp silently.
    send_frame = text[text.index("bool GajaDevice::send_frame_") : text.index("/// Conclusive state")]
    assert "log_send_blocked_(now, false)" in send_frame and "log_send_blocked_(now, true)" in send_frame


def test_firmware_version_is_037_everywhere() -> None:
    """fw_version_ (hello `fw`) and the yaml firmware_version move together."""
    header = DEVICE_H.read_text(encoding="utf-8")
    yaml = YAML.read_text(encoding="utf-8")
    assert 'fw_version_{"gaja-esphome/0.3.7"}' in header
    assert 'firmware_version: "gaja-esphome/0.3.7"' in yaml
    # The yaml comment chain must still explain 0.3.6 (the chain is extended, not replaced).
    assert "0.3.6: the uplink rides out a stalled server" in yaml

"""Server-side Wi-Fi event journal ingestion and diagnostics store (protocol v1.13).

Firmware >= 0.3.10 retains Wi-Fi / IP / boot milestones in RTC memory across disconnects and
software resets, then uploads undelivered batches as ``wifi_events`` once WebSocket ``ready`` is
complete and the device is idle. Older firmware never sends this message and never sets
``caps.wifi_journal``; both cases remain completely silent.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

MAX_EVENTS_PER_DEVICE = 100
MAX_BATCH_EVENTS = 64
U32_MAX = 2**32 - 1  # boot, uptime_ms, ms and dropped are uint32 counters on the device
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

# ESP-IDF wifi_err_reason_t (esp_wifi_types_generic.h).
WIFI_ERR_REASONS: dict[int, str] = {
    1: "UNSPECIFIED",
    2: "AUTH_EXPIRE",
    3: "AUTH_LEAVE",
    4: "ASSOC_EXPIRE",
    5: "ASSOC_TOOMANY",
    6: "NOT_AUTHED",
    7: "NOT_ASSOCED",
    8: "ASSOC_LEAVE",
    9: "ASSOC_NOT_AUTHED",
    10: "DISASSOC_PWRCAP_BAD",
    11: "DISASSOC_SUPCHAN_BAD",
    12: "BSS_TRANSITION_DISASSOC",
    13: "IE_INVALID",
    14: "MIC_FAILURE",
    15: "4WAY_HANDSHAKE_TIMEOUT",
    16: "GROUP_KEY_UPDATE_TIMEOUT",
    17: "IE_IN_4WAY_DIFFERS",
    18: "GROUP_CIPHER_INVALID",
    19: "PAIRWISE_CIPHER_INVALID",
    20: "AKMP_INVALID",
    21: "UNSUPP_RSN_IE_VERSION",
    22: "INVALID_RSN_IE_CAP",
    23: "802_1X_AUTH_FAILED",
    24: "CIPHER_SUITE_REJECTED",
    25: "TDLS_PEER_UNREACHABLE",
    26: "TDLS_UNSPECIFIED",
    27: "SSP_REQUESTED_DISASSOC",
    28: "NO_SSP_ROAMING_AGREEMENT",
    29: "BAD_CIPHER_OR_AKM",
    30: "NOT_AUTHORIZED_THIS_LOCATION",
    31: "SERVICE_CHANGE_PERCLUDES_TS",
    32: "UNSPECIFIED_QOS",
    33: "NOT_ENOUGH_BANDWIDTH",
    34: "MISSING_ACKS",
    35: "EXCEEDED_TXOP",
    36: "STA_LEAVING",
    37: "END_BA",
    38: "UNKNOWN_BA",
    39: "TIMEOUT",
    46: "PEER_INITIATED",
    47: "AP_INITIATED",
    48: "INVALID_FT_ACTION_FRAME_COUNT",
    49: "INVALID_PMKID",
    50: "INVALID_MDE",
    51: "INVALID_FTE",
    67: "TRANSMISSION_LINK_ESTABLISH_FAILED",
    68: "ALTERATIVE_CHANNEL_OCCUPIED",
    200: "BEACON_TIMEOUT",
    201: "NO_AP_FOUND",
    202: "AUTH_FAIL",
    203: "ASSOC_FAIL",
    204: "HANDSHAKE_TIMEOUT",
    205: "CONNECTION_FAIL",
    206: "AP_TSF_RESET",
    207: "ROAMING",
    208: "ASSOC_COMEBACK_TIME_TOO_LONG",
    209: "SA_QUERY_TIMEOUT",
    210: "NO_AP_FOUND_W_COMPATIBLE_SECURITY",
    211: "NO_AP_FOUND_IN_AUTHMODE_THRESHOLD",
    212: "NO_AP_FOUND_IN_RSSI_THRESHOLD",
}

RESET_REASONS: dict[int, str] = {
    0: "unknown",
    1: "power-on",
    2: "external",
    3: "software",
    4: "panic",
    5: "interrupt_watchdog",
    6: "task_watchdog",
    7: "watchdog",
    8: "deep_sleep",
    9: "brownout",
    10: "sdio",
    11: "usb",
    12: "jtag",
    13: "efuse",
    14: "power_glitch",
    15: "cpu_lockup",
}

# Per-device diagnostics store: device_id -> {"boot", "uptime_ms", "reset_reason", "dropped", "updated_ts", "events": deque}
WIFI_EVENTS: dict[str, dict[str, Any]] = {}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_u32(value: Any) -> bool:
    return _is_int(value) and 0 <= value <= U32_MAX


def _clean_str(value: Any, max_len: int = 32) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _CONTROL.sub("", value).strip()[:max_len]
    return cleaned or None


def wifi_reason_name(kind: str, reason: int | None) -> str | None:
    if reason is None:
        return None
    if kind == "disconnected":
        return WIFI_ERR_REASONS.get(reason, f"REASON_{reason}")
    if kind == "boot":
        return RESET_REASONS.get(reason, f"RESET_{reason}")
    if kind == "got_ip":
        return "ip_changed" if reason else "ip_unchanged"
    if kind == "scan_done":
        return "ok" if reason == 0 else "failed"
    return str(reason)


def get_wifi_events_log() -> dict[str, dict[str, Any]]:
    """Return a JSON-serializable snapshot of recent Wi-Fi events per device."""
    return {
        dev_id: {
            "boot": entry["boot"],
            "uptime_ms": entry["uptime_ms"],
            "reset_reason": entry["reset_reason"],
            "dropped": entry["dropped"],
            "updated_ts": entry["updated_ts"],
            "events": list(entry["events"]),
        }
        for dev_id, entry in WIFI_EVENTS.items()
    }


def record_wifi_events(device_id: str, msg: dict[str, Any]) -> None:
    """Validate, log at INFO, and retain a ``wifi_events`` batch from ``device_id``."""
    boot = msg.get("boot")
    uptime_ms = msg.get("uptime_ms")
    events_raw = msg.get("events")
    if not (_is_u32(boot) and _is_u32(uptime_ms) and isinstance(events_raw, list)):
        log.debug("device %s sent malformed wifi_events payload; ignored", device_id)
        return

    reset_reason = _clean_str(msg.get("reset_reason"), 32) or "unknown"
    dropped_val = msg.get("dropped", 0)
    dropped = dropped_val if _is_u32(dropped_val) else 0

    recv_ts = time.time()
    store = WIFI_EVENTS.get(device_id)
    if store is None:
        store = {
            "boot": boot,
            "uptime_ms": uptime_ms,
            "reset_reason": reset_reason,
            "dropped": 0,
            "updated_ts": recv_ts,
            "events": deque(maxlen=MAX_EVENTS_PER_DEVICE),
        }
        WIFI_EVENTS[device_id] = store
    else:
        store["boot"] = boot
        store["uptime_ms"] = uptime_ms
        store["reset_reason"] = reset_reason
        store["updated_ts"] = recv_ts
    store["dropped"] += dropped

    if dropped > 0:
        log.warning(
            "device %s Wi-Fi journal overwrote %d event(s) while offline (boot=%d reset=%s)",
            device_id,
            dropped,
            boot,
            reset_reason,
        )

    for item in events_raw[:MAX_BATCH_EVENTS]:
        if not isinstance(item, dict):
            continue
        ev_ms = item.get("ms")
        ev_kind = _clean_str(item.get("kind"), 24)
        if not (_is_u32(ev_ms) and ev_kind):
            continue
        ev_boot_raw = item.get("boot", boot)
        if not _is_u32(ev_boot_raw):
            continue
        ev_boot = ev_boot_raw

        ev_reason = item.get("reason") if _is_int(item.get("reason")) else None
        ev_bssid = _clean_str(item.get("bssid"), 24)
        ev_channel = item.get("channel") if _is_int(item.get("channel")) else None
        ev_rssi = item.get("rssi") if _is_int(item.get("rssi")) else None
        rname = wifi_reason_name(ev_kind, ev_reason)

        if ev_boot == boot:
            est_ts: float | None = recv_ts - (uptime_ms - ev_ms) / 1000.0
            when = datetime.fromtimestamp(est_ts).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        else:
            est_ts = None
            when = f"previous boot {ev_boot} (+{ev_ms}ms)"

        parts = [f"device {device_id} wifi_event at={when} kind={ev_kind}"]
        if ev_reason is not None:
            if rname and rname != str(ev_reason):
                parts.append(f"reason={rname}({ev_reason})")
            else:
                parts.append(f"reason={ev_reason}")
        if ev_bssid is not None:
            parts.append(f"bssid={ev_bssid}")
        if ev_channel is not None:
            parts.append(f"ch={ev_channel}")
        if ev_rssi is not None:
            parts.append(f"rssi={ev_rssi}dBm")
        if ev_boot == boot:
            parts.append(f"ms={ev_ms}")
        log.info(" ".join(parts))

        store["events"].append({
            "ms": ev_ms,
            "boot": ev_boot,
            "kind": ev_kind,
            "reason": ev_reason,
            "reason_name": rname,
            "bssid": ev_bssid,
            "channel": ev_channel,
            "rssi": ev_rssi,
            "est_ts": est_ts,
            "when": when,
            "received_ts": recv_ts,
        })

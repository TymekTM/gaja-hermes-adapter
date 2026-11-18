"""Desktop client configuration: a TOML file at %APPDATA%\\Gaja\\desktop.toml (Linux: ~/.config/gaja/desktop.toml).

The file holds the device token, so it must stay private (the file is written with a warning header;
on Windows the user profile is already per-user, on POSIX we chmod 600).

All load/save takes an explicit path (tests pass one under tmp_path); ``default_config_path()`` only
computes the location and never touches the server's own ~/.config/gaja files.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import platformdirs

#: The tools the client knows. A tool is advertised only if it is listed here, it is implemented
#: on this machine (its backend import succeeds) and its backend actually works.
ALL_TOOLS = ["shell", "open", "screenshot", "type", "keys", "mouse", "clipboard"]

#: Offered in the hello; the server picks one (48000 first per docs/DESKTOP.md).
DEFAULT_OUT_RATES = [48000, 24000, 16000]

_TOKEN_WARNING = (
    "# Gaja desktop client configuration.\n"
    "# This file contains the device token: keep it private (it gives voice access to Gaja).\n"
)

_ENV_VAR = "GAJA_DESKTOP_CONFIG"


@dataclass
class DesktopConfig:
    """Everything the client needs. Fields map 1:1 to the settings window."""

    # Device connection (docs/DEVICE.md listener).
    url: str = "ws://127.0.0.1:8792/device"
    device_id: str = "pc-desktop"
    token: str = ""
    lang: str = "pl"
    # Audio (docs/DESKTOP.md): one capture source, one or more playback devices.
    # ``input`` is a device name substring, or "loopback:<output name substring>" to hear what an
    # output plays. "" = the system default microphone.
    input: str = ""
    # Playback device name substrings, e.g. ["CABLE Input", "Speakers"]; "" = system default.
    outputs: list[str] = field(default_factory=lambda: [""])
    out_rates: list[int] = field(default_factory=lambda: list(DEFAULT_OUT_RATES))
    # Half duplex: no wake detection and no mic upload while Gaja's audio plays (default, because the
    # capture source usually hears the playback). Set false only with a headphone/echo-free setup.
    half_duplex: bool = True
    # Wake word ("Hej Gaja", the bundled hej_gaja.tflite) and its pre-roll ring buffer.
    wake_enabled: bool = True
    wake_model: str = ""  # optional path to a model .json (next to the .tflite); "" = bundled
    preroll_ms: int = 1500
    # Push-to-talk global hotkey (pynput combo string) and its enable flag.
    ptt_enabled: bool = True
    hotkey: str = "ctrl+alt+g"
    # Which PC tools are enabled (a subset of ALL_TOOLS). Also gated by what the platform supports.
    tools: list[str] = field(default_factory=lambda: list(ALL_TOOLS))

    def sanitised(self) -> DesktopConfig:
        """A copy with obviously broken values repaired (bad tool names, nonsense preroll, rates)."""
        cfg = dataclasses.replace(self)
        cfg.tools = [t for t in ALL_TOOLS if t in self.tools]
        cfg.outputs = [o for o in self.outputs if isinstance(o, str)] or [""]
        seen: set[int] = set()
        cfg.out_rates = [int(r) for r in self.out_rates
                         if 8000 <= int(r) <= 48000 and not (int(r) in seen or seen.add(int(r)))]
        cfg.out_rates = cfg.out_rates or list(DEFAULT_OUT_RATES)
        cfg.preroll_ms = min(max(0, int(self.preroll_ms)), 5000)
        cfg.device_id = (self.device_id or "pc-desktop").strip()[:64] or "pc-desktop"
        cfg.url = (self.url or DesktopConfig.url).strip()
        cfg.hotkey = (self.hotkey or DesktopConfig.hotkey).strip()
        return cfg


def default_config_path() -> Path:
    """%APPDATA%\\Gaja\\desktop.toml on Windows, ~/.config/gaja/desktop.toml elsewhere.

    ``GAJA_DESKTOP_CONFIG`` overrides the location (useful for tests and for a portable install).
    """
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(platformdirs.user_config_dir("Gaja", appauthor=False))
    else:
        base = Path(platformdirs.user_config_dir("gaja", appauthor=False))
    return base / "desktop.toml"


def _dump(cfg: DesktopConfig) -> dict:
    """The config as nested TOML tables (grouped, so the file stays readable by hand)."""
    return {
        "device": {"url": cfg.url, "device_id": cfg.device_id, "token": cfg.token, "lang": cfg.lang},
        "audio": {
            "input": cfg.input,
            "outputs": list(cfg.outputs),
            "out_rates": [int(r) for r in cfg.out_rates],
            "half_duplex": cfg.half_duplex,
        },
        "wake": {"enabled": cfg.wake_enabled, "model": cfg.wake_model, "preroll_ms": int(cfg.preroll_ms)},
        "ptt": {"enabled": cfg.ptt_enabled, "hotkey": cfg.hotkey},
        "tools": {"enabled": list(cfg.tools)},
    }


def save_config(cfg: DesktopConfig, path: Path) -> Path:
    """Write the config atomically; the token warning goes into a header comment. Returns the path."""
    import tomli_w  # noqa: PLC0415 (lazy: only writing needs it, reading is stdlib tomllib)

    path.parent.mkdir(parents=True, exist_ok=True)
    body = tomli_w.dumps(_dump(cfg))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_TOKEN_WARNING + body, encoding="utf-8")
    os.replace(tmp, path)
    if platform.system() != "Windows":
        try:
            os.chmod(path, 0o600)  # the token lives here
        except OSError:
            pass
    return path


def load_config(path: Path) -> DesktopConfig:
    """Load the config; a missing or partly broken file yields defaults for whatever is absent."""
    cfg = DesktopConfig()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, tomllib.TOMLDecodeError, UnicodeDecodeError, OSError):
        return cfg
    if not isinstance(data, dict):
        return cfg
    device = data.get("device") or {}
    audio = data.get("audio") or {}
    wake = data.get("wake") or {}
    ptt = data.get("ptt") or {}
    tools = data.get("tools") or {}
    cfg.url = _str(device.get("url"), cfg.url)
    cfg.device_id = _str(device.get("device_id"), cfg.device_id)
    cfg.token = _str(device.get("token"), cfg.token)
    cfg.lang = _str(device.get("lang"), cfg.lang)
    cfg.input = _str(audio.get("input"), cfg.input)
    outputs = audio.get("outputs")
    if isinstance(outputs, list):
        cfg.outputs = [o for o in outputs if isinstance(o, str)]
    rates = audio.get("out_rates")
    if isinstance(rates, list):
        cfg.out_rates = [r for r in rates if isinstance(r, int) and not isinstance(r, bool)]
    cfg.half_duplex = _bool(audio.get("half_duplex"), cfg.half_duplex)
    cfg.wake_enabled = _bool(wake.get("enabled"), cfg.wake_enabled)
    cfg.wake_model = _str(wake.get("model"), cfg.wake_model)
    cfg.preroll_ms = _int(wake.get("preroll_ms"), cfg.preroll_ms)
    cfg.ptt_enabled = _bool(ptt.get("enabled"), cfg.ptt_enabled)
    cfg.hotkey = _str(ptt.get("hotkey"), cfg.hotkey)
    enabled = tools.get("enabled")
    if isinstance(enabled, list):
        cfg.tools = [t for t in enabled if isinstance(t, str)]
    return cfg.sanitised()


def _str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default

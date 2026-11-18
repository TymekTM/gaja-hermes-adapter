"""`gaja-desktop` command line:

    gaja-desktop run [--no-tray]   connect as a device (wake word, PTT, tray icon, PC tools)
    gaja-desktop devices           list capture (incl. loopback) and playback devices
    gaja-desktop settings          open the settings window

Config: %APPDATA%\\Gaja\\desktop.toml (Linux: ~/.config/gaja/desktop.toml), see config.py.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys
from typing import Any

from . import FW, __version__
from .config import DesktopConfig, default_config_path, load_config

log = logging.getLogger("gaja_desktop")


def main(argv: list[str] | None = None) -> int:
    from .win import make_dpi_aware

    make_dpi_aware()  # before any tool/tray import: screenshots and mouse must share one pixel space
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="gaja-desktop", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"gaja-desktop {__version__} ({FW})")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="alternative config file (default: %%APPDATA%%\\Gaja\\desktop.toml)")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", parents=[common], help="connect as a Gaja device")
    run_p.add_argument("--no-tray", action="store_true", help="headless: no tray icon, log the state")
    sub.add_parser("devices", parents=[common], help="list capture, loopback and playback devices")
    sub.add_parser("settings", parents=[common], help="open the settings window")
    args = parser.parse_args(argv)

    path = default_config_path() if not getattr(args, "config", None) else pathlib.Path(args.config)
    cfg = load_config(path)
    if args.command == "devices":
        return _cmd_devices()
    if args.command == "settings":
        return _cmd_settings(cfg, path)
    return _cmd_run(cfg, tray=not args.no_tray)


# ---- run ---------------------------------------------------------------------------------


def _cmd_run(cfg: DesktopConfig, *, tray: bool) -> int:
    from .connection import DeviceClient
    from .hotkey import Hotkey

    if not cfg.token:
        log.warning("no device token in the config (%s) — the server will refuse the hello",
                    default_config_path())
    tray_ui: Any = None
    client = DeviceClient(cfg)
    if tray:
        from .tray import Tray

        tray_ui = Tray(client, cfg, on_quit=None)
        client.on_state = tray_ui.update_state
    else:
        client.on_state = lambda state: log.info("state: %s", state)

    async def runner() -> int:
        loop = asyncio.get_running_loop()
        hk: Hotkey | None = None
        if cfg.ptt_enabled:
            # The hotkey callback runs on the pynput thread: marshal into the loop.
            hk = Hotkey(cfg.hotkey, lambda: loop.call_soon_threadsafe(
                lambda: asyncio.create_task(client.push_to_talk())))
            hk.start()
        try:
            await asyncio.create_task(client.run())
        except asyncio.CancelledError:
            pass
        finally:
            if hk is not None:
                hk.stop()
        return 0

    if tray_ui is not None:
        tray_ui.start()  # False on headless machines; the run continues anyway
    try:
        code = asyncio.run(runner())
    except KeyboardInterrupt:
        log.info("interrupted")
        code = 0
    finally:
        client.stop()
        if tray_ui is not None:
            tray_ui.stop()
    return code


# ---- devices -----------------------------------------------------------------------------


def _cmd_devices() -> int:
    from .audio import list_input_devices, list_output_devices

    inputs = list_input_devices()
    print("Capture devices (use a name substring in [audio] input):")
    for dev in inputs:
        kind = "loopback of output" if dev.loopback else "microphone/cable"
        print(f"  [{kind:>19}] {dev.name}  (channels: {dev.channels})")
    print("\nPlayback devices (name substrings in [audio] outputs; name@rate pins the rate):")
    for dev in list_output_devices():
        print(f"  {dev.name}  (channels: {dev.channels})")
    print("\nTip: to hear what an output plays set input = \"loopback:<output name>\".")
    return 0


# ---- settings -----------------------------------------------------------------------------


def _cmd_settings(cfg: DesktopConfig, path: Any) -> int:
    from .settings_ui import SettingsWindow

    result = SettingsWindow(cfg, config_path=path).show()
    if result is None:
        print("cancelled; config unchanged")
        return 1
    print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

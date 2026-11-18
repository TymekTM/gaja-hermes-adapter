"""Tray icon (pystray): state at a glance, mute-listening toggle, "PC actions" switch (reconnects
without `caps.pc`), Settings… (tkinter window), Quit.

`pystray` is imported lazily: `--no-tray` and headless machines never touch it. The icon image is
drawn with PIL at runtime — no assets to bundle.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from .config import DesktopConfig

log = logging.getLogger(__name__)

#: internal client state -> (tooltip word, dot colour)
STATE_COLORS = {
    "connecting": ("connecting", (250, 179, 135)),
    "idle": ("connected", (166, 227, 161)),
    "listening": ("listening", (137, 180, 250)),
    "follow_up": ("listening", (137, 180, 250)),
    "speaking": ("speaking", (250, 227, 137)),
    "error": ("error", (243, 139, 168)),
}


def make_icon_image(state: str = "connecting", muted: bool = False) -> Any:
    """A 64 px PIL image: coloured dot (hollow when muted), 'G' in the middle."""
    from PIL import Image, ImageDraw  # noqa: PLC0415 (lazy)

    _, color = STATE_COLORS.get(state, STATE_COLORS["error"])
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    box = (4, 4, 60, 60)
    if muted:
        draw.ellipse(box, outline=color, width=6)
    else:
        draw.ellipse(box, fill=color)
    draw.text((24, 18), "G", fill=(30, 30, 30, 255))
    return img


class Tray:
    """Owns the pystray icon; `run()` blocks (call it from its own thread)."""

    def __init__(self, client: Any, cfg: DesktopConfig, on_quit: Callable[[], None] | None = None) -> None:
        self.client = client
        self.cfg = cfg
        self.on_quit = on_quit
        self.icon: Any = None
        self.error: str | None = None
        self._mute_var: Any = None
        self._pc_var: Any = None

    # ---- lifecycle --------------------------------------------------------------------------

    def start(self) -> bool:
        """Build the icon and run it on a daemon thread. False when no tray backend exists."""
        try:
            import pystray  # noqa: PLC0415 (lazy)

            self.icon = pystray.Icon("gaja-desktop", icon=make_icon_image(),
                                     title="Gaja: starting", menu=self._build_menu(pystray))
            thread = threading.Thread(target=self.icon.run, name="gaja-tray", daemon=True)
            thread.start()
            return True
        except Exception as exc:  # noqa: BLE001 - headless Linux has no tray
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("tray unavailable: %s", self.error)
            return False

    def stop(self) -> None:
        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception:  # noqa: BLE001
                pass
            self.icon = None

    # ---- wiring -----------------------------------------------------------------------------

    def update_state(self, state: str) -> None:
        """The client's on_state callback: refresh tooltip + icon."""
        if self.icon is None:
            return
        label, _ = STATE_COLORS.get(state, (state, (0, 0, 0)))
        muted = getattr(self.client, "muted", False)
        suffix = " (muted)" if muted else ""
        try:
            self.icon.title = f"Gaja: {label}{suffix}"
            self.icon.icon = make_icon_image(state, muted)
        except Exception:  # noqa: BLE001 - a dead tray must not kill the client
            log.exception("tray update failed")

    def _build_menu(self, pystray: Any) -> Any:
        client = self.client
        self._mute_var = pystray.CheckMenuItem(
            "Listen (mic on)", self._toggle_mute, checked=lambda item: not client.muted)
        self._pc_var = pystray.CheckMenuItem(
            "PC actions (agent tools)", self._toggle_pc, checked=lambda item: client.pc_enabled)
        settings_item = pystray.MenuItem("Settings…", self._open_settings, enabled=True)
        quit_item = pystray.MenuItem("Quit", self._quit)
        return pystray.Menu(self._mute_var, self._pc_var, pystray.Menu.SEPARATOR, settings_item,
                            pystray.Menu.SEPARATOR, quit_item)

    # ---- menu actions (run on the tray thread) ----------------------------------------------

    def _toggle_mute(self, icon: Any, item: Any) -> None:
        self.client.set_muted(not self.client.muted)
        self.update_state(self.client.state)

    def _toggle_pc(self, icon: Any, item: Any) -> None:
        self.client.set_pc_enabled(not self.client.pc_enabled)  # reconnects with new caps
        self.update_state(self.client.state)

    def _open_settings(self, icon: Any, item: Any) -> None:
        from .settings_ui import open_settings_in_thread  # noqa: PLC0415 (lazy)

        open_settings_in_thread(self.cfg)

    def _quit(self, icon: Any, item: Any) -> None:
        log.info("quit from the tray")
        self.client.stop()
        if self.on_quit is not None:
            self.on_quit()
        self.stop()

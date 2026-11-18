"""The settings window (tkinter): server URL, device id, token, audio devices from the live lists,
wake word, hotkey, and the enabled tools. Saves to the config file; the client picks changes up on
its next run (or after `request_reconnect()` for the parts the caller applies live).

Runs in whatever thread calls `show()` — the CLI puts it in the main thread for `gaja-desktop
settings` and in a worker thread from the tray menu.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Callable

try:  # headless boxes may lack tkinter entirely; only the guarded paths need it
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover - depends on the interpreter build
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

from . import __version__
from .config import ALL_TOOLS, DesktopConfig, default_config_path, save_config

log = logging.getLogger(__name__)


class SettingsWindow:
    """One modal-ish window bound to a DesktopConfig."""

    def __init__(self, cfg: DesktopConfig, on_save: Callable[[DesktopConfig], None] | None = None,
                 *, config_path=None) -> None:
        self.cfg = cfg
        self.on_save = on_save
        self.path = config_path or default_config_path()
        self._result: DesktopConfig | None = None
        self._closed = threading.Event()

    # ---- data helpers ------------------------------------------------------------------------

    def _input_choices(self) -> list[str]:
        try:
            from .audio import list_input_devices  # noqa: PLC0415

            devices = list_input_devices()
            plain = [d.name for d in devices if not d.loopback]
            loop = [f"loopback:{d.name}" for d in devices if d.loopback]
            return [""] + plain + loop
        except Exception as exc:  # noqa: BLE001 - no audio stack on this machine
            log.warning("cannot list capture devices: %s", exc)
            return ["", self.cfg.input] if self.cfg.input else [""]

    def _output_choices(self) -> list[str]:
        try:
            from .audio import list_output_devices  # noqa: PLC0415

            return [""] + [d.name for d in list_output_devices()]
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot list playback devices: %s", exc)
            return ["", *self.cfg.outputs]

    # ---- UI ---------------------------------------------------------------------------------

    def show(self) -> DesktopConfig | None:
        """Build and run the window (blocking). Returns the saved config, or None when cancelled."""
        if tk is None:
            raise RuntimeError("tkinter is not available in this Python build")
        root = tk.Tk()
        root.title(f"Gaja desktop settings — {__version__}")
        root.resizable(False, True)
        self._build(root)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()
        return self._result

    def _build(self, root: tk.Tk) -> None:
        cfg = self.cfg
        pad = {"padx": 8, "pady": 4}
        frame = ttk.Frame(root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # device connection
        group = ttk.LabelFrame(frame, text="Device connection", padding=8)
        group.pack(fill=tk.X, **pad)
        url = tk.StringVar(value=cfg.url)
        device_id = tk.StringVar(value=cfg.device_id)
        token = tk.StringVar(value=cfg.token)
        lang = tk.StringVar(value=cfg.lang)
        _row(group, 0, "Server URL", ttk.Entry(group, textvariable=url, width=44))
        _row(group, 1, "Device id", ttk.Entry(group, textvariable=device_id, width=24))
        token_entry = ttk.Entry(group, textvariable=token, width=44, show="•")
        _row(group, 2, "Token", token_entry)
        _row(group, 3, "Language", ttk.Combobox(group, textvariable=lang, width=8, values=["pl", "en", "auto"]))

        # audio
        group = ttk.LabelFrame(frame, text="Audio", padding=8)
        group.pack(fill=tk.X, **pad)
        input_var = tk.StringVar(value=cfg.input)
        input_box = ttk.Combobox(group, textvariable=input_var, width=44, values=self._input_choices())
        _row(group, 0, "Input (mic or loopback:<output>)", input_box)
        outputs_var = tk.StringVar(value=", ".join(cfg.outputs))
        out_box = ttk.Combobox(group, textvariable=outputs_var, width=44,
                               values=self._output_choices())
        _row(group, 1, "Outputs (comma separated, name@rate allowed)", out_box)
        half_duplex = tk.BooleanVar(value=cfg.half_duplex)
        ttk.Checkbutton(group, variable=half_duplex,
                        text="Half duplex (no mic while Gaja speaks — keep on unless headphones)").grid(
            row=2, column=0, columnspan=2, sticky=tk.W, **pad)

        # wake word & PTT
        group = ttk.LabelFrame(frame, text="Wake word and push-to-talk", padding=8)
        group.pack(fill=tk.X, **pad)
        wake_on = tk.BooleanVar(value=cfg.wake_enabled)
        ttk.Checkbutton(group, variable=wake_on, text='Wake word "Hej Gaja"').grid(
            row=0, column=0, sticky=tk.W, **pad)
        preroll = tk.StringVar(value=str(cfg.preroll_ms))
        _row(group, 1, "Pre-roll ms (0..5000)", ttk.Entry(group, textvariable=preroll, width=8))
        ptt_on = tk.BooleanVar(value=cfg.ptt_enabled)
        ttk.Checkbutton(group, variable=ptt_on, text="Push-to-talk hotkey").grid(
            row=2, column=0, sticky=tk.W, **pad)
        hotkey = tk.StringVar(value=cfg.hotkey)
        _row(group, 3, "Hotkey", ttk.Entry(group, textvariable=hotkey, width=16))

        # tools
        group = ttk.LabelFrame(frame, text="PC tools (Hermes runs these without asking!)", padding=8)
        group.pack(fill=tk.X, **pad)
        tool_vars = {name: tk.BooleanVar(value=name in cfg.tools) for name in ALL_TOOLS}
        for i, name in enumerate(ALL_TOOLS):
            ttk.Checkbutton(group, variable=tool_vars[name], text=name).grid(
                row=i // 4, column=i % 4, sticky=tk.W, padx=6, pady=2)

        # buttons
        buttons = ttk.Frame(frame, padding=8)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Save", command=lambda: self._save(
            root, url, device_id, token, lang, input_var, outputs_var, half_duplex, wake_on,
            preroll, ptt_on, hotkey, tool_vars)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side=tk.RIGHT, padx=4)

    def _save(self, root: tk.Tk, url: tk.StringVar, device_id: tk.StringVar, token: tk.StringVar,
              lang: tk.StringVar, input_var: tk.StringVar, outputs_var: tk.StringVar,
              half_duplex: tk.BooleanVar, wake_on: tk.BooleanVar, preroll: tk.StringVar,
              ptt_on: tk.BooleanVar, hotkey: tk.StringVar, tool_vars: dict[str, tk.BooleanVar]) -> None:
        cfg = self.cfg
        cfg.url = url.get().strip() or cfg.url
        cfg.device_id = device_id.get().strip() or cfg.device_id
        cfg.token = token.get().strip()
        cfg.lang = lang.get().strip() or "pl"
        cfg.input = input_var.get().strip()
        cfg.outputs = [part.strip() for part in outputs_var.get().split(",") if part.strip()] or [""]
        cfg.half_duplex = bool(half_duplex.get())
        cfg.wake_enabled = bool(wake_on.get())
        try:
            cfg.preroll_ms = min(max(0, int(preroll.get())), 5000)
        except ValueError:
            pass
        cfg.ptt_enabled = bool(ptt_on.get())
        cfg.hotkey = hotkey.get().strip() or cfg.hotkey
        cfg.tools = [name for name in ALL_TOOLS if tool_vars[name].get()]
        fixed = cfg.sanitised()
        for field in dataclasses.fields(DesktopConfig):  # keep the caller's object, repaired values
            setattr(cfg, field.name, getattr(fixed, field.name))
        save_config(cfg, self.path)
        self._result = cfg
        if self.on_save is not None:
            try:
                self.on_save(cfg)
            except Exception:  # noqa: BLE001 - the UI must survive callback errors
                log.exception("on_save callback failed")
        root.destroy()
        self._closed.set()


def _row(group: ttk.LabelFrame, row: int, label: str, widget: tk.Widget) -> None:
    ttk.Label(group, text=label).grid(row=row, column=0, sticky=tk.W, padx=4, pady=3)
    widget.grid(row=row, column=1, sticky=tk.W, padx=4, pady=3)


def open_settings_in_thread(cfg: DesktopConfig, on_save: Callable[[DesktopConfig], None] | None = None) -> threading.Thread:
    """The tray's Settings… entry: tkinter in its own thread so the tray keeps working."""

    def run() -> None:
        try:
            SettingsWindow(cfg, on_save).show()
        except Exception:  # noqa: BLE001 - no display server / tkinter missing
            log.exception("settings window failed")

    thread = threading.Thread(target=run, name="gaja-settings", daemon=True)
    thread.start()
    return thread

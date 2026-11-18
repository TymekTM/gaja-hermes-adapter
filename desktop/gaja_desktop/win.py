"""Windows process tweaks that must happen once, before anything touches pixels or input.

DPI awareness: without it Windows virtualises the desktop at e.g. 125/150 %, and mss screenshots
and pynput mouse coordinates are then each scaled by the OS — screenshot pixels and mouse pixels
no longer describe the same space, and the image is blurry on top. Per-monitor awareness makes
both APIs see physical pixels. Must be set before the first GDI/SendInput call (so: at startup,
before any tool or tray import).
"""

from __future__ import annotations

import ctypes
import sys

#: shcore: PROCESS_PER_MONITOR_DPI_AWARE (the Win 8.1+ shim; the modern
#: SetProcessDpiAwarenessContext(-4) is not available via plain windll argument passing).
PER_MONITOR_DPI_AWARE = 2


def make_dpi_aware() -> bool:
    """Best-effort per-monitor DPI awareness (Windows only, idempotent). True when the process
    is (or already was) DPI-aware; False on other platforms or when every call fails."""
    if sys.platform != "win32":
        return False
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(PER_MONITOR_DPI_AWARE)
        return True
    except Exception:
        pass  # already set, or shcore missing (Win 7): try the older call below
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return True
    except Exception:
        return False

# PyInstaller spec for the Gaja desktop client (Windows; PyInstaller cannot cross-build from Linux).
# Build:  cd desktop && uv run pyinstaller gaja-desktop.spec --noconfirm
# Result: dist/gaja-desktop.exe (one file, windowed; the wake-word model is bundled inside).

# -*- mode: python ; coding: utf-8 -*-
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

_desktop_dir = Path(SPECPATH).resolve()  # .../desktop
# The wake-word model is versioned once, in firmware/esphome/models/ (the ESP32 firmware uses the
# same files); copy it into the package at build time so PyInstaller bundles it (docs/DESKTOP.md).
# gaja_desktop/models/ is gitignored — a plain checkout has no copy until this runs.
_models_dst = _desktop_dir / "gaja_desktop" / "models"
_models_dst.mkdir(parents=True, exist_ok=True)
for _name in ("hej_gaja.tflite", "hej_gaja.json"):
    _src = _desktop_dir.parent / "firmware" / "esphome" / "models" / _name
    if _src.exists():
        shutil.copy2(_src, _models_dst / _name)

datas = [
    # The wake-word model, copied from firmware/esphome/models/ just above.
    ("gaja_desktop/models/hej_gaja.tflite", "gaja_desktop/models"),
    ("gaja_desktop/models/hej_gaja.json", "gaja_desktop/models"),
    # pymicro-wakeword / pymicro-features load their TensorFlow Lite shared library from their
    # own package directory at runtime (ctypes), so it must ride along as data.
    *collect_data_files("pymicro_wakeword"),
    *collect_data_files("pymicro_features"),
]

# The same shared library, explicitly as binaries (collect_data_files may classify it either way;
# ctypes loads it from the package dir, so it just has to be there next to the package).
binaries = [*collect_dynamic_libs("pymicro_wakeword")]

a = Analysis(
    ["gaja_desktop/__main__.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        # Backends are chosen dynamically at runtime (pynput: pynput._util.<backend> — the win32
        # backend lives in pynput/_util/win32.py(+win32_vks.py); xorg for Linux dev builds):
        "soundcard.mediafoundation",   # WASAPI on Windows
        "soundcard.pulseaudio",        # PulseAudio/PipeWire on Linux (dev builds)
        "pystray._win32",
        "pynput.keyboard",
        "pynput.mouse",
        "pynput._util.win32",
        "pynput._util.win32_vks",
        "pynput._util.xorg",
        # pymicro-features' compiled frontend is a top-level extension module
        # (micro_features_cpp.pyd / .so, not inside the pymicro_features package).
        "pymicro_wakeword",
        "pymicro_features",
        "micro_features_cpp",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "scipy", "matplotlib", "PyQt5", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="gaja-desktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # windowed: no console flashes when launched from Explorer
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

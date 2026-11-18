# PyInstaller spec for the Gaja desktop client (Windows; PyInstaller cannot cross-build from Linux).
# Build:  cd desktop && uv run pyinstaller gaja-desktop.spec --noconfirm
# Result: dist/gaja-desktop.exe (one file, windowed; the wake-word model is bundled inside).

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [
    # The wake-word model, copied from firmware/esphome/models/ into the package (see README).
    ("gaja_desktop/models/hej_gaja.tflite", "gaja_desktop/models"),
    ("gaja_desktop/models/hej_gaja.json", "gaja_desktop/models"),
    # pymicro-wakeword / pymicro-features load their TensorFlow Lite shared library from their
    # own package directory at runtime (ctypes), so it must ride along as data.
    *collect_data_files("pymicro_wakeword"),
    *collect_data_files("pymicro_features"),
]

binaries = []

a = Analysis(
    ["gaja_desktop/__main__.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        # Backends are chosen dynamically at runtime:
        "soundcard.mediafoundation",   # WASAPI on Windows
        "soundcard.pulseaudio",        # PulseAudio/PipeWire on Linux (dev builds)
        "pystray._win32",
        "pynput.keyboard",
        "pynput.mouse",
        "pynput._util.windows",
        "pynput._util.linux",
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

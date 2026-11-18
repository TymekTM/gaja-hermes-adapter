"""The PyInstaller spec must be executable and complete: the model files it references must
exist (copied from firmware/ by the spec itself), and every hidden import must be a real module
in this install (pynput._util.windows was not — the backend module is pynput._util.win32)."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path
from typing import Any

DESKTOP = Path(__file__).resolve().parent.parent
SPEC = DESKTOP / "gaja-desktop.spec"


class FakeAnalysis:
    def __init__(self, scripts, **kwargs: Any) -> None:
        self.scripts = scripts
        self.kwargs = kwargs
        self.datas = kwargs["datas"]
        self.binaries = kwargs["binaries"]
        self.pure = None


def run_spec() -> FakeAnalysis:
    """Execute the spec like PyInstaller does, with Analysis/PYZ/EXE stubbed and SPECPATH set."""
    recorded: dict[str, Any] = {}

    def fake_pyz(pure: Any) -> None:
        recorded["pure"] = pure

    def fake_exe(*args: Any, **kwargs: Any) -> None:
        recorded["exe_args"] = args

    module = runpy.run_path(
        str(SPEC),
        init_globals={
            "SPECPATH": str(DESKTOP),
            "DISTPATH": str(DESKTOP / "dist"),
            "SPECPATH_": None,
            "Analysis": FakeAnalysis,
            "PYZ": fake_pyz,
            "EXE": fake_exe,
            "TOC": list,
            "BUNDLE": fake_exe,
            "COLLECT": fake_exe,
            "Python": object,
            "bootloader_no_entries": 0,
        },
        run_name="gaja_desktop_spec_for_test",
    )
    analysis = module["a"]
    assert isinstance(analysis, FakeAnalysis)
    return analysis


def test_spec_copies_the_wake_word_model_into_the_package() -> None:
    analysis = run_spec()
    copied = DESKTOP / "gaja_desktop" / "models"
    assert (copied / "hej_gaja.tflite").is_file()
    assert (copied / "hej_gaja.json").is_file()
    # the files exist, so the datas entries PyInstaller must pack are real paths
    srcs = [Path(src) for src, _dest in analysis.datas]
    for rel in ("gaja_desktop/models/hej_gaja.tflite", "gaja_desktop/models/hej_gaja.json"):
        assert DESKTOP / rel in [s if s.is_absolute() else DESKTOP / s for s in srcs], rel


def test_spec_hidden_imports_all_resolve() -> None:
    analysis = run_spec()
    hidden = analysis.kwargs["hiddenimports"]
    assert hidden, "the spec lost its hiddenimports"
    missing = []
    for name in hidden:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    assert missing == [], f"hidden imports that do not exist in this install: {missing}"
    # the regression this guards against: the win32 backend module, not a 'windows' guess
    assert "pynput._util.win32" in hidden and "pynput._util.win32_vks" in hidden
    assert "micro_features_cpp" in hidden  # top-level compiled extension of pymicro-features


def test_spec_bundles_the_wakeword_shared_library_as_a_binary() -> None:
    analysis = run_spec()
    libs = [str(src) for src, _dest in analysis.binaries]
    assert any("tensorflowlite" in lib for lib in libs), libs


def test_workflow_build_step_exists() -> None:
    workflow = DESKTOP.parent / ".github" / "workflows" / "desktop.yml"
    text = workflow.read_text(encoding="utf-8")
    assert "pyinstaller gaja-desktop.spec" in text
    assert "uv run pytest -q" in text  # tests still run before every build

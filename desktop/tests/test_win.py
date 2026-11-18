"""make_dpi_aware: the Windows calls, with ctypes.windll faked (no Windows box needed)."""

from __future__ import annotations

import ctypes
from typing import Any

import pytest

from gaja_desktop.win import PER_MONITOR_DPI_AWARE, make_dpi_aware


class FakeFunc:
    def __init__(self, calls: list[str], name: str, *, error: Exception | None = None) -> None:
        self._calls = calls
        self._name = name
        self._error = error

    def __call__(self, *args: Any) -> int:
        self._calls.append(self._name)
        if self._error is not None:
            raise self._error
        return 0


class FakeDll:
    """One fake DLL (like ctypes.windll.shcore): attributes are recording functions."""

    def __init__(self, calls: list[str], dll: str, errors: dict[str, Exception]) -> None:
        self._calls = calls
        self._dll = dll
        self._errors = errors

    def __getattr__(self, func: str) -> FakeFunc:
        return FakeFunc(self._calls, f"{self._dll}.{func}", error=self._errors.get(func))


class FakeWinDll:
    """ctypes.windll stand-in: any attribute is a fake DLL (shcore, user32, …)."""

    def __init__(self, calls: list[str], errors: dict[str, Exception]) -> None:
        self._calls = calls
        self._errors = errors

    def __getattr__(self, dll: str) -> FakeDll:
        return FakeDll(self._calls, dll, self._errors)


@pytest.fixture
def win32(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr("sys.platform", "win32")
    return calls


def test_dpi_awareness_is_set_per_monitor_on_windows(win32, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ctypes, "windll", FakeWinDll(win32, {}), raising=False)
    assert make_dpi_aware() is True
    assert win32 == ["shcore.SetProcessDpiAwareness"]


def test_dpi_awareness_value_is_per_monitor(win32, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, ...]] = []

    class Spy:
        def __getattr__(self, func: str) -> Any:
            def call(*args: Any) -> int:
                seen.append((func, args))
                return 0
            return call

    monkeypatch.setattr(ctypes, "windll", type("W", (), {"shcore": Spy()}), raising=False)
    assert make_dpi_aware() is True
    assert seen == [("SetProcessDpiAwareness", (PER_MONITOR_DPI_AWARE,))]
    assert PER_MONITOR_DPI_AWARE == 2


def test_dpi_awareness_falls_back_to_set_process_dpi_aware(win32, monkeypatch: pytest.MonkeyPatch) -> None:
    # shcore missing or refusing (e.g. awareness already set): the older user32 call is tried once.
    monkeypatch.setattr(ctypes, "windll", FakeWinDll(win32, {"SetProcessDpiAwareness": OSError()}),
                        raising=False)
    assert make_dpi_aware() is True
    assert win32 == ["shcore.SetProcessDpiAwareness", "user32.SetProcessDPIAware"]


def test_dpi_awareness_all_calls_failing_is_false_but_never_raises(win32, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ctypes, "windll", FakeWinDll(
        win32, {"SetProcessDpiAwareness": OSError(), "SetProcessDPIAware": OSError()}), raising=False)
    assert make_dpi_aware() is False
    assert make_dpi_aware() is False


def test_dpi_awareness_is_a_noop_elsewhere() -> None:
    assert make_dpi_aware() is False  # this box is not win32

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from gaja.text.style import Style


class TTS(Protocol):
    name: str
    sample_rate: int
    languages: frozenset[str]

    async def synthesize(self, text: str, style: Style, language: str) -> np.ndarray: ...

    async def warm_up(self) -> None: ...


_REGISTRY: dict[str, Callable[..., TTS]] = {}


def register_tts(name: str) -> Callable[[Callable[..., TTS]], Callable[..., TTS]]:
    def decorator(factory: Callable[..., TTS]) -> Callable[..., TTS]:
        _REGISTRY[name] = factory
        return factory

    return decorator


def get_tts(name: str, **cfg: Any) -> TTS:
    if name not in _REGISTRY:
        module = name.split("-", 1)[0]
        try:
            importlib.import_module(f"gaja.tts.{module}_tts")
        except ModuleNotFoundError as exc:
            raise ValueError(f"unknown TTS engine {name!r}") from exc
    if name not in _REGISTRY:
        raise ValueError(f"unknown TTS engine {name!r}")
    return _REGISTRY[name](**cfg)


class EnginePool:
    """Engines selectable by name; each is built and warmed on first use, then kept for the process."""

    def __init__(self, factories: Mapping[str, Callable[[], TTS]]) -> None:
        self._factories = dict(factories)
        self._loaded: dict[str, TTS] = {}
        self._locks = {name: asyncio.Lock() for name in self._factories}

    def names(self) -> list[str]:
        return list(self._factories)

    def loaded(self) -> list[str]:
        return sorted(self._loaded)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    async def get(self, name: str) -> TTS:
        if name in self._loaded:
            return self._loaded[name]
        async with self._locks[name]:
            if name not in self._loaded:
                engine = await asyncio.to_thread(self._factories[name])
                await engine.warm_up()
                self._loaded[name] = engine
        return self._loaded[name]


def apply_energy(audio: np.ndarray, energy: float) -> np.ndarray:
    return np.clip(audio * energy, -1.0, 1.0).astype(np.float32, copy=False)

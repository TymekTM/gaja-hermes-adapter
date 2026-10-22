from __future__ import annotations

import concurrent.futures
import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Literal

import numpy as np

from gaja.config import VADConfig

SAMPLE_RATE = 16000
WINDOW = 512
WINDOW_MS = WINDOW * 1000 / SAMPLE_RATE
NEG_HYSTERESIS = 0.15

ProbFn = Callable[[np.ndarray], float]
SmartTurnFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class VadEvent:
    kind: Literal["speech_start", "speech_end", "speculate", "speculation_cancel"]
    audio: np.ndarray | None = None


class Endpointer:
    """Turns 16 kHz float samples into speech events with pre-roll.

    speech_start is only emitted once the segment holds ``barge_in_ms`` of speech windows (a cough,
    click or a single loud window never interrupts anything); a segment that ends before that is
    dropped silently. While ``set_strict(True)`` (Gaja is talking), confirmation needs
    ``barge_in_strict_ms`` and a higher probability, which keeps speaker echo from barging in.

    With spec_silence_ms set, emits "speculate" (audio so far) once the silence reaches it, then
    "speculation_cancel" if speech resumes before the real speech_end. A segment cut at
    max_utterance_s ends with speech_end; the speech that carries on is a continuation and does not
    emit another speech_start.

    With smart_turn enabled, at smart_turn_check_ms silence the classifier evaluates whether
    the speaker finished:
    - complete (prob >= threshold): commits speech_end early;
    - incomplete: extends the silence limit up to max_incomplete_silence_ms.
    """

    def __init__(
        self,
        prob: ProbFn,
        cfg: VADConfig,
        smart_turn: SmartTurnFn | Any | None = None,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self._prob = prob
        self._threshold = cfg.threshold
        self._strict_threshold = max(cfg.threshold, cfg.strict_threshold)
        self._min_silence = math.ceil(cfg.min_silence_ms / WINDOW_MS)
        self._min_speech = math.ceil(cfg.min_speech_ms / WINDOW_MS)
        self._confirm = max(1, math.ceil(cfg.barge_in_ms / WINDOW_MS))
        self._confirm_strict = max(self._confirm, math.ceil(cfg.barge_in_strict_ms / WINDOW_MS))
        self._max_windows = math.ceil(cfg.max_utterance_s * 1000 / WINDOW_MS)
        spec = math.ceil(cfg.spec_silence_ms / WINDOW_MS) if cfg.spec_silence_ms > 0 else 0
        self._spec_silence = spec if 0 < spec < self._min_silence else 0
        self._pre_roll: deque[np.ndarray] = deque(maxlen=math.ceil(cfg.pre_roll_ms / WINDOW_MS))
        self._rest = np.zeros(0, dtype=np.float32)
        self._strict = False
        self._speaking = False
        self._confirmed = False
        self._continuation = False
        self._speculating = False
        self._chunks: list[np.ndarray] = []
        self._speech_windows = 0
        self._strict_windows = 0
        self._silence_windows = 0

        # Smart Turn v3 semantic endpointing
        self._smart_turn_enabled = bool(cfg.smart_turn and smart_turn is not None)
        self._smart_turn = smart_turn
        self._executor = executor
        self._smart_turn_threshold = cfg.smart_turn_threshold
        self._check_silence = (
            math.ceil(cfg.smart_turn_check_ms / WINDOW_MS) if cfg.smart_turn_check_ms > 0 else self._min_silence
        )
        self._max_incomplete_silence = max(
            self._min_silence, math.ceil(cfg.max_incomplete_silence_ms / WINDOW_MS)
        )
        self._smart_turn_checked = False
        self._smart_turn_pending = False
        self._smart_turn_future: concurrent.futures.Future[float] | None = None
        self._incomplete = False

    @property
    def speaking(self) -> bool:
        """True while a confirmed user utterance is in progress."""
        return self._speaking and self._confirmed

    @property
    def in_speech(self) -> bool:
        """True while a speech segment is in progress, confirmed or not."""
        return self._speaking

    def reset(self) -> None:
        """Forget everything buffered (a device re-opened its mic: the old audio is stale)."""
        self._reset()
        self._rest = np.zeros(0, dtype=np.float32)
        self._continuation = False

    def set_strict(self, strict: bool) -> None:
        self._strict = strict

    def feed(self, samples: np.ndarray) -> list[VadEvent]:
        data = np.concatenate([self._rest, samples]) if self._rest.size else samples
        full = len(data) // WINDOW * WINDOW
        self._rest = data[full:].copy()
        events: list[VadEvent] = []
        for offset in range(0, full, WINDOW):
            events.extend(self._window(data[offset : offset + WINDOW]))
        return events

    def flush(self) -> list[VadEvent]:
        if not self._speaking:
            return []
        if not self._confirmed:
            self._reset()
            return []
        return [self._end()]

    def _call_smart_turn(self, audio: np.ndarray) -> float:
        assert self._smart_turn is not None
        if hasattr(self._smart_turn, "predict_probability"):
            return float(self._smart_turn.predict_probability(audio))
        return float(self._smart_turn(audio))

    def _window(self, window: np.ndarray) -> list[VadEvent]:
        probability = self._prob(window)
        if not self._speaking:
            self._pre_roll.append(window)
            if probability < self._threshold:
                self._continuation = False
                return []
            self._speaking = True
            self._confirmed = self._continuation
            self._continuation = False
            self._chunks = list(self._pre_roll)
            self._pre_roll.clear()
            self._speech_windows = 1
            self._strict_windows = 1 if probability >= self._strict_threshold else 0
            self._silence_windows = 0
            self._smart_turn_checked = False
            self._smart_turn_pending = False
            self._incomplete = False
            return self._confirm_events()

        self._chunks.append(window)
        events: list[VadEvent] = []
        if probability >= self._threshold:
            self._speech_windows += 1
            if probability >= self._strict_threshold:
                self._strict_windows += 1
            self._silence_windows = 0
            self._smart_turn_checked = False
            self._smart_turn_pending = False
            self._incomplete = False
            if self._smart_turn_future is not None:
                self._smart_turn_future.cancel()
                self._smart_turn_future = None
            if self._speculating:
                self._speculating = False
                events.append(VadEvent("speculation_cancel"))
            events.extend(self._confirm_events())
        elif probability < self._threshold - NEG_HYSTERESIS:
            self._silence_windows += 1

        # Check completed async smart turn future
        if self._smart_turn_pending and self._smart_turn_future is not None and self._smart_turn_future.done():
            try:
                st_prob = self._smart_turn_future.result()
            except Exception:
                st_prob = 1.0
            self._smart_turn_pending = False
            self._smart_turn_future = None
            if st_prob >= self._smart_turn_threshold:
                if not self._confirmed:
                    self._reset()
                    return events
                return [*events, self._end()]
            else:
                self._incomplete = True

        # Trigger smart turn check at check_silence
        if (
            self._smart_turn_enabled
            and self._confirmed
            and not self._smart_turn_checked
            and not self._smart_turn_pending
            and self._silence_windows >= self._check_silence
            and self._speech_windows >= self._min_speech
        ):
            self._smart_turn_checked = True
            audio_so_far = np.concatenate(self._chunks)
            if self._executor is not None:
                self._smart_turn_pending = True
                self._smart_turn_future = self._executor.submit(self._call_smart_turn, audio_so_far)
            else:
                try:
                    st_prob = self._call_smart_turn(audio_so_far)
                except Exception:
                    st_prob = 1.0
                if st_prob >= self._smart_turn_threshold:
                    if not self._confirmed:
                        self._reset()
                        return events
                    return [*events, self._end()]
                else:
                    self._incomplete = True

        silence_limit = (
            self._max_incomplete_silence
            if (self._incomplete or self._smart_turn_pending)
            else self._min_silence
        )

        if self._silence_windows >= silence_limit:
            if not self._confirmed:
                self._reset()
                return events
            return [*events, self._end()]

        if len(self._chunks) >= self._max_windows:
            if not self._confirmed:
                self._reset()
                return events
            forced = self._end()
            self._continuation = True
            return [*events, forced]

        if (
            self._spec_silence
            and self._confirmed
            and not self._speculating
            and self._silence_windows >= self._spec_silence
            and self._speech_windows >= self._min_speech
        ):
            self._speculating = True
            events.append(VadEvent("speculate", np.concatenate(self._chunks)))
        return events

    def _confirm_events(self) -> list[VadEvent]:
        if self._confirmed:
            return []
        if self._strict:
            if self._strict_windows < self._confirm_strict:
                return []
        elif self._speech_windows < self._confirm:
            return []
        self._confirmed = True
        return [VadEvent("speech_start")]

    def _end(self) -> VadEvent:
        long_enough = self._speech_windows >= self._min_speech
        audio = np.concatenate(self._chunks) if long_enough else None
        self._reset()
        return VadEvent("speech_end", audio)

    def _reset(self) -> None:
        self._speaking = False
        self._confirmed = False
        self._speculating = False
        self._chunks = []
        self._speech_windows = 0
        self._strict_windows = 0
        self._silence_windows = 0
        self._pre_roll.clear()
        self._smart_turn_checked = False
        self._smart_turn_pending = False
        if self._smart_turn_future is not None:
            self._smart_turn_future.cancel()
            self._smart_turn_future = None
        self._incomplete = False


def load_silero_prob() -> ProbFn:
    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad(onnx=True)

    def prob(window: np.ndarray) -> float:
        return float(model(torch.from_numpy(window), SAMPLE_RATE).reshape(-1)[0])

    return prob


@lru_cache(maxsize=1)
def _silero_onnx_session() -> Any:
    """One shared onnxruntime session for the packaged silero_vad.onnx (opset 16, the same file
    ``silero_vad.load_silero_vad(onnx=True)`` uses), without importing silero_vad/torch (~450 MB RSS)."""
    import importlib.util
    from pathlib import Path

    import onnxruntime as ort

    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("silero_vad package not found")
    path = Path(spec.submodule_search_locations[0]) / "data" / "silero_vad.onnx"
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=opts)


def load_silero_prob_onnx() -> ProbFn:
    """Silero VAD through onnxruntime with numpy state: numerically identical to the torch wrapper
    (MAXDIFF 0.0 over constant/ramp windows, bench/results/ram/silero-onnx.txt) and keeps torch out
    of the process. Each call returns an independent ProbFn (per-session RNN state); the underlying
    InferenceSession is shared."""
    session = _silero_onnx_session()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)

    def prob(window: np.ndarray) -> float:
        nonlocal state, context
        if window.ndim != 1:
            raise ValueError(f"expected a 1-D window of {SAMPLE_RATE / 31.25:.0f} samples, got shape {window.shape}")
        x = np.concatenate([context, window.reshape(1, -1)], axis=1)
        out, new_state = session.run(
            None, {"input": x, "state": state, "sr": np.array(SAMPLE_RATE, dtype="int64")}
        )
        state = new_state
        context = x[:, -64:]
        return float(out.reshape(-1)[0])

    return prob


def load_silero(backend: str = "onnx") -> ProbFn:
    """VAD backend selector: "onnx" (default, no torch) or "torch" (silero_vad wrapper, ~450 MB)."""
    if backend == "torch":
        return load_silero_prob()
    if backend == "onnx":
        return load_silero_prob_onnx()
    raise ValueError(f"unknown vad backend {backend!r}")

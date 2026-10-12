from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 8
MAX_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  # 128,000 samples
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

DEFAULT_CACHE_DIR = Path("/home/ttm-server/.cache/gaja-models/smart-turn")
DEFAULT_MODEL_NAME = "smart-turn-v3.2-cpu.onnx"
FALLBACK_MODEL_NAME = "smart-turn-v3.1-cpu.onnx"


def compute_slaney_mel_filters(
    sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS
) -> np.ndarray:
    """Compute Slaney mel filter bank matrix of shape (n_fft // 2 + 1, n_mels).

    Numerically identical to librosa / WhisperFeatureExtractor mel filters (< 1e-15).
    """
    min_hz, max_hz = 0.0, sr / 2.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - 0.0) / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
        h = np.asanyarray(hz, dtype=np.float64)
        mel = np.empty_like(h)
        linear = h < min_log_hz
        mel[linear] = h[linear] / f_sp
        mel[~linear] = min_log_mel + np.log(h[~linear] / min_log_hz) / logstep
        return mel

    def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
        m = np.asanyarray(mel, dtype=np.float64)
        hz = np.empty_like(m)
        linear = m < min_log_mel
        hz[linear] = f_sp * m[linear]
        hz[~linear] = min_log_hz * np.exp(logstep * (m[~linear] - min_log_mel))
        return hz

    m_pts = np.linspace(hz_to_mel(min_hz), hz_to_mel(max_hz), n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    bins = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)

    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        up_mask = (bins >= left) & (bins <= center)
        if center > left:
            weights[i, up_mask] = (bins[up_mask] - left) / (center - left)
        down_mask = (bins >= center) & (bins <= right)
        if right > center:
            weights[i, down_mask] = (right - bins[down_mask]) / (right - center)
        # Slaney area normalization
        weights[i] *= 2.0 / (right - left)

    return weights.T  # Shape: (201, 80)


_MEL_FILTERS: np.ndarray | None = None
_HANN_WINDOW: np.ndarray | None = None


def get_mel_filters() -> np.ndarray:
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        _MEL_FILTERS = compute_slaney_mel_filters()
    return _MEL_FILTERS


def get_hann_window() -> np.ndarray:
    global _HANN_WINDOW
    if _HANN_WINDOW is None:
        _HANN_WINDOW = (np.hanning(N_FFT + 1)[:-1]).astype(np.float32)
    return _HANN_WINDOW


def truncate_or_pad_audio(audio: np.ndarray, max_samples: int = MAX_SAMPLES) -> np.ndarray:
    """Keep the last 8 seconds of audio or pad with zeros at the beginning."""
    if audio.ndim > 1:
        audio = audio.flatten()
    if len(audio) > max_samples:
        return audio[-max_samples:]
    elif len(audio) < max_samples:
        return np.pad(audio, (max_samples - len(audio), 0), mode="constant", constant_values=0)
    return audio


def extract_features(audio: np.ndarray) -> np.ndarray:
    """Extract Whisper-compatible 80-channel log-mel features for Smart Turn.

    Follows the reference preprocessing:
    - 8 s input (128,000 samples at 16 kHz)
    - Zero-mean unit-variance normalization on the waveform
    - STFT with 400 window, 160 hop, Hann window, reflect padding
    - 80 Slaney mel filter channels
    - log10(clamp(mel, min=1e-10))
    - dynamic range clamp to max - 8.0, normalized via (log_mel + 4.0) / 4.0
    - Output shape: (1, 80, 800) float32
    """
    audio = truncate_or_pad_audio(audio)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Zero-mean unit-variance normalization (matches WhisperFeatureExtractor do_normalize=True)
    std = float(np.sqrt(np.var(audio) + 1e-7))
    normed = (audio - float(np.mean(audio))) / std

    # STFT with reflect padding
    pad = N_FFT // 2
    padded = np.pad(normed, (pad, pad), mode="reflect")
    window = get_hann_window()
    mel_filters = get_mel_filters()

    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH] * window
    # Drop the trailing frame to get exactly 800 frames
    stft = np.fft.rfft(frames, n=N_FFT)[:-1]
    mags = np.abs(stft) ** 2

    mel_spec = mags @ mel_filters
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_spec = np.log10(mel_spec).T  # Shape: (80, 800)

    max_val = float(np.max(log_spec))
    log_spec = np.maximum(log_spec, max_val - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return np.expand_dims(log_spec.astype(np.float32), axis=0)  # Shape: (1, 80, 800)


def resolve_smart_turn_model(path: str | Path | None = None) -> Path:
    """Resolve the path to the Smart Turn ONNX model file."""
    if path is not None:
        candidate = Path(path).resolve()
        if candidate.is_file():
            return candidate

    candidates = [
        DEFAULT_CACHE_DIR / DEFAULT_MODEL_NAME,
        DEFAULT_CACHE_DIR / FALLBACK_MODEL_NAME,
        Path("models") / DEFAULT_MODEL_NAME,
        Path("models") / FALLBACK_MODEL_NAME,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Smart Turn model not found at {path!r} or fallback paths: {candidates}"
    )


SmartTurnPredictor = Callable[[np.ndarray], float]


class SmartTurn:
    """Smart Turn v3 semantic endpointing classifier.

    Predicts whether a speaker finished their turn from raw 16 kHz audio.
    Runs ONNX inference on CPU using single-thread execution mode.
    """

    def __init__(self, model_path: str | Path | None = None, threads: int = 1) -> None:
        self._model_path = resolve_smart_turn_model(model_path)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(self._model_path), sess_options=so)
        self._input_name = self._session.get_inputs()[0].name
        log.info("Loaded Smart Turn model from %s (threads=%d)", self._model_path, threads)

    @property
    def model_path(self) -> Path:
        return self._model_path

    def predict_probability(self, audio: np.ndarray) -> float:
        """Return the probability [0.0, 1.0] that the speaker has completed their turn."""
        if audio.size == 0:
            return 1.0
        features = extract_features(audio)
        outputs = self._session.run(None, {self._input_name: features})
        return float(outputs[0][0].item())

    def predict(self, audio: np.ndarray, threshold: float = 0.5) -> bool:
        """Return True if complete, False if incomplete."""
        return self.predict_probability(audio) >= threshold

    def __call__(self, audio: np.ndarray) -> float:
        return self.predict_probability(audio)

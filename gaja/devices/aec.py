"""Server-side echo cancellation for full-duplex devices (protocol v1.11, docs/FULL_DUPLEX.md).

A ``caps.ref`` device streams its microphone AND a reference signal -- a 16 kHz decimated copy of
the samples it feeds to its speaker -- as tagged binary frames. This module removes the acoustic
echo (Gaja's own voice picked up by the microphone) from the mic stream before the existing VAD
barge-in path sees it, which is what lets a user interrupt a reply without Gaja interrupting
herself. It reuses the proven web-client machinery: the endpointer's strict mode raises the
barge-in bar while ``_playback_until`` is in the future (``Session.handle_audio``), and a
confirmed ``speech_start`` during playback cancels the reply (``Session._interrupt_speech``).

Implementation: one long frequency-domain adaptive filter (multi-delay-block frequency-domain
LMS, MDF, numpy only -- PyPI has no maintained AEC binding: ``speexdsp-python`` does not exist,
``webrtc-audio-processing`` 0.1.3 is a 2019 pre-alpha source build, and the cancellation must be
deterministic and testable here). The filter spans the whole plausible mic-vs-ref offset PLUS the
room impulse response, so it learns the device's transport delay (TX DMA + acoustic path + ADC
DMA, stable per boot) and the reverb TOGETHER -- no separate delay estimation to get wrong, and
per-bin power normalisation keeps the adaptation stable for speech where a time-domain
block-NLMS over correlated taps diverges. Details:

- Overlap-save: each 20 ms block is cancelled with one FFT of the reference window ending at the
  block; the adaptive impulse response is re-constrained to the modelled tail every block.
- Mic blocks are processed only once the reference has advanced past them (the device alternates
  mic/ref frames); if the reference stops (playback ended) the buffered tail is processed with a
  frozen filter instead of stalling the stream.
- Adaptation freezes while the mic peak exceeds the whole reference window's peak (Geigel-style
  double-talk guard), so it cannot learn to cancel the USER.
- A convergence gate outputs silence while the filter is still learning: on the first playback
  after boot, and (capped at 1 s) at the onset of each later reply, where a filter that converged
  on short replies is still adapting to new content; the filter persists across replies. The
  reference resumes aligned with the mic after a gap (the reference of silence is silence: the
  device sends ref frames only while playback is audible). Without the gate, the strict
  VAD could confirm a speech segment on the decaying warm-up residual of Gaja's own voice. The
  gate tracks cancellation QUALITY (residual vs predicted echo energy) and always lifts after
  GATE_MAX_BLOCKS so a genuine interruption can never be muted forever.

Frame sizes are whatever the caller streams (the device sends 20 ms / 320 samples); the filter
runs on fixed internal blocks and buffers the rest. Everything is float32 in [-1, 1] (the
pipeline's PCM16 -> float convention).
"""

from __future__ import annotations

import numpy as np

# Frame tags of the device protocol v1.11 (caps.ref): every binary frame carries a 1-byte tag.
# The firmware keeps matching constants in components/gaja_device/frame_tag.h;
# tests/test_device_aec.py asserts that both sides agree.
FRAME_MIC = 0x00
FRAME_REF = 0x01

RATE = 16000
BLOCK = 320  # 20 ms internal processing block
TAIL = 3072  # 192 ms: the transport delay (TX/ADC DMA + ring pacing, <= ~150 ms) + reverb
FFT = 4096  # MDF transform length (>= TAIL + BLOCK: overlap-save validity)
MU = 0.5  # per-bin normalised step
REF_MARGIN_BLOCKS = 1  # process a block only when ref covers block_end + this much
STARVED_BLOCKS = 4  # mic blocks without new ref before the filter runs unfed (frozen weights)
GATE_MIN_BLOCKS = 8  # minimum zero-output gate (see EchoCanceller.process)
GATE_MAX_BLOCKS = 90  # the gate always lifts after this many blocks (1.8 s), speech or not
GATE_REPLY_GAP_BLOCKS = 50  # a reference gap this long (1 s) means a new reply: re-gate its onset
GATE_REPLY_MAX_BLOCKS = 50  # ... for at most 1 s (lifts after GATE_MIN_BLOCKS when converged)
GATE_RESIDUAL_RATIO = 0.08  # gate lifts once residual energy < this share of the echo estimate
ENERGY_FLOOR = 1e-7  # ~-70 dBFS block energy; below it a signal counts as silence


class EchoCanceller:
    """Adaptive echo canceller for one device session.

    ``push_ref`` feeds reference samples (tag-0x01 frames), ``process`` cancels mic samples
    (tag-0x00 frames) and returns cleaned audio. Samples whose reference has not arrived yet are
    held back (bounded: the device alternates mic/ref frames, and a reference stream that stops
    starves the buffer loose), so the total returned over time equals the total fed -- no loss,
    no duplication. Until reference audio arrives the mic passes through unchanged, so
    pre-playback frames are untouched.
    """

    def __init__(self, rate: int = RATE) -> None:
        self._rate = rate
        self._wf = np.zeros(FFT // 2 + 1, dtype=complex)  # frequency-domain filter
        self._bin_power: np.ndarray | None = None  # smoothed |X(f)|^2 for the step normalisation
        self._err_ema = ENERGY_FLOOR  # smoothed residual block energy
        self._echo_ema = ENERGY_FLOOR  # smoothed predicted-echo block energy (gate reference)
        self._gated_blocks = 0
        self._gate_max = GATE_MAX_BLOCKS
        self._hist_len = FFT  # reference window history (the overlap-save window)
        self._ref_hist = np.zeros(self._hist_len, dtype=np.float32)
        self._ref_total = 0  # absolute ref samples received
        self._mic_total = 0  # absolute mic samples received
        self._processed = 0  # absolute mic samples processed
        self._pending = np.zeros(0, dtype=np.float32)  # mic received but not yet processed
        self._ref_seen = False
        self._starved = False
        self._starved_blocks = 0

    # ---- protocol surface ---------------------------------------------------------------

    def push_ref(self, samples: np.ndarray) -> None:
        """Reference audio (16 kHz float), e.g. from a tag-0x01 frame."""
        samples = np.asarray(samples, dtype=np.float32).ravel()
        if self._starved and samples.size and self._mic_total > self._ref_total:
            # Playback resumes after a reference gap (the device sends ref frames only while
            # audible, the mic keeps streaming): the reference of the gap is silence. Re-align
            # the reference timeline with the mic, or every reply after the first is uncancelled.
            gap = self._mic_total - self._ref_total
            if gap >= GATE_REPLY_GAP_BLOCKS * BLOCK:
                # A new reply: a filter that converged on a short first reply does not
                # generalise to new content at once, and its onset residual can be long enough
                # for the VAD to barge in on Gaja's own voice. Gate the onset again (shorter).
                self._gated_blocks = 0
                self._gate_max = GATE_REPLY_MAX_BLOCKS
                self._err_ema = self._echo_ema = ENERGY_FLOOR
            self._append_hist(self._ref_hist, np.zeros(min(gap, self._hist_len), dtype=np.float32))
            self._ref_total += gap
        self._append_hist(self._ref_hist, samples)
        self._ref_total += samples.size
        if samples.size:
            self._ref_seen = True
            self._starved = False
            self._starved_blocks = 0

    def process(self, mic: np.ndarray) -> np.ndarray:
        """Cancel the echo in mic samples; returns cleaned audio.

        The returned length may lag the input by a block or two while a reference frame is
        outstanding; totals are conserved exactly.
        """
        mic = np.asarray(mic, dtype=np.float32).ravel()
        if not self._ref_seen:
            return mic
        self._mic_total += mic.size
        self._pending = np.concatenate([self._pending, mic])
        out = np.zeros(0, dtype=np.float32)
        while self._pending.size >= BLOCK:
            block = self._pending[:BLOCK]
            if not self._ref_ready_for(self._processed + BLOCK):
                break
            out = np.concatenate([out, self._process_block(block)])
            self._processed += BLOCK
            self._pending = self._pending[BLOCK:]
        return out

    def reset(self) -> None:
        """Forget filter and alignment (a reconnect: the device re-learned nothing yet)."""
        self._wf[:] = 0.0
        self._bin_power = None
        self._err_ema = self._echo_ema = ENERGY_FLOOR
        self._gated_blocks = 0
        self._gate_max = GATE_MAX_BLOCKS
        self._pending = np.zeros(0, dtype=np.float32)
        self._processed = self._mic_total  # never re-process buffered audio after a reset
        self._starved = False
        self._starved_blocks = 0

    # ---- internals -----------------------------------------------------------------------

    def _ref_ready_for(self, block_end: int) -> bool:
        """True when the reference stream covers ``block_end`` (+ margin), or has starved."""
        if self._starved:
            return True
        if self._ref_total >= block_end + REF_MARGIN_BLOCKS * BLOCK:
            return True
        # The reference stopped arriving (playback ended mid-stream): after a few blocks of mic
        # without any ref, run the frozen filter over the buffered tail instead of stalling.
        if self._mic_total - self._ref_total > STARVED_BLOCKS * BLOCK:
            self._starved = True
            return True
        return False

    @staticmethod
    def _append_hist(hist: np.ndarray, data: np.ndarray) -> None:
        if data.size == 0:
            return
        if data.size >= hist.size:
            hist[:] = data[-hist.size :]
            return
        hist[: -data.size] = hist[data.size :]
        hist[-data.size :] = data

    def _process_block(self, block: np.ndarray) -> np.ndarray:
        # Overlap-save: the reference window ends at the block's end (absolute ref samples).
        end = self._processed + BLOCK
        window = np.zeros(FFT, dtype=np.float32)
        lo = end - FFT
        # Intersect [lo, end) with what actually exists: [max(0, ref_total - hist_len), ref_total).
        src_lo = max(lo, 0, self._ref_total - self._hist_len)
        src_hi = min(end, self._ref_total)
        if src_hi > src_lo:
            hist_base = self._ref_total - self._hist_len
            dst = src_lo - lo
            window[dst : dst + (src_hi - src_lo)] = self._ref_hist[src_lo - hist_base : src_hi - hist_base]
        x = np.fft.rfft(window)
        power = np.abs(x) ** 2
        if self._bin_power is None:
            self._bin_power = power.copy()  # seed with the real scale: a floor would explode
        else:
            self._bin_power = 0.9 * self._bin_power + 0.1 * power
        # irfft yields float64; the cleaned audio feeds the VAD's ONNX model, which takes float32 only.
        echo = np.fft.irfft(x * self._wf, FFT)[-BLOCK:].astype(np.float32)
        error = block - echo
        echo_energy = float(np.dot(echo, echo))
        self._echo_ema = 0.9 * self._echo_ema + 0.1 * echo_energy
        self._err_ema = 0.9 * self._err_ema + 0.1 * float(np.dot(error, error))

        # Double-talk guard (Geigel-style): the microphone peak exceeding the whole reference
        # window's peak means near-end speech dominates -- freeze the adaptation for this block
        # so the filter cannot learn to cancel the USER. (No echo-energy condition here: at boot
        # the estimate is exactly zero and the filter must adapt from there.)
        double_talk = float(np.abs(block).max()) > float(np.abs(window).max())
        if not double_talk:
            step_denom = self._bin_power + 1e-3 * float(self._bin_power.max()) + 1e-9
            e_padded = np.zeros(FFT, dtype=np.float32)
            e_padded[FFT - BLOCK :] = error
            e_fft = np.fft.rfft(e_padded)
            self._wf += (MU * np.conj(x) * e_fft) / step_denom
            # MDF constraint: keep the impulse response inside the modelled tail. Without this
            # the unconstrained frequency weights wrap energy around the block and the filter
            # never settles (textbook multi-delay-block frequency-domain LMS).
            w_time = np.fft.irfft(self._wf, FFT)
            w_time[TAIL:] = 0.0
            self._wf = np.fft.rfft(w_time)

        # Convergence gate (see module docstring): mute while the filter is still learning.
        # Quality criterion relative to the PREDICTED echo, so it is independent of playback
        # level; the hard GATE_MAX_BLOCKS bound guarantees a genuine interruption is never
        # muted forever (with the user speaking, the residual IS the speech).
        self._gated_blocks += 1
        if self._gated_blocks <= GATE_MIN_BLOCKS or (
            self._gated_blocks <= self._gate_max
            and self._echo_ema > ENERGY_FLOOR
            and self._err_ema > GATE_RESIDUAL_RATIO * self._echo_ema
        ):
            return np.zeros(BLOCK, dtype=np.float32)
        return error

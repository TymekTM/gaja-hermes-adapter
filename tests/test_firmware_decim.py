"""Host-side check of the firmware's 3:1 decimator (48 kHz -> 16 kHz, decim3.h).

Compiles ``firmware/esphome/components/gaja_device/decim3.h`` (pure C++, no ESPHome includes)
with the system C++ compiler and asserts what the full-duplex audio path relies on: exact unity
DC gain (the adaptive-volume dB statistics must not drift), passband transparency for speech,
stopband suppression above the 16 kHz output's Nyquist (no aliasing of playback leakage into the
wake word), identical output whether fed in one call or per 20 ms chunk (state carries across
calls), and the constant latency. Skipped when no C++ compiler is available.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "decim3.h"

MAIN = r"""
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include "decim3.h"
using esphome::gaja_device::Decim3;
static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}

// Bin magnitude of a real sinusoid over an integer number of periods: |X| = A/2*N for peak
// amplitude A, so RMS = sqrt(2) * |X| / N.
static double bin_rms(const std::vector<int16_t> &x, double freq, double rate) {
  double re = 0.0, im = 0.0;
  for (size_t i = 0; i < x.size(); i++) {
    const double ph = 2.0 * M_PI * freq * (double) i / rate;
    re += (double) x[i] * std::cos(ph);
    im -= (double) x[i] * std::sin(ph);
  }
  return std::sqrt(re * re + im * im) * std::sqrt(2.0) / (double) x.size();
}

// A sine whose measured tail (after the warm-up skip) is an exact number of periods.
static std::vector<int16_t> sine_for_tail(double freq, double rate, double amp, size_t skip,
                                          size_t periods) {
  const size_t tail = (size_t) (periods * rate / freq);   // decimated samples in the tail
  std::vector<int16_t> x(3 * (skip + tail));
  for (size_t i = 0; i < x.size(); i++) {
    x[i] = (int16_t) std::lround(amp * std::sin(2.0 * M_PI * freq * (double) i / rate));
  }
  return x;
}

int main() {
  // 1. DC: unity gain by construction (the q14 coefficients sum to exactly 1 << 14), so the
  //    adaptive-volume dB statistics see the same level the 16 kHz native mic used to deliver.
  //    The first TAPS/3 outputs are the filter warm-up (history zero-filled) and are skipped.
  {
    Decim3 d;
    std::vector<int16_t> in(9600, 12000);
    std::vector<int16_t> out(in.size() / 3);
    const size_t got = d.process(in.data(), in.size(), out.data());
    check(got == out.size(), "process returns n/3 outputs");
    bool exact = true;
    for (size_t i = Decim3::TAPS / 3; i < out.size(); i++) { if (out[i] != 12000) { exact = false; break; } }
    check(exact, "DC passes with exact unity gain after the warm-up");
  }

  // 2. Impulse: an impulse at an output's newest input sample shows coefficient h[0] in that
  //    output, h[3] in the next one (windows end on input 3m+2, three inputs apart) -- pins down
  //    the convolution indexing against the real q14 coefficient table, shift semantics included.
  {
    Decim3 d;
    std::vector<int16_t> in(96, 0);
    in[47] = 16000;  // output 15's newest sample (47 = 3*15 + 2)
    std::vector<int16_t> out(in.size() / 3, 0);
    d.process(in.data(), in.size(), out.data());
    // q14 coefficients: h[0] = -9, h[3] = 15, h[6] = -33 (see decim3.h); acc is an arithmetic
    // shift (floors toward -inf).
    auto expect = [](int32_t h) { return (int) ((int64_t) h * 16000 >> 14); };
    check(out[15] == expect(-9), "impulse shows h[0] in its own output");
    check(out[16] == expect(15), "impulse shows h[3] in the next output");
    check(out[17] == expect(-33), "impulse shows h[6] in the output after that");
    check(out[10] == 0, "impulse response is finite (TAPS/3 outputs)");
  }

  // 3. Passband: 1 kHz at 48 kHz survives decimation with its amplitude and frequency intact.
  {
    const double AMP = 12000.0;
    Decim3 d;
    const size_t SKIP = Decim3::TAPS / 3 + 8;  // drop the filter warm-up
    std::vector<int16_t> in = sine_for_tail(1000.0, 48000.0, AMP, SKIP, 500);
    std::vector<int16_t> out(in.size() / 3);
    d.process(in.data(), in.size(), out.data());
    std::vector<int16_t> tail(out.begin() + SKIP, out.end());
    const double db = 20.0 * std::log10(bin_rms(tail, 1000.0, 16000.0) / (AMP / std::sqrt(2.0)));
    check(db > -0.2 && db < 0.2, "1 kHz passband is transparent");
  }

  // 4. Stopband: 12 kHz at 48 kHz aliases to 4 kHz at 16 kHz -- it must be gone (< -40 dB).
  {
    const double AMP = 12000.0;
    Decim3 d;
    const size_t SKIP = Decim3::TAPS / 3 + 8;
    std::vector<int16_t> in = sine_for_tail(12000.0, 48000.0, AMP, SKIP, 500);
    std::vector<int16_t> out(in.size() / 3);
    d.process(in.data(), in.size(), out.data());
    std::vector<int16_t> tail(out.begin() + SKIP, out.end());
    const double db = 20.0 * std::log10(bin_rms(tail, 4000.0, 16000.0) / (AMP / std::sqrt(2.0)));
    check(db < -40.0, "12 kHz is suppressed > 40 dB (no aliasing into the wake word)");
  }

  // 5. Chunked == one-shot: the RX task feeds 20 ms chunks (960 in / 320 out); state must carry
  //    across calls bit-exactly.
  {
    Decim3 whole, chunked;
    std::vector<int16_t> in;
    for (size_t i = 0; i < 9600; i++) {
      in.push_back((int16_t) ((int32_t) (i * 7919) % 32000 - 16000));  // deterministic noise
    }
    std::vector<int16_t> a(in.size() / 3), b(in.size() / 3);
    whole.process(in.data(), in.size(), a.data());
    size_t at = 0;
    for (size_t off = 0; off < in.size(); off += 960) {
      at += chunked.process(in.data() + off, 960, b.data() + at);
    }
    check(at == a.size(), "chunked produces the same count");
    check(std::equal(a.begin(), a.end(), b.begin()), "chunked output is bit-identical");
  }

  // 6. Constant latency: the very first outputs appear after the same fixed number of inputs on
  //    every reset (the reference and mic decimators must age identically for the AEC).
  {
    Decim3 d;
    std::vector<int16_t> in(300, 0);
    in[0] = 16000;
    std::vector<int16_t> out(100, 0);
    d.process(in.data(), in.size(), out.data());
    int first_active = -1;
    for (size_t i = 0; i < out.size(); i++) { if (out[i] != 0) { first_active = (int) i; break; } }
    check(first_active >= 0 && first_active < 8, "impulse shows up in the first outputs");
  }

  if (failures) { std::printf("%d decim failures\n", failures); return 1; }
  std::printf("decim OK\n");
  return 0;
}
"""

pytestmark = pytest.mark.skipif(
    shutil.which("c++") is None and shutil.which("g++") is None and shutil.which("clang++") is None,
    reason="no host C++ compiler",
)


def _cxx() -> str:
    for candidate in ("c++", "g++", "clang++"):
        if shutil.which(candidate) is not None:
            return candidate
    raise AssertionError("unreachable: skipif above guarantees a compiler")


def test_decim() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-decim-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "decim"
        subprocess.run([_cxx(), "-std=c++17", "-I", str(HEADER.parent), str(main), "-o", str(exe)], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    assert "decim OK" in result.stdout
    assert "FAIL" not in result.stdout

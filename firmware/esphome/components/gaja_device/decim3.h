#pragma once

// 3:1 polyphase FIR decimator, 48 kHz -> 16 kHz, int16 in/out. Pure C++ (no ESPHome includes)
// so it can be unit-tested on the host (tests/test_firmware_decim.py).
//
// Full-duplex I2S shares ONE bit-clock between the microphone and the speaker (ESP-IDF full
// duplex on a single controller), so both directions run at the ES8311-verified 48000 Hz and
// the mic path (wake word + uplink, both 16 kHz) is decimated on the device. The playback
// reference for the server-side echo canceller is decimated with the SAME filter, so both
// streams carry the identical (linear-phase) group delay and stay sample-aligned relative to
// each other; the residual absolute offset is handled by the server's delay search.
//
// Filter: 48 taps, Hamming-windowed sinc, unity DC gain (q14 coefficients, exact sum 1 << 14),
// symmetric (linear phase, (TAPS-1)/2 = 23.5 input samples of delay at 48 kHz). Response
// (numpy check, q14 quantised): flat to ~6 kHz, -0.7 dB @ 7 kHz, -6 dB @ 8 kHz (output
// Nyquist), -21.6 dB @ 9 kHz, -53.5 dB @ 10 kHz, <= -60 dB above 11.6 kHz.

#include <cstdint>
#include <cstring>

namespace esphome::gaja_device {

class Decim3 {
 public:
  static constexpr size_t TAPS = 48;
  /// Output samples produced per call: input must be a multiple of 3.
  static size_t out_for(size_t in_samples) { return in_samples / 3; }
  /// Filter delay in INPUT samples (half the symmetric tap count).
  static constexpr size_t DELAY_IN = (TAPS - 1) / 2;

  void reset() { this->hist_len_ = 0; }

  /// Decimate `n` input samples (n % 3 == 0) into `out` (n/3 samples). Returns the number of
  /// output samples written. State carries across calls (the history is primed once, then the
  /// filter runs continuously).
  size_t process(const int16_t *in, size_t n, int16_t *out) {
    if (n < 3) {
      return 0;
    }
    size_t produced = 0;
    size_t consumed = 0;  // input samples absorbed from `in` for complete outputs
    // Output m corresponds to input index 3m, i.e. the newest tap is at 3m, going back TAPS.
    // History: the previous calls' last (TAPS-1) samples.
    while (consumed + 3 <= n) {
      // Build the 48-sample window ending at the current output's newest input sample.
      int16_t window[TAPS];
      // Newest sample for this output is in[consumed + 2] (the third of this group) so that
      // successive outputs consume 3 inputs and the window slides by 3.
      const size_t have = this->hist_len_;
      // Fill from history (oldest first).
      const size_t from_hist = TAPS - 3 < have ? TAPS - 3 : have;
      size_t w = 0;
      for (size_t i = 0; i < from_hist; i++) {
        window[w++] = this->hist_[have - from_hist + i];
      }
      // Then the three new samples.
      window[w++] = in[consumed];
      window[w++] = in[consumed + 1];
      window[w++] = in[consumed + 2];
      // w may be < TAPS only on the very first call(s): zero-pad at the front (filter warm-up).
      if (w < TAPS) {
        memmove(window + (TAPS - w), window, w * sizeof(int16_t));
        memset(window, 0, (TAPS - w) * sizeof(int16_t));
        w = TAPS;
      }
      int64_t acc = 0;
      for (size_t k = 0; k < TAPS; k++) {
        acc += (int64_t) COEFFS_Q14[k] * (int32_t) window[TAPS - 1 - k];
      }
      acc >>= 14;
      if (acc > 32767) {
        acc = 32767;
      } else if (acc < -32768) {
        acc = -32768;
      }
      out[produced++] = (int16_t) acc;
      consumed += 3;
      // Keep the last (TAPS - 3) samples as history for the next output.
      const size_t keep = TAPS - 3;
      if (w == TAPS) {
        // window currently holds the last TAPS inputs ending at consumed+2; slide by 3.
        memmove(this->hist_, window + 3, keep * sizeof(int16_t));
      }
      this->hist_len_ = keep;
    }
    return produced;
  }

 protected:
  static constexpr int32_t COEFFS_Q14[TAPS] = {
      -9,   -20,  -12,   15,    39,    25,   -33,   -85,   -54,   68,   168,  103,
      -126, -307, -186,  226,   549,   336,  -419,  -1067, -710,  1018, 3452, 5220,
      5222, 3452, 1018,  -710,  -1067, -419,  336,   549,   226,  -186, -307, -126,
      103,  168,  68,    -54,   -85,   -33,  25,    39,    15,   -12,  -20,  -9,
  };
  int16_t hist_[TAPS - 3]{};
  size_t hist_len_{0};
};

}  // namespace esphome::gaja_device

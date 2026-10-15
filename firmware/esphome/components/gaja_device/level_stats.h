#pragma once

// Speech-level statistics for the adaptive reply volume (and the "did anybody speak?" gate).
//
// Pure C++ (no ESPHome includes) so it is unit-tested on the host (tests/test_firmware_level_stats.py).
//
// Design (one tracker per mic-open window; clear() at every window start):
// - Each 20 ms PCM16 frame is reduced to its RMS *around the frame mean* (a DC offset in the mic
//   path must not look like level), in dBFS.
// - The window's noise floor is the minimum frame level seen. Every mic window on this
//   half-duplex device starts with the mic's own silence (it was off during playback), so the
//   window minimum is a true noise floor; a cleared window keeps the floor honest.
// - A frame is "speech" when its level exceeds floor + SPEECH_GATE_DB. Percentiles are taken over
//   speech frames only, so endpoint silence and zero-padded flushes do not drag the p90 down.
// - speech_seen() (>= SPEECH_MIN_FRAMES gated frames) answers "did anybody talk in this window",
//   used to tell the two listen_end meanings apart on a no-AEC device.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace esphome::gaja_device {

// 1 dB bins over the possible range of 20 ms frame RMS levels (dBFS).
static constexpr int LEVEL_DB_MIN = -90;
static constexpr int LEVEL_DB_MAX = 0;
static constexpr int LEVEL_DB_BINS = LEVEL_DB_MAX - LEVEL_DB_MIN + 1;

// A frame counts as speech when it is this far above the window's noise floor.
static constexpr float SPEECH_GATE_DB = 6.0f;
// ... and a window counts as "somebody spoke" only with this many gated frames (60 ms at 20 ms).
static constexpr uint32_t SPEECH_MIN_FRAMES = 3;

class LevelTracker {
 public:
  void clear() {
    memset(this->all_hist_, 0, sizeof(this->all_hist_));
    memset(this->speech_hist_, 0, sizeof(this->speech_hist_));
    this->frames_ = 0;
    this->speech_frames_ = 0;
    this->floor_db_ = LEVEL_DB_MAX;  // sinks to the window minimum as frames arrive
    this->max_db_ = LEVEL_DB_MIN;
  }

  /// ``len`` bytes of PCM16 LE mono (any length; odd trailing bytes ignored).
  void note_frame(const uint8_t *frame, size_t len) {
    const size_t samples = len / 2;
    if (samples == 0) {
      return;
    }
    const int16_t *pcm = (const int16_t *) (const void *) frame;
    // RMS around the frame mean: sum/sum_sq let us subtract the DC offset in closed form.
    int64_t sum = 0;
    int64_t sum_sq = 0;
    for (size_t i = 0; i < samples; i++) {
      sum += pcm[i];
      sum_sq += (int64_t) pcm[i] * pcm[i];
    }
    const double mean = (double) sum / (double) samples;
    double var = (double) sum_sq / (double) samples - mean * mean;
    if (var < 0.0) {
      var = 0.0;  // rounding
    }
    const float rms = (float) sqrt(var);
    const float db = 20.0f * log10f(rms / 32768.0f + 1e-12f);
    const int bin = LevelTracker::db_to_bin_(db);

    this->all_hist_[bin]++;
    this->frames_++;
    if (db > this->max_db_) {
      this->max_db_ = db;
    }
    if (db < this->floor_db_) {
      this->floor_db_ = db;  // the floor only sinks within a window
    }
    if (db > this->floor_db_ + SPEECH_GATE_DB) {
      this->speech_hist_[bin]++;
      this->speech_frames_++;
    }
  }

  uint32_t frames() const { return this->frames_; }
  uint32_t speech_frames() const { return this->speech_frames_; }
  float floor_db() const { return this->floor_db_; }
  float max_db() const { return this->max_db_; }
  /// True when the window contains real speech (not just the odd noise spike).
  bool speech_seen() const { return this->speech_frames_ >= SPEECH_MIN_FRAMES; }

  /// Fraction in [0,1] of the *speech* frames' level distribution, in dBFS; NAN when nobody spoke.
  float speech_percentile_db(float fraction) const {
    if (this->speech_frames_ == 0) {
      return NAN;
    }
    const float target = fraction * (float) this->speech_frames_;
    uint32_t acc = 0;
    for (int bin = 0; bin < LEVEL_DB_BINS; bin++) {
      acc += this->speech_hist_[bin];
      if ((float) acc >= target) {
        return (float) (bin + LEVEL_DB_MIN);
      }
    }
    return (float) LEVEL_DB_MAX;
  }

 private:
  static int db_to_bin_(float db) {
    int bin = (int) std::ceil(db) - LEVEL_DB_MIN;
    if (bin < 0) {
      bin = 0;
    } else if (bin >= LEVEL_DB_BINS) {
      bin = LEVEL_DB_BINS - 1;
    }
    return bin;
  }

  uint32_t all_hist_[LEVEL_DB_BINS]{};
  uint32_t speech_hist_[LEVEL_DB_BINS]{};
  uint32_t frames_{0};
  uint32_t speech_frames_{0};
  float floor_db_{LEVEL_DB_MAX};  // minimum frame level seen in this window
  float max_db_{LEVEL_DB_MIN};
};

}  // namespace esphome::gaja_device

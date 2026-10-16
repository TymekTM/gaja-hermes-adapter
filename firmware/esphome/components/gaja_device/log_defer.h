#pragma once

// Deferred log lines for the web UI log stream (pure C++, unit-tested on the host:
// tests/test_firmware_log_defer.py). Every log line pushed to an open web UI / SSE log stream
// costs network time, and a busy stream measurably stalls the device's downlink mid-reply. So
// while the device is busy the lines are only recorded here, and the main loop forwards them
// once the device is idle again.
//
// A byte ring of records [u16 length][u8 level][bytes]. When a new line does not fit, the OLDEST
// lines are discarded (counted in dropped()): the most recent history is the useful part. NOT
// thread-safe on its own: the owner serialises push()/pop() with one mutex. Storage is external
// (``attach``: PSRAM).

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace esphome::gaja_device {

class LogDefer {
 public:
  static constexpr size_t HEADER = 3;
  static constexpr size_t MAX_LINE = 512;  // longer lines are truncated

  void attach(uint8_t *buffer, size_t capacity) {
    this->buf_ = buffer;
    this->cap_ = capacity;
    this->head_ = this->tail_ = this->used_ = 0;
    this->lines_ = 0;
    this->dropped_ = 0;
  }
  bool attached() const { return this->buf_ != nullptr && this->cap_ > HEADER + MAX_LINE; }

  /// Record one line (truncated to MAX_LINE). Evicts the oldest lines to make room.
  void push(uint8_t level, const char *msg, size_t len) {
    if (!this->attached()) {
      return;
    }
    if (len > MAX_LINE) {
      len = MAX_LINE;
    }
    const size_t need = HEADER + len;
    while (this->cap_ - this->used_ < need) {
      this->discard_oldest_();
      this->dropped_++;
    }
    const uint8_t hdr[HEADER] = {(uint8_t) (len & 0xFF), (uint8_t) (len >> 8), level};
    this->write_(hdr, HEADER);
    this->write_((const uint8_t *) msg, len);
    this->lines_++;
  }

  /// Copy the oldest line into ``out`` (at least MAX_LINE + 1 bytes; NUL-terminated) and remove it.
  /// Returns false when empty.
  bool pop(uint8_t &level, char *out, size_t &len) {
    if (this->lines_ == 0) {
      return false;
    }
    uint8_t hdr[HEADER];
    this->read_(hdr, HEADER);
    len = (size_t) hdr[0] | ((size_t) hdr[1] << 8);
    level = hdr[2];
    this->read_((uint8_t *) out, len);
    out[len] = '\0';
    this->lines_--;
    return true;
  }

  size_t lines() const { return this->lines_; }
  /// Lines discarded since the last take_dropped().
  uint32_t take_dropped() {
    const uint32_t d = this->dropped_;
    this->dropped_ = 0;
    return d;
  }

 protected:
  void write_(const uint8_t *src, size_t n) {
    const size_t first = n < this->cap_ - this->head_ ? n : this->cap_ - this->head_;
    memcpy(this->buf_ + this->head_, src, first);
    memcpy(this->buf_, src + first, n - first);
    this->head_ = (this->head_ + n) % this->cap_;
    this->used_ += n;
  }
  void read_(uint8_t *dst, size_t n) {
    const size_t first = n < this->cap_ - this->tail_ ? n : this->cap_ - this->tail_;
    if (dst != nullptr) {
      memcpy(dst, this->buf_ + this->tail_, first);
      memcpy(dst + first, this->buf_, n - first);
    }
    this->tail_ = (this->tail_ + n) % this->cap_;
    this->used_ -= n;
  }
  void discard_oldest_() {
    uint8_t hdr[HEADER];
    this->read_(hdr, HEADER);
    this->read_(nullptr, (size_t) hdr[0] | ((size_t) hdr[1] << 8));
    this->lines_--;
  }

  uint8_t *buf_{nullptr};
  size_t cap_{0};
  size_t head_{0};
  size_t tail_{0};
  size_t used_{0};
  size_t lines_{0};
  uint32_t dropped_{0};
};

}  // namespace esphome::gaja_device

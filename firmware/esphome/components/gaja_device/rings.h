#pragma once

// SPSC byte rings for the Gaja device audio paths. Pure C++ (no ESPHome includes) so they can
// be unit-tested on the host (tests/test_firmware_rings.py).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace esphome::gaja_device {

/// Single-producer/single-consumer byte ring buffer (drops new data when full).
/// Only the producer writes ``head_``, only the consumer writes ``tail_`` (the peek/advance split
/// keeps the playback path lock-free). Indices are monotonically increasing counters; the modulo
/// happens on access, so ``clear()`` never corrupts the producer's position.
class SpscRing {
 public:
  void attach(uint8_t *buffer, size_t capacity) {
    this->buffer_ = buffer;
    this->capacity_ = capacity;
  }
  bool attached() const { return this->buffer_ != nullptr && this->capacity_ > 0; }

  /// Consumer side only (e.g. the main loop): pretend everything was consumed. The producer may
  /// keep pushing concurrently; resetting from any other thread corrupts the indices (that is why
  /// there is no reset()).
  void clear() {
    const size_t head = this->head_.load(std::memory_order_acquire);
    this->tail_.store(head, std::memory_order_release);
  }

  /// Producer side. Returns false when the ring was full and the data was dropped.
  bool push(const uint8_t *data, size_t len);
  /// Consumer side: copy up to ``len`` bytes without consuming them.
  size_t peek(uint8_t *dst, size_t len) const;
  /// Consumer side: consume ``len`` bytes previously peeked.
  void advance(size_t len);
  /// Free space the producer may write right now.
  size_t space() const {
    const size_t head = this->head_.load(std::memory_order_relaxed);
    const size_t tail = this->tail_.load(std::memory_order_acquire);
    return this->capacity_ - (head - tail);
  }
  size_t available() const {
    return this->head_.load(std::memory_order_acquire) - this->tail_.load(std::memory_order_relaxed);
  }
  uint32_t dropped() const { return this->dropped_.load(std::memory_order_relaxed); }

 protected:
  uint8_t *buffer_{nullptr};
  size_t capacity_{0};
  std::atomic<size_t> head_{0};  // write position (producer)
  std::atomic<size_t> tail_{0};  // read position (consumer)
  std::atomic<uint32_t> dropped_{0};
};

/// Rolling pre-wake window: the microphone task keeps overwriting the oldest audio. Positions are
/// absolute counters, so a reader can pin a snapshot (start position taken at one instant) and
/// peek it while the producer keeps running; it only goes stale once the producer laps it.
class RollingRing {
 public:
  void attach(uint8_t *buffer, size_t capacity) {
    this->buffer_ = buffer;
    this->capacity_ = capacity;
  }
  bool attached() const { return this->buffer_ != nullptr && this->capacity_ > 0; }

  /// Producer side; overwrites the oldest bytes when full.
  void push(const uint8_t *data, size_t len);
  /// Absolute position of the oldest retained byte ([tail_pos, head_pos) is the live window).
  size_t tail_pos() const;
  /// Absolute position one past the newest byte.
  size_t head_pos() const;
  /// Copy ``len`` bytes starting at absolute position ``abs`` (within the retained window).
  size_t peek_abs(size_t abs, uint8_t *dst, size_t len) const;

 protected:
  uint8_t *buffer_{nullptr};
  size_t capacity_{0};
  std::atomic<size_t> head_{0};  // absolute bytes ever written
};

inline bool SpscRing::push(const uint8_t *data, size_t len) {
  if (this->capacity_ == 0) {
    return false;
  }
  const size_t head = this->head_.load(std::memory_order_relaxed);
  const size_t tail = this->tail_.load(std::memory_order_acquire);
  const size_t used = head - tail;
  if (len > this->capacity_ - used) {
    this->dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;  // drop NEW data; the consumer keeps what it already has
  }
  const size_t start = head % this->capacity_;
  const size_t first = std::min(len, this->capacity_ - start);
  memcpy(this->buffer_ + start, data, first);
  if (len > first) {
    memcpy(this->buffer_, data + first, len - first);
  }
  this->head_.store(head + len, std::memory_order_release);
  return true;
}

inline size_t SpscRing::peek(uint8_t *dst, size_t len) const {
  const size_t head = this->head_.load(std::memory_order_acquire);
  const size_t tail = this->tail_.load(std::memory_order_relaxed);
  len = std::min(len, head - tail);
  const size_t start = tail % this->capacity_;
  const size_t first = std::min(len, this->capacity_ - start);
  memcpy(dst, this->buffer_ + start, first);
  if (len > first) {
    memcpy(dst + first, this->buffer_, len - first);
  }
  return len;
}

inline void SpscRing::advance(size_t len) {
  const size_t head = this->head_.load(std::memory_order_acquire);
  const size_t tail = this->tail_.load(std::memory_order_relaxed);
  len = std::min(len, head - tail);
  this->tail_.store(tail + len, std::memory_order_release);  // indices stay monotonic
}

inline void RollingRing::push(const uint8_t *data, size_t len) {
  if (this->capacity_ == 0) {  // preroll disabled
    return;
  }
  // Keep only the newest window, but still count every byte: head_pos() is "bytes ever written".
  const size_t retain = std::min(len, this->capacity_);
  data += len - retain;
  const size_t head = this->head_.load(std::memory_order_relaxed);
  const size_t new_head = head + len;
  const size_t start = (new_head - retain) % this->capacity_;
  const size_t first = std::min(retain, this->capacity_ - start);
  memcpy(this->buffer_ + start, data, first);
  if (retain > first) {
    memcpy(this->buffer_, data + first, retain - first);
  }
  this->head_.store(new_head, std::memory_order_release);
}

inline size_t RollingRing::head_pos() const { return this->head_.load(std::memory_order_acquire); }

inline size_t RollingRing::tail_pos() const {
  const size_t head = this->head_.load(std::memory_order_acquire);
  return head > this->capacity_ ? head - this->capacity_ : 0;
}

inline size_t RollingRing::peek_abs(size_t abs, uint8_t *dst, size_t len) const {
  if (this->capacity_ == 0) {
    return 0;
  }
  const size_t head = this->head_.load(std::memory_order_acquire);
  const size_t tail = head > this->capacity_ ? head - this->capacity_ : 0;
  if (abs < tail) {
    return 0;  // overwritten (producer lapped us); caller decides how to handle the gap
  }
  len = std::min(len, head - abs);
  const size_t start = abs % this->capacity_;
  const size_t first = std::min(len, this->capacity_ - start);
  memcpy(dst + 0, this->buffer_ + start, first);
  if (len > first) {
    memcpy(dst + first, this->buffer_, len - first);
  }
  return len;
}

}  // namespace esphome::gaja_device

#pragma once

// Bounded FIFO of server text messages with CLASS-AWARE overflow (pure C++, unit-tested on the host:
// tests/test_firmware_text_queue.py). NOT thread-safe on its own: the owner serialises push()/pop()
// with one mutex (the WebSocket task pushes, the main loop pops).
//
// Overflow rules (the item type carries ``bool droppable``, see server_msg.h):
//   * a droppable (log-only) message is discarded once the queue is more than half full;
//   * when the queue is completely full, the OLDEST droppable message is evicted to make room;
//   * a control message is never evicted. If the queue is full of control messages the NEW one is
//     refused (the caller logs it) -- the control messages already queued are what the state machine
//     still has to process in order.
//
// Storage is external (``attach``: PSRAM). Items are addressed through an index array so that
// evicting from the middle moves bytes, not whole 1 KiB items.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace esphome::gaja_device {

template<typename Item, size_t N> class TextQueue {
  static_assert(N > 0 && N <= 255, "TextQueue indices are uint8_t");

 public:
  enum class Result : uint8_t {
    PUSHED,          // enqueued
    PUSHED_EVICTED,  // enqueued after evicting the oldest droppable message
    DROPPED_NEW,     // the new message was discarded (droppable under pressure, or no room)
  };

  /// ``storage`` must hold N items and outlive the queue.
  void attach(Item *storage) {
    this->slots_ = storage;
    this->count_ = 0;
    for (size_t i = 0; i < N; i++) {
      this->free_[i] = (uint8_t) i;
    }
    this->free_count_ = N;
  }
  bool attached() const { return this->slots_ != nullptr; }
  size_t size() const { return this->count_; }
  static constexpr size_t capacity() { return N; }

  /// Producer side. ``item.droppable`` classifies the message.
  Result push(const Item &item) {
    if (this->slots_ == nullptr) {
      return Result::DROPPED_NEW;
    }
    if (item.droppable && this->count_ > N / 2) {
      return Result::DROPPED_NEW;  // log-only traffic never competes with control messages
    }
    Result result = Result::PUSHED;
    if (this->count_ >= N) {
      size_t victim = N;
      for (size_t i = 0; i < this->count_; i++) {
        if (this->slots_[this->order_[i]].droppable) {
          victim = i;
          break;
        }
      }
      if (victim == N) {
        return Result::DROPPED_NEW;  // all control: keep what is queued, refuse the new one
      }
      this->free_[this->free_count_++] = this->order_[victim];
      memmove(&this->order_[victim], &this->order_[victim + 1], (this->count_ - victim - 1) * sizeof(uint8_t));
      this->count_--;
      result = Result::PUSHED_EVICTED;
    }
    const uint8_t slot = this->free_[--this->free_count_];
    this->slots_[slot] = item;
    this->order_[this->count_++] = slot;
    return result;
  }

  /// Consumer side: oldest message into ``out``. False when empty.
  bool pop(Item &out) {
    if (this->count_ == 0) {
      return false;
    }
    const uint8_t slot = this->order_[0];
    out = this->slots_[slot];
    memmove(&this->order_[0], &this->order_[1], (this->count_ - 1) * sizeof(uint8_t));
    this->count_--;
    this->free_[this->free_count_++] = slot;
    return true;
  }

 protected:
  Item *slots_{nullptr};
  uint8_t order_[N]{};  // slot ids, oldest first
  uint8_t free_[N]{};   // free slot ids (stack)
  size_t count_{0};
  size_t free_count_{0};
};

}  // namespace esphome::gaja_device

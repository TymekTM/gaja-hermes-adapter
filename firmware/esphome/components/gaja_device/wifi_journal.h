#pragma once

// Pure C++ RTC-backed Wi-Fi event journal (no ESP-IDF / ESPHome includes; host-tested).
//
// Records compact Wi-Fi / IP / boot milestones inside an `RTC_NOINIT_ATTR` ring buffer so events
// captured while the device is offline — and across software resets, panics or watchdogs — survive
// until the next WebSocket session is ready to upload them.
//
// `WifiJournalStorage<N>` MUST remain trivially default-constructible and standard-layout so the
// C++ runtime startup code never zero-initializes `.rtc_noinit` on warm resets. Callers provide
// external synchronization when appending from `esp_event` and draining from the main loop.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace esphome::gaja_device {

enum class WifiEventKind : uint8_t {
  BOOT = 1,
  DISCONNECTED = 2,
  CONNECTED = 3,
  GOT_IP = 4,
  LOST_IP = 5,
  SCAN_DONE = 6,
};

inline constexpr uint8_t WIFI_EVENT_FLAG_HAS_BSSID = 1u << 0;
inline constexpr uint8_t WIFI_EVENT_FLAG_HAS_CHANNEL = 1u << 1;
inline constexpr uint8_t WIFI_EVENT_FLAG_HAS_RSSI = 1u << 2;

inline const char *wifi_event_kind_str(WifiEventKind kind) {
  switch (kind) {
    case WifiEventKind::BOOT:
      return "boot";
    case WifiEventKind::DISCONNECTED:
      return "disconnected";
    case WifiEventKind::CONNECTED:
      return "connected";
    case WifiEventKind::GOT_IP:
      return "got_ip";
    case WifiEventKind::LOST_IP:
      return "lost_ip";
    case WifiEventKind::SCAN_DONE:
      return "scan_done";
    default:
      return "unknown";
  }
}

struct WifiEventRecord {
  uint32_t ms;       // monotonic milliseconds since the event's boot
  uint32_t boot;     // boot number when the event happened (1-based)
  uint16_t reason;   // disconnect reason, esp_reset_reason_t, scan status, or ip_changed flag
  WifiEventKind kind;
  uint8_t channel;   // primary channel when WIFI_EVENT_FLAG_HAS_CHANNEL is set
  int8_t rssi;       // dBm when WIFI_EVENT_FLAG_HAS_RSSI is set
  uint8_t flags;     // WIFI_EVENT_FLAG_HAS_*
  uint8_t bssid[6];  // AP MAC when WIFI_EVENT_FLAG_HAS_BSSID is set
};

template <size_t N>
struct WifiJournalStorage {
  uint32_t magic;
  uint16_t version;
  uint16_t capacity;
  uint32_t boot;
  uint32_t head;      // monotonic 32-bit write sequence (wraps mod 2^32)
  uint32_t tail;      // monotonic 32-bit oldest undelivered sequence
  uint32_t dropped;   // undelivered events overwritten since last mark_delivered()
  uint16_t reset_reason;
  uint16_t reserved;
  uint32_t checksum;  // IEEE CRC-32 over header [0..checksum) + entries[]
  WifiEventRecord entries[N];
};

struct WifiJournalSnapshot {
  uint32_t ack_until;
  uint32_t dropped;
  size_t count;
};

namespace detail {

inline uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len) {
  static constexpr uint32_t NIBBLE[16] = {
      0x00000000u, 0x1DB71064u, 0x3B6E20C8u, 0x26D930ACu,
      0x76DC4190u, 0x6B6B51F4u, 0x4DB26158u, 0x5005713Cu,
      0xEDB88320u, 0xF00F9344u, 0xD6D6A3E8u, 0xCB61B38Cu,
      0x9B64C2B0u, 0x86D3D2D4u, 0xA00AE278u, 0xBDBDF21Cu,
  };
  for (size_t i = 0; i < len; i++) {
    uint8_t byte = data[i];
    crc ^= byte;
    crc = (crc >> 4) ^ NIBBLE[crc & 0x0Fu];
    crc = (crc >> 4) ^ NIBBLE[crc & 0x0Fu];
  }
  return crc;
}

}  // namespace detail

template <size_t N = 64>
class WifiJournal {
  static_assert(N > 0 && (N & (N - 1)) == 0,
                "WifiJournal capacity must be a power of two for clean uint32_t wraparound");

 public:
  static constexpr uint32_t MAGIC = 0x47574A31u;  // "GWJ1"
  static constexpr uint16_t VERSION = 1;
  static constexpr size_t CAPACITY = N;

  void attach(WifiJournalStorage<N> *storage) {
    this->storage_ = storage;
    this->ready_ = false;
  }
  bool attached() const { return this->storage_ != nullptr; }

  static uint32_t compute_checksum(const WifiJournalStorage<N> &s) {
    const uint8_t *base = reinterpret_cast<const uint8_t *>(&s);
    uint32_t crc = 0xFFFFFFFFu;
    crc = detail::crc32_update(crc, base, offsetof(WifiJournalStorage<N>, checksum));
    crc = detail::crc32_update(
        crc, reinterpret_cast<const uint8_t *>(s.entries), sizeof(s.entries));
    return crc ^ 0xFFFFFFFFu;
  }

  // Full validation including the CRC over the whole storage (~1.3 KiB of slow RTC memory). Writes
  // (begin_boot, append, mark_delivered) run it so a bit flip in RTC memory freezes the journal until the
  // next boot wipes it, instead of being sealed under a fresh CRC. Writes are rare (a Wi-Fi event, a
  // delivered batch); the reads polled from the idle loop (pending, snapshot) use the O(1) ready().
  //
  // A reset landing inside a write (between the record copy and the checksum store, ~100 us) leaves a
  // CRC mismatch, and the next boot starts an empty journal: whole-struct checksums trade that small
  // window for simple, layout-agnostic validation.
  bool is_valid() const {
    if (this->storage_ == nullptr) {
      return false;
    }
    const WifiJournalStorage<N> &s = *this->storage_;
    if (s.magic != MAGIC || s.version != VERSION || s.capacity != (uint16_t) N || s.boot == 0) {
      return false;
    }
    const uint32_t used = (uint32_t) (s.head - s.tail);
    if (used > N) {
      return false;
    }
    return s.checksum == compute_checksum(s);
  }

  void begin_boot(uint16_t reset_reason, bool force_reset) {
    if (this->storage_ == nullptr) {
      return;
    }
    WifiJournalStorage<N> &s = *this->storage_;
    if (force_reset || !this->is_valid()) {
      std::memset(&s, 0, sizeof(s));
      s.magic = MAGIC;
      s.version = VERSION;
      s.capacity = (uint16_t) N;
      s.boot = 1;
      s.head = 0;
      s.tail = 0;
      s.dropped = 0;
      s.reset_reason = reset_reason;
      s.reserved = 0;
      s.checksum = compute_checksum(s);
      this->ready_ = true;
      return;
    }
    s.boot += 1;
    s.reset_reason = reset_reason;
    s.checksum = compute_checksum(s);
    this->ready_ = true;
  }

  void append(uint32_t ms,
              WifiEventKind kind,
              uint16_t reason = 0,
              const uint8_t *bssid = nullptr,
              uint8_t channel = 0,
              int8_t rssi = 0,
              uint8_t flags = 0) {
    if (!this->ready() || !this->is_valid()) {
      return;
    }
    WifiJournalStorage<N> &s = *this->storage_;
    if ((uint32_t) (s.head - s.tail) >= N) {
      s.tail += 1;
      s.dropped += 1;
    }
    WifiEventRecord &rec = s.entries[s.head % N];
    rec.ms = ms;
    rec.boot = s.boot;
    rec.reason = reason;
    rec.kind = kind;
    rec.channel = channel;
    rec.rssi = rssi;
    rec.flags = flags;
    if (bssid != nullptr) {
      std::memcpy(rec.bssid, bssid, 6);
    } else {
      std::memset(rec.bssid, 0, 6);
    }
    s.head += 1;
    s.checksum = compute_checksum(s);
  }

  WifiJournalSnapshot snapshot(WifiEventRecord *out, size_t max_count) const {
    if (!this->ready() || out == nullptr || max_count == 0) {
      return WifiJournalSnapshot{0, 0, 0};
    }
    const WifiJournalStorage<N> &s = *this->storage_;
    const uint32_t avail = (uint32_t) (s.head - s.tail);
    const size_t n = std::min((size_t) avail, max_count);
    for (size_t i = 0; i < n; i++) {
      out[i] = s.entries[(uint32_t) (s.tail + (uint32_t) i) % N];
    }
    return WifiJournalSnapshot{s.tail + (uint32_t) n, s.dropped, n};
  }

  void mark_delivered(const WifiJournalSnapshot &snap) {
    if (!this->ready() || !this->is_valid()) {
      return;
    }
    WifiJournalStorage<N> &s = *this->storage_;
    const int32_t advance = (int32_t) (snap.ack_until - s.tail);
    if (advance > 0) {
      const uint32_t avail = (uint32_t) (s.head - s.tail);
      const uint32_t step = std::min((uint32_t) advance, avail);
      s.tail += step;
    }
    const uint32_t acked_drops = std::min(s.dropped, snap.dropped);
    s.dropped -= acked_drops;
    s.checksum = compute_checksum(s);
  }

  size_t pending() const {
    if (!this->ready()) {
      return 0;
    }
    return (size_t) (this->storage_->head - this->storage_->tail);
  }

  uint32_t boot() const { return this->storage_ != nullptr ? this->storage_->boot : 0; }
  uint32_t dropped() const { return this->storage_ != nullptr ? this->storage_->dropped : 0; }
  uint16_t reset_reason() const { return this->storage_ != nullptr ? this->storage_->reset_reason : 0; }

 private:
  // O(1) sanity check for the polled reads; writes additionally verify the CRC (is_valid).
  bool ready() const {
    return this->ready_ && this->storage_ != nullptr && this->storage_->magic == MAGIC &&
           (uint32_t) (this->storage_->head - this->storage_->tail) <= N;
  }

  WifiJournalStorage<N> *storage_{nullptr};
  bool ready_{false};  // begin_boot ran on the attached storage
};

}  // namespace esphome::gaja_device

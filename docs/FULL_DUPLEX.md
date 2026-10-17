# Full-duplex Gaja Atom (design note, protocol v1.11)

Goal: the user can interrupt Gaja mid-reply. Today the firmware is strictly half-duplex — one
I2S bus shared by mic and speaker, the mic is stopped while the speaker plays, hello advertises
`caps.aec = false`. Fallback: everything below is gated on a `duplex` option; `duplex: false`
keeps the current behaviour byte-for-byte.

## 1. Can stock ESPHome i2s_audio do RX+TX at once? — No.

Both the i2s_audio microphone and the i2s_audio speaker take an exclusive mutex on their shared
parent `I2SAudioComponent` for as long as their driver runs, and each allocates only ONE
direction of the port:

- `esphome/components/i2s_audio/i2s_audio.h:69`: `Mutex lock_;` in `I2SAudioComponent` (one
  instance per `i2s_audio:` block = one I2S port).
- Mic, `i2s_audio/microphone/i2s_audio_microphone.cpp:98`: `if (!this->parent_->try_lock()) {
  return false; }` — held until `stop_driver_()`; `:112`: `err = i2s_new_channel(&chan_cfg,
  NULL, &this->rx_handle_);` (RX handle only).
- Speaker, `i2s_audio/speaker/i2s_audio_speaker.cpp:270`: `i2s_new_channel(&chan_cfg,
  &this->tx_handle_, NULL);` (TX only); on any start failure or stop it calls
  `this->parent_->unlock()` (`:273`, `:282`, `:305`, `:327`).

So "Parent bus is busy": the mic holds the lock while RUNNING, the speaker can only start while
the mic is fully stopped, and vice versa. Two `i2s_audio:` blocks (the ESP32-S3 has two
controllers, `soc/esp32s3/include/soc/soc_caps.h:228`: `#define SOC_I2S_NUM (2U)`) do not help:
the board wires ONE set of clock pins to the ES8311 (lrclk GPIO3, bclk GPIO17, mclk GPIO11), a
GPIO can be driven by only one controller, and ESPHome's i2s_audio pin schema is
output-only — the codec needs the clocks from whichever controller runs, so both directions
must live on ONE controller.

## 2. The minimal change that works: one custom duplex driver on the port

ESP-IDF explicitly supports full duplex on one controller when one `i2s_new_channel` call
allocates both directions:

- `esp_driver_i2s/include/driver/i2s_common.h:116-121`: "If both tx_handle and rx_handle are
  not NULL, it means this I2S controller will work at full-duplex mode, the RX and TX channels
  will be allocated on a same I2S port in this case. (...) full-duplex mode can't guarantee
  TX/RX channels write/read synchronously, they can only share the clock signals for now."
- `esp_driver_i2s/include/driver/i2s_std.h:302-304` (note on `i2s_channel_init_std_mode`):
  "When initialize the STD mode with a same configuration as another channel on a same port,
  these two channels can constitude as full-duplex mode automatically".

Consequences, designed around:

- **One shared clock → one sample rate.** The ES8311 is verified at 48000 Hz (at 16000 the ADC
  returns white noise — yaml header, verified on device). So both directions run at 48 kHz,
  std/Philips, 16-bit, left slot (mic ADC data on left, speaker on left — verified). The mic
  path gets an on-device 3:1 polyphase decimator 48k→16k (new pure header `decim3.h`, host
  tested like the other `tests/test_firmware_*.py` headers). MCLK stays on
  (mclk_multiple 256 → 12.288 MHz, same as the verified speaker config).
- **The duplex bus is owned by a new gaja_device platform pair**, not by i2s_audio:
  `microphone: platform: gaja_device` and `speaker: platform: gaja_device`
  (`components/gaja_device/microphone|speaker/`). They implement the stock interfaces, so
  micro_wake_word, sound_level, media_player(speaker) and gaja_device keep working unchanged:
  - `microphone.h:19-52`: `class Microphone` needs only `start()/stop()` + `state_` transitions
    + `data_callbacks_`; `microphone_source.h:16,23`: `MicrophoneSource::start()/stop()`
    delegate to `this->mic_->start()/stop()`; sources handle gain (gain_factor) and channel
    masking, and "cannot convert sample rates" — hence the on-device decimation.
  - `speaker.h`: `play()`, `start()`, `stop()`, `has_buffered_data()` are the pure virtuals;
    base `set_volume()` already forwards to a linked `audio_dac` — the ES8311 component is
    I2C-only (`es8311/es8311.cpp` writes codec registers; no I2S code), so it stays as-is and
    keeps handling DAC volume/mute.
- **Reference signal for AEC comes free**: the driver pairs every RX chunk with the TX samples
  fed in the same wall-clock window (both decimated to 16k with the SAME decimator, so equal
  group delay) and pushes them into two PSRAM rings together. The residual mic↔ref offset (TX
  DMA depth + acoustic path + ADC DMA depth, tens of ms, stable per boot) is absorbed by the
  server-side delay search (below).
- Amp discipline unchanged: NS4150B enable stays ALWAYS_OFF, on 50 ms before playback, off
  after drain. The mic simply no longer has to release anything.

## 3. Echo: how barge-in must not fire on Gaja's own voice

Options evaluated:

**(a) On-device AEC with ESP-SR AFE (S3 + PSRAM are available).** ESPHome has no AFE
integration (micro_wake_word runs tflite-micro; the build tree has
`espressif__esp-tflite-micro`, no esp-sr). esp-sr would enter as a new IDF managed component
via `esp32.add_idf_component(...)`; its AFE AEC footprint (internal RAM, extra tasks, sdkconfig
expectations) is unverified against this build (4 MB playback ring + WiFi + WS already resident),
a bad interaction shows up as a runtime reboot, and none of it can be validated in this
environment (no device). Unverified-risk — rejected for this iteration.

**(b) Device sends mic + time-aligned playback reference; server cancels echo — CHOSEN.**
Python AEC packages were checked WITHOUT installing: `speexdsp-python` does not exist on PyPI
(404 on `pypi.org/pypi/speexdsp-python/json`); `webrtc-audio-processing` exists but version
0.1.3 is from 2019, marked Pre-Alpha, sdist/armv7 wheel only (source build with swig) — no
maintained binding with x86_64 wheels. So the AEC is implemented in-repo
(`gaja/devices/aec.py`) on numpy/scipy (both already dependencies): a coarse delay estimator
(cross-correlation over 0..240 ms) + frequency-domain block NLMS echo cancellation, fully
deterministic and unit-testable with synthetic echo (reference convolved + delayed +
attenuated), which is exactly the test the task requires. This reuses the proven web-client
path: `pipeline.py:456` `self._endpointer.set_strict(time.perf_counter() < self._playback_until)`
(VAD raises the barge-in bar while Gaja talks — its docstring in `gaja/audio/vad.py:33`
says it "keeps speaker echo from barging in"), and `pipeline.py:582-587` VAD `speech_start` →
`_interrupt_speech(reason="barge_in")`. For devices, `controller.py:304-305` already skips
`listen_end` for `caps.aec` clients, so an AEC device's mic window stays open through the
reply. Bandwidth cost: one extra 640 B frame per 20 ms during playback (~32 kB/s, Wi-Fi).

**(c) Wake-word-only barge-in during playback (no AEC).** micro_wake_word would listen to the
raw mic — i.e. to Gaja's own voice. A busy wake is accepted at `score >= barge_in_min_score`
(0.9; this firmware sends the constant 0.97, `controller.py:182`), and a no-AEC busy wake is
accepted WITHOUT verification (`controller.py:194-199`: `own_voice = busy and not
self.caps.aec` → pre-roll skipped, `verify = ... and not own_voice`) — so a single false trigger
on Gaja's own speech would cut the reply. Kept only as the `duplex: false` fallback (today's
behaviour, unchanged).

## 4. Protocol v1.11 (delta over v1.10)

- hello `caps`: `aec: true` + new `ref: true` when `duplex: true` (firmware
  `gaja-esphome/0.3.0`). `ref: false` = old half-duplex framing (raw PCM, no tags).
- Device→server binary frames, only when `caps.ref`: each frame carries a 1-byte tag:
  `0x00` = mic PCM16 16 kHz, `0x01` = playback reference (the decimated copy of what the device
  is feeding its speaker over the same 20 ms window). Mic frames keep flowing during playback;
  ref frames are sent while playback is audible.
- New server→device `{"type":"playback_stop"}`: the session cancelled the reply (VAD barge-in
  or a superseding wake). The device flushes its playback ring (up to ~44 s of audio buffered —
  unlike a browser, it would otherwise keep playing), stops the speaker, amp off, mic keeps
  streaming.
- Wake during playback (`caps.aec` devices): no longer interrupts at wake time; the wake is
  verified on the AEC-cleaned pre-roll and the interrupt happens when it is ACCEPTED (a false
  wake no longer cuts the reply). No-AEC devices keep the v1.10 behaviour (immediate cancel,
  polluted pre-roll skipped, no verification).

## 5. Phases

0. This note.
1. `decim3.h` (host-tested) + duplex I2S driver + gaja_device mic/speaker platforms +
   `duplex` state machine (mic never stops, WAKE_BEEP without mic pause, uplink during
   playback) + yaml switch. Compiles.
2. Wake-word barge-in during playback on the device (accept `wake` in PLAYING/LINGER/
   PREPARE_PLAY), server `playback_stop`, protocol doc.
3. Server AEC (`gaja/devices/aec.py` + tests with synthetic echo), tagged-frame routing,
   deferred wake interrupt for `caps.aec`, v1.11 doc, version bumps (protocol v1.11,
   `gaja-esphome/0.3.0` in yaml `firmware_version` and `gaja_device.h` `fw_version_`).

Everything hardware-dependent is UNVERIFIED until the lead flashes and tests: duplex I2S clocks
with the ES8311, 48k decimated mic level (gain_factor 4 may need rebalancing), AEC quality on
the real acoustic path, mww false-trigger rate during playback.

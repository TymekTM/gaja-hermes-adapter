# TODO

## Parakeet STT follow-ups

Parakeet (sherpa-onnx, CPU) is the turn engine since 2026-10-18 and faster-whisper small stays on the GPU for wake
verification ([STT_RESULTS.md, section 9](STT_RESULTS.md#9-parakeet-tdt-06b-v3-sherpa-onnx-cpu)). Open:
1. Check the live wake accept rate (the wake log in `/api/diagnostics`, admin panel diagnostics) and a day of turn transcripts against the bench.
2. The English set (`bench/data/stt/en`) was not measured with Parakeet.
3. Both engines are resident: about 1.1 GB more adapter RAM. If Parakeet can verify wakes as well as
   faster-whisper, whisper can go (frees ~380 MB VRAM).

## Device downlink stalls (stutters) — open, 2026-10-18

- Symptom: gaps mid-reply, then `device send failed (timeout)` after 12 s and a reconnect; the recovered
  reply then plays smoothly. Device side: `downlink stalled N ms mid-reply (0 B buffered)`.
- Measured with `ss -tni` during turns: lost segments (`lost:21`, `lost:42`), RTO backoff to 5-7 s, Send-Q
  stuck with the peer window still open (~64 KB) -> segments vanish, the device is not just slow to read.
- With the device idle the link is clean: 1000 x 1400 B pings at 2-2.5 Mbit/s, 0% loss; `POST
  /api/devices/<id>/say` of a 15 s sentence played without gaps (worst downlink gap 123 ms). The loss
  appears only during a real turn (uplink streaming -> PREPARE_PLAY -> PLAYING).
- Server pacing (9802c91, `[devices] downlink_lead_s`) did not remove it: a 12 s full outage still
  happened, and single 100 ms frames took > 1 s. Keep or retune once the cause is known.
- Next (daytime, on the device): check whether the outage starts at listen_end / PREPARE_PLAY (I2S
  mic->speaker switch); try `CONFIG_SPIRAM_TRY_ALLOCATE_WIFI_LWIP: n` (Wi-Fi/lwIP buffers in internal RAM),
  and a device-side prebuffer before playback starts.
- 2026-10-19 morning, same fw 0.2.5 and pacing, no changes overnight: user reports no stutters; 7 replies
  with worst downlink gap 51-943 ms, no send timeouts. One reconnect right after a long reply
  (`uplink ring lapped; skipping 20464 ms` -> `websocket transport error` in WAKE_BEEP), recovered at once.
  Not conclusive: the bad evening may be Wi-Fi conditions rather than the turn itself; keep watching.
- 2026-10-19 10:34: stalls again on 0.2.5 in the third long reply of a session (six 4-6 s stalls with
  0 B buffered, then the 12 s send timeout); the recovered reply and a 3.7 MB one after it were clean.
  Each reconnect also wiped the local LLM context (fixed in 8536eb6). Experiment running: fw 0.2.6t
  (~/gaja-esp-ota-026t, 0.2.5 + `CONFIG_SPIRAM_TRY_ALLOCATE_WIFI_LWIP: n` + debug heap sensors), flashed
  10:53. If stalls persist, roll back to ~/gaja-esp-ota-025 and look at the prebuffer / PREPARE_PLAY timing.

## Full-duplex firmware 0.3.0 — speaker silent

- On 0.3.0 nothing is audible: no boot chime, no wake beep, no replies, although turns run and the duplex
  speaker task writes I2S (the AEC reference arrives). The device was rolled back to 0.2.5 (built from
  37b6963 in ~/gaja-esp-ota-025). Suspects: ES8311 DAC init/unmute on the duplex bus, amp enable timing.
- Server-side AEC float64 crash fixed in a92ce66 (would have dropped every converged duplex turn).
- 0.2.5 double beep: a wake utterance holding only "Hej Gaja" opens a follow-up window, which beeps again.

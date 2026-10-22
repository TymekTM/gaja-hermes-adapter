# TODO

## Parakeet STT follow-ups

Parakeet (sherpa-onnx, CPU) is the turn engine since 2026-10-18 and faster-whisper small stays on the GPU for wake
verification ([STT_RESULTS.md, section 9](STT_RESULTS.md#9-parakeet-tdt-06b-v3-sherpa-onnx-cpu)). Open:
1. Check the live wake accept rate (the wake log in `/api/diagnostics`, admin panel diagnostics) and a day of turn transcripts against the bench.
2. The English set (`bench/data/stt/en`) was not measured with Parakeet.
3. Both engines are resident: about 1.1 GB more adapter RAM. If Parakeet can verify wakes as well as
   faster-whisper, whisper can go (frees ~380 MB VRAM).

## Household follow-ups

1. Per-person voices: `Person.tts` exists in `gaja/household.py` but nothing reads it yet; replies use the
   global TTS voice for everyone.
2. The admin panel has no household page: people, profiles and bindings are edited only through
   `GET`/`PUT /api/admin/household`.
3. Speaker identity comes only from a bound phone or a device bound to one person; voice ID is not built.

## Device downlink stalls (stutters) — open, 2026-10-18

Update 2026-10-22: on the full-duplex line (0.3.2–0.3.4) the user heard no stutters in live replies
(0.3.2 on 2026-10-21, 0.3.4 on 2026-10-22), so the stalls may belong to the 0.2.x half-duplex bus
switching. Keep this open until a few days of 0.3.x use confirm it.

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
- 0.2.6t did not help: five stalls > 1 s (up to 7 s) and a 12 s send timeout on 2026-10-21 11:22. Wi-Fi/lwIP
  buffer placement is not the cause.
- Related device-side mechanism (esp_websocket_client 1.8.0): a send whose write times out
  (`wlen == 0 && need_write != 0`) calls `esp_websocket_client_abort_connection` -> "websocket transport
  error" and a reconnect. Any TCP stall longer than the send timeout kills the link. 0.3.2 owns the TCP
  transport (`cfg.ext_transport`) and only writes when `select()` reports room; a socket full for 3 s
  reconnects on purpose ("uplink blocked for N ms").

## Full-duplex firmware 0.3.x

- 0.3.0 was silent: the duplex speaker never got its DOUT pin (fixed in 0.3.1, e4b836b).
- 0.3.1: wakes dropped the link while sending the pre-roll (the abort above); 0.3.2 gates writes on
  socket room. One 3 s block was still seen on the first wake after boot, later wakes were clean.
- First wake after an idle (2026-10-21 12:19, 2026-10-22 20:14): the adapter handled the wake text ~4 s
  after the device sent it, so the 3 s uplink guard reconnected a healthy link and the supersede dropped
  the turn. Likely the server's asyncio loop stalling on page-in after idle (12 GB swapped at the time).
  0.3.6 waits 8 s and logs the socket state while blocked; the stall itself is open (see the RAM work,
  and silero VAD running on the event loop, `gaja/audio/vad.py`).
- Server-side AEC does not work on the Atom VoiceS3R (dump of 2026-10-21): during replies the echo sits
  at ~-7 dBFS against ~-26 dBFS user speech, 5-13 % of the mic samples clip (software gain_factor 4),
  and the echo delay drifts 40-100 ms between replies (the reference is aligned by arrival). Even an
  offline least-squares filter only removes 8-18 dB, so Gaja barged in on her own voice. Since then
  `[devices] speech_barge_in = false` (default) serves a duplex device as half duplex: listen_end after
  the utterance, barge-in by the on-device wake word only. Turning speech barge-in back on would need
  mic headroom (gain_factor 1 on the uplink + server-side gain after AEC), sample-clock-stamped reference
  frames, and a nonlinear residual suppressor.
- Live test of 2026-10-21 on 0.3.2 with speech barge-in off: the device ignored listen_end while a reply
  played and kept streaming, so the server took Gaja's echo for follow-up turns (a self-talk loop), and
  "Hej Gaja" said during a reply did not interrupt it (the user's voice is ~19 dB under the clipped echo).
  0.3.3 (9459086) streams through a reply only when the server cancels echo (`aec` in ready) and drops the
  second beep after a bare "Hej Gaja". 0.3.3 is flashed but not yet tested live; wake-word barge-in over
  the reply is expected to stay weak (next steps: a quieter reply volume, or a button to interrupt).
- Server-side AEC float64 crash fixed in a92ce66 (would have dropped every converged duplex turn).
- 0.2.5 double beep: a wake utterance holding only "Hej Gaja" opens a follow-up window, which beeps again.

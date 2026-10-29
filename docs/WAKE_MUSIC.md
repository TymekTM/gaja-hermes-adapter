# "Hej Gaja" wake word under music — measurement and recommendation

The on-device microWakeWord model (`firmware/esphome/models/hej_gaja.tflite`, probability_cutoff
0.9608 as quantised by ESPHome from the 0.961 grid value, sliding_window_size 5) was tuned with
backgrounds at SNR 0–12 dB that are mostly noise-like (AudioSet/FMA picks). Speech into the mic
arrives quiet (≈ -36 dBFS RMS active before the software gain_factor 4), and music from another
speaker in the same room lands at similar or higher level, i.e. SNR 0 dB and below. Three options
are compared:

(a) lower the probability cutoff, accepting more false wakes (the server re-verifies every wake
    with faster-whisper anyway);
(b) put a noise-suppression prefilter before the wake model (WebRTC NS, portable to the ESP32-S3);
(c) retrain with harder, music-specific augmentation.

## Method

`bench/wake_music.py` (one command, seeded, result JSON in gitignored `bench/results/wake_music.json`)
reuses the streaming scorer of the training harness (1.5 s state flush, sliding-window average over
5 probabilities, quantised cutoff comparison, 2 s refractory) and the device front end end-to-end:

* speech: real wake takes under `data/wakeword/recordings/*/wake/` (194 takes: 97 via
  the device mic "usb uac", 97 via a line-in interface), active-speech RMS scaled to the measured
  Atom level (-36 dBFS primary; -42 dBFS sensitivity point, which matches the median of the quiet
  recording sessions), music mixed at SNR against that level;
* device chain: software gain 4 with int16 saturation, exactly the `gain_factor: 4` stage the
  firmware applies before the wake model and the uplink;
* music for the mixes and the music-only false-accept stream: local music files (`--music`, not in
  the repo; the run below used ≈ 2.9 min of unique real music) looped with random offsets to fill 10 of 30 minutes, plus a synthetic chord/drums bed (labelled synthetic below)
  for the remaining 20 minutes;
* false-accept sets at the same cutoffs: that music-only stream (30 min per level, -36 and -30 dBFS
  RMS), ambient room recordings (369 files, 6.08 h), 114 hard-negative near-miss takes,
  and 4 device-captured non-wake speech-over-music clips (scored as recorded, post-gain, with a
  30 s music pre-roll instead of the silence flush — "primed", the live-like protocol);
* the same grid with a prefilter: WebRTC noise suppression (pip package `webrtc-noise-gain`,
  APM NS level 2 + its high-pass filter) applied before the software gain, the way an ESP32-S3
  port would sit in the mic pipeline;
* device-captured wake attempts over music (5 clips, post-gain uplink audio) scored under two
  protocols: fresh (silence flush) and primed (music pre-roll). They measure where such
  audio actually sat relative to each cutoff.

Cutoffs are compared exactly as the firmware does: `int(cutoff*255)/255` (so the 0.961 grid point
compares against 0.9608). Recall cells count a take as detected when the window average exceeds the
cutoff anywhere in the clip; the music-content of each SNR cell is a different random segment, so
single-cell values carry a few points of jitter (n=194, binomial SE ≈ 3–4 pp).

Reproduce:

```
uv venv bench/.venv --python 3.11
uv pip install --python bench/.venv/bin/python soundfile scipy tflite-runtime webrtc-noise-gain
git clone https://github.com/OHF-Voice/micro-wake-word <mww>
uv pip install --python bench/.venv/bin/python -e <mww>
bench/.venv/bin/python bench/wake_music.py --recordings <recordings root> \
    --music <music files...> \
    --incident <captured wake attempts over music...> \
    --room-speech <captured non-wake speech over music...>
```

## Recall vs SNR (speech at -36 dBFS, gain 4 with clipping)

All values: share of 194 takes detected. Source: `bench/results/wake_music.json`
(`recall.*.scores`, no filter).

| SNR dB | @0.961 | @0.9 | @0.8 | @0.7 | @0.6 | @0.5 |
|---|---|---|---|---|---|---|
| clean      | 0.892 | 0.918 | 0.954 | 0.959 | 0.969 | 0.969 |
| +10        | 0.840 | 0.876 | 0.907 | 0.923 | 0.928 | 0.933 |
| +5         | 0.747 | 0.789 | 0.840 | 0.861 | 0.876 | 0.902 |
| 0          | 0.541 | 0.608 | 0.670 | 0.742 | 0.758 | 0.794 |
| -5         | 0.232 | 0.340 | 0.454 | 0.526 | 0.603 | 0.655 |
| -10        | 0.113 | 0.149 | 0.227 | 0.258 | 0.314 | 0.361 |

Device-mic takes only (n=97, the deployment-representative subset; same source key `by_mic.atom`):

| SNR dB | @0.961 | @0.9 | @0.8 | @0.7 | @0.6 | @0.5 |
|---|---|---|---|---|---|---|
| clean      | 0.856 | 0.876 | 0.928 | 0.938 | 0.959 | 0.959 |
| +10        | 0.794 | 0.825 | 0.887 | 0.897 | 0.907 | 0.918 |
| +5         | 0.722 | 0.753 | 0.825 | 0.845 | 0.856 | 0.887 |
| 0          | 0.557 | 0.598 | 0.629 | 0.691 | 0.722 | 0.742 |
| -5         | 0.278 | 0.381 | 0.515 | 0.536 | 0.598 | 0.639 |
| -10        | 0.144 | 0.196 | 0.258 | 0.299 | 0.351 | 0.402 |

At the quiet-speech sensitivity point (-42 dBFS, `recall.none@-42`) the picture is the same within
2–3 pp (e.g. SNR 0: 0.546 @0.961 → 0.629 @0.9 → 0.691 @0.8).

**Reading:** the model ranks fine but is badly calibrated under music at SNR ≤ 0 — at the deployed
cutoff it loses ≈ 35 pp at 0 dB and ≈ 66 pp at -5 dB relative to clean. That flat-vs-music shape is
a training-data gap (option c), not a model-capability ceiling: at SNR 0, recall at 0.5 is still
0.794, so a lower cutoff alone recovers +7 pp at 0 dB and +11 pp at -5 dB.

## Recall with the WebRTC NS prefilter (level 2, before gain)

Delta vs no filter at -36 dBFS (all takes; `recall.webrtc@-36` minus `recall.none@-36`):

| SNR dB | Δ@0.961 | Δ@0.9 | Δ@0.8 | Δ@0.7 | Δ@0.6 | Δ@0.5 |
|---|---|---|---|---|---|---|
| clean | 0.000 | 0.000 | -0.026 | -0.026 | -0.026 | -0.005 |
| +10   | +0.011 | 0.000 | 0.000 | +0.005 | +0.010 | +0.026 |
| +5    | -0.036 | +0.031 | +0.005 | 0.000 | +0.021 | +0.016 |
| 0     | -0.010 | +0.021 | +0.005 | -0.010 | -0.011 | 0.000 |
| -5    | -0.010 | -0.005 | -0.021 | -0.011 | -0.026 | +0.015 |
| -10   | +0.016 | -0.005 | -0.031 | -0.021 | -0.015 | +0.015 |

WebRTC NS does not rescue music-masked speech: deltas stay within ±3.6 pp with no sign at the SNRs
that matter (0 and -5 dB), and clean recall at mid cutoffs gets *worse* (-2.6 pp @0.8). The noise
it removes (stationary, ≈ -10 dB on synthetic stationary noise) is not what makes music hard —
the harmonic overlap is. Source: `bench/results/wake_music.json`.

## False accepts per hour at the same cutoffs

Source: `bench/results/wake_music.json` (`false_accepts.*`; music stream = 10 min real looped +
20 min synthetic per level; ambient = 369 files / 6.08 h; hard negatives = 114 takes, share
triggering, plus an FA/h equivalent if the set looped; room speech = 4 device captures, primed).

| set | filter | @0.961 | @0.9 | @0.8 | @0.7 | @0.6 | @0.5 |
|---|---|---|---|---|---|---|---|
| music only -36 dBFS | none   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| music only -30 dBFS | none   | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| music only -36 dBFS | webrtc | 0.00 | 0.00 | 0.00 | 0.00 | 2.00 | 4.00 |
| music only -30 dBFS | webrtc | 0.00 | 0.00 | 0.00 | 2.00 | 2.00 | 4.00 |
| ambient room        | none   | 1.15 | 3.45 | 5.59 | 7.89 | 10.19 | 15.12 |
| ambient room        | webrtc | 1.81 | 2.79 | 4.93 | 8.55 | 13.15 | 19.39 |
| hard negatives, share triggering | none | 0.044 | 0.053 | 0.070 | 0.070 | 0.096 | 0.114 |
| hard negatives, share triggering | webrtc | 0.035 | 0.035 | 0.061 | 0.061 | 0.079 | 0.105 |
| room speech over music, share triggering | none | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |

Music itself false-accepts at **zero** rate at every cutoff without a prefilter, at both levels —
the false-wake pressure comes from speech-like room content (ambient, near-misses), not from
music. NS moves ambient FA in *opposite* directions at neighbouring cutoffs (worse at 0.961:
1.15→1.81/h; better at 0.9: 3.45→2.79/h) — no consistent benefit.

## Device-captured wake attempts over music

5 device captures of a missed wake over music, scored as recorded (post-gain uplink audio), no
filter, under two protocols: fresh (silence flush) and primed (30 s music pre-roll). Source:
`bench/results/wake_music.json` (`incident_captures.none`).

Fresh, all five score 1.000. Primed, they sit at 0.892–1.000: four pass the deployed 0.9608
cutoff, all five pass at 0.8. Audio like this is borderline rather than hopeless, consistent with
the recall table's SNR ≈ 0 row. The pre-roll is concatenated music, not the room's reverberant
continuation, so these scores are indicative, not a replay.

One of the 4 non-wake speech-over-music captures (a song's spoken outro) scores 1.000 under every
cutoff and filter: a cutoff cannot remove this false accept, only the server's whisper
re-verification does. Source: `false_accepts.room_speech_over_music`.

## Recommendation

Combine (a) now with (c) next; skip (b).

1. **(a) Lower the cutoff from 0.961 to ≈ 0.9** (ESPHome-quantised 0.8980). Expected gains on real
   takes: +6.7 pp at SNR 0, +10.8 pp at -5 dB, +2.6 pp clean (n=194, `recall.none@-36`). Cost on the
   device side: ambient-room false accepts rise from 1.15/h to 3.45/h (7→21 events over 6.08 h),
   and hard-negative triggering from 4.4% to 5.3%. With the existing whisper re-verification the
   residual cost is small: at ~10 h/day of ordinary room noise, 0.961 produces ≈ 12 device-side
   wake attempts/day (≈ 0.5 of them near-miss speech that whisper may accept), 0.9 produces ≈ 35
   attempts/day (≈ 1.9 near-miss). The near-miss acceptance share is the assumption to watch; the
   hard-negative set is only 114 takes.
2. **(c) Retrain with music-specific, harder augmentation** as the durable fix. The current config
   (`wakeword/gaja_ww/config.py`) mixes backgrounds at SNR 0–12 dB drawn from noise/music pools;
   the measurement says the gap is at SNR ≤ 0 under *harmonic* music. Concretely: dedicate a
   background pool to real music tracks (the kind played where the device sits) at SNR -5…+5 dB,
   keep the competing-speech minority, and add the device-captured wake attempts over music as
   positives once labelled. The 0.5-cutoff recall (0.79 at SNR 0, 0.66 at -5) shows the model can
   separate these; training at that difficulty should lift recall at a *fixed* cutoff instead of
   buying it with false accepts.
3. **(b) Not recommended.** WebRTC NS (level 2, the portable form) leaves recall within ±3.6 pp
   everywhere that matters, worsens clean recall at mid cutoffs, and moves ambient FA in both
   directions depending on the cutoff. It does not address the harmonic-masking mechanism, and it
   adds an always-on DSP stage to the ESP32-S3 for no measured benefit.

## Where the data is too thin to trust

* **Music-only FA**: ≈ 2.9 min of unique real music, looped with offsets to 10 min (plus 20 min
  synthetic). The 0.00/h result is reassuring but the real-music coverage is narrow; a longer,
  more varied recording of music playing in the room would make it solid.
* **Incident captures (n=5) and room-speech captures (n=4, one FA)**: tiny; they locate the
  operating point on the cutoff axis but cannot establish rates.
* **Ambient FA/h at 0.961 rests on 7 events in 6.08 h** (Poisson 95% CI ≈ 0.5–2.4/h); at lower
  cutoffs the counts are healthier (21–92 events).
* **Recall cells**: n=194 takes from one speaker through two mic chains, with a different random
  music segment per cell — per-cell sampling noise ≈ ±3–4 pp on top of content jitter; trends
  across cutoffs and SNRs are robust, single cells are not.
* **SNR mixing reference**: SNR is defined against active-speech RMS at the -36 dBFS anchor
  measured on the device mic; rooms with different speaker-to-mic distance shift the whole SNR
  axis, not the shape of the trade-off.

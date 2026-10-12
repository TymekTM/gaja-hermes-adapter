# Prosody styles for Piper (PL) and Paradee (EN)

Per-sentence expressiveness is approximated with cheap prosody controls: speaking rate, pitch
shift, energy, and short pauses. No extra model, no GPU. Post-processing costs a few milliseconds
per sentence on CPU.

Code: `gaja/text/style.py` (tag parsing, presets), `gaja/tts/prosody.py` (engine parameters,
post-processing), `tests/test_style.py`, `tests/test_prosody.py`, `bench/prosody_demo.py`.

## Tags

Tags are written in square brackets, anywhere in the LLM output, and are case-insensitive.

| Tag | Polish alias(es) | Effect |
|---|---|---|
| `[neutral]` | `[neutralnie]` | Baseline. |
| `[happy]` | `[radośnie]`, `[wesoło]` | Emotion preset (see table). |
| `[excited]` | `[entuzjastycznie]`, `[ekscytująco]` | Emotion preset. |
| `[calm]` | `[spokojnie]` | Emotion preset, plus a short pause after. |
| `[sad]` | `[smutno]` | Emotion preset, plus a longer pause after. |
| `[serious]` | `[poważnie]` | Emotion preset, plus a short pause after. |
| `[fast]` | `[szybko]` | Multiplies rate by 1.18. |
| `[slow]` | `[wolno]` | Multiplies rate by 0.86. |
| `[laugh]` | `[śmiech]`, `[śmieje się]` | Pause after (0.12 s). No synthesized laughter. |
| `[sigh]` | `[westchnienie]`, `[wzdycha]` | Pause before (0.25 s) and after (0.20 s). |

Parsing rules (`parse_styled`):

- Tags apply to the text that follows them, up to the end of the sentence (`. ! ? …` or newline).
  A sentence end clears the tags, so the next sentence starts neutral unless it has its own tag.
- Several tags can be combined: `[fast][happy] ...`. The last emotion tag sets the preset.
  `fast`/`slow` multiply the rate, and events accumulate.
- A tag after non-empty text starts a new segment, so `[happy] Super! [calm] Teraz.` yields two
  segments.
- Unknown tags are dropped and never spoken. Stray or unterminated brackets are removed.
- Whitespace left behind by removed tags is collapsed.
- Tag names are bounded (at most 40 characters inside brackets, no nested brackets).
- Abbreviations such as `np.`, `dr.`, `godz.`, `tzn.` do not end a sentence.

## Presets

Values are the `Style` fields: `rate` (1.0 = neutral), `pitch_semitones`, `energy` (gain),
and `events`. Ranges are clamped to rate 0.8–1.25, pitch ±2.5 st, energy 0.6–1.4.

| Style | Emotion | Rate | Pitch (st) | Energy | Events |
|---|---|---|---|---|---|
| neutral | neutral | 1.00 | 0.0 | 1.00 | – |
| happy | happy | 1.05 | +1.2 | 1.10 | – |
| excited | excited | 1.12 | +2.0 | 1.20 | – |
| calm | calm | 0.92 | −0.6 | 0.85 | pause after 0.08 s |
| sad | sad | 0.85 | −1.5 | 0.80 | pause after 0.15 s |
| serious | serious | 0.95 | −1.0 | 1.05 | pause after 0.05 s |
| fast | neutral | 1.18 | 0.0 | 1.00 | – |
| slow | neutral | 0.86 | 0.0 | 1.00 | – |
| laugh | neutral | 1.00 | 0.0 | 1.00 | pause after 0.12 s |
| sigh | neutral | 1.00 | 0.0 | 1.00 | pause before 0.25 s, after 0.20 s |

## Engine mapping

- Piper (PL, 22050 Hz): `length_scale = (1 / rate) ** 1.5`. Piper's duration responds to
  `length_scale` with an exponent of about 0.65, so the 1.5 power compensates and the requested
  rate is reached (the plain `1 / rate` mapping gave about 1.10 for a requested 1.18).
  `noise_scale` is 0.667 plus an emotion delta (happy +0.08, excited +0.05, calm -0.07, sad -0.12,
  serious -0.07) and `0.1 * (energy - 1)`, clamped to [0.4, 0.95]. `noise_w_scale` is fixed at 0.8:
  its emotion deltas added rhythm variance that made the tempo mapping unreliable.
- Supertonic (PL, 22050 Hz output): `speed = rate`, clamped to the engine range [0.7, 2.0].
- Paradee (EN, 24000 Hz): `speed = rate`, then `apply_post` (core v1.5, `gaja/tts/paradee_tts.py`).
- MMS (PL, 16000 Hz, CC-BY-NC, candidate only): `apply_energy` only, so it has no pitch or pause
  controls. Pending a core-final request if MMS is ever promoted.
- Piper, Supertonic and Paradee apply `apply_post` to the audio: the pitch shift, the energy gain,
  a soft clip, and the pauses. Piper's own `volume` argument is not used, so energy is applied once.

## Pitch shift method

`pitch_shift` keeps the duration unchanged:

1. WSOLA time-stretch by `r = 2^(st/12)`. Frame 1024 samples at 22.05 kHz, hop 512, search ±128
   samples, Hann window, with a correlation search for the best overlap point.
2. Rational resample by `r` (`Fraction(r).limit_denominator(64)`, `scipy.signal.resample_poly`).
   The audio plays `r` times faster, so the pitch is multiplied by `r`, and the duration is back
   to the original.

Alternatives I measured and rejected:

- `librosa.effects.pitch_shift`: about 60 ms per 4 s sentence, over budget. Its polyphase
  resampler is also unsupported for this non-integer sample rate.
- `rubberband` CLI: about 530 ms per sentence including process start and I/O.
- Phase vocoder: 70–130 ms per sentence, pitch inconsistent across the sentence, and energy loss.

WSOLA was the only method that met the budget with acceptable accuracy. Its known weakness is a
slight roughness on large shifts; I did not evaluate that perceptually beyond the metrics below.

## Measured results

Measured with `bench/prosody_demo.py` through the production engine classes: `PiperTTS` at 2 threads
(gosia and darkman), `SupertonicTTS` M4, 6 steps, 2 threads, 22050 Hz, and `ParadeeTTS` at 1 thread.
Each style renders one demo sentence (for rate, pitch and loudness). WER is the Whistle round-trip on
the bench sentences for that language, 12 per style, on the post-processed audio.

Source JSON: `bench/results/prosody_piper_gosia.json`, `prosody_piper_darkman.json`,
`prosody_supertonic.json`, `prosody_paradee.json`. All four runs were quiet (start 1-min loadavg
0.92 to 1.45, `contaminated: false`, rule in each JSON). The older `prosody_piper.json` (no loadavg)
and the earlier tables in this file are superseded and are not used for any claim here.

Columns: rate and pitch are measured against the neutral audio, with the preset in brackets. Loudness
is RMS dB against neutral. The gate is "mean WER within +0.05 of neutral".

Piper gosia (`length_scale = (1/rate)^1.5`):

| style | rate (preset) | pitch st (preset) | loudness dB | WER mean | delta vs neutral | gate |
|---|---|---|---|---|---|---|
| neutral | 1.00 (1.00) | +0.00 (0.0) | +0.00 | 0.181 | +0.000 | ok |
| happy | 1.05 (1.05) | +1.26 (+1.2) | +1.23 | 0.225 | +0.044 | ok |
| excited | 1.12 (1.12) | +1.97 (+2.0) | +2.62 | 0.243 | +0.062 | **FAIL** |
| calm | 0.94 (0.92) | -0.82 (-0.6) | -1.29 | 0.212 | +0.031 | ok |
| sad | 0.85 (0.85) | -1.65 (-1.5) | -1.74 | 0.226 | +0.045 | ok |
| serious | 0.98 (0.95) | -1.23 (-1.0) | +0.90 | 0.220 | +0.039 | ok |
| fast | 1.21 (1.18) | -0.23 (0.0) | +1.05 | 0.185 | +0.004 | ok |
| slow | 0.86 (0.86) | +0.02 (0.0) | -0.03 | 0.220 | +0.039 | ok |

Piper darkman (`bench/results/prosody_piper_darkman.json`):

| style | rate (preset) | pitch st (preset) | loudness dB | WER mean | delta vs neutral | gate |
|---|---|---|---|---|---|---|
| neutral | 1.00 (1.00) | +0.00 (0.0) | +0.00 | 0.276 | +0.000 | ok |
| happy | 1.02 (1.05) | +1.29 (+1.2) | -0.28 | 0.368 | +0.092 | **FAIL** |
| excited | 1.08 (1.12) | +1.92 (+2.0) | +1.55 | 0.283 | +0.007 | ok |
| calm | 0.94 (0.92) | -0.56 (-0.6) | -2.81 | 0.253 | -0.023 | ok |
| sad | 0.85 (0.85) | -1.18 (-1.5) | -2.44 | 0.233 | -0.043 | ok |
| serious | 0.97 (0.95) | -0.71 (-1.0) | -0.29 | 0.316 | +0.040 | ok |
| fast | 1.19 (1.18) | +0.18 (0.0) | +0.04 | 0.269 | -0.007 | ok |
| slow | 0.88 (0.86) | -0.14 (0.0) | -0.66 | 0.281 | +0.005 | ok |

Supertonic-3 M4 (`bench/results/prosody_supertonic.json`):

| style | rate (preset) | pitch st (preset) | loudness dB | WER mean | delta vs neutral | gate |
|---|---|---|---|---|---|---|
| neutral | 1.00 (1.00) | +0.00 (0.0) | +0.00 | 0.161 | +0.000 | ok |
| happy | 1.04 (1.05) | +0.93 (+1.2) | +0.83 | 0.100 | -0.061 | ok |
| excited | 1.12 (1.12) | +1.06 (+2.0) | +1.48 | 0.117 | -0.044 | ok |
| calm | 0.93 (0.92) | -0.96 (-0.6) | -1.76 | 0.133 | -0.028 | ok |
| sad | 0.84 (0.85) | -1.00 (-1.5) | -2.22 | 0.164 | +0.003 | ok |
| serious | 0.96 (0.95) | -0.95 (-1.0) | +0.04 | 0.219 | +0.058 | **FAIL** |
| fast | 1.14 (1.18) | +0.06 (0.0) | +0.12 | 0.181 | +0.020 | ok |
| slow | 0.86 (0.86) | +0.42 (0.0) | -0.04 | 0.233 | +0.072 | **FAIL** |

Paradee-8M EN, 1 thread, `apply_post` path (`bench/results/prosody_paradee.json`):

| style | rate (preset) | pitch st (preset) | loudness dB | WER mean | delta vs neutral | gate |
|---|---|---|---|---|---|---|
| neutral | 1.00 (1.00) | +0.00 (0.0) | +0.00 | 0.141 | +0.000 | ok |
| happy | 1.02 (1.05) | +1.44 (+1.2) | +0.77 | 0.156 | +0.015 | ok |
| excited | 1.09 (1.12) | +2.16 (+2.0) | +1.49 | 0.165 | +0.024 | ok |
| calm | 0.97 (0.92) | -0.32 (-0.6) | -1.46 | 0.124 | -0.017 | ok |
| sad | 0.92 (0.85) | -1.19 (-1.5) | -2.01 | 0.108 | -0.033 | ok |
| serious | 0.97 (0.95) | -1.09 (-1.0) | +0.42 | 0.133 | -0.008 | ok |
| fast | 1.14 (1.18) | +0.17 (0.0) | -0.16 | 0.162 | +0.021 | ok |
| slow | 0.93 (0.86) | +0.12 (0.0) | -0.01 | 0.121 | -0.020 | ok |

Observations:

- Four of 32 style rows fail the gate: gosia excited (+0.062), darkman happy (+0.092), Supertonic
  serious (+0.058) and slow (+0.072). A flag means the WER rose on that engine, so these styles are
  candidates for a listening check. It is not proof of worse speech. The gate also moves with the
  noise: neutral gosia WER was 0.233 in an earlier run with the same settings and is 0.181 here.
  Compare WER deltas only within one engine. Absolute WER differs between voices.
- Piper rates are within 0.04 of the preset for every style (largest: darkman excited, 1.08 vs
  1.12). Supertonic's fast is 0.04 under its preset. Paradee's sad and slow measure 0.07 above their
  presets (0.92 vs 0.85, 0.93 vs 0.86), so Paradee's style rate is less extreme than requested. Its
  rate mapping has not been changed.
- Piper's pitch follows its presets within 0.4 st for happy, excited, sad and calm, using
  `pitch_shift` (Piper cannot change its own pitch). Paradee follows its presets within about 0.3 st
  for the same four styles.
- Supertonic's excited measures +1.06 st against a +2.0 preset, and its sad measures -1.00 against
  -1.5. Serious is close to its preset. pyin on this male voice is a likely cause, so treat the
  pitch column as approximate for Supertonic.
- Excited and happy are distinguished mainly by rate and loudness on Piper, and by rate on
  Supertonic. The styles are not validated by listening.

## Demo files

One demo sentence per style. The style wav is about 26 to 29 s and the conversation wav is 8 to 9 s,
both under the 30 s limit. Timestamps for each style are in the `demo_styles_<engine>.json` next to
each wav.

| file | length | what to listen for |
|---|---|---|
| `bench/out/tts/demo_styles_piper_gosia.wav` | 27.20 s | happy vs excited (excited should be brighter and faster, and may sound harsh; it failed the WER gate); calm and sad slower and lower, with pauses after; fast and slow clearly different in tempo |
| `bench/out/tts/demo_conversation_piper_gosia.wav` | 7.75 s | the emotion changes at each `[tag]` boundary, and no tag word is spoken |
| `bench/out/tts/demo_styles_piper_darkman.wav` | 26.02 s | happy (failed the WER gate): does it sound happy or just different? calm and sad should be audibly quieter |
| `bench/out/tts/demo_conversation_piper_darkman.wav` | 7.83 s | same as gosia conversation; compare the voice character with gosia |
| `bench/out/tts/demo_styles_supertonic.wav` | 29.08 s | serious and slow (both failed the gate): is serious flatter, and is slow only a tempo change? overall pitch range is narrower than Piper |
| `bench/out/tts/demo_conversation_supertonic.wav` | 9.08 s | tag-driven emotion changes on the male voice; check for artefacts at the pitch shift |
| `bench/out/tts/demo_styles_paradee.wav` | 28.00 s | EN pitch and rate changes (energy-only before core v1.5); sad and slow should sound slow but not pitch-warped |
| `bench/out/tts/demo_conversation_paradee.wav` | 8.26 s | EN conversation tags; the same checks as above |

These files are for a person to listen to. No listening test was run, and nothing here says a style
sounds right.

## Post-processing cost

- `apply_post` on a 3–4 s sentence: about 11–15 ms CPU (`time.process_time`), including the
  resample. This is an earlier measurement at load 11–27, not re-measured in round 2 and not in a
  JSON, so it is an upper bound. Budget: under 20 ms per 3 s sentence.
- Pauses add silence but no compute.

## Polish LLM system-prompt snippet

```
Odpowiadaj po polsku. Ton zaznaczaj na początku zdania znacznikiem: [radośnie], [smutno],
[spokojnie], [poważnie], [szybko], [wolno], [śmiech], [westchnienie]; działa do końca zdania.
Np.: [radośnie] Udało się! [poważnie] Termin: jutro, o 9:00.
```

Token count: NOT VERIFIED. The LFM2.5 tokenizer is not cached locally, and the llama-server
`/tokenize` endpoint was down during measurement. Estimate: about 70–80 tokens at roughly 3 characters
per token for Polish.

## Limitations

- Pitch and rate are approximations. Piper's prosody is not controllable beyond rate and noise,
  and the pitch shift is a signal-level change, not a true intonation model.
- The WER impact is measured with Whistle round-trip STT on post-processed audio. STT is a proxy
  for intelligibility, not for naturalness or whether the emotion is perceived.
- The F0 and rate figures are signal proxies (pyin median and trimmed speech duration), computed on
  one demo sentence per style. WER uses 12 bench sentences per style.
- The Piper `noise_scale` emotion deltas were tuned by hand for one step: excited was reduced from
  +0.13 to +0.05 after it failed the WER gate. `noise_w_scale` emotion deltas were removed. The
  deltas are not otherwise validated.
- `laugh` and `sigh` are pauses only; no synthesized laughter or breath sounds.
- The tags are only as reliable as the LLM's compliance with the system prompt. Unknown tags are
  stripped, but a model that never emits tags gives neutral speech.
- No subjective listening test was done.

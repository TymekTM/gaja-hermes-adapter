# TTS benchmark results (round 2, 2026-10-09)

Host: i5-8300H (4C/8T), 15 GB RAM, GTX 1050 Ti 4 GB. Hermes holds about 991 MiB of VRAM, so TTS runs on CPU (onnxruntime). The table below is rendered by `bench/tts_report.py` from `bench/results/`. Nothing in it was typed by hand. Every row names its source JSON.

This round replaces the 2026-10-08 tables. Old numbers that were not re-measured are kept in the last sections and are labelled as such.

## How to reproduce

One documented command:

```
bash bench/tts_final.sh all
```

It runs in this order: Piper (4 voices x main and probe), Supertonic (4 runs), Paradee and MMS (4 runs), the style demos (4 runs), the faster-whisper proxy, then the report. Every timed run is wrapped in `flock /tmp/gaja-bench.lock`. Before each one the 1-min loadavg must be below 1.5. If not, the run waits up to 180 s and then goes ahead, and its JSON is marked contaminated. Piper runs with `--piper-threads 2`, the production value. Paradee runs with `threads 1`, also production. Stages can run one at a time: `bash bench/tts_final.sh piper|supertonic|en-mms|demos|proxy|report`.

The full run takes well over the 10-minute foreground limit. Run it in the background and wait for the log.

## Metric definitions

- **RTF** = audio seconds / synthesis seconds, warm (2 discarded warm-up calls). RTF > 1 is faster than real time. Target: RTF > 1.
- **TTFA** = wall seconds from text in to first audio. Piper streams per sentence. Its items are single sentences, so its TTFA is that sentence's synthesis time. Supertonic, MMS and Paradee synthesise the whole utterance, so their TTFA equals synthesis time. The `(short)` value is the median over the 3–6 word sentences only. Target: TTFA < 0.3 s.
- **Whistle WER** = round-trip through Whistle (cactus-needle). Language is forced per set. The text is normalised by `tts_bench.normalize` (lowercase, numbers spelled out with num2words, punctuation stripped), then jiwer computes WER. This is the mean and median over the main set (12 sentences).
- **faster-whisper WER** = round-trip through faster-whisper `small` (CPU, int8, beam 5, language `pl`), mean over the main and probe sets (24 sentences). Produced by `bench/tts_quality_proxy.py`. It is a proxy for intelligibility, not a listening test. Polish only.
- **Peak dBFS / clip frac** = maximum peak level over all wavs in the voice, and the largest fraction of samples with |x| >= 0.999 in any wav. Both come from `bench/tts_quality_proxy.py`.
- **Max pause (s)** = longest run of silence (frames at or below -45 dBFS, 10 ms frames) between the first and last speech frame, maximum over the voice's wavs. Pauses of 0.3 s or more are counted in the JSON as `n_internal_pauses_ge_0_3s`.
- **Load 1-min start -> end** = `/proc/loadavg` at the start and end of each timed run. The end value includes the run itself. Only the start value decides contamination.
- **Contaminated** = start 1-min loadavg above 2.0 (`QUIET_LOAD` in `bench/tts_bench.py`). RTF and TTFA come from quiet runs when a voice has one. A row with no quiet run is labelled CONTAMINATED. A "timing from quiet run only" row also shows which run was excluded.
- **Cold load** = model load time in seconds, reported separately from RTF and TTFA.

## Measured results (round 2)

<!-- TTS_TABLE:BEGIN -->
### Polish (PL)

Main set (12 sentences) and probe set (12 sentences) per voice. RTF and TTFA come only from runs whose start 1-min loadavg was <= 2.0 when one exists. Whistle WER uses the main set. faster-whisper `small` (CPU, int8, beam 5) runs on both sets; WER does not depend on load.

| Engine / config | Role | SR (Hz) | Device | Threads | RTF median (min-max) | TTFA median (short) | Cold load (s) | Streaming | Prosody controls | Whistle WER mean / median | faster-whisper WER mean | Peak dBFS / clip frac | Max pause (s) | Load 1-min start -> end | Status | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Piper pl_PL-darkman-medium | alternative | 22050 | cpu (onnxruntime, threads=2, production PiperTTS) | 2 | 16.17 (14.26-16.75) | 0.23 (0.17) | 4.64 | sentence-level (production PiperTTS; items are single sentences, TTFA = their synth time) | style -> Piper length/noise, WSOLA pitch, pauses, soft clip (apply_post) | 0.296 / 0.273 | 0.117 | -0.4 / 0.0000 | 0.42 | 5.83 -> 6.98 ; 1.40 -> 1.37 | timing from quiet run only; contaminated: main (start load 5.83) | `tts_piper_final_darkman.json`, `tts_piper_final_darkman_probe.json` |
| Piper pl_PL-gosia-medium | production default | 22050 | cpu (onnxruntime, threads=2, production PiperTTS) | 2 | 15.74 (14.14-17.11) | 0.27 (0.15) | 2.22 | sentence-level (production PiperTTS; items are single sentences, TTFA = their synth time) | style -> Piper length/noise, WSOLA pitch, pauses, soft clip (apply_post) | 0.170 / 0.169 | 0.079 | -0.4 / 0.0000 | 0.46 | 1.30 -> 1.57 ; 7.61 -> 8.98 | timing from quiet run only; contaminated: probe (start load 7.61) | `tts_piper_final_gosia.json`, `tts_piper_final_gosia_probe.json` |
| Piper pl_PL-mc_speech-medium | candidate | 22050 | cpu (onnxruntime, threads=2, production PiperTTS) | 2 | 11.36 (5.31-16.57) | 0.29 (0.16) | 5.20 | sentence-level (production PiperTTS; items are single sentences, TTFA = their synth time) | style -> Piper length/noise, WSOLA pitch, pauses, soft clip (apply_post) | 0.260 / 0.177 | 0.136 | -0.4 / 0.0000 | 0.38 | 9.43 -> 9.45 ; 3.34 -> 3.21 | CONTAMINATED: main (start load 9.43), probe (start load 3.34) | `tts_piper_final_mc_speech.json`, `tts_piper_final_mc_speech_probe.json` |
| Piper pl_PL-bass-high | candidate | 22050 | cpu (onnxruntime, threads=2, production PiperTTS) | 2 | 2.51 (2.34-2.56) | 1.63 (1.00) | 3.09 | sentence-level (production PiperTTS; items are single sentences, TTFA = their synth time) | style -> Piper length/noise, WSOLA pitch, pauses, soft clip (apply_post) | 0.407 / 0.411 | 0.205 | -0.4 / 0.0000 | 0.18 | 1.46 -> 2.44 ; 8.66 -> 9.31 | timing from quiet run only; contaminated: probe (start load 8.66) | `tts_piper_final_bass-high.json`, `tts_piper_final_bass-high_probe.json` |
| Supertonic-3 M4, 6 steps, 22.05 kHz | config in config.toml | 22050 | cpu (onnxruntime, threads=2, steps=6, voice=M4, speed=1.05, sr=22050) | 2 | 4.66 (3.14-5.45) | 1.16 (0.67) | 1.55 | no (full utterance synthesis) | style -> speed (rate), WSOLA pitch, pauses, soft clip (apply_post) | 0.099 / 0.045 | 0.019 | -7.8 / 0.0000 | 0.44 | 1.70 -> 2.28 ; 1.49 -> 2.03 | quiet (start load <= 2.0) | `tts_supertonic_final.json`, `tts_supertonic_final_probe.json` |
| Supertonic-3 M4, 6 steps, native 44.1 kHz | reference only | 44100 | cpu (onnxruntime, threads=2, steps=6, voice=M4, speed=1.05, sr=44100) | 2 | 4.79 (4.19-5.39) | 1.08 (0.64) | 0.78 | no (full utterance synthesis) | style -> speed (rate), WSOLA pitch, pauses, soft clip (apply_post) | 0.148 / 0.097 | 0.025 | -8.1 / 0.0000 | 0.42 | 1.49 -> 2.20 ; 1.47 -> 1.81 | quiet (start load <= 2.0) | `tts_supertonic_native_M4.json`, `tts_supertonic_native_M4_probe.json` |
| MMS mms-tts-pol, CPU (CC-BY-NC) | candidate, licence-limited | 16000 | cpu (fp32) | torch default | 2.90 (2.27-3.78) | 1.58 (0.92) | 2.31 | no (full utterance synthesis) | energy only (no pitch or pauses) | 0.244 / 0.298 | 0.126 | -1.9 / 0.0000 | 0.86 | 1.47 -> 3.03 ; 1.40 -> 2.17 | quiet (start load <= 2.0) | `tts_mms_final_cpu.json`, `tts_mms_final_cpu_probe.json` |

### English (EN)

Whistle WER on the EN main set; faster-whisper proxy is PL-only, so it is n/a here.

| Engine / config | Role | SR (Hz) | Device | Threads | RTF median (min-max) | TTFA median (short) | Cold load (s) | Streaming | Prosody controls | Whistle WER mean / median | faster-whisper WER mean | Peak dBFS / clip frac | Max pause (s) | Load 1-min start -> end | Status | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Paradee-8M, 1 thread | production EN default | 24000 | cpu (1 thread) | 1 | 9.65 (8.70-10.46) | 0.43 (0.32) | 5.03 | no | style -> speed (rate), WSOLA pitch, pauses, soft clip (apply_post) | 0.141 / 0.143 | n/a | n/a | n/a | 1.46 -> 1.54 ; 1.46 -> 1.54 | quiet (start load <= 2.0) | `tts_paradee_final_en.json`, `tts_paradee_final_en_probe.json` |

### Not re-measured (old runs, no loadavg; do not compare with the tables above)

| Config | Lang | Device | SR (Hz) | RTF median (min-max) | TTFA median | Whistle WER mean | Cold load (s) | Failed | Status | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| Paradee-8M, 1 thread | pl | cpu (1 thread) | 24000 | 6.33 (6.14-6.54) | 1.06 | 0.920 | 6.32 | 0 | NOT RE-MEASURED; no loadavg recorded | `tts_paradee_t1.json` |
| Paradee-8M, 1 thread | en | cpu (1 thread) | 24000 | 6.38 (6.17-6.47) | 0.67 | 0.141 | 6.32 | 0 | NOT RE-MEASURED; no loadavg recorded | `tts_paradee_t1.json` |
| Paradee-8M, 2 threads | pl | cpu (2 threads) | 24000 | 9.50 (9.16-9.62) | 0.70 | 0.912 | 6.31 | 0 | NOT RE-MEASURED; no loadavg recorded | `tts_paradee_t2.json` |
| Paradee-8M, 2 threads | en | cpu (2 threads) | 24000 | 9.45 (9.23-9.60) | 0.45 | 0.141 | 6.31 | 0 | NOT RE-MEASURED; no loadavg recorded | `tts_paradee_t2.json` |
| Paradee-8M, 4 threads | pl | cpu (4 threads) | 24000 | 12.16 (11.08-12.52) | 0.54 | 0.907 | 6.26 | 0 | NOT RE-MEASURED; no loadavg recorded | `tts_paradee_t4.json` |
| Paradee-8M, 4 threads | en | cpu (4 threads) | 24000 | 12.05 (10.62-12.64) | 0.34 | 0.141 | 6.26 | 0 | NOT RE-MEASURED; no loadavg recorded | `tts_paradee_t4.json` |
| Paradee-8M, Polish via espeak-ng IPA | pl | cpu (1 thread), pl via espeak-ng IPA | 24000 | 6.40 (6.16-6.53) | 0.82 | 0.938 | 6.40 | 0 | NOT RE-MEASURED; no loadavg recorded | `tts_paradee_pl_espeak.json` |
| Chatterbox-turbo fp32 (GPU) | en | cuda (fp32) | 24000 | 1.42 (1.23-1.52) | 1.97 | 0.169 | 15.95 | 4 | NOT RE-MEASURED; no loadavg recorded | `tts_chatterbox-turbo_fp32.json` |
| Chatterbox-turbo fp32, cfm steps 1 (GPU) | en | cuda (fp32) | 24000 | 1.53 (1.32-1.60) | 1.77 | 0.217 | 14.57 | 0 | NOT RE-MEASURED; no loadavg recorded | `tts_chatterbox-turbo_fp32_cfm1.json` |
| Chatterbox-turbo fp32, expandable segments (GPU) | en | cuda (fp32) | 24000 | 1.45 (1.15-1.57) | 2.09 | 0.200 | 14.82 | 3 | NOT RE-MEASURED; no loadavg recorded | `tts_chatterbox-turbo_fp32_expseg.json` |
| Chatterbox-turbo fp16 (GPU) | en | cuda (fp16) | 24000 | 0.36 (0.31-0.38) | 9.42 | 0.623 | 17.77 | 0 | NOT RE-MEASURED; no loadavg recorded | `tts_chatterbox-turbo_fp16.json` |
<!-- TTS_TABLE:END -->

Piper, Supertonic and MMS are Polish voices. Paradee is English-only. Paradee's Polish runs are in the not-re-measured section.

## Recommendation

**Polish default: Piper `pl_PL-gosia-medium` (production, unchanged).** It has the lowest faster-whisper WER of the Piper voices (0.079 mean over 24 sentences). Quiet-run main set: RTF 15.74 and TTFA median 0.27 s, under the 0.3 s target. It streams sentence by sentence, so the first sentence starts within that TTFA.

Darkman is faster to first audio (quiet probe TTFA 0.23 s) but has a worse faster-whisper WER (0.117). Mc_speech (0.136) and bass-high (0.205) have worse WER. Their runs are contaminated or their TTFA median is above target, so neither is a candidate for the default.

**Supertonic is the best voice on proxies but is not a streaming default.** Its faster-whisper WER is 0.019 at 22.05 kHz (the lowest on the table) and its Whistle median is 0.045. RTF is 4.66, which passes. Its TTFA median is 1.16 s, and 0.67 s even for short sentences. That fails the TTFA < 0.3 s criterion, so no request is filed to core-final to switch the default. Supertonic stays a config option, as in `config.toml`.

**English default: Paradee-8M, 1 thread (production).** Whistle WER 0.141 on the quiet EN run, RTF 9.65, TTFA median 0.43 s (short 0.32 s). It is the only EN engine in the set. Its prosody now goes through `apply_post` (see below).

**Chatterbox-turbo and Qwen3-TTS-Polish stay out of the defaults.** Chatterbox needs the GPU and has not been re-measured here. Qwen has no streaming API and a high WER (see the old-run section).

## Prosody state

- Piper, Supertonic and Paradee render through `gaja/tts/prosody.py` `apply_post`: speed from the style rate (Piper: length and noise too), WSOLA pitch, pauses, and soft clip. Core shipped the Paradee change as v1.5 in `docs/ARCHITECTURE.md` (line 246). The Paradee file is dated 08:41, before the 09:14 bench and 09:56 demo runs, so the Paradee numbers above already include it.
- MMS uses `apply_energy` only, so it has no pitch or pause controls. Listed as energy only.
- Demo wavs: `bench/out/tts/` (paths in `docs/PROSODY.md`).

## Limitations

- **No listening test.** Every quality number is a proxy: Whistle WER, faster-whisper WER, and clip, level and pause statistics. Nothing here says a voice sounds better. The demos are for a person to listen to.
- **Small sets.** 12 sentences per set, 24 for the faster-whisper proxy. Medians on 3–4 short items are noisy.
- **Load.** The box ran at loadavg 2–9 for much of the session (other agents' processes, the shared lock often held). Rows marked "timing from quiet run only" use the quiet run; rows marked CONTAMINATED have no quiet run, so their RTF and TTFA are not usable as production numbers.
- **Lead reference.** The lead measured gosia at RTF 21.25 on a quiet machine. This round measured 15.74 on a quiet main run (start load 1.30). The cause was not investigated. Use 15.74 as the figure from this round.
- **Not re-measured:** Paradee at 1, 2 and 4 threads and Polish via espeak-ng; all Chatterbox-turbo GPU variants (see the not-re-measured table in the block above). They have no loadavg recorded, so they are shown for reference only.

## Old runs (not re-measured in round 2)

The following tables are carried over from the 2026-10-08 version of this document. The source JSONs are named, but the numbers were not re-checked in this round, and they have no loadavg recorded.

### Qwen3-TTS-Polish (llama-tts, `ru` token)

| Configuration | Device | n (failed) | RTF median [min-max] | TTFA median (s) | Whistle WER median (mean) | Peak VRAM (MiB) | Source JSON |
|---|---|---|---|---|---|---|---|
| Backbone GPU, mmproj CPU | GPU+CPU | 12 (0) | 0.41 [0.32-0.45] | 21.7 | 0.856 (0.81) | 2649 | `tts_qwen-polish_ru_cpuvoc.json` |
| mmproj GPU, `-b/-ub 128` | GPU | 12 (0) | 1.14 [1.06-1.18] | 7.3 | 0.875 (0.97) | 4005 | `tts_qwen-polish_ru_gpuvoc_b128.json` |
| mmproj GPU, `-b/-ub 32` | GPU | 10 (2) | 1.13 [0.98-1.18] | 7.6 | 0.831 (0.82) | 4017 | `tts_qwen-polish_ru_gpuvoc_b32.json` |
| mmproj GPU, `-b/-ub 32`, speaker ref | GPU | 12 (0) | 1.15 [1.02-1.18] | 9.3 | 0.917 (1.10) | 3913 | `tts_qwen-polish_ru_gpuvoc_b32_spk.json` |

Language-token probe (3 PL sentences): WER median 0.800 for both `ru` and `en`; means 0.822 and 0.911 (`tts_qwen-polish_lang_ru.json`, `tts_qwen-polish_lang_en.json`). The GGUF has no Polish codec token, so `ru` is used. Qwen load time is a per-call estimate because `llama-tts` reloads the model on every call.

### Co-residency with LFM2.5-2.6B Q4_K_M (llama-server, port 8791)

| Configuration | VRAM total (MiB) | LLM gen tok/s | TTS RTF | TTS WER median | Source |
|---|---|---|---|---|---|
| LFM `-ngl 99` alone | 2828 | 34.0 | n/a | n/a | `bench/logs/lfm_tps_alone.json` |
| LFM `-ngl 12` alone | 1950 | 16.9 | n/a | n/a | `bench/logs/lfm_tps_ngl12.json` |
| LFM `-ngl 99` + Qwen (backbone GPU, mmproj CPU) | n/a | n/a | load fails (`cudaMalloc` 613 MiB OOM) | n/a | `bench/logs/` (qwen probe logs) |
| LFM `-ngl 12` + Qwen (backbone GPU, mmproj CPU) | 3648 | 9.1–13.8 (3 samples) | 0.32 (alone 0.41) | 0.835 (alone 0.856) | `tts_qwen-polish_ru_cpuvoc_coresident.json` |
| LFM `-ngl 12` + Chatterbox fp16 | 3738 | not measured | 0.356 (alone 0.362) | 0.619 (alone 0.586) | `tts_chatterbox-turbo_fp16_coresident.json` |

LFM tok/s during TTS comes from three samples (`bench/logs/lfm_tps_during_tts_{1,2,3}.json`). Qwen GPU-mmproj and Chatterbox fp32 were not tested together. Each needs about 3.0 GB, and with LFM at `-ngl 12` only about 2.1 GB is free. This is an estimate, not a measurement. At `-ngl 99` LFM leaves about 1.2 GB, which fits no tested Qwen or Chatterbox load.

### Chatterbox-turbo fp32 (GPU) and Paradee threads, before round 2

Chatterbox-turbo fp32 OOMs on long sentences (en09–en12, about 9–10 s audio) at 3 GB. Its fp16 transcripts are largely unintelligible. These runs are kept in the not-re-measured table of the block above (`tts_chatterbox-turbo_*.json`). Paradee at 1 thread is the production EN default. In the old, not re-measured runs, 2 and 4 threads had higher RTF (9.5 and 12.2 against 6.3 at 1 thread, `tts_paradee_t2.json`, `tts_paradee_t4.json`, `tts_paradee_t1.json`). Production stays at 1 thread, because the round-2 quiet run was taken at 1 thread and the thread sweep was not repeated.

## Artifacts

- Run script: `bench/tts_final.sh` (`all`, or one stage)
- Bench: `bench/tts_bench.py` (per-item timings, WER, loadavg, contaminated flag)
- Proxies: `bench/tts_quality_proxy.py` (faster-whisper WER, clip, level, pause), output `bench/results/tts_quality_proxy.json`
- Report: `bench/tts_report.py` (fills the block above from `bench/results/`)
- Results: `bench/results/tts_<name>.json` and `tts_<name>_probe.json`, named in each table row
- Style demos: `bench/prosody_demo.py`, output `bench/results/prosody_<engine>.json`
- Audio: `bench/out/tts/`
- Logs: `bench/logs/tts_final_piper.log`, `bench/logs/tts_final_rest.log`, `bench/logs/tts_final_bass_probe.log`

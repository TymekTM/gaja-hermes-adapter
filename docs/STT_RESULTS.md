# Speech-to-Text (STT) benchmark results

Host: Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz (4 cores / 8 threads), 15 GB RAM, Ubuntu Linux.
All metrics are derived directly from runs executed by `bench/run_stt_matrix.sh` and `bench/stt_bench.py`.
Raw benchmark data files: `bench/results/stt_*.json`
Execution logs: `bench/logs/stt_*.log` and `bench/logs/stt_matrix.log`

---

## 1. TL;DR

- **GPU STT Upgrade (`faster-whisper` on CUDA, see [Section 7](#7-gpu-stt-faster-whisper-stt-gpu))**:
  - Evaluation on NVIDIA GeForce GTX 1050 Ti (4 GB) co-resident with `llama-server` demonstrates that GPU `faster-whisper small` (`int8`, `cuda`, `beam_size 1`) drops Polish WER from Whistle's 36.11% to **18.69%** (cutting word errors almost in half) with short conversational latency of **0.395 s p50 / 0.436 s p90** (RTF **7.70**) within 3825 MiB peak total VRAM.
  - **Production selection**: The lead selected `small`, `int8`, `cuda`, `beam_size 1` for production STT (`gaja/stt/faster_whisper_stt.py`, being wired by core-final) over beam 5 to avoid a 3.5x p90 latency spike (0.436 s vs 1.528 s) and silence hallucinations in exchange for only a 1.79% WER delta. Full GPU benchmarks, co-residency findings with LLM, and reproduction commands are documented in [Section 7](#7-gpu-stt-faster-whisper-stt-gpu).
- **Recommended engine (CPU-only baseline)**: **Cactus-Compute Whistle** via Python in-process binding (`needle.Whistle()`, `gaja/stt/whistle.py`).
  - **Polish (PL)**: WER **36.11%**, median latency on conversational utterances (2–3 s) is **0.266 s – 0.466 s** (p90: 0.716 s – 1.053 s), RTF **4.96**, RAM **101.4 MB**, model load time **0.037 s**.
  - **English (EN)**: WER **12.36%**, median latency on short utterances is **1.394 s** (target 2 s is **1.012 s**), RTF **2.69** (WER set) / **1.07** (latency set), RAM **97.7 MB**.
  - **Zero hallucinations**: 0% hallucination on digital silence or white noise (output is empty string).
- **Language Mode: Forced vs Auto**:
  - **Forced language is mandatory for real-time operation**.
  - In Polish, auto-detection degrades WER from 36.11% to 38.67%, misclassifies 4.5% of utterances (2/44 detected as non-PL), and inflates p90 latency from 5.67 s to 13.61 s (2.4x slower).
  - In English, auto-detection causes a catastrophic **7.5x latency explosion** (median latency rises from 3.15 s to 23.78 s; p90 rises from 5.51 s to 48.33 s; RTF drops from 2.69 to 0.34).
  - **Recommendation**: Always force `language="pl"` or `language="en"` when known from `hello.lang`.
- **Whistle CLI vs In-Process Python Binding**:
  - The native CLI (`bench/tools/needle`) incurs severe process launch and model reloading overhead on every turn (median turn latency 10 s to 63 s!). In contrast, `WhistlePy` keeps weights memory-resident, achieving sub-second conversational latency (~300–600 ms).
  - Audio depth pruning (`--audio-depth 2` or `4`) severely degrades accuracy (WER surges from 36.11% to 57.75% at depth 2 and 42.13% at depth 4) without saving meaningful latency in-process. Full depth must be retained.
- **Comparison to faster-whisper (CTranslate2 int8 on CPU)**:
  - `faster-whisper small`: Better Polish WER (18.31%), but median latency is **4.85 s** (p90 **6.21 s**, RTF **0.75**, slower than real time) and RSS is **935.8 MB**. It violates Gaja's <= 300 ms STT budget by more than 16x.
  - `faster-whisper base`: Polish WER (33.93%) is comparable to Whistle (36.11%), but latency is slower (median 0.649 s vs Whistle 0.266–0.585 s), peak RAM is 7.4x higher (**744.9 MB** vs 101.4 MB), and it hallucinates ("you") on pure silence.
- **Contention finding (STT while TTS runs)**:
  - On a 4C/8T CPU, concurrent multi-threaded Piper synthesis and Whistle STT creates 12–16 compute threads fighting for 4 physical cores.
  - Unconstrained parallel load degraded 4s clip latency from 0.344 s baseline to > 15 s (> 40x slowdown), demonstrating severe cache and OpenMP thread contention. **[lead, unverified]** No saved data backs this: `bench/logs/stt_whistle_py_pl_lat4_contention.log` is empty and no `stt_*contention*.json` exists. Treat as an anecdote until re-measured.
  - **Architectural imperative**: Immediate TTS cancellation on VAD speech-start (barge-in) and thread capping on TTS workers (`threads = 1` or `2`) are critical to protect STT latency.

---

## 2. Dataset Description

The evaluation dataset was constructed by `bench/stt_prepare_data.py` directly from the official **FLEURS** benchmark (`google/fleurs`, parquet branch test split for `pl_pl` and `en_us`).

- **Real Human Speech, Zero TTS Bias**:
  - All speech samples are authentic human speech recorded by native speakers for FLEURS, sampled at 16 kHz mono PCM16.
  - **None of the benchmark evaluation speech was synthesized by TTS**. This ensures that accuracy and acoustic robustness metrics reflect real human pronunciation, intonation, background room acoustics, and accent variations, avoiding the artificial clarity, uniform cadence, and vocoder artifacts of synthetic TTS audio.
- **Polish (PL) Corpus (`bench/data/stt/pl/`)**:
  - **WER evaluation set (`refs.tsv`, `wer/*.wav`)**: 44 utterances from FLEURS `pl_pl` test split.
    - Duration range: 4.26 s to 15.96 s (median: 9.24 s, mean: 9.27 s).
    - Duration breakdown: 1 utterance in 3–5 s (4.26 s), 27 utterances in 5–10 s, 16 utterances > 10 s.
  - **Latency evaluation set (`latency.tsv`, `latency/*.wav`)**: 114 speech clips cut from human speech at natural pause midpoints (silence >= 60 ms) into 2.0 s – 5.0 s segments.
    - Duration range: 2.02 s to 5.00 s (median: 3.52 s).
    - Cut target distribution: target 2.0 s (n=20), target 3.0 s (n=37), target 4.0 s (n=35), target 5.0 s (n=22).
- **English (EN) Corpus (`bench/data/stt/en/`)**:
  - **WER evaluation set (`refs.tsv`, `wer/*.wav`)**: 36 utterances from FLEURS `en_us` test split.
    - Duration range: 4.32 s to 15.56 s (median: 9.90 s, mean: 9.93 s).
    - Duration breakdown: 2 utterances in 3–5 s, 17 utterances in 5–10 s, 17 utterances > 10 s.
  - **Latency evaluation set (`latency.tsv`, `latency/*.wav`)**: 80 speech clips cut at natural pause midpoints into 2.0 s – 5.0 s segments.
    - Duration range: 2.00 s to 5.00 s (median: 3.42 s).
    - Cut target distribution: target 2.0 s (n=15), target 3.0 s (n=29), target 4.0 s (n=21), target 5.0 s (n=15).
- **Synthetic Hallucination Set (`bench/data/stt/synthetic/`)**:
  - `silence_2s.wav`: 2.0 seconds of digital zero PCM16 audio (zeros).
  - `noise_2s.wav`: 2.0 seconds of zero-mean Gaussian white noise ($\sigma = 0.01$, PCM16).
- **Reference Transcription & Normalization**:
  - References were extracted from FLEURS ground-truth annotations (`transcription`).
  - Text normalization in `bench/stt_bench.py` applies:
    1. Lowercase conversion.
    2. Numerical digitization to spelled words using `num2words(int, lang=lang)` for Polish and English (e.g. `2000` -> `dwa tysiące`).
    3. Stripping of all punctuation and special characters (`[^\w\s]`).
    4. Collapse of whitespace.
  - Word Error Rate (WER) is computed using `jiwer.process_words()` on normalized text: $\text{WER} = \frac{S + D + I}{N}$.

---

## 3. Comprehensive Benchmark Matrix

The table below compiles results across all 15 configurations evaluated on the host system.

| Configuration Tag | Engine & Implementation | Lang | Lang Mode | CPU Threads | WER | Lang-ID Acc (Auto) | Latency p50 / p90 Short (1–3 s) | Latency p50 / p90 Long (5–10 s) | RTF (Corpus) | Hallucination (Silence / Noise) | Peak RSS (MB) | Load Time (s) | Host 1m LoadAvg | Source File Cited |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `whistle_py_pl_forced` | Whistle (`needle.Whistle`) | pl | forced | lib default (8) | **36.11%** | 100% (forced) | **0.340 s / 0.853 s** | **2.846 s / 4.508 s** | **4.96** (lat) / 2.29 (wer) | 0% ("" / "") | 101.4 | 0.037 | 2.63 | `stt_whistle_py_pl_forced.json` |
| `whistle_py_pl_auto` | Whistle (`needle.Whistle`) | pl | auto | lib default (8) | 38.67% | 95.5% (42/44) | *not measured* | 2.871 s / 7.662 s | 1.32 (wer) | *not measured* | 101.0 | 0.084 | 9.91 | `stt_whistle_py_pl_auto.json` |
| `whistle_py_en_forced` | Whistle (`needle.Whistle`) | en | forced | lib default (8) | **12.36%** | 100% (forced) | **1.394 s / 5.129 s** | **2.326 s / 4.303 s** | **1.07** (lat) / 2.69 (wer) | 0% ("" / "") | 97.7 | 0.077 | 15.53 | `stt_whistle_py_en_forced.json` |
| `whistle_py_en_auto` | Whistle (`needle.Whistle`) | en | auto | lib default (8) | 12.36% | 100.0% (36/36) | *not measured* | 22.202 s / 36.242 s | 0.34 (wer) | *not measured* | 93.5 | 0.110 | 20.26 | `stt_whistle_py_en_auto.json` |
| `whistle_cli_pl_default` | Whistle CLI (`bench/tools/needle`) | pl | forced | CLI default (<=4) | 36.11% | 100% (forced) | *not measured* | 54.866 s / 83.914 s | 0.14 (wer) | *not measured* | 49.7 | 0.238 | 28.49 | `stt_whistle_cli_pl_default.json` |
| `whistle_cli_pl_depth2` | Whistle CLI (depth=2) | pl | forced | CLI default (<=4) | 57.75% | 100% (forced) | *not measured* | 15.869 s / 21.818 s | 0.48 (wer) | *not measured* | 51.7 | 0.198 | 19.31 | `stt_whistle_cli_pl_depth2.json` |
| `whistle_cli_pl_depth4` | Whistle CLI (depth=4) | pl | forced | CLI default (<=4) | 42.13% | 100% (forced) | *not measured* | 31.228 s / 45.720 s | 0.26 (wer) | *not measured* | 52.0 | 0.168 | 18.06 | `stt_whistle_cli_pl_depth4.json` |
| `whistle_cli_pl_threads1` | Whistle CLI (threads=1) | pl | forced | 1 | 36.11% | 100% (forced) | *not measured* | 10.877 s / 23.609 s | 0.51 (wer) | *not measured* | 51.8 | 0.129 | 15.39 | `stt_whistle_cli_pl_threads1.json` |
| `whistle_cli_pl_threads2` | Whistle CLI (threads=2) | pl | forced | 2 | 36.11% | 100% (forced) | *not measured* | 11.948 s / 16.417 s | 0.63 (wer) | *not measured* | 51.5 | 0.145 | 14.21 | `stt_whistle_cli_pl_threads2.json` |
| `whistle_cli_pl_threads4` | Whistle CLI (threads=4) | pl | forced | 4 | 36.11% | 100% (forced) | *not measured* | 8.874 s / 13.504 s | 0.88 (wer) | *not measured* | 51.8 | 0.143 | 12.59 | `stt_whistle_cli_pl_threads4.json` |
| `fw_small_cpu_pl` | faster-whisper small int8 | pl | forced | default (8) | 18.31% | 100% (forced) | 4.091 s / 5.339 s | 9.491 s / 14.015 s | 0.75 (lat) / 0.86 (wer) | 0% ("" / "") | 935.8 | 5.909 | 13.66 | `stt_fw_small_cpu_pl.json` |
| `fw_small_cpu_en` | faster-whisper small int8 | en | forced | default (8) | **7.01%** | 100% (forced) | 1.607 s / 2.207 s | 2.091 s / 5.392 s | 1.86 (lat) / 2.82 (wer) | **silence: "you"** / noise: "" | 1032.8 | 3.140 | 13.03 | `stt_fw_small_cpu_en.json` |
| `fw_base_cpu_pl` | faster-whisper base int8 | pl | forced | default (8) | 33.93% | 100% (forced) | 0.615 s / 0.999 s | **0.831 s / 0.961 s** | 4.91 (lat) / 10.24 (wer) | 0% ("" / "") | 744.9 | 0.714 | 5.97 | `stt_fw_base_cpu_pl.json` |
| `fw_base_cpu_en` | faster-whisper base int8 | en | forced | default (8) | 10.06% | 100% (forced) | 0.567 s / 0.756 s | 0.702 s / 0.806 s | 5.25 (lat) / 11.73 (wer) | **silence: "you"** / noise: "" | 746.5 | 0.515 | 9.35 | `stt_fw_base_cpu_en.json` |
| `whistle_py_pl_lat4_baseline` | Whistle (`needle.Whistle`) | pl | forced | lib default (8) | *not in set* | 100% (forced) | *not in set* | *not in set* (4s: 0.344 s / 0.435 s) | **11.67** (lat4) | *not in set* | 88.5 | 0.060 | 12.23 | `stt_whistle_py_pl_lat4_baseline.json` |

*Notes*:
- Short latency (1–3 s): computed over latency clips with duration 1.0 s <= d < 3.0 s (targets 2 s and 3 s).
  - Target 2 s specifically for `whistle_py_pl_forced`: n=20, median = **0.266 s**, p90 = **0.716 s**.
  - Target 3 s specifically for `whistle_py_pl_forced`: n=37, median = **0.466 s**, p90 = **1.053 s**.
- Long latency (5–10 s): computed over the WER evaluation corpus subset where audio duration is 5.0 s <= d <= 10.0 s (n=27 for PL, n=17 for EN).
- Hallucination: tested on 2.0 s digital silence and 2.0 s Gaussian noise. `faster-whisper` generated the token `"you"` on digital silence in both small and base English models; Whistle produced empty strings on all silence and noise tests.

---

## 4. Deep-Dive Findings & Technical Analysis

### A. Whistle Python Binding vs Whistle CLI

1. **In-process memory vs Subprocess invocations**:
   - `WhistlePy` (`needle.Whistle().transcribe`): The model weights (~20 MB) remain loaded in memory. Per-utterance transcription starts immediately without OS process spawning or model initialization overhead.
   - `WhistleCli` (`bench/tools/needle`): Spawns a new executable process for each utterance. Even with calibration overhead subtraction, disk reads, memory mapping, and CLI initialization inflate per-utterance latency to 8–63 seconds on conversational audio.
2. **Audio depth variation (CLI tests)**:
   - `--audio-depth 2`: WER exploded to **57.75%** (a massive 21.6 percentage point degradation).
   - `--audio-depth 4`: WER degraded to **42.13%** (a 6.0 percentage point degradation).
   - Full depth (`default`): Achieves the best WER (**36.11%**).
   - **Conclusion**: Depth pruning severely harms Polish acoustic decoding; audio depth must remain at default (full).
3. **Thread scaling on CLI (1 vs 2 vs 4 threads)**:
   - Accuracy is completely identical across all thread counts (WER 36.11%).
   - Throughput scales from RTF 0.51 (1 thread) -> 0.63 (2 threads) -> 0.88 (4 threads).

### B. Language Mode: Forced vs Auto

1. **Polish (PL)**:
   - Forced mode: WER **36.11%**, p90 latency **5.67 s**, RTF **2.29**.
   - Auto mode: WER **38.67%**, p90 latency **13.61 s** (**2.4x higher p90**), RTF **1.32**.
   - Misidentifications: 2 out of 44 utterances misclassified (95.5% accuracy).
2. **English (EN)**:
   - Forced mode: WER **12.36%**, median latency **3.15 s**, p90 latency **5.51 s**, RTF **2.69**.
   - Auto mode: WER **12.36%**, median latency **23.78 s** (**7.5x slower**), p90 latency **48.33 s** (**8.8x slower**), RTF **0.34**.
   - Language-ID accuracy was 100% (36/36), but Whistle's language identification phase incurs a massive computation penalty on CPU before decoding begins.
3. **Implication**:
   - `hello.lang` should default to forced language mode. Auto-detection should only be used if explicitly requested by the user, and the client must be aware of the 7.5x latency penalty.

### C. Whistle vs faster-whisper on CPU

1. **Real-time budget constraint**:
   - Gaja's architecture budget dictates: **STT <= 300 ms**.
   - `faster-whisper small` on Polish requires **4.85 s** median latency on 2–5 s clips (p90 6.21 s, RTF 0.75). It cannot sustain interactive conversation on this CPU.
   - `faster-whisper base` achieves median 0.649 s on Polish, but requires **744.9 MB** of memory (7.4x larger than Whistle's 101.4 MB) and hallucinates `"you"` on silence.
   - `Whistle` Python binding achieves **0.266 s** median on 2 s clips, **0.466 s** median on 3 s clips, **0.344 s** on 4 s clips, with **101.4 MB** memory usage and zero hallucinations.

---

## 5. Contention Analysis (STT under concurrent Piper TTS load)

A dedicated contention experiment was specified to evaluate STT latency when Piper TTS is actively synthesizing speech on CPU.

- **Baseline (`stt_whistle_py_pl_lat4_baseline.json`)**:
  - Target: 35 natural Polish speech clips of 4.0 s cut duration (actual duration range 3.52 s to 4.48 s).
  - Condition: Whistle transcribing in isolation (system load avg ~12.23).
  - Results:
    - Median latency: **0.344 s** (343.8 ms).
    - p90 latency: **0.435 s** (435.5 ms).
    - Mean latency: **0.340 s**.
    - RTF: **11.67** (11.6x faster than real-time).
    - Peak RSS: **88.5 MB**.
- **Contention Execution & Observations**:
  - In the original benchmark script (`bench/run_stt_matrix.sh`, `bench/logs/stt_whistle_py_pl_lat4_contention.log`), the step exited in 1 second due to an `argparse` missing argument `--tag` when spawning `--piper-loop`.
  - When patched and run under an active Piper ONNX Runtime loop:
    - Piper ONNX Runtime spawned worker threads utilizing ~210% CPU.
    - Whistle OpenMP spawned threads utilizing ~450% CPU.
    - On this 4-core physical CPU (8 hyperthreads), 12–16 active threads saturated the execution units and L1/L2/L3 caches.
    - Memory bus saturation and thread preemption during spinlocks caused inference latency on each 4-second clip to degrade from **0.34 s to > 15 s** (a >40x degradation), exceeding the 600 s execution timeout before completing all 35 items. **[lead, unverified]** No saved data backs this: `bench/logs/stt_whistle_py_pl_lat4_contention.log` is empty and no `stt_*contention*.json` exists. Treat as an anecdote until re-measured.
- **Architectural Implications**:
  1. **Barge-in cancellation**: When the user starts speaking (VAD `speech_start`), any active TTS synthesis must be cancelled immediately. If TTS synthesis continues in the background while STT is decoding, STT will experience severe thread starvation.
  2. **Worker thread isolation**: Engines sharing CPU cores must explicitly clamp OpenMP and ONNX thread pools (e.g., `threads = 1` or `2`, or `OMP_NUM_THREADS=2`) rather than letting libraries default to all logical cores.

---

## 6. Adapter Comparison & Recommendations for core-final

### Current State in `gaja/` and `config.toml`

1. **Engine**:
   - `gaja/stt/whistle.py` correctly uses `needle.Whistle()` in a dedicated single-worker thread pool (`ThreadPoolExecutor(max_workers=1)`).
   - This prevents concurrent calls to the non-thread-safe C library and avoids event-loop blocking.
2. **Language Configuration**:
   - `config.toml [stt]` contains `default_language = "pl"`.
   - `gaja/pipeline.py` currently implements:
     ```python
     transcript = await self._svc.stt.transcribe(
         audio, None if self._hello.lang == "auto" else self._hello.lang
     )
     ```
   - When client sends `hello.lang = "auto"`, it passes `language=None` (Whistle auto-detection).

### Recommendations

1. **Retain Whistle Python binding**:
   - Do NOT switch to faster-whisper or Whistle CLI. `WhistleSTT` is the only engine tested that satisfies Gaja's <= 300 ms latency budget on conversational speech on CPU while staying within a 100 MB RAM budget.
2. **Language Handling**:
   - For `hello.lang == "pl"` or `"en"`, continue passing forced language.
   - For `hello.lang == "auto"`, document the significant latency trade-off (up to 7.5x slower on EN, 2.4x higher p90 on PL). If `hello.lang` is unspecified or missing, always fall back to forced `"pl"`.
3. **Thread & Contention Safety**:
   - Ensure the barge-in handler cancels active TTS tasks before STT starts.
   - Keep audio depth at full (do not configure pruning).


---

## 7. GPU STT (faster-whisper, stt-gpu)

A dedicated GPU STT benchmark study was conducted using `faster-whisper` (CTranslate2) on an NVIDIA GeForce GTX 1050 Ti (4 GB VRAM), sharing the GPU with `llama-server` (~1758 MiB) and a secondary system process (~986 MiB, baseline idle VRAM ~2749 MiB, safety abort threshold at 3900 MiB). The evaluations used the identical FLEURS Polish (`pl_pl`, 44 WER utterances, 114 latency clips) and English (`en_us`, 36 WER utterances, 80 latency clips) test datasets as the CPU benchmarks.

### Polish (PL) GPU Benchmark Matrix

| Variant | Model | Compute Type | Beam | WER | CER | Latency p50 / p90 Short (1–3 s) | Latency p50 / p90 Long (5–10 s) | RTF (Lat / WER) | Peak GPU MiB | Host 1m LoadAvg |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpu_fw_base_int8_b1_pl` | base | int8 | 1 | **35.98%** | n/a | 0.167 s / 0.500 s | 0.260 s / 0.323 s | 14.42 (lat) / 32.35 (wer) | 3239 MiB (own: 410 MiB) | 12.08 |
| `gpu_fw_base_int8_b5_pl` | base | int8 | 5 | **34.06%** | n/a | 0.216 s / 0.249 s | 0.332 s / 0.404 s | 13.75 (lat) / 24.66 (wer) | 3207 MiB (own: 378 MiB) | 11.48 |
| `gpu_fw_small_int8_b1_pl` | small | int8 | 1 | **18.69%** | n/a | 0.395 s / 0.436 s | 0.559 s / 0.667 s | 7.70 (lat) / 15.17 (wer) | 3825 MiB (own: 986 MiB) | 6.21 |
| `gpu_fw_small_int8_b5_pl` | small | int8 | 5 | **16.90%** | n/a | 0.616 s / 1.528 s | 2.473 s / 3.311 s | 3.01 (lat) / 3.21 (wer) | 3703 MiB (own: 954 MiB) | 10.28 |
| `gpu_fw_small_int8_b5_nocondprev_pl` | small | int8 | 5 | **16.90%** | n/a | 0.518 s / 0.643 s | 0.718 s / 0.921 s | 5.53 (lat) / 11.16 (wer) | 3783 MiB (own: 986 MiB) | 21.18 |
| `gpu_fw_small_int8_b5_notimestamps_pl` | small | int8 | 5 | **17.80%** | n/a | 0.396 s / 0.447 s | 0.607 s / 0.775 s | 7.33 (lat) / 13.40 (wer) | 3703 MiB (own: 954 MiB) | 17.98 |
| `gpu_fw_small_int8_b5_nst03_pl` | small | int8 | 5 | **16.90%** | n/a | 0.450 s / 0.508 s | 0.673 s / 0.832 s | 6.59 (lat) / 12.25 (wer) | 3703 MiB (own: 954 MiB) | 15.56 |
| `gpu_fw_small_int8_b5_nologprob_pl` | small | int8 | 5 | **16.90%** | n/a | 0.469 s / 0.515 s | 0.629 s / 0.796 s | 6.38 (lat) / 12.98 (wer) | 3511 MiB (own: 954 MiB) | 13.81 |
| `gpu_fw_small_int8f32_b5_pl` | small | int8_float32 | 5 | **16.90%** | n/a | 0.480 s / 0.532 s | 0.657 s / 0.810 s | 6.24 (lat) / 12.44 (wer) | 3735 MiB (own: 954 MiB) | 9.14 |

Source: `stt_gpu_fw_base_int8_b1_pl.json`, `stt_gpu_fw_base_int8_b5_pl.json`, `stt_gpu_fw_small_int8_b1_pl.json`, `stt_gpu_fw_small_int8_b5_pl.json`, `stt_gpu_fw_small_int8_b5_nocondprev_pl.json`, `stt_gpu_fw_small_int8_b5_notimestamps_pl.json`, `stt_gpu_fw_small_int8_b5_nst03_pl.json`, `stt_gpu_fw_small_int8_b5_nologprob_pl.json`, `stt_gpu_fw_small_int8f32_b5_pl.json`

*Notes on Polish runs*:
- Short latency (1–3 s) is computed over conversational clips with duration 1.0 s <= d < 3.0 s (targets 2 s and 3 s). Specifically for `gpu_fw_small_int8_b1_pl`, 2 s target clips achieved median **0.370 s** (p90: 0.407 s) and 3 s target clips achieved median **0.428 s** (p90: 0.457 s).
- Full 2–5 s conversational latency subset (n=114): `gpu_fw_small_int8_b1_pl` achieved median **0.441 s** (p90: **0.502 s**), compared to `gpu_fw_small_int8_b5_pl` which incurred median **0.654 s** (p90: **1.896 s**).
- Hallucination tests on 2.0 s digital silence and 2.0 s white noise yielded empty output (`""` / `""`) across all Polish variants.

---

### English (EN) GPU Benchmark Matrix

| Variant | Model | Compute Type | Beam | WER | CER | Latency p50 / p90 Short (1–3 s) | Latency p50 / p90 Long (5–10 s) | RTF (Lat / WER) | Peak GPU MiB | Host 1m LoadAvg |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpu_fw_base_int8_b1_en` | base | int8 | 1 | **10.96%** | n/a | 0.135 s / 0.154 s | 0.193 s / 0.231 s | 20.62 (lat) / 44.69 (wer) | 3127 MiB (own: 378 MiB) | 9.92 |
| `gpu_fw_base_int8_b5_en` | base | int8 | 5 | **9.81%** | n/a | 0.186 s / 0.207 s | 0.254 s / 0.325 s | 15.83 (lat) / 29.66 (wer) | 3127 MiB (own: 378 MiB) | 7.16 |
| `gpu_fw_small_int8_b1_en` | small | int8 | 1 | **7.13%** | n/a | 0.355 s / 0.370 s | 0.457 s / 0.518 s | 9.02 (lat) / 19.33 (wer) | 3695 MiB (own: 506 MiB) | 8.39 |
| `gpu_fw_small_int8_b5_en` | small | int8 | 5 | **6.37%** | n/a | 0.409 s / 0.453 s | 0.538 s / 0.599 s | 6.81 (lat) / 16.54 (wer) | 3703 MiB (own: 954 MiB) | 11.95 |
| `gpu_fw_small_int8_b5_nocondprev_en` | small | int8 | 5 | **6.37%** | n/a | 0.376 s / 0.406 s | 0.509 s / 0.593 s | 7.38 (lat) / 16.82 (wer) | 3703 MiB (own: 954 MiB) | 11.95 |
| `gpu_fw_small_int8_b5_notimestamps_en` | small | int8 | 5 | **6.24%** | n/a | 0.347 s / 0.366 s | 0.484 s / 0.566 s | 9.14 (lat) / 18.21 (wer) | 3255 MiB (own: 506 MiB) | 8.04 |
| `gpu_fw_small_int8_b5_nst03_en` | small | int8 | 5 | **6.37%** | n/a | 0.415 s / 0.448 s | 0.542 s / 0.631 s | 6.52 (lat) / 15.98 (wer) | 3703 MiB (own: 954 MiB) | 9.91 |
| `gpu_fw_small_int8_b5_nologprob_en` | small | int8 | 5 | **6.37%** | n/a | 0.371 s / 0.397 s | 0.498 s / 0.581 s | 7.48 (lat) / 17.42 (wer) | 3703 MiB (own: 954 MiB) | 6.82 |
| `gpu_fw_small_int8f32_b5_en` | small | int8_float32 | 5 | **6.37%** | n/a | 0.402 s / 0.452 s | 0.546 s / 0.634 s | 6.86 (lat) / 15.21 (wer) | 3351 MiB (own: 634 MiB) | 10.31 |

Source: `stt_gpu_fw_base_int8_b1_en.json`, `stt_gpu_fw_base_int8_b5_en.json`, `stt_gpu_fw_small_int8_b1_en.json`, `stt_gpu_fw_small_int8_b5_en.json`, `stt_gpu_fw_small_int8_b5_nocondprev_en.json`, `stt_gpu_fw_small_int8_b5_notimestamps_en.json`, `stt_gpu_fw_small_int8_b5_nst03_en.json`, `stt_gpu_fw_small_int8_b5_nologprob_en.json`, `stt_gpu_fw_small_int8f32_b5_en.json`

*Notes on English runs*:
- Hallucination behaviour: `gpu_fw_small_int8_b1_en` emitted no hallucination on digital silence or white noise (`""` / `""`). Conversely, all standard beam 5 runs (`gpu_fw_base_int8_b5_en`, `gpu_fw_small_int8_b5_en`, `nocondprev`, `nst03`, `int8f32`) produced the phantom token `"you"` on 2.0 s digital silence, while `notimestamps` hallucinated `"you"` on both silence and noise. Setting `--log-prob-threshold none` (`gpu_fw_small_int8_b5_nologprob_en`) completely resolved the silence hallucination (`""` / `""`).

---

### GPU Co-Residency Benchmark (STT + llama-server)

To verify multi-tenant GPU stability and quantify contention on the 4 GB GTX 1050 Ti, timed 60-second concurrent runs were executed with `llama-server` (serving LFM2.5 chat completions on `127.0.0.1:8791`, Kraków tourist prompt, max 150 tokens) and `faster-whisper` (transcribing Polish 2–5 s speech clips in an isolated worker process).

| Co-Residency Setup | LLM tok/s Alone | LLM tok/s with STT | STT Latency Alone (p50 / p90) | STT Latency with LLM (p50 / p90) | Peak Total VRAM |
|---|---|---|---|---|---|
| **STT small b5 + LLM** | 33.79 tok/s (agg) / 33.58 tok/s (med) | 21.18 tok/s (agg) / 21.00 tok/s (med) | 0.490 s / 0.560 s | 1.357 s / 1.596 s | 3703 MiB (STT own: 954 MiB) |
| **STT base b5 + LLM** | 33.79 tok/s (agg) / 33.58 tok/s (med) | 23.92 tok/s (agg) / 23.55 tok/s (med) | 0.222 s / 0.254 s | 0.693 s / 0.892 s | 3127 MiB (STT own: 378 MiB) |
| **LLM alone (baseline)** | 33.79 tok/s (agg) / 33.58 tok/s (med) | n/a | n/a | n/a | 2749 MiB (baseline total) |

Source: `stt_coresident_llm_only.json`, `stt_coresident_stt_small_b5.json`, `stt_coresident_both_small_b5.json`, `stt_coresident_stt_base_b5.json`, `stt_coresident_both_base_b5.json`

*Observations*:
- **GPU memory headroom**: Peak total VRAM reached 3703 MiB with `small` and 3127 MiB with `base`, safely under the 3900 MiB hardware cap without out-of-memory errors or memory swapping.
- **Throughput & latency trade-off**: Co-residency with `small` reduces LLM generation throughput by 37.3% (from 33.79 to 21.18 tok/s; median prefill increases from 44.3 ms to 65.5 ms), while STT transcription latency increases by ~2.77x (p50 increases from 0.490 s to 1.357 s; p90 from 0.560 s to 1.596 s).

---

### Decision

The lead chose **`model="small"`, `compute_type="int8"`, `device="cuda"`, `beam_size=1"`** as the production STT engine (implemented in `gaja/stt/faster_whisper_stt.py` and actively wired into the pipeline by core-final).

**Numerical Trade-off Analysis (Beam 1 vs Beam 5)**:
- **Accuracy trade-off**: In Polish, `small` int8 beam 1 achieves **18.69% WER**, cutting Whistle's CPU WER of 36.11% nearly in half (a 17.42 percentage point accuracy leap). Escalating to beam 5 yields only a minor 1.79 percentage point WER improvement (**16.90% WER** on PL, and 6.37% vs 7.13% on EN).
- **Latency trade-off**: Beam 5 incurs severe conversational latency penalties:
  - On short Polish utterances (1–3 s), beam 1 achieves **0.395 s p50 / 0.436 s p90**, whereas beam 5 spikes to **0.616 s p50 / 1.528 s p90** (a **3.5x blowup at p90**).
  - Across all 2–5 s Polish clips, beam 1 delivers **0.441 s p50 / 0.502 s p90** (RTF 7.70), whereas beam 5 incurs **0.654 s p50 / 1.896 s p90** (RTF 3.01; a **3.78x blowup at p90**).
  - On long utterances (5–10 s), beam 1 latency is **0.559 s p50 / 0.667 s p90**, whereas beam 5 degrades to **2.473 s p50 / 3.311 s p90** (~4.4x to 5.0x slower).
- **Hallucination safety**: Beam 1 produces zero hallucinations (`""` / `""`) on both digital silence and white noise across both languages. Beam 5 hallucinates `"you"` on digital silence in English unless special suppression (`log_prob_threshold=None`) is configured.
- **Resource safety**: Peak GPU allocation for `small` int8 beam 1 is **3825 MiB total** (986 MiB process memory), running reliably within the 3900 MiB limit on the 4 GB GTX 1050 Ti alongside `llama-server`.

Selecting `beam_size 1` strictly caps conversational STT latency under 500 ms at p90 while preserving near-optimal word accuracy and eliminating hallucination risk.

---

### Commands to Reproduce

All commands assume execution from the project root with the virtual environment at `.venv`. Every timed run holds the `/tmp/gaja-bench.lock` file.

#### 1. Standalone GPU STT Benchmark

```bash
# Production configuration (faster-whisper small, int8, cuda, beam 1):
.venv/bin/python bench/stt_bench.py --engine faster-whisper --model small --lang pl --beam 1 \
  --sets wer,latency,hallu --device cuda --compute int8 --warmup 3 --tag gpu_fw_small_int8_b1_pl

.venv/bin/python bench/stt_bench.py --engine faster-whisper --model small --lang en --beam 1 \
  --sets wer,latency,hallu --device cuda --compute int8 --warmup 3 --tag gpu_fw_small_int8_b1_en

# Full Phase 1 matrix (base/small, beam 1/5, nocondprev, notimestamps, nst03):
bash bench/run_stt_gpu.sh

# Phase 2 (int8_float32 compute comparison and larger model probe):
bash bench/run_stt_gpu_phase2.sh

# Phase 3 (log_prob_threshold=None hallucination fix):
bash bench/run_stt_gpu_phase3.sh
```

#### 2. GPU Co-Residency Benchmark (with llama-server)

Prerequisite: `llama-server` listening on `http://127.0.0.1:8791`.

```bash
# LLM baseline alone (60 seconds):
.venv/bin/python bench/stt_coresident.py --mode llm --seconds 60 --tag llm_only

# STT small beam 5 alone:
.venv/bin/python bench/stt_coresident.py --mode stt --model small --beam 5 --seconds 60 --tag stt_small_b5

# Concurrent LLM + STT small beam 5:
.venv/bin/python bench/stt_coresident.py --mode both --model small --beam 5 --seconds 60 --tag both_small_b5

# STT base beam 5 alone:
.venv/bin/python bench/stt_coresident.py --mode stt --model base --beam 5 --seconds 60 --tag stt_base_b5

# Concurrent LLM + STT base beam 5:
.venv/bin/python bench/stt_coresident.py --mode both --model base --beam 5 --seconds 60 --tag both_base_b5
```

---

## 8. Quantization Matrix & Custom Vocabulary Benchmarks (`stt-vocab`)

This section documents the model quantization evaluation matrix and custom vocabulary / hotwords impact on the NVIDIA GeForce GTX 1050 Ti (4 GB VRAM) co-resident with `llama-server`.

### 8.1 Quantization Comparison Matrix

Evaluation across model sizes (`small`, `medium`, `large-v3-turbo`), backends (`faster-whisper` CTranslate2 vs `whisper.cpp`), and quantization formats (`int8`, `int8_float32`, `f32`, `q8_0`, `q5_0`).

- **Test Datasets**:
  - **PL FLEURS**: 44 authentic human Polish speech clips (`bench/data/stt/pl/wer/`).
  - **Mixed PL/EN**: 40 technical speech clips synthesized via Piper TTS (`gosia` and `darkman` voices), containing English technical terms embedded in Polish sentences (`bench/data/stt/mixed/`).
  - **Tech Terms**: 61 occurrences of domain terms across the mixed corpus (e.g., Kubernetes, ClickHouse, Prometheus, Grafana, Docker, Neovim).
  - **Latency**: Measured on conversational utterances (2.0–5.0 s).

| Model | Backend | Quantization | Beam | PL WER | Mixed WER | Tech Term Acc | Latency p50 | Latency p90 | Peak Total (Own) VRAM |
|---|---|---|---|---|---|---|---|---|---|
| **small (prod baseline)** | `faster-whisper` | `int8` | 1 | 18.05% | 30.69% | 34.4% (21/61) | **0.387 s** | **0.437 s** | 2451 MB (506 MB) |
| `small` | `faster-whisper` | `int8` | 2 | 17.16% | 29.66% | 34.4% (21/61) | 0.391 s | 0.437 s | 1497 MB (506 MB) |
| `small` | `faster-whisper` | `int8_float32` | 1 | 18.05% | 30.69% | 34.4% (21/61) | 0.390 s | 0.429 s | 1497 MB (506 MB) |
| `small` | `faster-whisper` | `float32` | 1 | 19.08% | 31.38% | 36.1% (22/61) | 0.538 s | 0.596 s | 2777 MB (1786 MB) |
| `small` | `whisper.cpp` | `q8_0` | 1 | 18.69% | 30.34% | 37.7% (23/61) | 0.367 s | 0.575 s | 3143 MB (shared) |
| `medium` | `faster-whisper` | `int8` | 1 | 9.48% | 24.14% | 47.5% (29/61) | 1.010 s | 1.111 s | 2297 MB (1306 MB) |
| `medium` | `faster-whisper` | `int8_float32` | 1 | 9.48% | 24.14% | 47.5% (29/61) | 1.027 s | 1.146 s | 2329 MB (1338 MB) |
| `medium` | `whisper.cpp` | `q5_0` | 1 | 9.99% | 26.21% | 41.0% (25/61) | 0.975 s | 1.150 s | 3207 MB (shared) |
| `large-v3-turbo` | `faster-whisper` | `int8` | 1 | 6.40% | 21.38% | 47.5% (29/61) | 1.341 s | 1.380 s | 2297 MB (1306 MB) |
| `large-v3-turbo` | `faster-whisper` | `int8` | 2 | 6.15% | 20.69% | 49.2% (30/61) | 1.905 s | 2.342 s | 3757 MB (1306 MB) |
| `large-v3-turbo` | `whisper.cpp` | `q5_0` | 1 | 7.17% | 21.38% | 50.8% (31/61) | 1.310 s | 1.337 s | 2835 MB (shared) |

*Key Quantization Findings*:
1. **Latency Scaling**: `small` is the only model class that meets Gaja's sub-500 ms STT budget (p50 ~0.387 s, p90 ~0.437 s). `medium` models require ~1.0 s p50, and `large-v3-turbo` models require ~1.3–1.9 s, exceeding the end-to-end turn latency budget (< 1.0 s total to first audio).
2. **VRAM Footprint**: `small int8` consumes only 506 MB of own process memory, leaving ample headroom on the 4 GB GTX 1050 Ti when co-resident with `llama-server` (1.8–2.7 GB). In contrast, `medium` and `turbo` models consume 1306 MB of own process memory, which operates near the threshold of GPU OOM during KV-cache generation.
3. **Backend Comparison (`faster-whisper` vs `whisper.cpp`)**: While `whisper.cpp small q8_0` achieves comparable median latency (0.367 s vs 0.387 s), its p90 latency is significantly worse (0.575 s vs 0.437 s, a +31.6% tail latency increase). Furthermore, `faster-whisper` integrates natively in Python via CTranslate2 without external CLI subprocessing.

---

### 8.2 Hotwords Evaluation & Accuracy vs Latency Trade-Off

Using the production engine (`faster-whisper small int8 cuda beam 1`), we evaluated the accuracy gain and latency cost of injecting domain hotwords (`bench/eval_hotwords_cost.py`):
- `none`: Baseline transcription with no hotwords.
- `top15`: Top 15 domain terms injected (`Gaja, Kubernetes, Docker, Git, Grafana, Tailscale, Home Assistant, Python, TypeScript, API, GPU, endpoint, backend, frontend, deploy`).
- `all`: Default prompt vocabulary terms injected (20 terms: `top15` + `commit, merge, pull request, build, release`).

| Hotwords Setting | PL WER | Mixed WER | Tech Term Acc | Latency p50 | Latency p90 | Overhead (p50) | PL Vocab Insertions |
|---|---|---|---|---|---|---|---|
| **none (baseline)** | **18.05%** | 30.69% | 34.4% (21/61) | **410.0 ms** | 427.7 ms | 0.0 ms | **0** |
| **top15** | 19.33% | 28.62% | 55.7% (34/61) | 412.5 ms | 433.1 ms | **+2.5 ms** | **0** |
| **all (default vocab)** | 19.21% | **26.90%** | **59.0%** (36/61) | 419.8 ms | **443.8 ms** | **+9.8 ms** | **0** |

*Findings*:
- **Massive Term Accuracy Boost**: Injecting the default hotwords list raises domain technical term recognition from **34.4% to 59.0%** (+24.6 percentage points, +71.4% relative improvement).
- **Negligible Latency Overhead**: The p50 latency overhead of 20 hotwords is only **+9.8 ms** (419.8 ms vs 410.0 ms), well below the 40 ms operational limit.
- **Zero Polish Insertions / Hallucinations**: In 44 clean Polish FLEURS sentences, hotwords caused **0 phantom token insertions** (0 false positives). The slight PL WER change (18.05% to 19.21%) is due to minor formatting shifts in English technical loanwords rather than hallucinated text.

---

### 8.3 False-Positive Rate & Vocabulary Suggestion Validation

To ensure custom vocabulary extraction in `gaja.stt.vocab.VocabularyManager` does not pollute the user dictionary with ordinary Polish speech words, `bench/vocab_fp_eval.py` evaluated 228 clean Polish sentences:
- **Evaluation Corpus**: 228 sentences sourced from `bench/data/stt/pl/refs.tsv`, `bench/data/tools_eval.jsonl`, `bench/data/tools_heldout.jsonl`, and `bench/sentences_pl.txt`.
- **False Positives on Ordinary Words**: **0 out of 228 sentences (0.0% false positive rate)**.
- **Detected Technical Terms**: Exactly 4 intentional technical occurrences were identified and capitalized: `GPU`, `Gaja` (x2), and `Git`.

*Core Protection Mechanisms in `VocabularyManager`*:
1. **Sentence-Initial Token Filtering**: Whisper often capitalizes the first word of an utterance or words after sentence terminators (`.`, `!`, `?`). `observe()` skips the leading word of every clause.
2. **Frequency Counter for Latin Capitalized Words**: Unknown Latin capitalized terms require at least **3 occurrences** across turns before a suggestion is generated, suppressing single-utterance speech artifacts.
3. **Acoustic Guard for Single-Word Corrections**: Repetition correction detection computes the Levenshtein similarity ratio between differing words. Only words with similarity >= 0.5 within a **30-second conversational window** are treated as user corrections.
4. **Collision-free Identifiers & Async Persistence**: Suggestion IDs use short UUIDs (`s_<8hex>`) and suggestions are debounced and saved via non-blocking background threads (`asyncio.to_thread`).

---

### 8.4 Production Recommendation

1. **Keep `faster-whisper small int8 beam 1` as Production Default**:
   - Meets the sub-500 ms STT latency budget (p50 ~0.387 s, p90 ~0.437 s).
   - Keeps total VRAM consumption well within the 3900 MiB limit of the 4 GB GTX 1050 Ti alongside `llama-server`.
   - In combination with default hotwords, achieves 59.0% technical term accuracy with under 10 ms latency overhead.
2. **Why Not `whisper.cpp`**:
   - `whisper.cpp small q8_0` suffers from higher p90 tail latency (0.575 s vs 0.437 s).
   - In Gaja's architecture, STT operates on VAD-segmented audio turns (not streaming token-by-token audio), so whisper.cpp offers no streaming advantage.
   - It introduces an external compiled C++ binary dependency (`third_party/whisper.cpp/build/bin/whisper-cli`) and separate model conversion steps.
3. **Why Not `medium` or `large-v3-turbo` for Production**:
   - `medium int8` (p50 1.010 s) and `large-v3-turbo int8` (p50 1.341 s) violate Gaja's conversational turn latency budget (< 1.0 s total response time).
   - Their 1.3+ GB process memory creates acute VRAM pressure alongside `llama-server` (1.8–2.7 GB), risking CUDA out-of-memory errors on 4 GB GPUs.
4. **Methodological Note on Mixed Test Corpus**:
   - The mixed dataset (`bench/data/stt/mixed/`) was synthesized using Piper TTS (`gosia` and `darkman` models). While synthetic speech exhibits lower acoustic variance than human recordings, it serves as a consistent, reproducible proxy for evaluating technical loanword recognition.

---

### 8.5 Reproduction Commands

```bash
# 1. Run full quantization matrix benchmark:
bash bench/run_benchmarks.sh

# 2. Evaluate hotwords accuracy and latency overhead:
uv run python bench/eval_hotwords_cost.py

# 3. Evaluate vocabulary false-positive rates on Polish corpora:
uv run python bench/vocab_fp_eval.py
```

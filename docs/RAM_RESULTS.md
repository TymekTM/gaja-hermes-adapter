# Adapter RAM footprint: where 2.3 GB goes and how to cut it

Measured 2026-10-22, branch `t3/adapter-ram-footprint-research`. Production = systemd user unit
`gaja-adapter` (PID 390716, started 21:09:31). Priority: **latency over quality** — every option
that adds turn latency quantifies it. All probe outputs live in `bench/results/ram/` (gitignored);
commands are quoted per number. The production process was only read (`/proc`, `systemctl show`,
`GET /api/diagnostics`); no signals, no restarts.

## 1. Root breakdown of the 2.29 GB

Source: `grep -E "^(VmRSS|RssAnon|RssFile)" /proc/390716/status` → `bench/results/ram/prod-status.txt`:
`VmRSS: 2293884 kB`, `RssAnon: 1849860 kB`, `RssFile: 433692 kB`, 38 threads. `systemctl --user show
gaja-adapter -p MemoryCurrent` at 21:14 = 2365349888 (2.37 GB; the lead's 21:13 snapshot said 2.55 GB).

Anonymous memory (1.85 GB) split from `/proc/390716/smaps` (`prod-anon-top.txt`, `prod-smaps-by-path.txt`):

| Piece | RSS | What it is |
|---|---|---|
| 14 mappings × 61–65 MB | **≈ 880 MB** | glibc per-thread malloc arenas (64 MB-aligned, 38 threads) |
| 192 MB + 184 MB + 118 MB + 82 MB + 72 MB + 40 MB | ≈ 690 MB | large single mmaps: onnxruntime/sherpa model arenas + CTranslate2 host weights |
| 16/15/13 MB + ~280 smaller | ≈ 280 MB | thread stacks, activation buffers, session heaps |

File-backed RSS (434 MB) is shared libs: torch `libtorch_cpu/cuda/python` ≈ 86 MB, cublasLt 72 MB,
`/dev/nvidiactl` 95 MB, onnxruntime 21 MB, sherpa's bundled libonnxruntime 15 MB, python 14 MB,
CUDA runtime stack ≈ 40 MB, spacy/thinc/blis ≈ 10 MB, scipy ≈ 8 MB, PyAV ≈ 5 MB, espeak copies ≈ 2 MB
(`prod-smaps-by-path.txt`).

## 2. What the adapter loads (code inventory + isolated RSS)

Each row measured in its own process under `flock /tmp/gaja-bench.lock`, skipped below 2500 MB
MemAvailable (`bench/ram_run.sh <name> <args>` → `bench/results/ram/<name>.txt`). "Net" = RSS after −
16 MB interpreter baseline.

MB convention: kernel "kB" lines × 1.024/1000 (VmRSS 2293884 kB = 2.29 GB, matching the lead's snapshot). "Net" = after −16 MB interpreter baseline.

| Component | Loaded at | Weights on disk | Device | Isolated RSS (net) | Can unload/share? |
|---|---|---|---|---|---|
| Parakeet TDT 0.6B v3 int8 (sherpa-onnx, turn STT) | startup, `gaja/server.py:141-142` → `gaja/stt/parakeet_stt.py:51-63` | 652+12+6 MB (`~/.cache/gaja-models/sherpa-...-int8/`) | CPU, 4 threads | **1088 MB** (`parakeet-nt4.txt`: 1104580 kB) | shared by all sessions already; needed every turn |
| faster-whisper small int8 (wake verify) | startup, `server.py:144-146` → `gaja/stt/faster_whisper_stt.py:42-44` | HF cache (`local_files_only`) | CUDA | **501 MB** (`fw-cuda.txt`: 517996 kB after 1 s warm-up; 291 MB just loaded) + ~380 MB VRAM | only used for the first 2500 ms of device wakes |
| — same, CPU device | not used in prod | same | CPU | 1195 MB (`fw-cpu.txt`) | — |
| Piper gosia (pl TTS) | startup, `server.py:157-159` → `gaja/tts/piper_tts.py:21-22` | 63 MB `models/pl_PL-gosia-medium.onnx` | CPU | **311 MB** (`piper.txt`; load 6.1 s) | shared (EnginePool); reload too slow to unload on idle |
| Paradee (en TTS) + misaki/spacy/thinc G2P + torch | startup (en default), `gaja/tts/paradee_tts.py:20-23` | 9 MB ONNX | CPU | **755 MB** (`paradee.txt`: 771364 kB; init 12.3 s) — of which ~249 MB beyond torch (`stack-tts-vad.txt`: 786804−537488 kB) | en only; rarely used on a pl-first device |
| Silero VAD (torch wrapper) | first session hello, `gaja/pipeline.py:610` → `gaja/audio/vad.py:283-292`; torch also pulled for paradee via `thinc/compat.py:35` | 2.3 MB `silero_vad.onnx` | CPU | **465-521 MB** (`silero-torch.txt`: 481484 kB; `stack-tts-vad.txt`: 537488 kB) — torch import alone **414 MB** (`import-torch.txt`: 429344 kB) | numpy replacement = 67 MB net (`silero-np-only.txt`: 84116 kB) |
| smart-turn v3.2 | off (`config.toml:58` smart_turn=false) | 8.7 MB | CPU | — (not loaded) | already opt-in |
| Whistle (STT fallback) | lazy on engine failure, `gaja/stt/select.py:150-156` | — | CPU | not measured | already lazy |
| supertonic TTS | lazy on hello selection (`server.py:152-154`, `gaja/tts/supertonic_tts.py:31`) | ~1 GB per `config.toml:95` | CPU | not measured (RAM budget) | already lazy |
| matcha-pl (1.1 GB in `~/.cache/gaja-models/`) | never — no code reference | 1.1 GB | — | — | dead cache, not RAM |
| sounds (`/play`) | per request, `gaja/api/devices.py:69-104`, not cached | ≤20 MB/300 s each | — | — | no resident cost |
| Python + framework floor | `import gaja.server` | — | — | **125 MB** net (`import-gaja-server.txt`: 142136 kB; pulls scipy via `gaja/devices/audio.py:13`, `gaja/tts/prosody.py:6`) | — |

`import faster_whisper` also maps PyAV (av.libs ≈ 5 MB; verified: only `import faster_whisper` puts
`av` in `sys.modules` of the candidate imports tested).

Sanity: the isolated numbers sum to ~2.9 GB vs 2.29 GB measured in production — the difference is
shared pages (torch/onnxruntime/numpy libs, shared arenas) counted twice per process.

## 3. What shipped on this branch

One implementation commit (plus this doc), everything behind config:

1. **`feat(ram): torch-free silero VAD backend + lazy non-default TTS preload`**
   - `[vad] backend = "onnx"` (new default, `gaja/audio/vad.py:296-347`): same `silero_vad.onnx`,
     onnxruntime + numpy state, **no torch import** (test asserts `"torch" not in sys.modules`),
     one shared InferenceSession instead of one per session. Outputs bit-identical to the torch
     wrapper: MAXDIFF 0.000e+00 (`silero-onnx.txt`) and `GAJA_RAM_BENCH=1 pytest -k matches_torch`
     passes. VAD cost: 67 MB net vs 465-521 MB.
   - `[tts] preload_languages = ["pl"]` (repo `config.toml`, `gaja/config.py` + `server.py`
     `_preload_engines`): paradee loads on the first en use instead of startup.
   - Combined effect once deployed: **≈ −712 MB** (779 MB net torch-VAD+paradee stack → 67 MB net
     numpy VAD, `stack-tts-vad.txt` 795496 kB vs `silero-np-only.txt` 84116 kB), at the cost of the
     first en turn/hello paying ~12.3 s once per boot (Paradee init, `stack-tts-vad.txt`). pl path
     unchanged.

## 4. Options table

| Option | RAM saving | Latency / accuracy cost | Effort | Risk | Status |
|---|---|---|---|---|---|
| VAD onnx backend + lazy paradee (above) | **−712 MB** measured (isolated) | 0 ms on pl turns; first en turn +12.3 s once/boot | done | low (rollback: `backend="torch"`, `preload_languages=null`) | **shipped** |
| `wake_engine = "parakeet"` (== `engine`): the wake path already shares the turn recognizer (`server.py:143-146` skips a second STT when equal; `pipeline.py:913`, `devices/controller.py:451` fall back to `stt`) | **−501 MB** RSS (`fw-cuda.txt`) + ~380 MB VRAM (TODO.md:10) + drops PyAV | wake verify p50 480 ms (`parakeet_eval.json` `verify_p50_ms`) vs fw 406/424 ms (STT_RESULTS.md:443) → **+60-75 ms** per wake, off the reply path; wake accepts 144/194 (score 3.0) or 169/194 (score 4.0) vs 179/194 fw; hard false accepts 4-7/114 vs 20/114 (STT_RESULTS.md:470-473); score 4.0 costs ~1 pp PL WER (STT_RESULTS.md:460) | config line | medium: 5-18 pp more repeated wake phrases | **not switched** (lead's UX call) |
| `MALLOC_ARENA_MAX=2` (+ `MALLOC_TRIM_THRESHOLD_=67108864`) in the unit | est. **−300-600 MB** of the ~880 MB arenas (estimate; unmeasurable read-only) | malloc lock contention with 38 threads — watch p90 STT | unit drop-in | medium | **proposed diff only** |
| Parakeet int8 → further quant | 0 (already int8; no smaller build on disk, encoder.int8.onnx = 652 MB) | — | — | — | no action |
| onnxruntime arena off (piper `_capped_session`) | −6 MB measured (`piper-arena-off.txt` 322268 vs `piper.txt` 327704 kB) | none seen | small | — | rejected (too small) |
| Unload piper/parakeet on idle | up to −1.1 GB while idle | reload 6.1 s (piper, `piper.txt`) on the first reply — violates latency-first | medium | wake-while-loading | rejected |
| Share models across sessions | VAD sessions now shared (this branch); STT/TTS/EnginePool already process-wide | — | done | — | shipped |
| Trim imports (scipy 8 MB, av 5 MB leaves with fw, pyarrow not imported) | ~10-15 MB | none | small | low | not done (noise) |
| jemalloc | unmeasured; same ballpark as arena cap, adds a dependency | — | medium | new .so in prod venv | not done |

Projected steady-state adapter RSS with the shipped commits **plus** `wake_engine = "parakeet"`:
2294 − 712 − 501 ≈ **1.08 GB** (−53%). With the arena cap on top, plausibly 0.5–0.8 GB (estimate,
needs a production restart to verify — do it in a maintenance window and compare
`systemctl --user show gaja-adapter -p MemoryCurrent` after a day).

## 5. Proposed systemd unit diff (NOT applied — production files are off-limits here)

```diff
# ~/.config/systemd/user/gaja-adapter.service
 [Service]
 Environment=HF_HUB_OFFLINE=1
+Environment=MALLOC_ARENA_MAX=2
+Environment=MALLOC_TRIM_THRESHOLD_=67108864
 MemorySwapMax=0
```

Apply with `systemctl --user daemon-reload && systemctl --user restart gaja-adapter` in a window
where the device is idle; keep `journalctl --user -u gaja-adapter` and the admin diagnostics
latencies under watch for a day (p90 `stt_ms` is the canary for arena lock contention).

## 6. Answers to the two open questions

- **TODO.md "Parakeet STT follow-ups" item 3** (both engines resident, ~1.1 GB more adapter RAM):
  confirmed — Parakeet 1088 MB + faster-whisper CUDA 501 MB host. Would Parakeet hold the wake
  quality? Not fully: 144/194 accepted wakes at today's hotword score 3.0 (74%) vs faster-whisper
  179/194 (92%); 169/194 (87%) at score 4.0, which costs ~1 pp PL WER on turns. False accepts are
  strictly better (4-7 vs 20 of 114 hard negatives). A rejected wake only means the user repeats
  the phrase (device already beeped). Whether 5-18 pp more repeats are acceptable is a UX decision.
- **Why torch is resident**: `load_silero_prob` (VAD) imports it directly, and Paradee's G2P chain
  (misaki → spacy → thinc/compat.py:35 `try: import torch`) imports it even with the ONNX VAD.
  Removing both (this branch) takes torch out of a pl-only day entirely.

# Setup (reproducible)

Everything below runs from the project root the repository root directory. Nothing here touches Hermes or `~/voice-agent`.

## 1. Python environment

Prerequisites: uv, Python 3.11 (uv can install it), NVIDIA driver with CUDA 12.x runtime, network access to GitHub and download.pytorch.org.

```bash
export UV_HTTP_TIMEOUT=300
uv sync
```

Key points in `pyproject.toml`:
- `torch==2.6.0` and `torchaudio==2.6.0` are direct dependencies, so `[tool.uv.sources]` routes them to the `pytorch-cu126` index. Without the direct pins uv resolves CPU/PyPI torch.
- The 13 `nvidia-*-cu12` wheels are pinned explicitly. `uv.lock` does not record the cu126 wheel's CUDA runtime deps, and `import torch` fails with `libcudnn.so.9` missing without them.
- `setuptools<81`: resemble-perth imports `pkg_resources`.
- `en-core-web-sm` 3.8.0 is declared as a wheel URL. misaki fetches it at runtime otherwise, and `uv sync` removes ad-hoc installs.
- `paradee` is installed from `git+https://github.com/sahilmahendrakar/paradee`.

Check:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.6.0+cu126 True NVIDIA GeForce GTX 1050 Ti`.

## 2. Download models

Large downloads: run detached with retries. The network was about 650 KB/s.

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_XET=1   # only if xet transfers stall
nohup bash -c 'for i in 1 2 3 4 5; do uv run python scripts/fetch_models.py && break; sleep 30; done; echo FETCH_EXIT=$?' > bench/logs/fetch_models.log 2>&1 &
```

`scripts/fetch_models.py` downloads, in order: Paradee (int8 ONNX only), Cactus-Compute/whistle, LiquidAI/LFM2.5-2.6B-GGUF (Q4_K_M only), ResembleAI/chatterbox (multilingual, narrowed file list), ResembleAI/chatterbox-turbo (all files). Cache: `~/.cache/huggingface/hub`.

## 3. llama.cpp (LLM server)

Source: `third_party/llama.cpp` at commit `dcd387a` (0.5.0-dev, ggml 0.25.3). Repo: https://github.com/ggml-org/llama.cpp

```bash
cd third_party/llama.cpp
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCUDAToolkit_ROOT=/opt/cuda-12.9 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda-12.9/bin/nvcc \
  -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j3
cd -
```

Notes:
- `CMAKE_CUDA_ARCHITECTURES=61` targets the GTX 1050 Ti (Pascal, sm_61).
- `-j3` keeps RAM use under control. A previous build was killed mid-compile at higher parallelism.
- Output: `third_party/llama.cpp/build/bin/llama-server`. Build log of the successful run: `third_party/build_llama_server.log` (ends with `BUILD_EXIT=0`).
- Shared libs (`libggml-cuda.so`, `libllama.so`, ...) are in `build/bin`. Keep `build/bin` next to the binary.
- The TTS CLI uses the same build tree. Build it with `cmake --build build --target llama-tts -j3` (log: `third_party/build_llama_tts.log`, ends with `BUILD_EXIT=0`). Output: `build/bin/llama-tts`.

Run (path to the GGUF is in the HF cache snapshot; the path is also recorded in `bench/logs/lfm_gguf_path.txt`):

```bash
GGUF=$(ls ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B-GGUF/snapshots/*/LFM2.5-2.6B-Q4_K_M.gguf | head -1)
third_party/llama.cpp/build/bin/llama-server \
  -m "$GGUF" --host 127.0.0.1 --port 8791 \
  -ngl 99 -c 2048 -np 1 --jinja --reasoning-budget 0
```

- `--reasoning-budget 0` is required. The LFM2.5 template always opens `<think>`; without this flag the reply is ~287 tokens of English reasoning before the answer.
- `-ngl 99` puts the whole model on the GPU (about 1.84 GB VRAM measured). `-ngl 12` uses about 0.96 GB and halves generation speed (34 → 17 tok/s). See `docs/TTS_RESULTS.md`.
- Stop it with `pkill -f '[l]lama-server -m'`. The bracket keeps pkill from matching the shell that runs the command.

Smoke test:

```bash
curl -s http://127.0.0.1:8791/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"Odpowiadaj po polsku, krótko."},{"role":"user","content":"Jaka jest dziś pogoda w Krakowie?"}],"max_tokens":200}'
```

Recorded outputs: `bench/out/llm_smoke_pl.json` (thinking on, 287 tokens) and `bench/out/llm_smoke_pl_nothink.json` (final Polish reply).

## 4. Test inputs

```bash
uv run python scripts/make_pl_test_audio.py
```

Uses the Piper voice `models/pl_PL-gosia-medium.onnx` (the adapter default) and writes `bench/out/input_pl_16k.wav` (16 kHz). Override with `--text`, `--voice`, `--out`. Requires `piper-tts`, which is in the project env.

The handoff test clip used by the end-to-end suite is generated the same way:

```bash
.venv/bin/python scripts/make_pl_test_audio.py --out bench/out/e2e/handoff_pl_16k.wav \
  --text "Zapytaj Hermesa, jaki jest dziś dzień tygodnia."
```

## 5. STT smoke test

```bash
uv run python scripts/smoke_stt.py bench/out/input_pl_16k.wav
```

Prints JSON with `load_s`, `transcribe_s`, `peak_rss_mb` and the Whistle result.

## 6. TTS smoke tests

```bash
uv run python scripts/smoke_tts.py paradee
uv run python scripts/smoke_tts.py chatterbox-turbo
uv run python scripts/smoke_tts.py chatterbox-mtl
```

Each run writes `bench/out/smoke_<model><suffix>_{en,pl}.wav` and prints one JSON line per language with `audio_s`, `wall_s`, `rtf`, `peak_rss_mb`, `vram_reserved_mb`, `load_s`.

Options (env vars):
- `TTS_DEVICE=cpu`: run on CPU. Adds `_cpu` to output names.
- `TTS_HALF=1`: chatterbox-mtl only. Loads on CPU, converts t3/s3gen/ve to fp16, moves to CUDA, runs under `torch.autocast`. Needed because fp32 on the 4 GB GPU hits OOM at load. Adds `_fp16` to output names.
- `TTS_CFG=0`: sets `cfg_weight` for chatterbox-mtl. No Polish improvement measured.

Full-precision GPU load of chatterbox-mtl fails with CUDA OOM. Use `TTS_DEVICE=cpu` or `TTS_HALF=1`.

## 7. Logs and outputs

- `bench/logs/`: download, sync and smoke logs.
- `bench/out/`: test audio and LLM outputs.
- `third_party/build_llama_server.log`: llama-server build log.
- Results and verdicts: `docs/MODELS.md`. TTS benchmark results: `docs/TTS_RESULTS.md`.

## 8. TTS benchmark (phase 2)

### Dependencies

Added with `uv add piper-tts jiwer num2words` (in `pyproject.toml`: `piper-tts>=1.8.0`, `jiwer>=4.0.0`, `num2words>=0.5.14`). Torch is unchanged (2.6.0+cu126). Log: `bench/logs/uv_add_tts_deps.log`.

### Qwen3-TTS-Polish (GGUF) download

Repo `mradermacher/Qwen3-TTS-Polish-GGUF`. Files used:
- `Qwen3-TTS-Polish.Q4_K_S.gguf` (988,747,104 bytes)
- `Qwen3-TTS-Polish.mmproj-Q8_0.gguf` (446,422,976 bytes)

```bash
.venv/bin/hf download mradermacher/Qwen3-TTS-Polish-GGUF \
  Qwen3-TTS-Polish.Q4_K_S.gguf Qwen3-TTS-Polish.mmproj-Q8_0.gguf README.md
```

Files land in `~/.cache/huggingface/hub/models--mradermacher--Qwen3-TTS-Polish-GGUF/snapshots/<rev>/`. Run detached (the link was about 650 KB/s). A partial from an earlier attempt, when the upstream revision changed, did not resume and was discarded. Check the file sizes above before using the GGUFs. Log: `bench/logs/dl_qwen_polish.log`.

### llama-tts flags that matter here

- `--no-mmproj-offload` keeps the mmproj on CPU. This is the fix for the GPU OOM at default batch size. A smaller context alone (`-c 2048`) did not fix it. The bench still uses `-c 2048`.
- The mmproj on GPU (faster vocoding, about 3x the RTF of the CPU vocoder) needs a smaller batch. `-b/-ub 128` works without the speaker file. With `--tts-speaker-file` it OOMs and aborts, so use `-b/-ub 32`.
- `--tts-lang ru` selects the closest available codec token for Polish. Polish has no token in this GGUF.

### Running the bench

```bash
# Piper speaker reference (CPU, once)
.venv/bin/python scripts/make_speaker_ref.py

# One engine at a time. Do not run two heavy benchmarks together.
.venv/bin/python bench/tts_bench.py --engine piper --langs pl
.venv/bin/python bench/tts_bench.py --engine paradee --threads 1 --langs pl,en
.venv/bin/python bench/tts_bench.py --engine chatterbox-turbo --langs en --cb-dtype fp32
.venv/bin/python bench/tts_bench.py --engine chatterbox-turbo --langs en --cb-dtype fp16 --tag fp16
.venv/bin/python bench/tts_bench.py --engine qwen-polish --langs pl --qwen-lang-pl ru \
  --qwen-mmproj-cpu --qwen-model "$QWEN_GGUF" --qwen-mmproj "$QWEN_MMPROJ"
.venv/bin/python bench/tts_bench.py --engine qwen-polish --langs pl --qwen-lang-pl ru --qwen-batch 32 \
  --qwen-speaker bench/out/tts/speaker_ref_piper.wav --qwen-model "$QWEN_GGUF" --qwen-mmproj "$QWEN_MMPROJ"
```

`$QWEN_GGUF` and `$QWEN_MMPROJ` are the two snapshot paths above.

Per-item OOMs and other errors are recorded as error rows and excluded from the summary. Result files are written to `bench/results/tts_<engine>[_<tag>].json`, audio to `bench/out/tts/<engine>[_<tag>]/`.

### Known limits

- Chatterbox fp32 runs out of memory on sentences of about 9 s audio or longer. Chunk the text per sentence.
- Chatterbox fp16 is about 4x slower than fp32 on this GPU and its speech is largely unintelligible.
- Qwen3-TTS reloads the model per call, so TTFA equals full synthesis time and each call pays about 3.5 s of load.
- Co-residency with LFM at `-ngl 99` is not possible for any tested TTS config (fails at load). Use `-ngl 12` for LFM, or run TTS on CPU.

## 9. STT benchmark (phase 3a)

### Dependencies

Added with `uv add pyarrow faster-whisper silero-vad` (in `pyproject.toml`: `pyarrow>=25.0.1`, `faster-whisper>=1.2.1`, `silero-vad>=6.2.3`). Torch is unchanged (2.6.0+cu126). Log: `bench/logs/uv_add_stt_deps.log`. Pre-change copy: `bench/logs/pyproject.before_stt.toml`.

### Native needle CLI (Whistle decoder knobs)

The `cactus-needle` Python API exposes neither `--audio-depth` nor `--threads`. The native CLI does. Copy it from the HF snapshot of `Cactus-Compute/needle3` (snapshot `2ae11323dc000f5e70c49f7403efa6af12ba9e67`):

```bash
mkdir -p bench/tools
S=~/.cache/huggingface/hub/models--Cactus-Compute--needle3/snapshots/2ae11323dc000f5e70c49f7403efa6af12ba9e67
cp "$S/linux-x86_64/needle" "$S/linux-x86_64/needle.h" bench/tools/
```

Fetch it with `hf download Cactus-Compute/needle3 --include 'linux-x86_64/*'` if the snapshot is missing. Only `bench/tools/needle` is used by the bench.

### Test data (FLEURS, streamed)

```bash
.venv/bin/python bench/stt_prepare_data.py            # pl 44 WER utts, en 36 WER utts, latency clips
.venv/bin/python bench/stt_prepare_data.py --latency-only   # rebuild latency clips from the WER wavs
```

Reads only the first 150 rows of `google/fleurs` `refs/convert/parquet` `pl_pl/test/0000.parquet` and `en_us/test/0000.parquet` over HTTP range reads. Output: `bench/data/stt/{pl,en}/` with `wer/`, `refs.tsv`, `latency/`, `latency.tsv`. The full parquet files (462 MB and 402 MB) are never downloaded.

### Benchmarks

```bash
# Full CPU matrix, sequential (about 1-1.5 h). Log: bench/logs/stt_matrix.log
nohup bash bench/run_stt_matrix.sh > bench/logs/stt_matrix.log 2>&1 &

# One engine config
.venv/bin/python bench/stt_bench.py --engine whistle --lang pl --sets wer,latency,hallu --tag whistle_py_pl_forced
.venv/bin/python bench/stt_bench.py --engine whistle-cli --lang pl --sets wer --depth 4 --threads 1 --tag example
.venv/bin/python bench/stt_bench.py --engine faster-whisper --model small --lang pl --tag fw_small_cpu_pl

# Streaming and VAD (run after the matrix, not concurrently)
.venv/bin/python bench/stt_stream_test.py --mode stream --chunks 0.1 0.2 0.5 1.0
.venv/bin/python bench/stt_stream_test.py --mode vad
```

Results: `bench/results/stt_<tag>.json`, `stt_stream_whistle.json`, `stt_vad_silero.json`. Metric definitions are in `docs/STT_RESULTS.md`.

## 10. Adapter: voices, Hermes, deploy

### Voice models

| Engine | Language | Files | Notes |
|---|---|---|---|
| `piper` | pl | `models/pl_PL-gosia-medium.onnx(.json)` | default for `pl` |
| `piper-darkman` | pl | `models/pl_PL-darkman-medium.onnx(.json)` | selectable with `hello.tts` |
| `paradee` | en | Paradee int8 ONNX in the HF cache (`scripts/fetch_models.py`) | default for `en` |
| `supertonic` | pl, en | model files must already exist locally; the loader uses `auto_download=False` | not default, loaded on first selection, about 1 GB RAM |

Piper voices are not in git. Copy the gosia voice from the bench tree once:

```bash
mkdir -p models
cp bench/models/piper/pl_PL-gosia-medium.onnx bench/models/piper/pl_PL-gosia-medium.onnx.json models/
```

### Hermes ACP worker

`[hermes] backend = "acp"` starts `hermes acp` as a persistent subprocess (JSON-RPC 2.0 over stdio) and
keeps it for the life of the adapter. `command` in `config.toml` points at the pipx-installed binary:

```toml
[hermes]
backend = "acp"
command = ["hermes", "acp"]  # or custom absolute path in config.local.toml
cwd = "~"
timeout_s = 300.0
```

- The adapter warms the worker at startup (`health()`, which spawns `hermes acp` and runs `initialize`).
  A cold first task took about 15 s in the live check. Warm-task latency is in `docs/HERMES.md` and the e2e handoff result.
- Each Gaja WebSocket session gets its own Hermes session (`session/new` on first handoff).
- If the worker is unavailable, the handoff emits `handoff{state:"error"}` and the adapter speaks
  `Hermes jest teraz niedostępny, nie udało się wykonać zadania.` The voice loop keeps working.
- `backend = "stub"` gives a fixed answer and is for tests only.

Check the real worker end to end with one read-only task:

```bash
flock /tmp/gaja-bench.lock env PYTHONPATH=. .venv/bin/python scripts/hermes_smoke.py
```

### Deploy (systemd user units)

See `deploy/README.md`. The units are linked, enabled and started with:

```bash
systemctl --user link  "$PWD/deploy/gaja-llm.service"
systemctl --user link  "$PWD/deploy/gaja-adapter.service"
systemctl --user daemon-reload
systemctl --user enable --now gaja-llm gaja-adapter
curl -s http://127.0.0.1:8790/health
```

Boot start needs lingering. Check it with `loginctl show-user $USER -p Linger`. Linger is `yes` on this machine.
The adapter binds to `127.0.0.1`. Remote browsers reach it through the existing Tailscale HTTPS proxy on :8443, whose
origin (e.g. `https://<host>.<tailnet>.ts.net:8443`) is listed in `[server] allowed_origins`; WS handshakes from any
other browser origin get HTTP 403. Agents must not run or change `tailscale serve` themselves.

### Checks

```bash
.venv/bin/python -m pytest tests                        # unit and protocol tests; live tests skip if the adapter is down
bench/e2e.sh                                            # real stack, all scenarios (one command)
bench/e2e.sh --pl-clips 3 --en-clips 2 --scenarios pl,text   # a subset; extra args go to scripts/e2e_suite.py
```

`bench/e2e.sh` runs `scripts/e2e_suite.py` under `flock /tmp/gaja-bench.lock`, records the load average before and
after, and writes `bench/out/e2e/run.log` and `bench/out/e2e/results.json`. The lock keeps timed runs from overlapping
with other benchmarks on this GPU and CPU.

The suite runs these scenarios over `WS /ws` (`--scenarios` takes `all` or a comma list of
`pl,en,text,barge,barge_think,handoff,routing`):
- `pl`: Polish audio turns from `bench/data/stt/pl/latency` (`--pl-clips`, default 10).
- `en`: English audio turns from `bench/data/stt/en/latency` (`--en-clips`, default 4).
- `text`: one typed Polish and one typed English chit-chat turn, answered locally.
- `barge`, `barge_think`: barge-in during `speaking` and during `thinking`.
- `handoff`: one real Hermes handoff with a harmless read-only prompt.
- `routing`: 4 read-only Hermes prompts that must hand off and 4 local prompts that must not, including the trap
  "Kim jest Hermes w mitologii?". It reports precision and recall.

Every timed turn is appended to `logs/turns.jsonl`. The summary reports p50 and p90 of `stt_ms`, `router_ms`,
`llm_ttft_ms`, `first_sentence_ms`, `tts_first_ms` and `e2e_first_audio_ms`. Per-turn metrics for audio turns also
record `stt_wait_ms` (time waiting for the STT engine / queue wait), `stt_majflt` (major page faults of the adapter
process during transcription) and `stt_audio_s` (audio duration transcribed). Turns with `stt_ms > 1500` log an INFO
diagnostic line.

### STT engine and VRAM budget

`stt.engine` in `config.toml` selects the transcriber:
- `faster-whisper` (production): `small`, `int8`, `device = "cuda"`, `beam_size = 1`, language forced from `hello.lang`.
  Speech is segmented by the Silero endpointer before it reaches STT. The model is built and warmed at startup.
- `whistle`: cactus-needle on CPU. It is selectable, and it is also the automatic fallback.

Fallback rule: if faster-whisper fails to build or warm up (for example CUDA out of memory), or a decode fails, the
adapter logs the error, serves from Whistle, and reports the stage and error in `/health` as `stt_fallback`. The failed
utterance is retried on Whistle. The adapter does not crash and does not touch the Hermes process.

Check which engine serves:

```bash
curl -s http://127.0.0.1:8790/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print({k: h[k] for k in ("stt_engine","stt_requested","stt_device","stt_fallback")})'
```

VRAM budget on the GTX 1050 Ti (4096 MiB). Measured with `nvidia-smi`:

| Process | VRAM |
|---|---|
| Hermes (`hermes gateway run`, shared with other tools) | ~986 MiB |
| llama-server (LFM2.5-2.6B Q4_K_M, `-ngl 99`, `-c 8192 -np 2 --no-kv-unified`) | 1856 MiB |
| faster-whisper `small` int8 (adapter process, including CUDA context) | 378 MiB |
| **Total** (2026-10-09) | 3225 MiB idle; peak 3353 MiB over 4 audio turns (200 ms sampling) |

Earlier STT bench peak: 3825 MiB total (cap 3900 MiB).

Keep total use at or below about 3900 MiB so that the GPU does not run out under load. Check with
`nvidia-smi --query-gpu=memory.used,memory.total --format=csv` after several turns. If faster-whisper hits an OOM,
`stt_fallback` shows it, and the adapter keeps running on Whistle. Note that the whole GPU is shared, so the numbers
move with other GPU users.

### Router, chat template and llama slots

- `deploy/gaja-llm.service` runs llama-server with `--chat-template-file deploy/lfm_nothink.jinja` and
  `-c 8192 -np 2 --no-kv-unified` (two slots of 4096 tokens; a unified cache could not restore an evicted slot).
  Native tool calls still parse with this template.
- Slot 0 is the chat (`llm.chat_slot`), slot 1 the router (`llm.router_slot`); every request pins `id_slot`.
- `deploy/gaja-adapter.service` sets `HF_HUB_OFFLINE=1` (model loads are local only) and waits up to 60 s for
  llama-server `/health` before starting.
- After editing either unit: `systemctl --user daemon-reload`, then `systemctl --user restart gaja-llm gaja-adapter`.
- Router v3 (`prompts/router3_pl.md`): one 1-token logprob call, classes LOCAL / TOOL / HERMES, `threshold = 0.5`
  (HERMES), `tool_threshold = 0.6` (TOOL), chosen from `bench/router3_eval.py` (`bench/results/router3_eval_r6.json`).
  Re-run it with the adapter's llama-server up: `flock /tmp/gaja-bench.lock .venv/bin/python bench/router3_eval.py`.
- Prefix caching: the hybrid LFM2.5 model reuses the KV prefix only for an exact repeat or a primed checkpoint. The
  adapter primes (`n_predict 0`, `cache_prompt`) the router slot with system prompt + the last 2 messages after each
  turn and on `speech_start`, and (with `llm.prime_after_turn`) the chat slot with the whole conversation after each
  turn. Expected per request: router ~30 evaluated tokens (~115 ms), chat 13–14 tokens. Check with `timings.cache_n`
  / `prompt_n` in a response, or the `prompt eval` lines in `journalctl --user -u gaja-llm`.

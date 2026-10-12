<img src="assets/icon.svg" width="64" height="64" alt="">

# Gaja Voice Adapter

Gaja is a Polish-first, low-latency, real-time voice assistant adapter designed for local deployment. It bridges real-time audio input from browser clients (WebSockets) and ESP32 IoT hardware with local speech recognition (STT), a lightweight local language model (LLM), fast local tool execution, expressive speech synthesis (TTS), and asynchronous task delegation to the autonomous Hermes agent.

---

## Features

- **Real-Time Voice Loop with Barge-In**: Sub-second voice streaming with Silero VAD, acoustic echo suppression, and Smart Turn end-of-speech boundary detection. User interruptions immediately cancel playing audio buffers.
- **Speculative Sentence Streaming**: Replies are chunked and dispatched to TTS on the first completed clause, allowing audio playback to begin while the language model is still generating.
- **Fast Local Tools (Variant C Direct Speak)**: Eight local tools (weather, time, calculator, NBP currency rates, Wikipedia summaries, units, server health, and Polish holidays) execute in ~1.5–2.5 s, speaking deterministic Polish replies directly without a redundant second LLM generation turn.
- **Hermes Agent Hand-Off**: Autonomous delegation of complex or long-running tasks (coding, research, system operations) to the Hermes agent via a persistent ACP protocol worker over stdio.
- **IoT Hardware & Devices Support**: Server protocol support for ESP32-S3 devices (e.g. M5Stack Atom/EchoS3R) running on-device wake-word detection over UDP/TCP.
- **Interactive Web Client & PWA**: Zero-framework web application featuring a WebGL fragment shader orb visualising state, push-to-talk, input device switching, low-signal watchdogs, and progressive web app capabilities.
- **On-Device Wake Word & Audio Recorder**: Support for "Hej Gaja" via microWakeWord, plus a built-in browser recorder (`/record.html`) for collecting training and evaluation samples.

---

## Architecture Overview

Detailed architectural contracts, message formats, and protocols are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
   [ Browser / PWA / ESP32 Device ]
                 │
           (PCM16 Audio over WebSocket / TCP)
                 ▼
      ┌─────────────────────┐
      │  Gaja Voice Adapter │  (FastAPI / asyncio)
      └──────────┬──────────┘
                 ├──────────────────────────────┐
                 ▼                              ▼
        [ Silero VAD / STT ]           [ Smart Turn EOS ]
        (Whistle / faster-whisper)
                 │
                 ▼
      [ Router v3 Logprob Classifier ]
        (1-token /completion on llama-server)
          │            │               │
      [ LOCAL ]    [ TOOL ]       [ HERMES ]
          │            │               │
     LFM2.5-2.6B    Variant C      Hermes ACP
     llama-server   Direct Speak   Background Task
          │            │               │
          └─────┬──────┘               ▼
                ▼              Progress & Results
       [ Text Normalizer ]
                │
                ▼
        [ Piper TTS / CPU ]
         (Gosia / Darkman)
                │
           (WAV Audio)
                ▼
      [ Client Playback ]
```

---

## Hardware Baseline & Latency Measurements

Gaja was built and benchmarked on commodity consumer hardware:
- **CPU**: Intel Core i5-8300H (4 cores / 8 threads @ 2.30 GHz)
- **RAM**: 15 GB DDR4
- **GPU**: NVIDIA GeForce GTX 1050 Ti (4 GB VRAM, Pascal sm_61)

Measured headline metrics from repository benchmarks:

- **Speech-to-Text (STT)** ([docs/STT_RESULTS.md](docs/STT_RESULTS.md)):
  - *WhistlePy (in-process Whisper on CPU)*: Median turn latency **266–585 ms** (RSS ~101 MB), maintaining real-time responsiveness without GPU contention.
- **Language Model & Routing** ([docs/LLM_RESULTS.md](docs/LLM_RESULTS.md)):
  - *LiquidAI LFM2.5-2.6B (Q4_K_M on llama-server)*: Alone generation speed **31.3–34.0 tokens/s** (29.4–31.9 ms/token) with prefill at **193.0–531.7 tokens/s**.
  - *Time to First Token (TTFT)*: **~200–250 ms** with warm prefix prompt cache.
  - *Router v2 / v3 Classifier*: Median classification latency **53.89 ms** (~54 ms, 1 token `/completion`), achieving **0.950 Precision**, **0.950 Recall**, and **0.950 F1** on an 80-case holdout dataset.
- **Text-to-Speech (TTS)** ([docs/TTS_RESULTS.md](docs/TTS_RESULTS.md)):
  - *Piper `pl_PL-gosia-medium` (Polish default on CPU)*: Real-Time Factor (RTF) **15.74**, Time to First Audio (TTFA) median **0.27 s** (< 0.3 s target), streaming sentence-by-sentence.
  - *Paradee-8M (English default on CPU)*: RTF **9.65**, TTFA median **0.43 s** (0.32 s for short sentences).
- **Fast Local Tools** ([docs/TOOLS.md](docs/TOOLS.md)):
  - *Variant C (Direct Speak)*: Median voice turn latency **2150 ms** (saving ~2.7 s compared to two-turn LLM generation), **99.1% accuracy**, and **0.0% hallucinations**.

---

## Quick Start

See [docs/SETUP.md](docs/SETUP.md) for full installation and system requirements.

### 1. Requirements

- Linux (Ubuntu 22.04+ recommended)
- Python 3.11+
- `uv` package manager
- `llama-server` (llama.cpp) with GPU support

### 2. Environment Setup

```bash
# Clone repository and synchronize dependencies
git clone <repo-url> gaja-hermes-adapter
cd gaja-hermes-adapter
uv sync
```

### 3. Run the Services

Start the local LLM server and the Gaja adapter:

```bash
# In terminal 1: Start llama-server (LFM2.5-2.6B)
llama-server -m models/lfm2.5-2.6b-q4_k_m.gguf --host 127.0.0.1 --port 8791 -ngl 99 -c 8192 -np 2 --chat-template-file deploy/lfm_nothink.jinja --reasoning-budget 0

# In terminal 2: Start Gaja adapter
uv run python -m gaja
```

Systemd user service templates and an automated installation script are provided in `deploy/` and `scripts/install_units.sh`.

### 4. Open Web Interface or Test via CLI

- **Web Browser**: Open `http://localhost:8790` in Chrome, Safari, or Firefox.
- **CLI Client**:
  ```bash
  # Send text query
  uv run python clients/cli_wav.py --text "Która jest godzina?"

  # Stream audio file
  uv run python clients/cli_wav.py bench/data/stt/01_pogoda_krakow.wav
  ```

---

## Configuration

Configuration is managed via Pydantic Settings in [gaja/config.py](gaja/config.py), evaluated in the following order of precedence (highest to lowest):

1. **Explicit keyword arguments** passed to `Settings(...)`.
2. **Environment variables** with the prefix `GAJA_` and double underscores (`__`) for nested keys:
   ```bash
   export GAJA_SERVER__PORT=8790
   export GAJA_LLM__TEMPERATURE=0.2
   ```
3. **Local overrides** in `config.local.toml` (git-ignored, optional):
   Used to configure machine-specific settings such as custom Tailscale origins or explicit Hermes binary paths.
4. **Project defaults** in `config.toml`.

### Example `config.local.toml`

```toml
[server]
# Add custom domain or Tailscale HTTPS origin for remote microphone access
allowed_origins = [
  "http://localhost:8790",
  "https://localhost:8790",
  "http://127.0.0.1:8790",
  "https://127.0.0.1:8790",
  "https://my-host.my-tailnet.ts.net:8443",
]

[hermes]
# Specify an absolute path if hermes is not in system PATH
command = ["/usr/local/bin/hermes", "acp"]
```

---

## Repository Layout

```
.
├── assets/             # Icons and visual assets
├── bench/              # Benchmark evaluation scripts, datasets, and results
│   ├── data/           # Holdout evaluation sets (routing, tools, quality, audio)
│   └── results/        # Raw JSON benchmark records
├── clients/            # CLI test clients, real adapter checks, and mock server
├── deploy/             # Systemd user service templates and jinja chat templates
├── docs/               # Architecture, protocols, hardware notes, and benchmarks
│   ├── ADAPTER.md      # Core pipeline contract and lifecycle
│   ├── ARCHITECTURE.md # System architecture, interfaces, and cross-stream contracts
│   ├── DEVICE.md       # IoT hardware protocol specification
│   ├── HERMES.md       # Hermes ACP integration analysis
│   ├── LLM_RESULTS.md  # Routing and LLM benchmarks
│   ├── STT_RESULTS.md  # Speech-to-text benchmark results
│   ├── TOOLS.md        # Local tools evaluation and specifications
│   ├── TTS_RESULTS.md  # Speech synthesis benchmarks
│   └── WEB_CLIENT.md   # Web client and PWA documentation
├── gaja/               # Core application package
│   ├── audio/          # VAD, audio chunking, and Smart Turn detection
│   ├── hermes/         # Hermes ACP client and background task lifecycle
│   ├── llm/            # llama-server client and Router v3 classification
│   ├── text/           # Text normalization and phonetic preprocessing
│   ├── tools/          # Async local tool implementations
│   ├── config.py       # Pydantic configuration hierarchy
│   ├── pipeline.py     # Real-time turn orchestrator
│   └── server.py       # FastAPI application and WebSocket endpoints
├── prompts/            # Production system and router prompts
├── scripts/            # Unit install scripts and testing utilities
├── tests/              # Pytest test suite
└── web/                # Zero-build single-page web client and PWA
```

---

## Development

### Running Tests

```bash
# Run unit and integration tests (fast)
uv run pytest -q --ignore=tests/test_web_e2e.py

# Run Playwright E2E browser tests (requires browser dependencies)
flock /tmp/gaja-e2e.lock uv run pytest -q tests/test_web_e2e.py
```

### Running Benchmarks

Benchmark scripts are located in `bench/`:
- **STT evaluation**: `bash bench/run_stt_matrix.sh`
- **TTS synthesis benchmarks**: `bash bench/tts_final.sh all`
- **Routing evaluation**: `uv run python bench/routing_eval.py`
- **Tool evaluation**: `uv run python bench/tools_eval.py`

All timed benchmark executions should be wrapped under `flock /tmp/gaja-bench.lock`.

---

## Licence

Distributed under the MIT Licence. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Tymoteusz Bielski.

---

## Po polsku

Gaja to adapter asystentki głosowej zoptymalizowany pod kątem języka polskiego, działający w czasie rzeczywistym z minimalnym opóźnieniem na sprzęcie konsumenckim.

### Główne cechy

- **Pętla głosowa z przerwaniem wypowiedzi (barge-in)**: Natychmiastowe zatrzymanie odtwarzania głosu po wykryciu mowy użytkownika (Silero VAD + echo guard + Smart Turn).
- **Strumieniowanie spekulatywne**: Rozpoczęcie syntezy mowy (TTS) po wygenerowaniu pierwszego zdania przez model LLM.
- **Szybkie narzędzia lokalne (Wariant C Direct Speak)**: Pogoda, czas, kalkulator, kursy walut NBP, Wikipedia, status serwera i święta w ~1.5–2.5 s bez zbędnej drugiej tury LLM.
- **Współpraca z agentem Hermes**: Automatyczne przekazywanie złożonych zadań (kodowanie, research, system) do agenta Hermes przez protokół ACP.
- **Klient webowy i PWA**: Interfejs w przeglądarce (`http://localhost:8790`) z kulą WebGL reagującą na stan asystenta.
- **Własne słowo wybudzające ("Hej Gaja")**: Narzędzie do nagrywania próbek pod adresem `/record.html`.

### Szybki start

```bash
# Instalacja zależności
uv sync

# Uruchomienie adaptera
uv run python -m gaja
```

Więcej informacji w plikach [docs/SETUP.md](docs/SETUP.md) oraz [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

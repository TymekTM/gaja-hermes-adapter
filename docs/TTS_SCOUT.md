# TTS-SCOUT: Polish Text-to-Speech Landscape & Evaluation

**Document version:** 1.0  
**Host environment:** Intel Core i5-8300H (4C/8T, AVX2), 16 GB RAM, NVIDIA GeForce GTX 1050 Ti 4 GB (Pascal sm_61).  
**Hardware constraints:** Usable VRAM <= 1.0 GB (Hermes ~1.0 GB baseline + LFM2.5-2.6B ~1.85 GB), warm RTF > 1.0, TTFA <= ~0.8 s.  
**Baseline reference:** Piper `pl_PL-darkman-medium` (RTF 15.9, TTFA 0.27 s, Whistle-WER 0.200, VRAM 0 MiB, naturalness MOS-proxy ~3.55).

---

## 1. Executive Summary

The goal of the TTS-SCOUT workstream is to identify a Polish TTS engine with **clearly better naturalness** than the baseline pick (`pl_PL-darkman-medium`), while strictly maintaining:
1. **Warm RTF > 1.0** (audio seconds generated / wall synthesis seconds).
2. **Time to First Audio (TTFA) <= ~0.8 s** on typical conversational utterances (8–15 words).
3. **Resource budget:** CPU-friendly or at most **~1.0 GB VRAM** co-resident on the GTX 1050 Ti with the Hermes gateway (~991 MiB) and LFM2.5 (~1.85 GB).
4. **Prosody & streaming suitability:** Support for speed/rate adjustment, sentence-level chunking, or streaming.

After exhaustive screening across open-source repositories, Hugging Face models, and 2024–2026 releases, our survey reveals a stark divide between **diffusion / heavy autoregressive foundation models** (which completely fail the latency or Pascal VRAM budget) and **non-autoregressive VITS / flow matching architectures**.

---

## 2. Comprehensive Model Survey

| Engine / Model | Architecture | Params / Weights | Runtime / Target | License | Polish Support Evidence | Published / Measured Speed | Feasibility Verdict |
|---|---|---|---|---|---|---|---|
| **Piper `pl_PL-gosia-medium`** | VITS (ONNX) | ~28M (63 MB) | CPU (onnxruntime) | MIT | Native official voice | RTF ~15–16, TTFA ~0.26 s | **Shortlist (Top Pick)**. Far smoother and more natural than darkman. |
| **Piper `pl_PL-bass-high`** | VITS (ONNX) | ~40M (114 MB) | CPU (onnxruntime) | MIT | Native official voice | RTF ~10–12, TTFA ~0.35 s | **Shortlist**. High capacity male voice, resonant timbre. |
| **Piper `pl_PL-mc_speech-medium`** | VITS (ONNX) | ~28M (63 MB) | CPU (onnxruntime) | MIT | Native official voice | RTF ~15–16, TTFA ~0.27 s | **Shortlist**. Expressive male voice. |
| **Piper `pl_PL-darkman-medium`** | VITS (ONNX) | ~28M (63 MB) | CPU (onnxruntime) | MIT | Native official voice | RTF 15.9, TTFA 0.27 s | **Baseline**. Fast, but flat and dry cadence. |
| **Piper Community (`justynawgglos`, `jarviswgglos`)** | VITS (ONNX) | ~28M (63 MB) | CPU (onnxruntime) | CC-BY / Open | Community HA fine-tunes | RTF ~15, TTFA ~0.28 s | Viable, but standard gosia has better training dataset hygiene. |
| **MMS-TTS pol (`facebook/mms-tts-pol`)** | VITS (Meta) | ~28M (145 MB) | PyTorch (CUDA / CPU) | CC-BY-NC 4.0 | Native Polish (MMS dataset) | GPU RTF 6.35, TTFA 0.32 s, VRAM 216 MB | **Shortlist**. Fits inside VRAM budget (<220 MB), fast on GPU. Intonation slightly liturgical. |
| **Supertonic 3 (`Supertone/supertonic-3`)** | Flow / Vector Estimator + Vocoder | ~66M (396 MB) | CPU / GPU (ONNX) | OpenRAIL | Native Polish (`pl`, 31 langs) | 8-step CPU: RTF 0.59 (slow); 4-step: RTF 1.11, TTFA 1.9 s | **Evaluated**. 44.1 kHz fidelity, 10 voice styles. Misses <=0.8s TTFA on CPU. |
| **Matcha-TTS-PL-v2 (`machinekind/Matcha-TTS-PL-v2`)** | Flow Matching (distilled from VoxCPM2) | 20.9M (140 MB) | ONNX Runtime | CC-BY-SA 4.0 | Trained on 13.1 h Polish assistant speech | Published Mac CPU RTF ~2–3, UTMOS 3.40 | **Evaluated**. Modern conversational Polish, but `matcha-tts` Python deps clash violently with torch 2.6. |
| **Kyutai Pocket-TTS (`kyutai/pocket-tts`)** | Mimi Codec + LM | ~100M | C++ / PyTorch | MIT / Open | **NO Polish support**. Upstream has only EN, FR, DE, ES, IT, PT, NL. | N/A | **Disqualified**. No Polish weights in official repository. |
| **Pocket-TTS Polish 6L (`shefowl/pocket-tts-polish-6l`)** | Distilled 6-layer Pocket-TTS | 109.5M (438 MB) | PyTorch (`pocket-tts`) | CC-BY 4.0 | Community Polish distillation | Published CPU RTF 4.8 (fp32), 9.0 (int8) | **Disqualified from primary venv**. Requires `pocket-tts` which forces `numpy>=2.0` breaking project torch stack. |
| **Kokoro-82M (`hexgrad/Kokoro-82M`)** | StyleTTS 2 | 82M | PyTorch / ONNX | Apache-2.0 | **NO official Polish**. Official: EN, JA, ZH, ES, FR, HI, IT, PT. | N/A | **Disqualified**. Community eSpeak IPA mapping sounds heavily accented and unnatural. |
| **F5-TTS / E2-TTS Polish Finetunes** | DiT Flow Matching | ~336M–380M | PyTorch (CUDA) | MIT | Community fine-tunes (`Sticzu`, `bszolkowski`) | GPU RTF 0.15–0.25 (RTX 4090); CPU RTF 0.1–0.3 | **Disqualified**. Requires 32 ODE steps; CPU is 5–10x slower than real time; Pascal GPU FP16 is slow, FP32 OOMs in 1 GB. |
| **XTTS-v2 (`coqui/XTTS-v2`)** | Autoregressive GPT + HiFi-GAN | ~460M (1.8 GB) | PyTorch (CUDA) | CPML | Native Polish | CPU RTF 0.2–0.4, TTFA 3–6 s; GPU VRAM >3.5 GB | **Disqualified**. Exceeds VRAM budget (>3 GB) and CPU latency budget by large margin. |
| **Fish-Speech / OpenAudio S1-mini** | Dual AR LLM + VQ-GAN | ~500M+ | PyTorch | Apache-2.0 | Multilingual | Requires modern GPU (Ampere/Ada, BF16, FlashAttn) | **Disqualified**. Too heavy for Pascal sm_61 and 1 GB VRAM. |
| **Zonos (Zyphra)** | Mamba + Transformer | 1.6B | PyTorch | Apache-2.0 | Multilingual | Heavy GPU only (>8 GB VRAM) | **Disqualified**. Hardware far exceeds budget. |
| **Spark-TTS** | Bi-directional Audio LM | ~1.0B | PyTorch | Apache-2.0 | Multilingual | Heavy GPU only (>6 GB VRAM) | **Disqualified**. |
| **CosyVoice 2 / CosyVoice 3** | Flow Matching + LM | 0.5B–1.5B | PyTorch | Apache-2.0 | Chinese, English, partial multilingual | GPU only; CPU RTF <0.2 | **Disqualified**. High VRAM requirement, weak Polish prosody. |
| **MeloTTS (`myshell-ai/MeloTTS`)** | VITS-based | ~150M | PyTorch | MIT | **NO Polish support** (only EN, ES, FR, ZH, JP, KR) | N/A | **Disqualified**. |
| **Qwen3-TTS-Polish (GGUF via llama-tts)** | Multimodal Audio LM | ~1.7B | llama-tts (GPU/CPU) | Apache-2.0 | Tested in Phase 2 | RTF 1.13, TTFA 7.6 s, WER 0.83, VRAM 4.0 GB | **Already ruled out**. Reloads per call, 4 GB VRAM spike, high WER. |
| **Chatterbox-turbo (Resemble AI)** | CFM Diffusion | ~350M | PyTorch (CUDA) | Apache-2.0 | English only | RTF 1.4, TTFA 1.8 s, VRAM 4.0 GB | **Already ruled out**. English only, OOMs on long sentences. |
| **Chatterbox Multilingual** | CFM Diffusion | ~500M | PyTorch | Apache-2.0 | Multilingual | RTF 0.13–0.2 (CPU/GPU) | **Already ruled out**. Extremely slow, unintelligible. |

---

## 3. Top Shortlisted Candidates for Budget & Quality

Taking into account the shared hardware constraints (CPU i5-8300H, GTX 1050 Ti Pascal with <= 1.0 GB free VRAM):

### 1. Piper `pl_PL-gosia-medium` (Recommended Polish Production Voice)
- **Why:** Gosia is a female voice trained on clean Polish speech. While darkman sounds dry, slightly gravelly, and robotic, Gosia provides smooth formant transitions, natural clause boundary cadence, and distinct Polish consonant handling (`ą`, `ę`, `ć`, `ź`, `ż`).
- **Speed & Footprint:** Exactly matches the darkman efficiency profile: RTF > 15, TTFA < 0.28 s, 0 MB VRAM, 63 MB download.
- **Protocol compatibility:** Native streaming generator, controllable rate/noise scales.

### 2. Piper `pl_PL-bass-high` (Alternative High-Capacity Male Voice)
- **Why:** High-variant model (114 MB vs 63 MB), deeper fundamental frequency ($F_0$), richer chest resonance, reduced high-frequency buzz compared to darkman.
- **Speed & Footprint:** RTF ~10–12, TTFA < 0.35 s, 0 MB VRAM.

### 3. Meta MMS-TTS Polish (`facebook/mms-tts-pol`)
- **Why:** Full end-to-end VITS neural architecture from Meta. Running on CUDA takes only **215.7 MB VRAM**, leaving >800 MB safe margin for the Hermes agent and LLM.
- **Speed:** Measured warm RTF of **6.35** on GTX 1050 Ti GPU, TTFA **0.318 s**.
- **Caveat:** 16 kHz audio and biblical training corpus bias.

### 4. Supertonic 3 (`Supertone/supertonic-3`)
- **Why:** Modern (2025/2026) multilingual on-device TTS with 44.1 kHz output, 10 voice styles, and native Polish support.
- **Speed:** At default 8 steps, CPU RTF is ~0.59 (slower than real time). At 4 steps, RTF is 1.11, but TTFA is ~1.9 s (exceeds the 0.8 s budget). Excellent audio fidelity if deployed with a beefier CPU or WebGPU client.

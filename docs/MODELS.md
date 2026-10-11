# Model candidates

Hardware: i5-8300H, 15 GB RAM, GTX 1050 Ti 4 GB (sm_61). About 1 GB of VRAM is held by the Hermes gateway, so about 3 GB is free.

Legend: **[verified]** = checked by running it here. **[card]** = from the model card or repo, not run here.

## Summary

| Model | Role | Polish | Streaming | License | Load | RAM / VRAM | Smoke result | RTF (this box) |
|---|---|---|---|---|---|---|---|---|
| Cactus-Compute/whistle (cactus-needle 3.1.2) | STT | Yes [card], transcribes Polish [verified] | `stream()` method exists [card] | Apache-2.0 [card, verify] | 0.5 s | — | Polish control clip transcribed (near-exact) | n/a |
| LiquidAI/LFM2.5-2.6B (Q4_K_M GGUF) | LLM via llama.cpp | Yes, replies in Polish [verified] | Token streaming via llama-server [card] | "lfm1.0" [card, verify] | see SETUP.md | ~1.75 GB VRAM with `-ngl 99` [verified] | Clean Polish reply with `--reasoning-budget 0` | n/a |
| sahilmahendrakar/Paradee-8M-v1.0 (int8 ONNX) | TTS | **No**, English only [card, verified by output] | No (`tts(text)` full utterance) | Apache-2.0 [card] | fast, CPU [verified] | CPU only | EN intelligible, PL not | 1st call only, see notes |
| ResembleAI/chatterbox-turbo | TTS | **No**, English only [card, verified by output] | No (`generate()` full utterance) | MIT [card] | ~few s after warm-up | RSS 4.8 GB, VRAM 2.95 GB [verified] | EN intelligible, PL not | 0.70 (EN, 1st call), 1.56 (PL, warm, non-Polish text) |
| ResembleAI/chatterbox (multilingual, 500M) — not a candidate, tested as reference | TTS | Listed as 23 languages incl. Polish [card] | No | MIT [card, verify] | 27 s (fp16 GPU) | GPU fp16: VRAM 1.78 GB; CPU fp32: RSS 6.9 GB | EN good; PL garbled, with and without `cfg_weight=0` | 0.13 (CPU fp32), 0.20 (GPU fp16) |

## Notes per model

### STT: Cactus-Compute/whistle
- Package: `cactus-needle==3.1.2`, API `needle.Whistle()`, `transcribe(path)`, `embed`, `stream`.
- Input: 16 kHz audio. Languages listed on card: en, de, fr, es, it, nl, pl.
- Test: `scripts/smoke_stt.py bench/out/input_pl_16k.wav`. Output text: "Dzień dobry, chciałem sprawdzić pogodę najotrowkrakowie, czy będzie padać też po południu?" Language detected: pl. Clip source: Piper pl_PL-darkman-medium (`scripts/make_pl_test_audio.py`).
- Caveat: Whistle detects English for the TTS outputs, so its language ID is not a reliable check for synthetic speech. Its Polish STT on natural-like audio is good.

### LLM: LiquidAI/LFM2.5-2.6B (GGUF Q4_K_M)
- Runs in llama.cpp (architecture `lfm2`). Built with CUDA, see SETUP.md.
- Chat template is ChatML-like and always appends `<|im_start|>assistant\n<think>`. Without intervention, the model spends ~287 tokens in English reasoning. `--reasoning-budget 0` skips it. The `enable_thinking` kwarg is ignored by this template.
- Tool calls: Pythonic by default (`<|tool_call_start|>...<|tool_call_end|>`) [card]. Not tested here.
- Published speed [card]: 113 tok/s on Ryzen AI Max+, 220 tok/s on M5 Max. These are not comparable to this GPU. Measured here, see `bench/out/llm_smoke_pl_nothink.json`.

### TTS-A: sahilmahendrakar/Paradee-8M-v1.0
- Kokoro-82M distillation, 8M params, ONNX int8 (9 MB), run via onnxruntime on CPU.
- G2P: misaki with espeak fallback (espeakng_loader bundled). `en_core_web_sm` is required and is declared in pyproject.toml.
- Published speed [card]: ~25x real time on M4 with PyTorch; ~18x with ONNX, 1 thread.
- Polish: not supported. Output for the Polish sentence is English-phonetic ("DEESIAN DABRIE TO JEST CROCKY TESS CINTASY MOY PO POLSCO" by Whistle).
- Streaming: none. `Paradee()(text)` returns the whole utterance.

### TTS-B: ResembleAI/chatterbox-turbo
- ChatterboxTurboTTS, 350M, English only. Supports paralinguistic tags (e.g. `[laugh]`) [card]. Voice cloning from a reference clip [card]. Output is watermarked (resemble-perth).
- Polish: not supported (card lists English only). Output for the Polish sentence: "Duzzi, Dubri, To Jusqueroki Test, Cyntisi, Mowi Pole Palscu" (Whistle).
- Streaming: none. `generate()` returns the full utterance.
- Needs `setuptools<81` (resemble-perth imports `pkg_resources`).

### Reference: ResembleAI/chatterbox (multilingual)
- ChatterboxMultilingualTTS, 500M, 23 languages incl. Polish [card]. Not one of the four candidates, but the only Chatterbox variant with Polish.
- Full fp32 on GPU fails: CUDA OOM during `.to(device)` (2.89 GB allocated, ~3 GB free).
- Working options: CPU fp32 (RTF 0.13, RSS 6.9 GB), or GPU fp16 (load on CPU, `.half()`, move to cuda, `torch.autocast` fp16; RTF ~0.20, VRAM 1.78 GB).
- Polish output is not intelligible: "Ding doloré, to his cricket, death, and pigeon movable polyacum" (fp16). With `cfg_weight=0` (fp16): "Ching Dolly tastes incredibly contested, fantasm Robert Paulisko". Not usable as-is.
- Not yet tested: `mradermacher/Qwen3-TTS-Polish-GGUF` (986 MB), already in the HF cache from earlier work. It is the most plausible Polish TTS to evaluate next.

## Hard target status
- TTS RTF > 1: met only by chatterbox-turbo in its first measurement (EN 0.70 cold, PL warm 1.56 but on non-Polish text, which is not a valid comparison for Polish output). Paradee is faster but English only.
- Streaming: no candidate TTS has a streaming API. Chunk at sentence level as a fallback.
- VRAM: LLM (~1.75 GB) + turbo (~2.95 GB) + Hermes (~0.99 GB) exceeds 4 GB. They cannot run together on the GPU as configured.

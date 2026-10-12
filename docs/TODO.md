# TODO

## Parakeet STT (sherpa-onnx) as an alternative to faster-whisper

Status: not started beyond a benchmark harness. The unfinished harness is on the local branch
`night/stt-parakeet` (commit "WIP: Parakeet (sherpa-onnx) STT benchmark harness"): a `parakeet` engine in
`bench/stt_bench.py`, `bench/run_parakeet_suite.py` and `bench/hotwords_tech.txt`. That branch also adds the
`sherpa-onnx` dependency.

To do:
1. Rebase the harness on `main` and run the suite: PL, EN and mixed PL/EN sets, 2 and 4 CPU threads, greedy vs
   beam with technical hotwords, and a run under CPU contention.
2. Compare against production faster-whisper small int8 on CUDA (0.395 s p50, PL WER 18.69%, see
   [STT_RESULTS.md](STT_RESULTS.md)), including VRAM: the 4 GB GPU already holds Hermes, llama-server and the
   adapter.
3. Adopt only if it keeps first audio under 1 s and improves Polish WER or technical-term accuracy; otherwise
   record the results in STT_RESULTS.md and close this item.

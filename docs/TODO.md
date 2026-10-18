# TODO

## Parakeet STT follow-ups

Parakeet (sherpa-onnx, CPU) is the turn engine since 2026-10-18 and faster-whisper small stays on the GPU for wake
verification ([STT_RESULTS.md, section 9](STT_RESULTS.md#9-parakeet-tdt-06b-v3-sherpa-onnx-cpu)). Open:
1. Check the live wake accept rate (the wake log in `/api/diagnostics`, admin panel diagnostics) and a day of turn transcripts against the bench.
2. The English set (`bench/data/stt/en`) was not measured with Parakeet.
3. Both engines are resident: about 1.1 GB more adapter RAM. If Parakeet can verify wakes as well as
   faster-whisper, whisper can go (frees ~380 MB VRAM).

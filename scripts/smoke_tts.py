import json
import os
import resource
import sys
import time

import numpy as np
import soundfile as sf
import torch

EN = "Hello, this is a short smoke test of the text to speech model."
PL = "Dzień dobry, to jest krótki test syntezy mowy po polsku."
OUT = "bench/out"


def peak_rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def peak_vram_mb():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_reserved() / 2**20)


def to_numpy(wav):
    if torch.is_tensor(wav):
        wav = wav.detach().float().cpu().numpy()
    return np.squeeze(np.asarray(wav, dtype=np.float32))


name = sys.argv[1]
device = os.environ.get("TTS_DEVICE", "cuda")
tag_suffix = "" if device == "cuda" else f"_{device}"

t0 = time.perf_counter()
if name == "paradee":
    from paradee import SAMPLE_RATE, Paradee

    tts = Paradee()
    sr = SAMPLE_RATE

    def synth(text):
        return tts(text)
elif name == "chatterbox-turbo":
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    tts = ChatterboxTurboTTS.from_pretrained(device=device)
    sr = tts.sr

    def synth(text):
        return tts.generate(text)
elif name == "chatterbox-mtl":
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    half = bool(os.environ.get("TTS_HALF"))
    tts = ChatterboxMultilingualTTS.from_pretrained(device="cpu" if half else device)
    if half:
        for m in (tts.t3, tts.s3gen, tts.ve):
            m.half().to(device)
        if tts.conds is not None:
            tts.conds = tts.conds.to(device)
        tts.device = device
        tag_suffix += "_fp16"
    sr = tts.sr

    def synth(text):
        lang = "pl" if text == PL else "en"
        extra = {"cfg_weight": float(os.environ["TTS_CFG"])} if os.environ.get("TTS_CFG") else {}
        with torch.autocast("cuda", dtype=torch.float16, enabled=half):
            return tts.generate(text, language_id=lang, **extra)
else:
    raise SystemExit(f"unknown model {name}")
load_s = time.perf_counter() - t0
print(f"[{name}] loaded in {load_s:.2f}s sr={sr} peak_rss_mb={peak_rss_mb()} vram_mb={peak_vram_mb()}", flush=True)

for tag, text in (("en", EN), ("pl", PL)):
    t1 = time.perf_counter()
    wav = to_numpy(synth(text))
    wall = time.perf_counter() - t1
    audio_s = len(wav) / sr
    path = f"{OUT}/smoke_{name}{tag_suffix}_{tag}.wav"
    sf.write(path, wav, sr)
    print(json.dumps({
        "model": name,
        "lang": tag,
        "audio_s": round(audio_s, 2),
        "wall_s": round(wall, 2),
        "rtf": round(audio_s / wall, 2),
        "file": path,
        "peak_rss_mb": peak_rss_mb(),
        "vram_reserved_mb": peak_vram_mb(),
        "load_s": round(load_s, 2),
    }, ensure_ascii=False), flush=True)

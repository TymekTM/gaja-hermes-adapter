import io
import wave

import soundfile as sf
from piper import PiperVoice

VOICE = "/home/ttm-server/voice-agent/pl_PL-darkman-medium.onnx"
OUT = "bench/out/tts/speaker_ref_piper.wav"
TEXT = "Dzień dobry, mogę pomóc w sprawdzeniu pogody i przypomnieć o spotkaniach w kalendarzu."

voice = PiperVoice.load(VOICE)
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    voice.synthesize_wav(TEXT, wf)
buf.seek(0)
audio, sr = sf.read(buf, dtype="float32")
sf.write(OUT, audio, sr, subtype="PCM_16")
print(f"wrote {OUT} sr={sr} dur={len(audio) / sr:.2f}s")

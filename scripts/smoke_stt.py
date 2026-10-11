import json
import resource
import sys
import time

from needle import Whistle

path = sys.argv[1] if len(sys.argv) > 1 else "bench/out/input_pl_16k.wav"

t0 = time.perf_counter()
model = Whistle()
load_s = time.perf_counter() - t0

t1 = time.perf_counter()
result = model.transcribe(path)
gen_s = time.perf_counter() - t1

print(json.dumps({
    "file": path,
    "load_s": round(load_s, 2),
    "transcribe_s": round(gen_s, 2),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    "result": result,
}, ensure_ascii=False, indent=2, default=str))

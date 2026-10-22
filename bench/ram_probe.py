"""RSS probes for the Gaja adapter's resident models. One scenario per process, run via bench/ram_run.sh.

Every scenario prints "RSS <kB> <stage>" lines, so the runner's output file holds the whole
measurement. Nothing here talks to the network or the production process.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # -I mode: make gaja importable


def rss_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("no VmRSS")


def tag(label: str) -> None:
    print(f"RSS {rss_kb()} kB {label}", flush=True)


class NumpySileroProb:
    """Silero VAD via onnxruntime only (no torch): same ONNX file, numpy state."""

    def __init__(self) -> None:  # noqa: ANN101
        import importlib.util
        from pathlib import Path

        import onnxruntime as ort

        spec = importlib.util.find_spec("silero_vad")  # locate without importing (no torch)
        path = str(Path(spec.submodule_search_locations[0]) / "data" / "silero_vad.onnx")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        self._state = None
        self._context = None

    def __call__(self, window) -> float:  # noqa: ANN001
        import numpy as np

        if self._state is None:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._context = np.zeros((1, 64), dtype=np.float32)
        x = np.concatenate([self._context, window.reshape(1, -1)], axis=1)
        out, state = self._session.run(None, {"input": x, "state": self._state, "sr": np.array(16000, dtype="int64")})
        self._state = state
        self._context = x[:, -64:]
        return float(out.reshape(-1)[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("scenario")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup-s", type=float, default=1.0)
    args = p.parse_args()
    print(f"CMD argv: {sys.argv}", flush=True)
    tag("start")

    s = args.scenario

    if s.startswith("import:"):
        mod = s[len("import:") :]
        importlib.import_module(mod)
        if "torch" in sys.modules:
            with open("/proc/self/maps") as f:
                maps = f.read()
            torch_libs = sorted({line.split()[-1] for line in maps.splitlines() if "site-packages/torch" in line and line.strip().endswith(".so")})
            print(f"TORCH-IN-MODULES torch libs mapped: {len(torch_libs)}", flush=True)
        tag(f"after import {mod}")
        return

    if s == "stack_tts_vad":
        # Production today: paradee (en default) + torch silero VAD in one process.
        import asyncio
        import time

        import numpy as np

        from gaja.audio.vad import load_silero_prob
        from gaja.tts.paradee_tts import ParadeeTTS

        prob = load_silero_prob()
        w = np.zeros(512, dtype=np.float32)
        for _ in range(10):
            prob(w)
        tag("after torch silero VAD")
        t0 = time.perf_counter()
        tts = ParadeeTTS(threads=1)
        tag(f"after Paradee() init ({time.perf_counter() - t0:.1f}s)")
        asyncio.run(tts.warm_up())
        tag("after paradee warm_up")
        return

    if s == "paradee":
        import asyncio
        import time

        from gaja.tts.paradee_tts import ParadeeTTS

        tts = ParadeeTTS(threads=1)
        tag("after Paradee() init")
        t0 = time.perf_counter()
        asyncio.run(tts.warm_up())
        tag(f"after paradee warm_up ({time.perf_counter() - t0:.1f}s)")
        return

    if s == "silero_np_only":
        import numpy as np

        prob = NumpySileroProb()
        w = np.zeros(512, dtype=np.float32)
        for _ in range(10):
            prob(w)
        tag("after numpy-only silero session + 10 windows")
        return

    if s == "parakeet":
        import asyncio

        from gaja.stt.parakeet_stt import ParakeetSTT

        stt = ParakeetSTT(num_threads=args.num_threads)
        tag("after OfflineRecognizer.from_transducer (int8)")
        asyncio.run(stt.warm_up(args.warmup_s))
        tag(f"after warm_up {args.warmup_s}s silence")
        return

    if s == "fw":
        import asyncio

        from gaja.stt.faster_whisper_stt import FasterWhisperSTT

        fw = FasterWhisperSTT(model="small", device=args.device, compute_type="int8", beam_size=1)
        tag(f"after WhisperModel small int8 {args.device}")
        asyncio.run(fw.warm_up(1.0))
        tag(f"after warm_up 1s silence {args.device}")
        return

    if s == "piper_arena_off":
        import asyncio
        import time
        from pathlib import Path

        import onnxruntime as ort

        import gaja.tts.piper_tts as pt

        def capped(model_path: Path, threads: int) -> ort.InferenceSession:
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])

        pt._capped_session = capped
        model = Path("/home/ttm-server/.t3/projects/gaja-hermes-adapter/models/pl_PL-gosia-medium.onnx")
        t0 = time.perf_counter()
        tts = pt.piper_gosia(model, threads=2)
        tag(f"after piper load arena-off ({time.perf_counter() - t0:.1f}s)")
        asyncio.run(tts.warm_up())
        tag("after piper warm_up arena-off")
        return

    if s == "piper":
        import asyncio
        import time
        from pathlib import Path

        from gaja.tts.piper_tts import piper_gosia

        model = Path(
            "/home/ttm-server/.t3/projects/gaja-hermes-adapter/models/pl_PL-gosia-medium.onnx"
        )
        t0 = time.perf_counter()
        tts = piper_gosia(model, threads=2)
        tag(f"after piper gosia load ({time.perf_counter() - t0:.1f}s)")
        asyncio.run(tts.warm_up())
        tag("after piper gosia warm_up")
        return

    if s == "silero_torch":
        import numpy as np

        from gaja.audio.vad import load_silero_prob

        prob = load_silero_prob()
        w = np.zeros(512, dtype=np.float32)
        for _ in range(10):
            prob(w)
        tag("after load_silero_prob (torch wrapper) + 10 windows")
        return

    if s == "silero_onnx":
        import numpy as np

        from gaja.audio.vad import load_silero_prob

        torch_prob = load_silero_prob()
        np_prob = NumpySileroProb()
        w = np.zeros(512, dtype=np.float32)
        for prob in (torch_prob, np_prob):
            for _ in range(10):
                prob(w)
        outs = [
            (torch_prob(np.full(512, v, dtype=np.float32)), np_prob(np.full(512, v, dtype=np.float32)))
            for v in (0.0, 0.25, -0.25, 0.9)
        ]
        maxdiff = max(abs(float(x) - float(y)) for x, y in outs)
        print(f"MAXDIFF torch-vs-numpy {maxdiff:.3e}", flush=True)
        tag("after torch wrapper + numpy wrapper + prob compare")
        return

    raise SystemExit(f"unknown scenario {s!r}")


if __name__ == "__main__":
    main()

import os
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "third_party/llama.cpp/build/bin/llama-server"
GGUF = Path((ROOT / "bench/logs/lfm_gguf_path.txt").read_text().strip())
# Port domyślnie 8792; do testów obok działającej produkcji (adapter trzyma 8792
# dla urządzeń IoT) ustaw BENCH_LLM_PORT, np. BENCH_LLM_PORT=8794.
PORT = int(os.environ.get("BENCH_LLM_PORT", "8792"))
URL = f"http://127.0.0.1:{PORT}"
TEMPLATES = ROOT / "bench/templates"

BASE_ARGS = ["-ngl", "99", "-c", "4096", "-np", "1", "--jinja", "-fa", "on"]
VARIANTS = {
    "budget0": ["--reasoning-budget", "0"],
    "empty_think": ["--chat-template-file", str(TEMPLATES / "lfm_empty_think.jinja")],
    "nothink": ["--chat-template-file", str(TEMPLATES / "lfm_nothink.jinja")],
    "nothink_budget0": ["--chat-template-file", str(TEMPLATES / "lfm_nothink.jinja"), "--reasoning-budget", "0"],
    # LFM3: GGUF nie ma osadzonego szablonu; plik z upstream models/templates/LFM3.jinja.
    # nothink = natywne --reasoning-budget 0 (model potrafi sam otworzyć <think> mimo pustego
    # bloku w prompcie, więc forsowanie po stronie serwera jest niezawodne).
    "lfm3": ["--chat-template-file", str(TEMPLATES / "LFM3.jinja")],
    "lfm3_nothink": ["--chat-template-file", str(TEMPLATES / "LFM3.jinja"), "--reasoning-budget", "0"],
}


def start(extra_args, log_name, base_args=None):
    args = BASE_ARGS if base_args is None else base_args
    cmd = [str(BIN), "-m", str(GGUF), "--host", "127.0.0.1", "--port", str(PORT), *args, *extra_args]
    log = open(ROOT / "bench/logs" / log_name, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early, see bench/logs/{log_name}")
        try:
            if requests.get(f"{URL}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy")


def stop(proc):
    proc.terminate()
    proc.wait(timeout=60)


def loadavg_1min():
    return float(Path("/proc/loadavg").read_text().split()[0])

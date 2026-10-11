from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import (
    ToolResult,
    format_number_pl,
    inflect_number_noun,
    polish_declension,
)


async def server_status(lang: str = "pl") -> ToolResult:
    """Read server health: uptime, load average, RAM, GPU memory, and systemd user services."""
    try:
        # Uptime
        uptime_s = 0.0
        try:
            with open("/proc/uptime", encoding="utf-8") as f:
                uptime_s = float(f.readline().split()[0])
        except Exception:
            pass

        days = int(uptime_s // 86400)
        hours = int((uptime_s % 86400) // 3600)

        # Loadavg
        try:
            load1, load5, load15 = os.getloadavg()
            load1 = round(load1, 2)
            load5 = round(load5, 2)
            load15 = round(load15, 2)
        except Exception:
            load1 = load5 = load15 = 0.0

        # RAM
        ram_total_gb = 0.0
        ram_used_gb = 0.0
        try:
            mem: dict[str, int] = {}
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem[parts[0].strip()] = int(parts[1].split()[0])
            total_kb = mem.get("MemTotal", 0)
            avail_kb = mem.get("MemAvailable", 0)
            ram_total_gb = round(total_kb / 1024 / 1024, 1)
            ram_avail_gb = round(avail_kb / 1024 / 1024, 1)
            ram_used_gb = round(max(0.0, ram_total_gb - ram_avail_gb), 1)
        except Exception:
            pass

        # GPU memory via nvidia-smi fixed arguments
        gpu_used = "N/A"
        gpu_total = "N/A"
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1.5)
            line = stdout.decode().strip()
            if line and "," in line:
                gpu_used, gpu_total = [x.strip() for x in line.split(",", 1)]
        except Exception:
            pass

        # Parse GPU memory in sensible rounded units (GB)
        gpu_used_gb: float | None = None
        gpu_total_gb: float | None = None
        if gpu_used != "N/A" and gpu_total != "N/A":
            m_u = re.search(r"(\d+(?:\.\d+)?)", gpu_used)
            m_t = re.search(r"(\d+(?:\.\d+)?)", gpu_total)
            if m_u and m_t:
                val_u = float(m_u.group(1))
                val_t = float(m_t.group(1))
                # Values from nvidia-smi memory.used are in MiB
                gpu_used_gb = round(val_u / 1024, 1)
                gpu_total_gb = round(val_t / 1024) if round(val_t / 1024, 1).is_integer() else round(val_t / 1024, 1)

        # Systemd user services status (fixed names)
        services: dict[str, str] = {"gaja-adapter": "unknown", "gaja-llm": "unknown"}
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl",
                "--user",
                "is-active",
                "gaja-adapter",
                "gaja-llm",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1.5)
            lines = stdout.decode().strip().splitlines()
            names = ["gaja-adapter", "gaja-llm"]
            for i, name in enumerate(names):
                if i < len(lines):
                    services[name] = lines[i].strip()
        except Exception:
            pass

        all_active = all(v == "active" for v in services.values())

        data = {
            "uptime_days": days,
            "uptime_hours": hours,
            "loadavg_1m": load1,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "gpu_used": gpu_used,
            "gpu_total": gpu_total,
            "services": services,
        }

        if lang == "en":
            gpu_str = (
                f", GPU {gpu_used_gb} of {gpu_total_gb} gigabytes"
                if gpu_used_gb is not None
                else ""
            )
            speak = (
                f"Server is up for {days} days, load average {load1}. "
                f"RAM {ram_used_gb} of {ram_total_gb} gigabytes{gpu_str}."
            )
        else:
            uptime_str = f"Serwer działa od {inflect_number_noun(days, 'dzień', case='gen')}, obciążenie {format_number_pl(load1)}."
            ram_str = f"RAM {format_number_pl(ram_used_gb)} z {inflect_number_noun(ram_total_gb, 'gigabajt', case='gen')}"
            if gpu_used_gb is not None and gpu_total_gb is not None:
                gpu_phrase = f", GPU {format_number_pl(gpu_used_gb)} z {inflect_number_noun(gpu_total_gb, 'gigabajt', case='gen')}"
            else:
                gpu_phrase = ""
            speak = f"{uptime_str} {ram_str}{gpu_phrase}."

        speak = normalize_for_tts(speak, lang)
        return ToolResult(data=data, speak=speak)
    except Exception as exc:
        err = f"Błąd sprawdzania serwera: {exc}"
        speak = "Nie udało się sprawdzić stanu serwera." if lang == "pl" else "Could not check server status."
        return ToolResult(data={"error": err}, speak=speak, error=err)

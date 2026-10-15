"""Host-side check of the firmware's adaptive-volume mapping.

Compiles ``firmware/esphome/components/gaja_device/volume_map.h`` with the system C++
compiler and asserts the clamping/linearity the firmware relies on. Skipped when no
C++ compiler is available (the firmware itself is compiled with ESPHome).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "volume_map.h"

MAIN = r"""
#include <cstdio>
#include <cmath>
#include "volume_map.h"
using esphome::gaja_device::map_db_to_volume;
static int failures = 0;
static void expect(float got, float want, const char *what) {
  if (std::fabs(got - want) > 1e-6f) { std::printf("FAIL %s: got %.6f want %.6f\n", what, got, want); failures++; }
}
int main() {
  const float q = -24.0f, l = -12.0f, vmin = 0.7f, vmax = 1.0f;  // firmware defaults
  expect(map_db_to_volume(-40.0f, q, l, vmin, vmax), vmin, "below range clamps to vol_min");
  expect(map_db_to_volume(q, q, l, vmin, vmax), vmin, "quiet_db -> vol_min");
  expect(map_db_to_volume(l, q, l, vmin, vmax), vmax, "loud_db -> vol_max");
  expect(map_db_to_volume(-6.0f, q, l, vmin, vmax), vmax, "above range clamps to vol_max");
  expect(map_db_to_volume((q + l) / 2, q, l, vmin, vmax), (vmin + vmax) / 2, "midpoint is linear");
  // strictly monotonic across the whole range
  float prev = -999.0f;
  for (int db = -90; db <= 0; ++db) {
    float v = map_db_to_volume((float) db, q, l, vmin, vmax);
    if (v < prev - 1e-6f) { std::printf("FAIL monotonicity at %d dB\n", db); failures++; }
    prev = v;
  }
  // degenerate config must not whisper
  expect(map_db_to_volume(-50.0f, -10.0f, -10.0f, vmin, vmax), vmax, "quiet>=loud returns vol_max");
  // a generic range works too (calibrated values will replace the defaults on the device)
  expect(map_db_to_volume(-30.0f, -30.0f, -20.0f, 0.4f, 0.9f), 0.4f, "generic range low end");
  expect(map_db_to_volume(-25.0f, -30.0f, -20.0f, 0.4f, 0.9f), 0.65f, "generic range midpoint");
  if (failures) return 1;
  std::printf("volume map OK\n");
  return 0;
}
"""

pytestmark = pytest.mark.skipif(shutil.which("g++") is None and shutil.which("c++") is None,
                                reason="no host C++ compiler")


def test_volume_map() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-volmap-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "volmap"
        subprocess.run(["c++", "-std=c++17", "-I", str(HEADER.parent), str(main), "-o", str(exe)], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    assert "volume map OK" in result.stdout

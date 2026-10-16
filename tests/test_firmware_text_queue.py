"""Host-side check of the firmware's class-aware text queue (G-3 / N-8).

Compiles ``firmware/esphome/components/gaja_device/text_queue.h`` (pure C++): FIFO order, the
half-full discard of droppable (log-only) messages, eviction of the oldest droppable message on a
full queue, and the invariant that a control message is never evicted.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "text_queue.h"

MAIN = r"""
#include <cstdio>
#include <vector>
#include "text_queue.h"
using namespace esphome::gaja_device;
static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}
struct Msg {
  int id;
  bool droppable;
  char pad[1000];
};
static Msg mk(int id, bool droppable) { Msg m{}; m.id = id; m.droppable = droppable; return m; }
using Q = TextQueue<Msg, 8>;
using R = Q::Result;

static std::vector<int> drain(Q &q) {
  std::vector<int> ids;
  Msg m;
  while (q.pop(m)) ids.push_back(m.id);
  return ids;
}

int main() {
  std::vector<Msg> store(8);
  Q q;
  q.attach(store.data());

  // FIFO.
  check(q.push(mk(1, false)) == R::PUSHED, "push control");
  check(q.push(mk(2, true)) == R::PUSHED, "push droppable below half");
  check(q.push(mk(3, false)) == R::PUSHED, "push control");
  check(q.size() == 3, "size");
  Msg out;
  check(q.pop(out) && out.id == 1, "fifo 1");
  check(q.pop(out) && out.id == 2, "fifo 2");
  check(q.pop(out) && out.id == 3, "fifo 3");
  check(!q.pop(out), "empty");

  // N-8: droppable traffic is discarded once the queue is more than half full (> 4 of 8).
  for (int i = 0; i < 4; i++) check(q.push(mk(10 + i, false)) == R::PUSHED, "fill 4 control");
  check(q.push(mk(20, true)) == R::PUSHED, "droppable at exactly half is still accepted");   // count 4 -> 5
  check(q.size() == 5, "size 5");
  check(q.push(mk(21, true)) == R::DROPPED_NEW, "droppable above half is discarded");
  check(q.push(mk(22, false)) == R::PUSHED, "control is still accepted above half");
  check(q.size() == 6, "size 6");
  auto got = drain(q);
  check((got == std::vector<int>{10, 11, 12, 13, 20, 22}), "order preserved, discarded one absent");

  // G-3: a full queue evicts the OLDEST droppable message, never a control message.
  check(q.push(mk(30, false)) == R::PUSHED, "c30");
  check(q.push(mk(31, true)) == R::PUSHED, "d31");
  check(q.push(mk(32, false)) == R::PUSHED, "c32");
  check(q.push(mk(33, true)) == R::PUSHED, "d33");          // count 4 (<= half): accepted
  check(q.push(mk(34, false)) == R::PUSHED, "c34");
  check(q.push(mk(35, false)) == R::PUSHED, "c35");
  check(q.push(mk(36, false)) == R::PUSHED, "c36");
  check(q.push(mk(37, false)) == R::PUSHED, "c37");         // now full (8)
  check(q.size() == 8, "full");
  check(q.push(mk(38, false)) == R::PUSHED_EVICTED, "control push evicts");
  check(q.size() == 8, "still full after eviction");
  check(q.push(mk(39, false)) == R::PUSHED_EVICTED, "second control push evicts the next droppable");
  // Both droppables (31, 33) are gone, every control message survived, in order.
  got = drain(q);
  check((got == std::vector<int>{30, 32, 34, 35, 36, 37, 38, 39}), "only droppables were evicted; order kept");

  // A full queue of control messages: the NEW message is refused, nothing queued is lost.
  for (int i = 0; i < 8; i++) check(q.push(mk(40 + i, false)) == R::PUSHED, "fill control");
  check(q.push(mk(50, false)) == R::DROPPED_NEW, "all-control full queue refuses the new control message");
  check(q.push(mk(51, true)) == R::DROPPED_NEW, "and the new droppable one");
  got = drain(q);
  check((got == std::vector<int>{40, 41, 42, 43, 44, 45, 46, 47}), "no control message lost");

  // Slots are recycled correctly across many cycles with mixed eviction.
  int next = 100;
  std::vector<int> expect_controls;
  for (int round = 0; round < 500; round++) {
    const bool ctl = (round % 3) != 0;
    const int id = next++;
    R r = q.push(mk(id, !ctl));
    if (ctl && r != R::DROPPED_NEW) expect_controls.push_back(id);
    if (round % 5 == 0) {
      Msg m;
      if (q.pop(m) && !m.droppable) {
        check(!expect_controls.empty() && expect_controls.front() == m.id, "controls come out in order, none skipped");
        if (!expect_controls.empty()) expect_controls.erase(expect_controls.begin());
      }
    }
  }
  check(q.size() <= 8, "never above capacity");
  // Everything still queued that is control must match the remaining expectation (in order).
  std::vector<int> still_controls;
  Msg m;
  while (q.pop(m)) if (!m.droppable) still_controls.push_back(m.id);
  check(still_controls == expect_controls, "no control message was ever lost or reordered");

  // Unattached queue is inert.
  Q dead;
  check(dead.push(mk(1, false)) == R::DROPPED_NEW, "unattached refuses");
  check(!dead.pop(out), "unattached empty");

  if (failures) { std::printf("%d text_queue failures\n", failures); return 1; }
  std::printf("text_queue OK\n");
  return 0;
}
"""

pytestmark = pytest.mark.skipif(
    shutil.which("c++") is None and shutil.which("g++") is None and shutil.which("clang++") is None,
    reason="no host C++ compiler",
)


def _cxx() -> str:
    for candidate in ("c++", "g++", "clang++"):
        if shutil.which(candidate) is not None:
            return candidate
    raise AssertionError("unreachable: skipif above guarantees a compiler")


def test_text_queue() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-text-queue-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "text_queue"
        subprocess.run(
            [_cxx(), "-std=c++17", "-fsanitize=address,undefined", "-I", str(HEADER.parent), str(main), "-o", str(exe)],
            check=True,
        )
        result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert "text_queue OK" in result.stdout, result.stdout + result.stderr
    assert "FAIL" not in result.stdout

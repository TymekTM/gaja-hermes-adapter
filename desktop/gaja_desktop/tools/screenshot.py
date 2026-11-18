"""`screenshot`: capture a monitor (or the whole virtual screen) as a JPEG kept under the wire limit.

The whole `pc_result` frame must stay under the listener's 1 MiB `ws_max_size`, so the base64 image
is kept at ≤ 900 KiB by stepping JPEG quality down and then downscaling (docs/DESKTOP.md).
"""

from __future__ import annotations

import base64
import io
from typing import Any

from . import Tool, ToolResult

#: Hard ceiling for `image.b64` (bytes of base64), chosen to keep the whole JSON frame < 1 MiB.
B64_LIMIT = 900 * 1024
JPEG_QUALITY_STEPS = [85, 70, 55, 40, 30]
SCALE_STEPS = [1.0, 0.75, 0.5, 0.35, 0.25]


def encode_jpeg_under_limit(image: Any, *, max_width: int = 1600,
                            b64_limit: int = B64_LIMIT) -> tuple[str, int, int] | None:
    """PIL image -> (b64, width, height) with `b64` ≤ limit; None when even the smallest step fails."""
    from PIL import Image  # noqa: PLC0415

    img: Image.Image = image
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, max(1, round(img.height * ratio))), Image.LANCZOS)
    for scale in SCALE_STEPS:
        candidate = img if scale == 1.0 else img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
        for quality in JPEG_QUALITY_STEPS:
            buf = io.BytesIO()
            candidate.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            if len(base64.b64encode(data)) <= b64_limit:
                return base64.b64encode(data).decode("ascii"), candidate.width, candidate.height
    return None


def grab_monitor(monitor: int) -> tuple[Any, list[dict[str, Any]]]:
    """mss grab of monitor 0 (all monitors combined) or 1..n; returns (PIL image, monitor list)."""
    import mss  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with mss.mss() as sct:
        monitors = sct.monitors  # [0] = all combined, [1..] = one per monitor
        if monitor < 0 or monitor >= len(monitors):
            raise IndexError(f"monitor {monitor} out of range (have {len(monitors) - 1})")
        shot = sct.grab(monitors[monitor])
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        listed = [{"index": i, "left": m["left"], "top": m["top"],
                   "width": m["width"], "height": m["height"]} for i, m in enumerate(monitors)]
        return img, listed


class ScreenshotTool(Tool):
    name = "screenshot"
    description = "Take a screenshot of a monitor"

    def available(self) -> bool:
        try:
            import mss  # noqa: PLC0415
            from PIL import Image  # noqa: F401, PLC0415

            with mss.mss() as sct:  # raises without a display (headless Linux)
                return bool(sct.monitors)
        except Exception:  # noqa: BLE001
            return False

    def run(self, args: dict[str, Any], timeout_s: float) -> ToolResult:
        monitor = self._int(args, "monitor", 0, 0, 16)
        if monitor is None:
            return ToolResult.bad_request("'monitor' must be an integer 0..16")
        max_width = self._int(args, "max_width", 1600, 200, 4000)
        if max_width is None:
            return ToolResult.bad_request("'max_width' must be an integer 200..4000")
        try:
            img, listed = grab_monitor(monitor)
        except IndexError as exc:
            return ToolResult.bad_request(str(exc))
        except Exception as exc:  # noqa: BLE001 - mss failures (lost display, …)
            return ToolResult(ok=False, error=f"screenshot failed: {type(exc).__name__}: {exc}")
        encoded = encode_jpeg_under_limit(img, max_width=max_width)
        if encoded is None:
            return ToolResult(ok=False, error="screenshot too large to encode under the frame limit")
        b64, width, height = encoded
        # The mapping the agent needs: image pixels -> mouse coordinates. `listed[monitor]` is the
        # captured rect in mouse space BEFORE downscaling (mss reports the virtual screen, whose
        # origin is negative when a monitor sits left of/above the primary).
        rect = listed[monitor]
        text = (
            f"monitor {monitor} of {len(listed) - 1} display(s): mouse-space rect "
            f"{rect['width']}x{rect['height']} at ({rect['left']},{rect['top']}); "
            f"image {width}x{height}, {len(b64) * 3 // 4 // 1024} KiB jpeg; "
            f"mouse = ({rect['left']} + img_x*{rect['width']}/{width}, "
            f"{rect['top']} + img_y*{rect['height']}/{height})"
        )
        return ToolResult(
            ok=True,
            text=text,
            image={"mime": "image/jpeg", "b64": b64, "width": width, "height": height},
            data={"monitors": listed},
        )

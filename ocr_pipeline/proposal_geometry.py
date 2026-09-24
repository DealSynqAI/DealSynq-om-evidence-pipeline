"""Explicit coordinate boundary for page-image model proposals.

The proposal contract is normalized xyxy. Unified Source Blocks use xywh.
Never clamp an invalid model box: doing so can silently assign content to the
wrong region while making the resulting package look schema-valid.
"""

from __future__ import annotations

import math


def xyxy_to_xywh(raw: object) -> list[float]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"Expected four xyxy coordinates, got {raw!r}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) for value in raw):
        raise ValueError(f"Non-numeric xyxy coordinates: {raw!r}")
    x0, y0, x1, y1 = (float(value) for value in raw)
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ValueError(f"Invalid normalized xyxy box: {raw!r}")
    return [x0, y0, x1 - x0, y1 - y0]

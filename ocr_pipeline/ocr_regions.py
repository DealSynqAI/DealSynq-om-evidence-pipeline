"""Regions for printed text that no PDF object bounds.

Inspection builds regions from PDF text, image, and vector objects. Text that
exists only in rendered pixels, such as a heading drawn into a background
image band, falls outside all of them. Without a region it can only be kept as
unresolved raw evidence. Here OCR lines outside every inspected region are
clustered into text lanes and given their own normal-text regions, so they are
extracted and validated like any other text.

Lanes that belong to an unbounded graphic are left alone and remain reviewable
raw evidence: short labels scattered across a page (map or diagram
annotations) and lanes surrounded by busy imagery (logos and photo captions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .models import PageInspection, Region

# More short lanes than this on one page reads as annotations of a graphic.
MAX_SHORT_LANES = 3
# Share of the lane's surroundings in its most common color; below it the lane sits on imagery.
MIN_PLAIN_SURROUNDINGS = 0.3


def _center(box: list[float]) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _inside(box: list[float], point: tuple[float, float], margin: float = 0.0) -> bool:
    x, y, width, height = box
    return x - margin <= point[0] <= x + width + margin and y - margin <= point[1] <= y + height + margin


def _union(lines: list[dict[str, Any]]) -> list[float]:
    x0 = min(line["coordinates"][0] for line in lines)
    y0 = min(line["coordinates"][1] for line in lines)
    x1 = max(line["coordinates"][0] + line["coordinates"][2] for line in lines)
    y1 = max(line["coordinates"][1] + line["coordinates"][3] for line in lines)
    return [x0, y0, x1 - x0, y1 - y0]


def cluster_text_lines(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group lines into vertical text lanes: left- or center-aligned, with small gaps."""
    clusters: list[list[dict[str, Any]]] = []
    for line in sorted(lines, key=lambda item: (float(item["coordinates"][1]), float(item["coordinates"][0]))):
        x, y, width, height = (float(value) for value in line["coordinates"])
        best: tuple[float, list[dict[str, Any]]] | None = None
        for cluster in clusters:
            px, py, pw, ph = (float(value) for value in cluster[-1]["coordinates"])
            gap = y - (py + ph)
            aligned = abs(x - px) <= 35 or abs((x + width / 2) - (px + pw / 2)) <= 60
            if aligned and -max(height, ph) <= gap <= max(30, 1.5 * max(height, ph)):
                score = abs(gap) + abs(x - px) / 4
                if best is None or score < best[0]:
                    best = (score, cluster)
        if best:
            best[1].append(line)
        else:
            clusters.append([line])
    return clusters


def _plain_surroundings(image: Image.Image, box: list[float]) -> float:
    """Share of the ring around a text box taken by its most common coarse color."""
    width, height = image.size
    x, y, w, h = box
    margin = max(1.5 * h, 8.0)
    outer = [max(0, int((x - margin) * width / 1000)), max(0, int((y - margin) * height / 1000)),
             min(width, int((x + w + margin) * width / 1000)), min(height, int((y + h + margin) * height / 1000))]
    pixels = np.asarray(image.crop(tuple(outer)).convert("RGB")) // 16
    ring = np.ones(pixels.shape[:2], dtype=bool)
    ring[max(0, int(y * height / 1000) - outer[1]):int((y + h) * height / 1000) - outer[1],
         max(0, int(x * width / 1000) - outer[0]):int((x + w) * width / 1000) - outer[0]] = False
    colors = pixels[ring].reshape(-1, 3)
    if not len(colors):
        return 0.0
    _values, counts = np.unique(colors, axis=0, return_counts=True)
    return float(counts.max() / len(colors))


def add_ocr_text_regions(
    inspection: PageInspection, ocr_lines: list[dict[str, Any]], image_path: Path,
    skip: Callable[[list[dict[str, Any]]], bool] | None = None,
) -> int:
    """Add a normal-text region for each OCR text lane outside every inspected region.

    ``skip`` lets the caller leave a lane for a more specific owner, such as a brand mark.
    """
    outside = [
        line for line in ocr_lines
        if "-ocr-" in str(line.get("evidence_id", ""))
        and str(line.get("text", "")).strip()
        and float(line.get("confidence", 0.0)) >= 0.8
        and not any(_inside(region.coordinates, _center(line["coordinates"]), 2)
                    for region in inspection.regions)
    ]
    lanes = [cluster for cluster in cluster_text_lines(outside) if not (skip and skip(cluster))]
    if not lanes:
        return 0

    def short(cluster: list[dict[str, Any]]) -> bool:
        return sum(len(str(line["text"]).split()) for line in cluster) <= 3

    if sum(short(cluster) for cluster in lanes) > MAX_SHORT_LANES:
        lanes = [cluster for cluster in lanes if not short(cluster)]
    with Image.open(image_path) as image:
        lanes = [cluster for cluster in lanes
                 if _plain_surroundings(image, _union(cluster)) >= MIN_PLAIN_SURROUNDINGS]
    added = 0
    next_order = max((region.reading_order for region in inspection.regions), default=0)
    for cluster in lanes:
        x0, y0, w, h = _union(cluster)
        x1, y1 = x0 + w, y0 + h
        added += 1
        inspection.regions.append(Region(
            region_id=f"p{inspection.page:03d}-ocr-text-{added:03d}",
            page=inspection.page, kind="normal_text",
            coordinates=[round(x0, 3), round(y0, 3), round(x1 - x0, 3), round(y1 - y0, 3)],
            reading_order=next_order + added,
            classification_method="RapidOCR text lane outside inspected PDF regions",
            confidence=round(min(float(line.get("confidence", 0.0)) for line in cluster), 5),
            metadata={
                "word_count": sum(len(str(line["text"]).split()) for line in cluster),
                "source_bbox_points": [
                    x0 * inspection.width_points / 1000, y0 * inspection.height_points / 1000,
                    x1 * inspection.width_points / 1000, y1 * inspection.height_points / 1000,
                ],
            },
        ))
    return added

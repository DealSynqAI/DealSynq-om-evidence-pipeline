"""Evidence-gated proposals for pages containing several physical regions.

The model supplies possible boundaries only. Printed text and pixel/PDF image
geometry are the independent witnesses; neither model labels nor model text
become source content.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import pdfplumber


def _xywh(box: list[float]) -> list[float]:
    return [float(box[0]), float(box[1]), float(box[2]) - float(box[0]), float(box[3]) - float(box[1])]


def _area(box: list[float]) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def _inter(left: list[float], right: list[float]) -> float:
    return (max(0.0, min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0]))
            * max(0.0, min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1])))


def _inside(box: list[float], line: dict[str, Any]) -> bool:
    x, y, w, h = line["coordinates"]
    return box[0] <= x + w / 2 <= box[0] + box[2] and box[1] <= y + h / 2 <= box[1] + box[3]


def _photo_tiles(image: Path) -> tuple[np.ndarray, list[list[float]], np.ndarray | None]:
    picture = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if picture is None:
        return np.zeros((20, 20), dtype=np.uint8), [], None
    picture = cv2.resize(picture, (400, 400), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    mask = np.zeros((20, 20), dtype=np.uint8)
    for row in range(20):
        for col in range(20):
            tile = picture[row * 20:(row + 1) * 20, col * 20:(col + 1) * 20]
            std = float(np.median(np.std(tile.reshape(-1, 3), axis=0)))
            saturation = float(np.median(hsv[row * 20:(row + 1) * 20, col * 20:(col + 1) * 20, 1]))
            chroma = float(np.mean(np.max(tile, axis=2) - np.min(tile, axis=2)))
            if std > 25 and (saturation > 18 or chroma > 25):
                mask[row, col] = 1
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for index in range(1, count):
        x, y, w, h, pixels = (int(value) for value in stats[index])
        if pixels >= 8 and w >= 2 and h >= 2:
            components.append([x * 50.0, y * 50.0, w * 50.0, h * 50.0])
    return mask, components, picture


def _color_metrics(picture: np.ndarray | None, box: list[float]) -> tuple[float, float, float]:
    if picture is None:
        return 0.0, 0.0, 0.0
    x0, y0 = max(0, int(box[0] * 0.4)), max(0, int(box[1] * 0.4))
    x1 = min(400, max(x0 + 1, int((box[0] + box[2]) * 0.4)))
    y1 = min(400, max(y0 + 1, int((box[1] + box[3]) * 0.4)))
    crop = cv2.resize(picture[y0:y1, x0:x1], (128, 128), interpolation=cv2.INTER_AREA)
    quantized = (crop // 32).reshape(-1, 3)
    _colors, counts = np.unique(quantized, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    finer = crop // 16
    colors, finer_counts = np.unique(finer.reshape(-1, 3), axis=0, return_counts=True)
    background = colors[np.argmax(finer_counts)]
    foreground = np.max(np.abs(finer.astype(np.int16) - background.astype(np.int16)), axis=2) >= 2
    share = float(np.mean(foreground))
    if not foreground.any():
        return entropy, share, 0.0
    _visible, visible_counts = np.unique(finer[foreground], axis=0, return_counts=True)
    weights = visible_counts / visible_counts.sum()
    visible_entropy = float(-(weights * np.log2(weights)).sum())
    return entropy, share, visible_entropy


def _colored_card_boundary(picture: np.ndarray | None, box: list[float]) -> list[float] | None:
    """Expand an OCR card only to a connected, nearly uniform printed panel."""
    if picture is None:
        return None
    x0, y0 = max(0, int(box[0] * 0.4)), max(0, int(box[1] * 0.4))
    x1 = min(400, max(x0 + 1, int((box[0] + box[2]) * 0.4)))
    y1 = min(400, max(y0 + 1, int((box[1] + box[3]) * 0.4)))
    lower = picture[y0 + (y1 - y0) // 2:y1, x0:x1].reshape(-1, 3)
    colored = lower[(lower.min(axis=1) > 170) & (lower.max(axis=1) < 245)
                    & (lower.mean(axis=1) < 235)]
    if len(colored) < 0.2 * len(lower):
        return None
    ink = np.median(colored, axis=0)
    mask = (np.max(np.abs(picture.astype(np.float32) - ink), axis=2) <= 12).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    best: tuple[float, list[float]] | None = None
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        overlap = max(0, min(x1, x + w) - max(x0, x)) * max(0, min(y1, y + h) - max(y0, y))
        if (overlap < 0.35 * (x1 - x0) * (y1 - y0)
                or area > 2.5 * (x1 - x0) * (y1 - y0)
                or w > 1.6 * (x1 - x0) or h > 1.8 * (y1 - y0)):
            continue
        if best is None or overlap > best[0]:
            best = (float(overlap), [x * 2.5, y * 2.5, w * 2.5, h * 2.5])
    if best is None:
        return None
    panel = best[1]
    x = min(box[0], panel[0])
    y = min(box[1], panel[1])
    right = max(box[0] + box[2], panel[0] + panel[2])
    bottom = max(box[1] + box[3], panel[1] + panel[3])
    # A logo can rise above a colored card. Extend only when a continuous
    # printed edge stays in the same narrow card lane above the panel.
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    bands = []
    for top in range(max(0, int(y) - 55), int(y) - 4, 5):
        sample = edges[max(0, int(top * 0.4)):max(1, int((top + 5) * 0.4)),
                       max(0, int(x * 0.4)):min(400, int(right * 0.4))]
        bands.append((top, bool(sample.size and np.mean(sample > 0) >= 0.025)))
    run: list[int] = []
    for top, supported in bands:
        run = [*run, top] if supported else []
        if len(run) >= 5:
            y = max(0.0, float(run[0]))
            x = max(0.0, x - 5)
            right = min(1000.0, right + 5)
            break
    return [x, y, right - x, bottom - y]


def _tile_share(mask: np.ndarray, box: list[float]) -> float:
    x0, y0 = max(0, int(box[0] // 50)), max(0, int(box[1] // 50))
    x1 = min(20, max(x0 + 1, int(np.ceil((box[0] + box[2]) / 50))))
    y1 = min(20, max(y0 + 1, int(np.ceil((box[1] + box[3]) / 50))))
    return float(np.mean(mask[y0:y1, x0:x1]))


def _ocr_clusters(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Link nearby lines in one column, never across a wide gutter."""
    usable = [line for line in lines if str(line.get("text", "")).strip()
              and "-ocr-" in str(line.get("evidence_id", ""))]
    neighbors = [[] for _ in usable]
    for i, left in enumerate(usable):
        ax, ay, aw, ah = left["coordinates"]
        for j in range(i + 1, len(usable)):
            bx, by, bw, bh = usable[j]["coordinates"]
            horizontal_gap = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
            vertical_gap = max(0.0, max(ay, by) - min(ay + ah, by + bh))
            if horizontal_gap <= 22 and vertical_gap <= 90:
                neighbors[i].append(j)
                neighbors[j].append(i)
    seen: set[int] = set()
    groups = []
    for start in range(len(usable)):
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        component = []
        while queue:
            current = queue.popleft()
            component.append(usable[current])
            for other in neighbors[current]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        groups.append(component)
    return groups


def _line_box(lines: list[dict[str, Any]], pad: float = 8) -> list[float]:
    x0 = min(line["coordinates"][0] for line in lines)
    y0 = min(line["coordinates"][1] for line in lines)
    x1 = max(line["coordinates"][0] + line["coordinates"][2] for line in lines)
    y1 = max(line["coordinates"][1] + line["coordinates"][3] for line in lines)
    x, y = max(0.0, x0 - pad), max(0.0, y0 - pad)
    return [x, y, min(1000.0, x1 + pad) - x, min(1000.0, y1 + pad) - y]


def propose_regions(image: Path, inspection: Any, lines: list[dict[str, Any]],
                    plan: dict[str, Any] | None, pdf: Path | None = None) -> list[dict[str, Any]]:
    """Return candidate receipts with explicit independent support or rejection."""
    mask, pixel_components, picture = _photo_tiles(image)
    rapid = [line for line in lines if "-ocr-" in str(line.get("evidence_id", ""))]
    proposals: list[dict[str, Any]] = []
    model_maps = [_xywh(item["bbox"]) for item in (plan or {}).get("blocks", [])
                  if item.get("type") == "map" and isinstance(item.get("bbox"), list)
                  and len(item["bbox"]) == 4
                  and all(isinstance(value, (int, float)) for value in item["bbox"])
                  and 0 < _area(_xywh(item["bbox"])) < 600_000]

    def add(kind: str, box: list[float], origin: str, *, model_type: str | None = None) -> None:
        if len(box) != 4 or not (0 <= box[0] < 1000 and 0 <= box[1] < 1000
                                  and 0 < box[2] <= 1000 - box[0] and 0 < box[3] <= 1000 - box[1]):
            return
        owned = [line for line in rapid if _inside(box, line)]
        texture = _tile_share(mask, box)
        color_entropy, foreground_share, foreground_entropy = (
            _color_metrics(picture, box) if kind == "photograph" else (0.0, 0.0, 0.0)
        )
        words = " ".join(str(line.get("text", "")) for line in owned)
        numeric = sum(bool(re.search(r"\d", str(line.get("text", "")))) for line in owned)
        support = {"rapidocr_lines": len(owned), "numeric_lines": numeric,
                   "photo_texture_tile_share": round(texture, 3),
                   "photo_color_entropy": round(color_entropy, 3),
                   "photo_foreground_share": round(foreground_share, 3),
                   "photo_foreground_entropy": round(foreground_entropy, 3),
                   "pixel_component_overlap": round(max((_inter(box, item) / max(1, _area(box))
                                                          for item in pixel_components), default=0), 3)}
        if kind == "photograph":
            map_conflict = any(_inter(box, candidate) /
                               max(1.0, min(_area(box), _area(candidate))) >= 0.65
                               for candidate in model_maps)
            accepted = (_area(box) >= 12_000 and texture >= 0.30
                        and (color_entropy >= 3.5
                             or foreground_share >= 0.25 and foreground_entropy >= 3.5)
                        and len(owned) <= 2 and not map_conflict)
            reason = ("natural pixel texture with sparse OCR" if accepted
                      else "photo lacks independent natural-image/OCR support or conflicts with map proposal")
        elif kind == "map":
            geo = bool(re.search(r"\b(?:road|street|avenue|highway|location|site|map|mile|mi|interstate)\b", words, re.I))
            accepted = geo and texture >= 0.12 and len(owned) >= 2 and _area(box) < 700_000
            reason = "pixel geography and printed labels" if accepted else "map lacks pixel/geography support"
        elif kind == "table":
            ys = {round(line["coordinates"][1] / 20) for line in owned}
            xs = {round(line["coordinates"][0] / 80) for line in owned}
            accepted = len(ys) >= 3 and len(xs) >= 2 and numeric >= 3
            reason = "OCR numeric grid" if accepted else "no independent OCR numeric grid"
        else:
            accepted = len(owned) >= 3 and len(words) >= 20
            reason = "OCR lines in physical region" if accepted else "insufficient OCR text"
        proposals.append({"kind": kind, "coordinates": [round(v, 3) for v in box],
                          "origin": origin, "model_type": model_type,
                          "support": support, "accepted": accepted, "reason": reason,
                          "ocr_evidence_ids": [line["evidence_id"] for line in owned]})

    # PDF image member geometry is independent of both OCR and the model.
    for region in inspection.regions:
        for member in region.metadata.get("source_member_bboxes", []):
            if len(member) != 4:
                continue
            x0, y0, x1, y1 = member
            box = [x0 * 1000 / inspection.width_points, y0 * 1000 / inspection.height_points,
                   (x1 - x0) * 1000 / inspection.width_points,
                   (y1 - y0) * 1000 / inspection.height_points]
            if 12_000 <= _area(box) <= 650_000:
                add("photograph", box, "pdf_image_member")
    if pdf is not None:
        with pdfplumber.open(pdf) as document:
            for member in document.pages[inspection.page - 1].images:
                box = [float(member["x0"]) * 1000 / inspection.width_points,
                       float(member["top"]) * 1000 / inspection.height_points,
                       (float(member["x1"]) - float(member["x0"])) * 1000 / inspection.width_points,
                       (float(member["bottom"]) - float(member["top"])) * 1000 / inspection.height_points]
                if (12_000 <= _area(box) <= 800_000 and box[0] >= 0 and box[1] >= 0
                        and box[0] + box[2] <= 1000 and box[1] + box[3] <= 1000):
                    add("photograph", box, "pdf_image_member")
    for box in pixel_components:
        if _area(box) <= 650_000:
            add("photograph", box, "pixel_component")
    for item in (plan or {}).get("blocks", []):
        raw = item.get("bbox")
        if not isinstance(raw, list) or len(raw) != 4 or not all(isinstance(v, (int, float)) for v in raw):
            continue
        box = _xywh(raw)
        if _area(box) > 700_000:
            continue
        model_type = str(item.get("type") or "")
        if model_type == "photograph":
            add("photograph", box, "model_boundary", model_type=model_type)
        elif model_type == "map":
            add("map", box, "model_boundary", model_type=model_type)
        elif model_type == "table":
            add("table", box, "model_boundary", model_type=model_type)
        elif model_type in {"text", "chart", "kpi_panel"} and _area(box) >= 25_000 and box[2] <= 450:
            # A model's chart label may actually be a sponsor/tenant card.
            in_box = [line for line in rapid if _inside(box, line)]
            words = " ".join(str(line.get("text", "")) for line in in_box)
            profile = bool(re.search(r"\b(?:sponsor|developer|owner|rating|revenue|total sf)\b", words, re.I))
            kind = "tenant_card" if (box[2] <= 350 and box[3] >= 300 and len(in_box) >= 5 and profile) else "text_panel"
            add(kind, box, "model_boundary", model_type=model_type)
    for group in _ocr_clusters(lines):
        if len(group) >= 3:
            box = _line_box(group)
            words = " ".join(str(line.get("text", "")) for line in group)
            profile = bool(re.search(r"\b(?:sponsor|developer|owner|rating|revenue|total sf)\b", words, re.I))
            kind = "tenant_card" if box[2] <= 350 and box[3] >= 280 and len(group) >= 7 and profile else "text_panel"
            if kind == "tenant_card":
                # A logo/name line can sit above a card's metrics and prose.
                # Extend only to an independently printed OCR line in the same lane.
                heading_lines = [line for line in rapid if line not in group
                                 and box[1] - 85 <= line["coordinates"][1] < box[1]
                                 and box[0] - 20 <= line["coordinates"][0]
                                 + line["coordinates"][2] / 2 <= box[0] + box[2] + 20
                                 and len(str(line.get("text", "")).strip()) >= 4]
                if heading_lines:
                    box = _line_box([*group, *heading_lines])
                panel = _colored_card_boundary(picture, box)
                if panel is not None:
                    box = panel
            add(kind, box, "ocr_cluster")
    return proposals


def select_regions(proposals: list[dict[str, Any]], lines: list[dict[str, Any]],
                   existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose nonoverlapping owners. Dense OCR takes precedence over photos."""
    rapid = {str(line.get("evidence_id")): line for line in lines
             if "-ocr-" in str(line.get("evidence_id", ""))}
    occupied = [(block["coordinates"], block["type"]) for block in existing
                if _area(block["coordinates"]) < 450_000]
    already_owned = {str(key) for block in existing
                     for key in block.get("provenance", {}).get("ocr_evidence_ids", [])}
    ranked = sorted((item for item in proposals if item["accepted"]),
                    key=lambda item: (item["kind"] == "tenant_card",
                                      item["origin"] == "pdf_image_member",
                                      item["kind"] in {"text_panel", "table", "map"},
                                      len(item["ocr_evidence_ids"]), _area(item["coordinates"])), reverse=True)
    chosen: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for item in ranked:
        box = item["coordinates"]
        if any(_inter(box, other) / max(1, min(_area(box), _area(other))) >= 0.75
               for other, other_type in occupied
               if not (item["kind"] == "photograph" and other_type in {"text", "heading", "footnote"}
                       and _area(other) < 10_000)):
            continue
        fresh = [key for key in item["ocr_evidence_ids"]
                 if key in rapid and key not in used_ids
                 and not (item["kind"] == "photograph" and key in already_owned)]
        if item["kind"] in {"text_panel", "tenant_card", "table"} and len(fresh) < 3:
            continue
        if item["kind"] == "photograph" and len(fresh) > 2:
            continue
        selected = {**item, "ocr_evidence_ids": fresh}
        chosen.append(selected)
        used_ids.update(fresh)
        occupied.append((box, item["kind"]))
    return chosen


def empty_table_artifacts(blocks: list[dict[str, Any]],
                          photos: list[list[float]]) -> list[dict[str, Any]]:
    """Reject blank PDF-grid bands crossing an independently supported photo."""
    return [block for block in blocks
            if block.get("type") == "table"
            and not block.get("provenance", {}).get("ocr_evidence_ids")
            and not any(str(row.get("label") or "").strip()
                        or any(str(cell.get("raw_value") or "").strip()
                               for cell in row.get("cells", []))
                        for row in block.get("content", {}).get("rows", []))
            and any(_inter(block["coordinates"], photo) /
                    max(1, _area(block["coordinates"])) >= 0.15 for photo in photos)]

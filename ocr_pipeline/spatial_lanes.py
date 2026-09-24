"""Conservative OCR geometry for a key/value list beside a prose column."""

from __future__ import annotations

import re
import statistics
from typing import Any

from .models import PageInspection, Region


def split_mixed_key_value_regions(
    inspection: PageInspection, ocr_lines: list[dict[str, Any]],
) -> int:
    """Split only when aligned pairs and a separate dense prose lane are clear."""
    if len(ocr_lines) < 15:
        return 0
    replacements: list[Region] = []
    split_count = 0
    for region in inspection.regions:
        x0, y0, width, height = region.coordinates
        review = region.metadata.get("paddle_table_review") or {}
        if (region.kind not in {"table", "visual", "unknown"} or width < 550 or height < 350
                or region.metadata.get("rows")
                or review.get("status") not in {None, "layout_only", "not_detected"}):
            replacements.append(region)
            continue
        contained = [line for line in ocr_lines
                     if x0 <= line["coordinates"][0] + line["coordinates"][2] / 2 <= x0 + width
                     and y0 <= line["coordinates"][1] + line["coordinates"][3] / 2 <= y0 + height]
        prose = [line for line in contained
                 if len(str(line.get("text") or "")) >= 55
                 and line["coordinates"][2] >= 120]
        if len(prose) < 5:
            replacements.append(region)
            continue
        prose_x = statistics.median(line["coordinates"][0] for line in prose)
        prose_starts = [line["coordinates"][0] for line in prose]
        prose_y = sorted(line["coordinates"][1] for line in prose)
        prose_gaps = [right - left for left, right in zip(prose_y, prose_y[1:])]
        if (max(prose_starts) - min(prose_starts) > 60
                or statistics.median(prose_gaps) >= 38):
            replacements.append(region)
            continue
        lane_mid = x0 + (prose_x - x0) / 2
        labels = sorted((line for line in contained
                         if line["coordinates"][0] < lane_mid
                         and 2 <= len(str(line.get("text") or "").strip()) <= 42
                         and re.search(r"[A-Za-z]", str(line.get("text") or ""))),
                        key=lambda line: line["coordinates"][1])
        values = [line for line in contained
                  if lane_mid <= line["coordinates"][0] < prose_x - 10
                  and line["coordinates"][0] + line["coordinates"][2] <= prose_x - 8]
        if len(labels) < 5 or len(values) < 5:
            replacements.append(region)
            continue
        label_x = [line["coordinates"][0] for line in labels]
        label_y = [line["coordinates"][1] for line in labels]
        label_gaps = [right - left for left, right in zip(label_y, label_y[1:])]
        if (max(label_x) - min(label_x) > 35
                or statistics.median(label_gaps) <= statistics.median(prose_gaps) + 8):
            replacements.append(region)
            continue
        pairs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for index, label in enumerate(labels):
            top = ((labels[index - 1]["coordinates"][1] + label["coordinates"][1]) / 2
                   if index else label["coordinates"][1] - 30)
            bottom = ((label["coordinates"][1] + labels[index + 1]["coordinates"][1]) / 2
                      if index + 1 < len(labels) else label["coordinates"][1] + 30)
            owned = sorted((line for line in values if top <= line["coordinates"][1] < bottom),
                           key=lambda line: (line["coordinates"][1], line["coordinates"][0]))
            if owned:
                pairs.append((label, owned))
        if (len(pairs) < 5 or len(pairs) < 0.75 * len(labels)
                or sum(any(re.search(r"\d", str(item.get("text") or "")) for item in owned)
                       for _, owned in pairs) < 5):
            replacements.append(region)
            continue
        value_right = max(item["coordinates"][0] + item["coordinates"][2]
                          for _, owned in pairs for item in owned)
        if value_right >= prose_x - 8:
            replacements.append(region)
            continue
        boundary = (value_right + prose_x) / 2
        table_top = max(y0, min(min([label["coordinates"][1],
                                     *(item["coordinates"][1] for item in owned)])
                                for label, owned in pairs) - 12)
        table_bottom = min(y0 + height, max(max([label["coordinates"][1] + label["coordinates"][3],
                                                 *(item["coordinates"][1] + item["coordinates"][3]
                                                   for item in owned)])
                                           for label, owned in pairs) + 12)
        rows = [["Field", "Value"]]
        cell_boxes: list[list[list[float] | None]] = [[None, None]]
        cell_ids: list[list[list[str]]] = [[[], []]]
        for label, owned in pairs:
            rows.append([str(label["text"]), "\n".join(str(item["text"]) for item in owned)])
            vx0 = min(item["coordinates"][0] for item in owned)
            vy0 = min(item["coordinates"][1] for item in owned)
            vx1 = max(item["coordinates"][0] + item["coordinates"][2] for item in owned)
            vy1 = max(item["coordinates"][1] + item["coordinates"][3] for item in owned)
            cell_boxes.append([label["coordinates"], [vx0, vy0, vx1 - vx0, vy1 - vy0]])
            cell_ids.append([[str(label["evidence_id"])],
                             [str(item["evidence_id"]) for item in owned]])
        replacements.append(Region(
            region_id=f"{region.region_id}-kv", page=region.page, kind="table",
            coordinates=[x0, table_top, boundary - x0, table_bottom - table_top],
            reading_order=region.reading_order,
            classification_method="RapidOCR spatial key-value lane split",
            confidence=min(region.confidence, min(float(item.get("confidence", 0))
                                               for label, owned in pairs for item in [label, *owned])),
            metadata={"rows": rows, "rows_source": "ocr_geometry",
                      "cell_coordinates": cell_boxes, "cell_evidence_ids": cell_ids,
                      "paddle_table_review": review},
        ))
        replacements.append(Region(
            region_id=f"{region.region_id}-prose", page=region.page, kind="normal_text",
            coordinates=[boundary, y0, x0 + width - boundary, height],
            reading_order=region.reading_order + 1,
            classification_method="RapidOCR spatial prose lane split",
            confidence=region.confidence,
        ))
        split_count += 1
    inspection.regions = replacements
    return split_count

"""Assign OCR and native PDF evidence lines to inspected regions."""

from __future__ import annotations

import re
import statistics
from typing import Any

from .common import _box_union, _center, _contains
from .models import PageInspection, Region


def _exclusive_region_lines(
    regions: list[Region], page_lines: list[dict[str, Any]],
    decisions: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Assign every OCR line to at most one logical region.

    PDF image objects often overlap semantic text and repeated footer artwork.
    A single evidence item must not silently support multiple root blocks.
    """
    result = {region.region_id: [] for region in regions}
    # Strong cell geometry beats a nearby chart; a broad PDF table finder box
    # without cell geometry does not beat the chart solely because it is called
    # a table. Decoration stays below native text.
    priority = {"table": 6.0, "visual": 5.0, "normal_text": 4.0,
                "unknown": 2.0, "decoration": 1.0}

    def cell_boxes(value: Any) -> list[list[float]]:
        if isinstance(value, (list, tuple)):
            if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
                return [list(value)]
            return [box for nested in value for box in cell_boxes(nested)]
        return []

    # A PDF image object can include a background extending into the next
    # text column. When OCR independently finds a substantial text lane inside
    # that image, evidence outside the physical member or far outside its text
    # lane is available to another region or a raw evidence block.
    def mostly_inside_member(line: dict[str, Any], members: list[list[float]]) -> bool:
        x, y, width, height = line["coordinates"]
        if width <= 0 or height <= 0:
            return False
        for mx, my, mw, mh in members:
            overlap_width = max(0.0, min(x + width, mx + mw) - max(x, mx))
            overlap_height = max(0.0, min(y + height, my + mh) - max(y, my))
            if overlap_width / width >= 0.90 and overlap_height / height >= 0.90:
                return True
        return False

    def near_member(line: dict[str, Any], members: list[list[float]]) -> bool:
        """Keep printed edge labels with their image when the overlap is clear."""
        if mostly_inside_member(line, members):
            return True
        x, y, width, height = line["coordinates"]
        if width <= 0 or height <= 0:
            return False
        for mx, my, mw, mh in members:
            overlap_width = max(0.0, min(x + width, mx + mw) - max(x, mx))
            overlap_height = max(0.0, min(y + height, my + mh) - max(y, my))
            overflow = max(mx - x, x + width - mx - mw,
                           my - y, y + height - my - mh, 0.0)
            if (overflow <= 20 and overlap_width / width >= 0.45
                    and overlap_height / height >= 0.45):
                return True
        return False

    visual_text_lanes: dict[str, tuple[float, float]] = {}
    for region in regions:
        if region.kind != "visual" or region.metadata.get("chart_type_hint"):
            continue
        members = cell_boxes(region.metadata.get("member_coordinates"))
        if not members:
            continue
        supported = [line for line in page_lines
                     if str(line.get("evidence_id", "")).startswith(f"p{region.page:03d}-ocr-")
                     and mostly_inside_member(line, members)
                     and len(str(line.get("text", "")).strip()) >= 3]
        if len(supported) >= 4:
            visual_text_lanes[region.region_id] = (
                min(float(line["coordinates"][0]) for line in supported),
                max(float(line["coordinates"][0]) + float(line["coordinates"][2])
                    for line in supported),
            )

    def score(region: Region, line: dict[str, Any]) -> float:
        area = region.coordinates[2] * region.coordinates[3] / 1_000_000
        value = priority.get(region.kind, 2.0)
        if region.kind == "table":
            cells = cell_boxes(region.metadata.get("cell_coordinates"))
            if cells and any(_contains(box, line) for box in cells):
                value += 2.0
            elif area >= 0.60 and not cells:
                value -= 2.5
        elif region.kind == "visual":
            members = cell_boxes(region.metadata.get("member_coordinates"))
            if members and mostly_inside_member(line, members):
                value += 0.5
            elif members:
                # The visual's padded search box may overlap a neighboring
                # paragraph. Prefer a genuine text region for words outside
                # every physical image member when both claim the line.
                value -= 1.5
        return value

    for line in page_lines:
        excluded_visual_lane = False
        def eligible(region: Region) -> bool:
            nonlocal excluded_visual_lane
            owner_box = region.metadata.get("ownership_coordinates") or region.coordinates
            near_visual_edge = (
                region.region_id in visual_text_lanes
                and near_member(line, cell_boxes(region.metadata.get("member_coordinates")))
                and _contains([
                    owner_box[0] - 20, owner_box[1] - 20,
                    owner_box[2] + 40, owner_box[3] + 40,
                ], line)
            )
            if not _contains(owner_box, line) and not near_visual_edge:
                return False
            if (
                region.region_id in visual_text_lanes
                and (
                    (
                        line.get("evidence_source") == "native_pdf_positioned_word"
                        and (
                            not near_member(
                                line, cell_boxes(region.metadata.get("member_coordinates")),
                            )
                            or not (
                                visual_text_lanes[region.region_id][0] - 25 <= _center(line["coordinates"])[0]
                                <= visual_text_lanes[region.region_id][1] + 25
                            )
                        )
                    ) or (
                        str(line.get("evidence_id", "")).startswith(f"p{region.page:03d}-ocr-")
                        and not near_member(
                            line, cell_boxes(region.metadata.get("member_coordinates")),
                        )
                    )
                )
            ):
                excluded_visual_lane = True
                return False
            return True

        candidates = [
            region for region in regions if eligible(region)
        ]
        if not candidates and excluded_visual_lane:
            lx, ly, lw, lh = line["coordinates"]
            cy = ly + lh / 2
            nearby_text = [region for region in regions if region.kind == "normal_text"
                           and region.coordinates[1] <= cy <= region.coordinates[1] + region.coordinates[3]
                           and lx <= region.coordinates[0] + region.coordinates[2] + 30
                           and lx + lw >= region.coordinates[0] - 30]
            if len(nearby_text) == 1:
                candidates = nearby_text
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda region: (
            -score(region, line),
            region.coordinates[2] * region.coordinates[3],
            -region.confidence, region.region_id,
        ))
        owner = ranked[0]
        if len(ranked) > 1:
            margin = score(ranked[0], line) - score(ranked[1], line)
            ambiguous = margin < 0.25 and ranked[0].kind != ranked[1].kind
            if decisions is not None:
                decisions.append({
                    "evidence_id": line["evidence_id"],
                    "owner_region_id": None if ambiguous else owner.region_id,
                    "ambiguous": ambiguous,
                    "candidates": [
                        {"region_id": region.region_id, "kind": region.kind,
                         "score": score(region, line)} for region in ranked
                    ],
                })
            if ambiguous:
                continue
        result[owner.region_id].append(line)
    return result


def _separate_visual_footnotes(
    inspection: PageInspection, lines_by_region: dict[str, list[dict[str, Any]]],
) -> dict[str, list[list[dict[str, Any]]]]:
    """Keep marked chart notes as text evidence, not discarded chart labels."""
    notes: dict[str, list[list[dict[str, Any]]]] = {}
    marker = re.compile(r"^\s*[*†‡]", re.UNICODE)
    for region in inspection.regions:
        if region.kind != "visual":
            continue
        region_lines = lines_by_region.get(region.region_id, [])
        selected_ids: set[str] = set()
        for line in region_lines:
            value = str(line.get("text", "")).strip()
            if len(value) < 20 or not re.search(r"[A-Za-z]{3}", value):
                continue
            box = line.get("coordinates") or []
            if len(box) != 4 or box[1] < region.coordinates[1] + region.coordinates[3] * 0.66:
                continue
            nearby_marker = next((other for other in region_lines if
                marker.fullmatch(str(other.get("text", "")).strip())
                and abs(other["coordinates"][1] - box[1]) <= 25
                and 0 <= box[0] - (other["coordinates"][0] + other["coordinates"][2]) <= 45
            ), None)
            if not marker.match(value) and nearby_marker is None:
                continue
            members = [line] + ([nearby_marker] if nearby_marker is not None else [])
            notes.setdefault(region.region_id, []).append(members)
            selected_ids.update(str(item["evidence_id"]) for item in members)
        if selected_ids:
            lines_by_region[region.region_id] = [
                line for line in region_lines if str(line["evidence_id"]) not in selected_ids
            ]
    return notes


def _augment_native_visual_evidence(
    inspection: PageInspection, page_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add non-duplicate native positioned words for detected visual regions."""
    result = list(page_lines)
    seen_candidates: set[tuple[str, tuple[float, ...]]] = set()
    native_index = 0
    for region in inspection.regions:
        if region.kind != "visual":
            continue
        candidates = list(region.metadata.get("native_visual_words", []))
        if str(region.metadata.get("chart_type_hint") or "").casefold() == "bar":
            candidates.extend(_reconstruct_vertical_values(candidates))
        candidates.extend(_reconstruct_vertical_words(candidates))
        for candidate in candidates:
            text = str(candidate.get("text", "")).strip()
            coordinates = [float(value) for value in candidate.get("coordinates", [])]
            if not text or len(coordinates) != 4:
                continue
            key = (text.casefold(), tuple(round(value, 2) for value in coordinates))
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            cx, cy = _center(coordinates)
            duplicate = any(
                str(line.get("text", "")).strip().casefold() == text.casefold()
                and abs(cx - _center(line["coordinates"])[0]) <= 15
                and abs(cy - _center(line["coordinates"])[1]) <= 45
                for line in result
            )
            if duplicate:
                continue
            native_index += 1
            result.append({
                "evidence_id": f"p{inspection.page:03d}-native-{native_index:04d}",
                "text": text, "confidence": 1.0, "coordinates": coordinates,
                "evidence_source": "native_pdf_positioned_word",
            })
    return result


def _augment_native_table_evidence(
    inspection: PageInspection, page_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = list(page_lines)
    for region in inspection.regions:
        if region.kind != "table":
            continue
        if region.metadata.get("rows_source") == "paddle":
            continue
        for word in region.metadata.get("native_panel_words") or []:
            result.append({
                "evidence_id": word["evidence_id"],
                "text": word["text"],
                "confidence": 1.0,
                "coordinates": word["coordinates"],
                "evidence_source": "native_pdf_positioned_panel_word",
            })
        rows = region.metadata.get("rows") or []
        coordinates = region.metadata.get("cell_coordinates") or []
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                text = str(value or "").strip()
                cell_coordinates = (
                    coordinates[row_index][column_index]
                    if row_index < len(coordinates) and column_index < len(coordinates[row_index])
                    else None
                )
                if not text or not cell_coordinates:
                    continue
                result.append({
                    "evidence_id": f"{region.region_id}-native-table-r{row_index:03d}-c{column_index:03d}",
                    "text": text, "confidence": 1.0, "coordinates": cell_coordinates,
                    "evidence_source": "native_pdf_table_cell",
                })
    return result


def _reconstruct_vertical_values(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reassemble rotated numeric labels split by native PDF extraction.

    Some PDF generators encode a rotated percentage as vertically stacked
    number fragments plus a percent sign. Geometry, rather than a value list,
    list, determines the reconstructed label.
    """
    tokens = [
        item for item in candidates
        if re.fullmatch(r"[\d.%]+", str(item.get("text", "")).strip())
        and len(item.get("coordinates", [])) == 4
    ]
    by_x: list[list[dict[str, Any]]] = []
    for token in sorted(tokens, key=lambda item: (_center(item["coordinates"])[0], item["coordinates"][1])):
        cx, _ = _center(token["coordinates"])
        group = next((group for group in by_x if abs(_center(group[0]["coordinates"])[0] - cx) <= 4), None)
        if group is None:
            by_x.append([token])
        else:
            group.append(token)
    reconstructed = []
    for x_group in by_x:
        clusters: list[list[dict[str, Any]]] = []
        for token in sorted(x_group, key=lambda item: item["coordinates"][1]):
            if not clusters:
                clusters.append([token])
                continue
            previous = clusters[-1][-1]["coordinates"]
            gap = token["coordinates"][1] - (previous[1] + previous[3])
            if gap <= 45:
                clusters[-1].append(token)
            else:
                clusters.append([token])
        for cluster in clusters:
            texts = [str(item["text"]).strip() for item in cluster]
            if "%" not in texts or not any(any(char.isdigit() for char in text) for text in texts):
                continue
            ordered = sorted(cluster, key=lambda item: item["coordinates"][1], reverse=True)
            value = "".join(
                str(item["text"])[::-1] if str(item["text"]).isdigit() else str(item["text"])
                for item in ordered
            )
            if not re.fullmatch(r"\d{1,3}(?:\.\d+)?%", value):
                continue
            reconstructed.append({
                "text": value, "coordinates": _box_union(*(item["coordinates"] for item in cluster)),
                "confidence": 1.0, "evidence_source": "native_pdf_rotated_value_reconstruction",
            })
    return reconstructed


def _reconstruct_vertical_words(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reassemble vertically encoded units and labels from positioned PDF tokens."""
    tokens = [
        item for item in candidates
        if re.fullmatch(r"[A-Za-z]{1,3}", str(item.get("text", "")).strip())
        and len(item.get("coordinates", [])) == 4
    ]
    x_groups: list[list[dict[str, Any]]] = []
    for token in sorted(tokens, key=lambda item: (_center(item["coordinates"])[0], item["coordinates"][1])):
        cx, _ = _center(token["coordinates"])
        group = next((items for items in x_groups if abs(_center(items[0]["coordinates"])[0] - cx) <= 4), None)
        if group is None:
            x_groups.append([token])
        else:
            group.append(token)
    reconstructed: list[dict[str, Any]] = []
    for x_group in x_groups:
        clusters: list[list[dict[str, Any]]] = []
        for token in sorted(x_group, key=lambda item: item["coordinates"][1]):
            if not clusters:
                clusters.append([token])
                continue
            previous = clusters[-1][-1]["coordinates"]
            gap = token["coordinates"][1] - (previous[1] + previous[3])
            if gap <= 12:
                clusters[-1].append(token)
            else:
                clusters.append([token])
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            text = "".join(str(item["text"]).strip() for item in cluster)
            if not 3 <= len(text) <= 16:
                continue
            reconstructed.append({
                "text": text,
                "coordinates": _box_union(*(item["coordinates"] for item in cluster)),
                "confidence": 1.0,
                "evidence_source": "native_pdf_vertical_word_reconstruction",
            })
    return reconstructed


def _split_visual_text_lines(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split image-backed text at clear column gutters and vertical whitespace."""
    if not lines:
        return []

    # A PDF may store an entire two-column page as one image. Vertical-only
    # segmentation then joins two unrelated paragraphs on the same baseline.
    # Require a substantial empty gutter across both OCR lines and native words.
    meaningful = [line for line in lines if len(str(line.get("text", "")).strip()) >= 2]
    columns: list[list[dict[str, Any]]] = [lines]
    if len(meaningful) >= 6:
        best: tuple[float, float] | None = None
        for cut in range(300, 701, 10):
            left = [line for line in meaningful if float(line["coordinates"][0])
                    + float(line["coordinates"][2]) <= cut - 12]
            right = [line for line in meaningful if float(line["coordinates"][0]) >= cut + 12]
            crossing = len(meaningful) - len(left) - len(right)
            if min(len(left), len(right)) < 3 or crossing > max(1, len(meaningful) // 20):
                continue
            score = min(len(left), len(right)) - abs(cut - 500) / 1000
            if best is None or score > best[0]:
                best = (score, float(cut))
        if best is not None:
            cut = best[1]
            columns = [
                [line for line in lines if _center(line["coordinates"])[0] < cut],
                [line for line in lines if _center(line["coordinates"])[0] >= cut],
            ]

    groups: list[list[dict[str, Any]]] = []
    for column in columns:
        ordered = sorted(column, key=lambda line: (
            float(line["coordinates"][1]), float(line["coordinates"][0]),
        ))
        if not ordered:
            continue
        heights = [max(1.0, float(line["coordinates"][3])) for line in ordered]
        median_height = statistics.median(heights)
        groups.append([ordered[0]])
        previous_bottom = float(ordered[0]["coordinates"][1]) + float(ordered[0]["coordinates"][3])
        for line in ordered[1:]:
            top = float(line["coordinates"][1])
            gap = top - previous_bottom
            if gap > max(10.0, median_height * 0.72):
                groups.append([line])
            else:
                groups[-1].append(line)
            previous_bottom = max(previous_bottom, top + float(line["coordinates"][3]))
    return groups

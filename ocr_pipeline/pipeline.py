from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
from functools import lru_cache
import difflib
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any
import unicodedata

from PIL import Image
from pypdf import PdfReader

from .comparison_layout import reconstruct_comparison_panel
from .inspection import inspect_pdf
from .map_geometry import ATLAS_URL, colored_mark_for_value, register_us_state_map, state_for_value
from .models import PageInspection, Region, SourceBlock, validate_source_blocks
from .source_observations import is_layout_glyph, represented_in_content
from .workers import QwenVisionClient, run_paddle_table_worker, run_rapidocr_worker


NUMBER = re.compile(r"(?:[$€£]\s*)?\(?\d[\d,.]*(?:\s*(?:%|x|million|billion|bn|mm|m|b|k))?\)?", re.I)
_TABLE_SCALAR = re.compile(
    r"\s*[+-]?(?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|x|million|billion|bn|mm|m|b|k)?\s*", re.I,
)
MAP_TERMS = {"geographic", "geography", "distribution by state", "map"}
CHART_TERMS = {"chart", "graph", "plot", "series", "axis", "legend"}
US_GEOGRAPHIES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
    "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
    "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
    "puerto rico",
}
PIPELINE_VERSION = "0.1.0"
PAGE_SCHEMA_VERSION = "unified-source-page/4.0"
COLLECTION_SCHEMA_VERSION = "unified-source-collection/3.0"
INSPECTION_SCHEMA_VERSION = "pdf-page-inspection/1.0"
INSPECTION_INDEX_SCHEMA_VERSION = "pdf-inspection-index/1.0"
VISION_DIAGNOSTIC_SCHEMA_VERSION = "opencv-region-diagnostic/1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _vision_summary(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "horizontal_line_count": int(features.get("horizontal_lines", 0)),
        "vertical_line_count": int(features.get("vertical_lines", 0)),
        "diagonal_line_count": int(features.get("diagonal_lines", 0)),
        "rectangle_candidate_count": int(features.get("rectangle_candidates", 0)),
        "bar_candidate_count": int(features.get("bar_candidates", 0)),
        "point_candidate_count": int(features.get("point_candidates", 0)),
        "legend_swatch_candidate_count": int(features.get("legend_swatch_candidates", 0)),
        "horizontal_axis_candidate_count": int(features.get("horizontal_axis_candidates", 0)),
        "vertical_axis_candidate_count": int(features.get("vertical_axis_candidates", 0)),
        "plot_boundary_detected": bool(features.get("plot_boundary_detected", False)),
        "table_grid_confidence": round(float(features.get("table_grid_confidence", 0.0)), 5),
        "saturated_pixel_fraction": round(float(features.get("saturated_pixel_fraction", 0.0)), 5),
    }


def _write_vision_diagnostic(
    diagnostics: Path, document_id: str, source_hash: str, region: Region,
    features: dict[str, Any],
) -> dict[str, str]:
    path = diagnostics / f"{region.region_id}.json"
    payload = {
        "schema_version": VISION_DIAGNOSTIC_SCHEMA_VERSION,
        "document_id": document_id,
        "source_sha256": source_hash,
        "page": region.page,
        "region_id": region.region_id,
        "region_coordinates": region.coordinates,
        "coordinate_formats": {
            "region_coordinates": "page-relative [x, y, width, height], normalized 0..1000",
            "feature_boxes": "region-relative [x, y, width, height], normalized 0..1000",
            "line_segments": "region-relative [x1, y1, x2, y2], normalized 0..1000",
        },
        "vision_summary": _vision_summary(features),
        "features": features,
    }
    _write_json(path, payload)
    return {
        "path": str(path.relative_to(diagnostics.parents[1])).replace("\\", "/"),
        "sha256": _sha256(path),
    }


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "document"


def parse_pages(spec: str | None, page_count: int) -> list[int]:
    if not spec:
        return list(range(1, page_count + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    invalid = sorted(page for page in pages if page < 1 or page > page_count)
    if invalid:
        raise ValueError(f"Pages outside 1..{page_count}: {invalid}")
    return sorted(pages)


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


def _contains(coordinates: list[float], line: dict[str, Any]) -> bool:
    x, y, w, h = coordinates
    lx, ly, lw, lh = line["coordinates"]
    cx, cy = lx + lw / 2, ly + lh / 2
    return x <= cx <= x + w and y <= cy <= y + h


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
            if members and any(_contains(box, line) for box in members):
                value += 0.5
        return value

    for line in page_lines:
        candidates = [
            region for region in regions
            if _contains(region.metadata.get("ownership_coordinates") or region.coordinates, line)
        ]
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


_SINGLE_CARD_VALUE = re.compile(
    r"[$€£]?\s*\d[\d,.]*(?:\s*[-–—]\s*\d[\d,.]*)?\s*(?:%|％|x)?",
    re.I,
)


def _claim_single_value_cards(
    regions: list[Region], page_lines: list[dict[str, Any]],
    lines_by_region: dict[str, list[dict[str, Any]]],
) -> set[str]:
    """Join a printed KPI value with its nearby caption across PDF regions.

    Native PDF table detection often isolates the large value as a 2x2 grid.
    A card is accepted only with one distinct OCR value and one nearby caption
    in its horizontal lane. Ranges retain their printed form, not a made-up
    exact number.
    """
    absorbed: set[str] = set()
    for region in regions:
        if region.kind != "table":
            continue
        rows = region.metadata.get("rows") or []
        if region.metadata.get("rows_source") == "paddle":
            continue
        if not rows or len(rows) > 4 or max((len(row) for row in rows), default=0) > 2:
            continue
        owned = lines_by_region.get(region.region_id, [])
        values = [line for line in owned if _SINGLE_CARD_VALUE.fullmatch(str(line.get("text", "")).strip())]
        if len(values) != 1:
            continue
        value = values[0]
        vx, vy, vw, vh = value["coordinates"]
        value_y = vy + vh / 2
        rx, _, rw, _ = region.coordinates
        captions = [
            line for line in page_lines
            if line["evidence_id"] != value["evidence_id"]
            and 3 <= len(str(line.get("text", "")).strip()) <= 90
            and not _SINGLE_CARD_VALUE.fullmatch(str(line.get("text", "")).strip())
            and rx - 25 <= line["coordinates"][0] + line["coordinates"][2] / 2 <= rx + rw + 25
            and 10 <= line["coordinates"][1] + line["coordinates"][3] / 2 - value_y <= 180
        ]
        if not captions:
            continue
        caption = min(captions, key=lambda line: (
            line["coordinates"][1] - vy,
            abs(line["coordinates"][0] + line["coordinates"][2] / 2 - (rx + rw / 2)),
        ))
        for owner_id, owner_lines in lines_by_region.items():
            if owner_id != region.region_id and caption in owner_lines:
                owner_lines.remove(caption)
                if not owner_lines:
                    owner_region = next((item for item in regions if item.region_id == owner_id), None)
                    if owner_region and owner_region.kind == "normal_text":
                        absorbed.add(owner_id)
                break
        if caption not in owned:
            owned.append(caption)
        raw = str(value["text"]).strip()
        numeric_value, unit, normalized_value = _numeric_value(raw)
        if re.search(r"\d\s*[-–—]\s*\d", raw):
            numeric_value = normalized_value = None
            unit = "percent" if "%" in raw or "％" in raw else "multiple" if raw.lower().endswith("x") else None
        region.metadata["single_card_binding"] = {
            "label": str(caption["text"]).strip(), "value": raw,
            "numeric_value": numeric_value, "unit": unit, "normalized_value": normalized_value,
            "label_evidence_id": caption["evidence_id"], "value_evidence_id": value["evidence_id"],
            "label_coordinates": caption["coordinates"], "value_coordinates": value["coordinates"],
            "visual_mark_coordinates": region.coordinates,
            "grounding_method": "unique OCR value and caption in one printed KPI card",
            "confidence": 0.78,
        }
    return absorbed


def _absorb_structured_visual_text(
    inspection: PageInspection, lines_by_region: dict[str, list[dict[str, Any]]],
    vision_plan: dict[str, Any] | None,
) -> set[str]:
    """Move exact chart label/value islands into their containing visual.

    PDF text layers can make an individual pie label a separate text region.
    Only a unique, exact model item with two unique OCR lines can cross that
    ownership boundary; general prose and titles remain independent blocks.
    """
    absorbed: set[str] = set()
    if not vision_plan:
        return absorbed

    def norm(value: str) -> str:
        return re.sub(r"\W+", "", value.casefold(), flags=re.UNICODE)

    for visual in inspection.regions:
        hint = visual.metadata.get("vision_plan") or {}
        index = hint.get("block_index")
        if visual.kind != "visual" or hint.get("type") not in {"chart", "kpi_panel"}:
            continue
        if not isinstance(index, int) or index >= len(vision_plan.get("blocks", [])):
            continue
        items = vision_plan["blocks"][index].get("items", [])
        for text_region in inspection.regions:
            if text_region.kind != "normal_text" or text_region.region_id in absorbed:
                continue
            tx, ty, tw, th = text_region.coordinates
            vx, vy, vw, vh = visual.coordinates
            if not (vx - 2 <= tx and vy - 2 <= ty and tx + tw <= vx + vw + 2 and ty + th <= vy + vh + 2):
                continue
            island_lines = lines_by_region.get(text_region.region_id, [])
            if len(island_lines) != 2:
                continue
            island_texts = {norm(str(line.get("text", ""))) for line in island_lines}
            matching = [item for item in items if {
                norm(str(item.get("label", ""))), norm(str(item.get("value", "")))
            } == island_texts and len(island_texts) == 2]
            if len(matching) != 1:
                continue
            combined = lines_by_region.get(visual.region_id, []) + island_lines
            label, value = matching[0]["label"], matching[0]["value"]
            if sum(norm(str(line.get("text", ""))) == norm(label) for line in combined) != 1:
                continue
            if sum(norm(str(line.get("text", ""))) == norm(value) for line in combined) != 1:
                continue
            lines_by_region[visual.region_id] = combined
            lines_by_region[text_region.region_id] = []
            absorbed.add(text_region.region_id)
    return absorbed


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


def _crop(image_path: Path, coordinates: list[float], output_path: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
        x, y, w, h = coordinates
        box = (
            max(0, round(x * width / 1000)), max(0, round(y * height / 1000)),
            min(width, round((x + w) * width / 1000)), min(height, round((y + h) * height / 1000)),
        )
        image.crop(box).save(output_path)


def _render_pages(pdf: Path, pages: list[int], output: Path, dpi: int, pdftoppm: Path | None) -> dict[int, Path]:
    executable = pdftoppm or (Path(shutil.which("pdftoppm")) if shutil.which("pdftoppm") else None)
    if executable is None or not executable.exists():
        raise RuntimeError("pdftoppm was not found; pass --pdftoppm with a Poppler executable")
    rendered = {}
    for page in pages:
        target = output / f"page-{page:03d}"
        subprocess.run([
            str(executable), "-f", str(page), "-l", str(page), "-singlefile",
            "-png", "-r", str(dpi), str(pdf), str(target),
        ], check=True, capture_output=True)
        rendered[page] = target.with_suffix(".png")
    return rendered


def _ocr_text(lines: list[dict[str, Any]]) -> str:
    # Image-backed PDF text may appear twice: RapidOCR contributes a full
    # phrase while positioned native words contribute each word separately.
    # Keep both in provenance, but render an overlapping word only once.
    phrases = [line for line in lines if
        "-native-" not in str(line.get("evidence_id", ""))
        and len(str(line.get("text", ""))) >= 20]

    def tokens(value: str) -> list[str]:
        return re.findall(r"\w+", _fold_token(value))

    def covered(native: dict[str, Any]) -> bool:
        needle = tokens(str(native.get("text", "")))
        if not needle:
            return False
        nx, ny = _center(native["coordinates"])
        for phrase in phrases:
            px, py, pw, ph = phrase["coordinates"]
            if not (px - 15 <= nx <= px + pw + 15 and py - 18 <= ny <= py + ph + 18):
                continue
            words = tokens(str(phrase.get("text", "")))
            if any(words[start:start + len(needle)] == needle for start in range(len(words) - len(needle) + 1)):
                return True
        return False

    visible = [line for line in lines if not (
        "-native-" in str(line.get("evidence_id", "")) and covered(line)
    )]
    ordered = sorted(visible, key=lambda line: (line["coordinates"][1], line["coordinates"][0]))
    return "\n".join(str(line["text"]) for line in ordered)


def _token_agreement(left: str, right: str) -> float | None:
    left_tokens = set(re.findall(r"\w+", left.casefold()))
    right_tokens = set(re.findall(r"\w+", right.casefold()))
    if not left_tokens or not right_tokens:
        return None
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 5)


def _fold_token(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in value if not unicodedata.combining(character))


def _hybrid_text(native: str, ocr: str) -> str:
    """Use complete OCR wording while repairing OCR spelling from native PDF tokens."""
    native_words = re.findall(r"[\w’'-]+", native, re.UNICODE)
    by_folded: dict[str, list[str]] = {}
    for word in native_words:
        by_folded.setdefault(_fold_token(word), []).append(word)

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        exact = by_folded.get(_fold_token(word))
        if exact:
            return exact[0]
        candidates = difflib.get_close_matches(_fold_token(word), by_folded, n=1, cutoff=0.90)
        return by_folded[candidates[0]][0] if candidates else word

    return re.sub(r"[\w’'-]+", replace, ocr, flags=re.UNICODE)


def _text_structure(raw: str) -> dict[str, Any] | None:
    """Recover paragraph and bullet semantics from line breaks without document vocabulary."""
    normalized_paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", raw)
        if paragraph.strip()
    ]
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines() if line.strip()]
    # Symbol fonts commonly expose bullets/arrows as Unicode private-use
    # characters. At the start of a line they carry list structure, not text.
    lines = [re.sub(r"^[\uF000-\uF8FF]\s*", "• ", line) for line in lines]
    bullet_indexes = [index for index, line in enumerate(lines) if re.match(r"^(?:[-•▪‣]|\d+[.)])\s+", line)]
    if not bullet_indexes:
        if len(normalized_paragraphs) <= 1:
            return None
        return {"paragraphs": normalized_paragraphs, "lists": []}
    first = bullet_indexes[0]
    intro_index = first - 1 if first and lines[first - 1].endswith(":") else None
    prose_lines = lines[:intro_index if intro_index is not None else first]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in prose_lines:
        current.append(line)
        if re.search(r"[.!?][\"'’)]?$", line):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    items = [
        re.sub(r"^(?:[-•▪‣]|\d+[.)])\s+", "", lines[index]).strip()
        for index in bullet_indexes
    ]
    return {
        "paragraphs": paragraphs,
        "lists": [{
            "intro": lines[intro_index] if intro_index is not None else None,
            "ordered": bool(re.match(r"^\d+[.)]\s+", lines[first])),
            "items": items,
        }],
    }


def _classify_visual(region: Region, lines: list[dict[str, Any]], features: dict[str, Any]) -> tuple[str, float, list[str]]:
    text = " ".join(str(line["text"]) for line in lines).lower()
    warnings: list[str] = []
    visual_hint = str(region.metadata.get("visual_hint") or "").strip().lower()
    if visual_hint == "chart":
        return "chart", max(0.80, float(region.confidence)), warnings
    if visual_hint == "photograph":
        return "photograph", max(0.82, float(region.confidence)), warnings
    map_term_hits = sum(term in text for term in MAP_TERMS)
    geography_hits = sum(
        len(str(line.get("text", "")).split()) <= 5
        and any(re.search(rf"\b{re.escape(name)}\b", str(line.get("text", "")), re.I) for name in US_GEOGRAPHIES)
        for line in lines
    )
    chart_term_hits = sum(term in text for term in CHART_TERMS)
    horizontal = int(features.get("horizontal_lines", 0))
    vertical = int(features.get("vertical_lines", 0))
    rectangles = int(features.get("rectangle_candidates", 0))
    bars = int(features.get("bar_candidates", 0))
    points = int(features.get("point_candidates", 0))
    swatches = int(features.get("legend_swatch_candidates", 0))
    horizontal_axes = int(features.get("horizontal_axis_candidates", 0))
    vertical_axes = int(features.get("vertical_axis_candidates", 0))
    plot_boundary = bool(features.get("plot_boundary_detected", False))
    table_grid_confidence = float(features.get("table_grid_confidence", 0.0))
    numeric = sum(bool(NUMBER.search(str(line["text"]))) for line in lines)
    axis_ticks = sum(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", str(line.get("text", "")).strip())) for line in lines)
    # A photograph or map collage can contain many apparent grid lines. A
    # visual table also needs a substantial density of printed numeric cells.
    numeric_density = numeric / max(1, len(lines))
    if table_grid_confidence >= 0.60 and horizontal >= 4 and vertical >= 3 and numeric_density >= 0.35:
        return "table", 0.76, warnings
    if geography_hits >= 2 and numeric >= 2:
        return "map", min(0.94, 0.72 + geography_hits * 0.04), warnings
    place_pattern = re.compile(r"\b(?:river|bay|district|street|road|hill|peninsula|lake|county)\b", re.I)
    printed_place_labels = sum(
        bool(place_pattern.search(str(line.get("text", ""))))
        for line in lines
    )
    if printed_place_labels >= 4 and points >= 8 and numeric_density < 0.35:
        return "map", 0.78, warnings
    # KPI cards contain prominent values and labels but no axes, marks, or plot
    # boundary. Detecting them separately prevents business vocabulary from
    # turning a metric panel into a chart.
    if _infer_chart_type(region, lines, features, None) == "kpi_panel":
        return "kpi_panel", 0.82, warnings
    axis_pair = horizontal_axes >= 1 and vertical_axes >= 1
    repeated_marks = bars >= 3 or (points >= 8 and axis_pair and axis_ticks >= 3)
    legend_marks = swatches >= 2 and (bars >= 2 or points >= 3)
    if numeric >= 2 and (repeated_marks or legend_marks or (plot_boundary and axis_pair and axis_ticks >= 3)):
        return "chart", 0.78, warnings
    prose_lines = sum(len(str(line.get("text", "")).split()) >= 4 for line in lines)
    if len(lines) >= 10 and prose_lines >= 5 and numeric_density < 0.20 and printed_place_labels < 3:
        return "normal_text", 0.75, ["visual page has substantial OCR prose without a supported data plot"]
    if numeric >= 2 and (map_term_hits or chart_term_hits):
        warnings.append("semantic wording requires visual-model confirmation; no structural plot evidence found")
    elif (horizontal + vertical + rectangles) >= 8 and numeric >= 2:
        warnings.append("ambiguous geometry requires visual-model confirmation")
    if len(lines) >= 3 and numeric == 0:
        return "normal_text", 0.62, ["image region treated as scanned text"]
    warnings.append("visual type could not be classified confidently")
    return "unclassified_visual", 0.35, warnings


def _block_type_for_text(region: Region, candidate_text: str | None = None) -> str:
    text = clean_text(region.native_text if candidate_text is None else candidate_text)
    font = float(region.metadata.get("median_font_size", 0) or 0)
    word_count = int(region.metadata.get("word_count", len(text.split())) or 0)
    y = region.coordinates[1]
    if y >= 890 or (font and font <= 8 and y >= 820):
        return "footnote"
    if "@" in text or re.search(r"\b\d{3}[-.) ]\d{3}[- ]\d{4}\b", text):
        return "contact"
    if word_count <= 14 and not text.endswith((".", ",", ";", ":")) and (font >= 14 or text.isupper() or text.istitle()):
        return "heading"
    return "text"


def _semantic_role_for_text(
    block_type: str, text: str, coordinates: list[float], page_number: int, nested: bool = False,
) -> str:
    folded = text.casefold()
    legal_signals = {
        "confidential", "important note", "legal notice", "disclaimer", "offer to sell",
        "solicitation", "securities", "offering materials", "terms and conditions",
    }
    if sum(signal in folded for signal in legal_signals) >= 2:
        return "legal_notice"
    if block_type == "heading":
        if nested:
            return "panel_heading"
        if coordinates[1] <= 220:
            return "document_title" if page_number == 1 else "page_title"
        return "section_heading"
    if block_type == "footnote":
        return "footnote"
    if block_type == "contact":
        return "contact_information"
    return "body_text"


def _line_box(lines: list[dict[str, Any]]) -> list[float]:
    x0 = min(float(line["coordinates"][0]) for line in lines)
    y0 = min(float(line["coordinates"][1]) for line in lines)
    x1 = max(float(line["coordinates"][0]) + float(line["coordinates"][2]) for line in lines)
    y1 = max(float(line["coordinates"][1]) + float(line["coordinates"][3]) for line in lines)
    return [x0, y0, x1 - x0, y1 - y0]


def _split_visual_text_lines(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split image-backed text using vertical whitespace, independent of document wording."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda line: (float(line["coordinates"][1]), float(line["coordinates"][0])))
    heights = [max(1.0, float(line["coordinates"][3])) for line in ordered]
    median_height = statistics.median(heights)
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
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


def _looks_like_brand_mark(
    lines: list[dict[str, Any]], page_lines: list[dict[str, Any]],
    document_token_pages: Counter[str] | None = None,
) -> bool:
    """Recognize short logo text from repetition or strong page-corner placement."""
    if not 1 <= len(lines) <= 3:
        return False
    text = " ".join(str(line.get("text", "")) for line in lines).strip()
    tokens = {
        _fold_token(token) for token in re.findall(r"[^\W\d_]{2,}", text, re.UNICODE)
        if _fold_token(token)
    }
    if not 1 <= len(tokens) <= 8 or NUMBER.search(text):
        return False
    letters = [character for character in text if character.isalpha()]
    if not letters or sum(character.isupper() for character in letters) / len(letters) < 0.72:
        return False
    own_ids = {str(line.get("evidence_id")) for line in lines}
    elsewhere = " ".join(
        str(line.get("text", "")) for line in page_lines
        if str(line.get("evidence_id")) not in own_ids
    ).casefold()
    repeated = sum(bool(re.search(rf"\b{re.escape(token)}\b", elsewhere)) for token in tokens)
    page_repetition = repeated / len(tokens) >= 0.75
    document_repetition = bool(document_token_pages) and (
        sum(document_token_pages.get(token, 0) >= 2 for token in tokens) / len(tokens) >= 0.75
    )
    x, y, width, height = _line_box(lines)
    in_vertical_corner_band = y >= 800 or y + height <= 200
    in_horizontal_corner_band = x <= 400 or x + width >= 600
    corner_signature = width <= 360 and height <= 140 and in_vertical_corner_band and in_horizontal_corner_band
    return page_repetition or document_repetition or corner_signature


def _inline_heading_subsection_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, native_threshold: float,
) -> list[SourceBlock] | None:
    """Split repeated inline all-caps labels into semantic subsection trees."""
    native_lines = [line.strip() for line in region.native_text.splitlines() if line.strip()]
    heading_pattern = re.compile(r"^([A-Z][A-Z0-9 &/'-]{2,}?):\s*(.*)$")
    headings = [
        (index, match.group(1).strip(), match.group(2).strip())
        for index, line in enumerate(native_lines)
        if (match := heading_pattern.match(line))
    ]
    # Repetition is the structural evidence. A lone colon-led line can be ordinary prose.
    if len(headings) < 2:
        return None
    ordered_ocr = sorted(lines, key=lambda line: (float(line["coordinates"][1]), float(line["coordinates"][0])))
    if len(ordered_ocr) != len(native_lines):
        return None

    positive_steps = [
        float(current["coordinates"][1]) - float(previous["coordinates"][1])
        for previous, current in zip(ordered_ocr, ordered_ocr[1:])
        if float(current["coordinates"][1]) > float(previous["coordinates"][1])
    ]
    typical_step = statistics.median(positive_steps) if positive_steps else 0.0
    blocks: list[SourceBlock] = []
    for position, (start, heading_text, first_body_text) in enumerate(headings, 1):
        end = headings[position][0] if position < len(headings) else len(native_lines)
        section_ocr = [dict(line) for line in ordered_ocr[start:end]]
        if not section_ocr:
            return None
        body_lines = [first_body_text, *native_lines[start + 1:end]]
        body_parts: list[str] = []
        for line_index, body_line in enumerate(body_lines):
            if line_index and typical_step:
                previous = section_ocr[line_index - 1]
                current = section_ocr[line_index]
                step = float(current["coordinates"][1]) - float(previous["coordinates"][1])
                if step > typical_step * 1.35:
                    body_parts.append("")
            body_parts.append(body_line)
        body_native = "\n".join(body_parts).strip()
        if not body_native:
            return None

        first_ocr = str(section_ocr[0].get("text", ""))
        ocr_match = heading_pattern.match(first_ocr)
        ocr_heading = ocr_match.group(1).strip() if ocr_match else heading_text
        section_ocr[0]["text"] = ocr_match.group(2).strip() if ocr_match else first_ocr
        section_box = _line_box(section_ocr)
        first_box = [float(value) for value in ordered_ocr[start]["coordinates"]]
        heading_fraction = min(0.72, max(0.12, (len(heading_text) + 1) / max(1, len(first_ocr))))
        heading_box = [first_box[0], first_box[1], first_box[2] * heading_fraction, first_box[3]]
        group_id = f"{region.region_id}-subsection-{position:03d}"
        heading_id = f"{group_id}-heading"
        body_region = Region(
            region_id=f"{group_id}-body", page=region.page, kind="normal_text",
            coordinates=section_box, reading_order=region.reading_order + position,
            classification_method="inline-heading subsection reconstruction",
            confidence=region.confidence, native_text=body_native,
            metadata={
                "word_count": len(re.findall(r"\w+", body_native, re.UNICODE)),
                "source_bbox_points": region.metadata.get("source_bbox_points"),
            },
        )
        body = _text_block(
            document_id, source_hash, inspection, body_region, section_ocr, image, native_threshold,
        )
        body.parent_block_id = group_id
        body.hierarchy_depth = 1
        heading_agreement = _token_agreement(heading_text, ocr_heading)
        heading_confidence = min(
            inspection.native_text_quality,
            0.5 + heading_agreement / 2 if heading_agreement is not None else region.confidence,
        )
        heading = SourceBlock(
            document_id=document_id, type="heading", page=region.page, block_id=heading_id,
            content={
                "text": heading_text,
                "evidence_text": {
                    "selected": "native", "native": heading_text, "ocr": ocr_heading,
                    "token_agreement": heading_agreement,
                },
            },
            coordinates=heading_box,
            extraction_method=["native PDF text", "Python inline-heading reconstruction"],
            confidence=heading_confidence, validation_status="passed",
            provenance=_provenance(source_hash, body_region, [], image),
            semantic_role="section_heading", parent_block_id=group_id,
            hierarchy_depth=1, heading_level=2,
        )
        group = SourceBlock(
            document_id=document_id, type="group", page=region.page, block_id=group_id,
            content={"role": "subsection", "child_block_ids": [heading_id, body.block_id]},
            coordinates=section_box,
            extraction_method=["Python inline-heading hierarchy"],
            confidence=min(heading.confidence, body.confidence),
            validation_status="passed" if body.validation_status == "passed" else "needs_review",
            warnings=[] if body.validation_status == "passed" else ["one or more child blocks require review"],
            provenance=_provenance(source_hash, body_region, [], image),
            semantic_role="document_subsection", child_block_ids=[heading_id, body.block_id],
        )
        blocks.extend([group, heading, body])
    return blocks


def _profile_biography_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, native_threshold: float,
) -> list[SourceBlock] | None:
    """Split a name-and-role lead line from the biography that follows it."""
    native_lines = [line.strip() for line in region.native_text.splitlines() if line.strip()]
    if len(native_lines) < 2 or len(" ".join(native_lines[1:]).split()) < 20:
        return None
    rx, ry, rw, rh = region.coordinates
    adjacent_portrait = any(
        candidate.kind == "visual"
        and str(candidate.metadata.get("visual_hint") or "").casefold() == "photograph"
        and (
            max(0.0, min(ry + rh, candidate.coordinates[1] + candidate.coordinates[3])
                - max(ry, candidate.coordinates[1]))
            / max(1.0, min(rh, candidate.coordinates[3]))
        ) >= 0.45
        and -20 <= rx - (candidate.coordinates[0] + candidate.coordinates[2]) <= 250
        for candidate in inspection.regions
    )
    if not adjacent_portrait:
        return None
    heading_text = native_lines[0]
    if not re.fullmatch(r"[^.!?:]{2,80}\s+[–—-]\s+[^.!?:]{2,60}", heading_text):
        return None
    if not 2 <= len(heading_text.split()) <= 12:
        return None
    ordered_ocr = sorted(lines, key=lambda line: (float(line["coordinates"][1]), float(line["coordinates"][0])))
    if len(ordered_ocr) < 2:
        return None
    heading_ocr = ordered_ocr[0]
    body_ocr = ordered_ocr[1:]
    body_native = "\n".join(native_lines[1:])
    group_id = f"{region.region_id}-profile-biography"
    heading_id = f"{group_id}-heading"
    body_region = Region(
        region_id=f"{group_id}-body", page=region.page, kind="normal_text",
        coordinates=_line_box(body_ocr), reading_order=region.reading_order + 1,
        classification_method="name-role biography reconstruction",
        confidence=region.confidence, native_text=body_native,
        metadata={
            "word_count": len(re.findall(r"\w+", body_native, re.UNICODE)),
            "source_bbox_points": region.metadata.get("source_bbox_points"),
        },
    )
    body = _text_block(
        document_id, source_hash, inspection, body_region, body_ocr, image, native_threshold,
    )
    body.parent_block_id = group_id
    body.hierarchy_depth = 1
    heading_ocr_text = str(heading_ocr.get("text", "")).strip()
    agreement = _token_agreement(heading_text, heading_ocr_text)
    heading = SourceBlock(
        document_id=document_id, type="heading", page=region.page, block_id=heading_id,
        content={
            "text": heading_text,
            "evidence_text": {
                "selected": "native", "native": heading_text, "ocr": heading_ocr_text or None,
                "token_agreement": agreement,
            },
        },
        coordinates=[float(value) for value in heading_ocr["coordinates"]],
        extraction_method=["native PDF text", "RapidOCR", "Python name-role heading reconstruction"],
        confidence=min(inspection.native_text_quality, 0.5 + agreement / 2 if agreement is not None else region.confidence),
        validation_status="passed", provenance=_provenance(source_hash, region, [heading_ocr], image),
        semantic_role="profile_name_and_role", parent_block_id=group_id,
        hierarchy_depth=1, heading_level=2,
    )
    group = SourceBlock(
        document_id=document_id, type="group", page=region.page, block_id=group_id,
        content={"role": "profile_biography", "child_block_ids": [heading_id, body.block_id]},
        coordinates=region.coordinates, extraction_method=["Python profile-biography hierarchy"],
        confidence=min(heading.confidence, body.confidence),
        validation_status="passed" if body.validation_status == "passed" else "needs_review",
        warnings=[] if body.validation_status == "passed" else ["one or more child blocks require review"],
        provenance=_provenance(source_hash, region, [], image), semantic_role="document_subsection",
        child_block_ids=[heading_id, body.block_id],
    )
    return [group, heading, body]


def _leading_heading_body_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, native_threshold: float,
) -> list[SourceBlock] | None:
    """Split a visually separated lead heading from the text that follows."""
    native_lines = [line.strip() for line in region.native_text.splitlines() if line.strip()]
    if len(native_lines) < 2:
        return None
    heading_text = native_lines[0]
    if (
        not 1 <= len(heading_text.split()) <= 14
        or heading_text.endswith((".", ",", ";", ":"))
    ):
        return None
    ordered_ocr = sorted(lines, key=lambda line: (float(line["coordinates"][1]), float(line["coordinates"][0])))
    if len(ordered_ocr) < 2:
        return None
    first_box = [float(value) for value in ordered_ocr[0]["coordinates"]]
    second_box = [float(value) for value in ordered_ocr[1]["coordinates"]]
    gap = second_box[1] - (first_box[1] + first_box[3])
    median_height = statistics.median(float(line["coordinates"][3]) for line in ordered_ocr)
    if gap < max(12.0, median_height * 0.60):
        return None

    body_lines = ordered_ocr[1:]
    body_native = "\n".join(native_lines[1:])
    group_id = f"{region.region_id}-heading-body"
    heading_id = f"{group_id}-heading"
    body_region = Region(
        region_id=f"{group_id}-body", page=region.page, kind="normal_text",
        coordinates=_line_box(body_lines), reading_order=region.reading_order + 1,
        classification_method="leading-heading whitespace reconstruction",
        confidence=region.confidence, native_text=body_native,
        metadata={
            "word_count": len(re.findall(r"\w+", body_native, re.UNICODE)),
            "source_bbox_points": region.metadata.get("source_bbox_points"),
        },
    )
    body = _text_block(
        document_id, source_hash, inspection, body_region, body_lines, image, native_threshold,
    )
    body.parent_block_id = group_id
    body.hierarchy_depth = 1
    ocr_heading = str(ordered_ocr[0].get("text", "")).strip()
    agreement = _token_agreement(heading_text, ocr_heading)
    heading_role = _semantic_role_for_text("heading", heading_text, first_box, region.page)
    heading_level = 1 if heading_role in {"document_title", "page_title"} else 2
    heading = SourceBlock(
        document_id=document_id, type="heading", page=region.page, block_id=heading_id,
        content={
            "text": heading_text,
            "evidence_text": {
                "selected": "native", "native": heading_text, "ocr": ocr_heading or None,
                "token_agreement": agreement,
            },
        },
        coordinates=first_box,
        extraction_method=["native PDF text", "RapidOCR", "Python leading-heading reconstruction"],
        confidence=min(
            inspection.native_text_quality,
            0.5 + agreement / 2 if agreement is not None else region.confidence,
        ),
        validation_status="passed", provenance=_provenance(source_hash, region, [ordered_ocr[0]], image),
        semantic_role=heading_role, parent_block_id=group_id, hierarchy_depth=1,
        heading_level=heading_level,
    )
    group = SourceBlock(
        document_id=document_id, type="group", page=region.page, block_id=group_id,
        content={"role": "section", "child_block_ids": [heading_id, body.block_id]},
        coordinates=region.coordinates,
        extraction_method=["Python leading-heading hierarchy"],
        confidence=min(heading.confidence, body.confidence),
        validation_status="passed" if body.validation_status == "passed" else "needs_review",
        warnings=[] if body.validation_status == "passed" else ["one or more child blocks require review"],
        provenance=_provenance(source_hash, region, [], image), semantic_role="document_section",
        child_block_ids=[heading_id, body.block_id],
    )
    return [group, heading, body]


def _parse_contact_details(raw_text: str, normalized: str) -> dict[str, Any]:
    """Recover common contact fields while preserving the original text as evidence."""
    lines = [re.sub(r"\s+", " ", line).strip(" |│") for line in raw_text.splitlines() if line.strip()]
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", normalized)
    phone_match = re.search(r"\b(?:\+?1[-. )]*)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b", normalized)
    website_match = re.search(
        r"(?<![@\w])(?:https?://|www\.)?[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}(?:/[^\s|│]*)?",
        normalized, re.I,
    )
    organization = None
    suffix = re.compile(r"\b(?:LLC|L\.L\.C\.|INC\.?|CORP\.?|LTD\.?|LLP|LP|PLC)\b", re.I)
    for line in lines[:3]:
        if not any(character.isdigit() for character in line) and (
            suffix.search(line) or (line.isupper() and 1 <= len(line.split()) <= 10)
        ):
            organization = line.rstrip(",")
            break

    name = None
    if lines and (email_match or phone_match):
        # Contact cards commonly print a person's name before a separator,
        # followed by email/phone. Do not infer a person from an organization,
        # address, or unstructured sentence.
        first = re.split(r"\s*[|│]\s*", lines[0], maxsplit=1)[0].strip()
        pieces = first.split()
        person_token = re.compile(r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*$", re.UNICODE)
        if (
            2 <= len(pieces) <= 4
            and all(person_token.fullmatch(piece) for piece in pieces)
            and not first.isupper()
            and not suffix.search(first)
            and first != organization
        ):
            name = first

    street = city = state = postal_code = None
    street_suffix = re.compile(
        r"\b(?:STREET|ST|AVENUE|AVE|ROAD|RD|DRIVE|DR|LANE|LN|BOULEVARD|BLVD|"
        r"PARKWAY|PKWY|HIGHWAY|HWY|COURT|CT|CIRCLE|CIR|TRAIL|TRL|WAY|SUITE|STE)\b",
        re.I,
    )
    for index, line in enumerate(lines):
        if re.search(r"\d", line) and street_suffix.search(line):
            street = line
            if index + 1 < len(lines):
                locality = re.match(r"^(.*?)(?:,\s*|\s+)([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$", lines[index + 1])
                if locality:
                    city, state, postal_code = locality.group(1).strip(), locality.group(2), locality.group(3)
            break
    address = None
    if any((street, city, state, postal_code)):
        address = {"street": street, "city": city, "state": state, "postal_code": postal_code}
    website = website_match.group(0) if website_match else None
    if website and not re.match(r"https?://", website, re.I):
        website = f"https://{website}"
    return {
        "name": name,
        "organization": organization,
        "address": address,
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
        "website": website,
    }


def _provenance(source_hash: str, region: Region, ocr_lines: list[dict[str, Any]], image: Path) -> dict[str, Any]:
    return {
        "source_sha256": source_hash,
        "region_id": region.region_id,
        "classification_method": region.classification_method,
        "reading_order": region.reading_order,
        "source_bbox_points": region.metadata.get("source_bbox_points"),
        "ocr_evidence_ids": [line["evidence_id"] for line in ocr_lines],
        "rendered_page": f"page-images/{image.name}",
    }


def _text_block(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, native_threshold: float,
) -> SourceBlock:
    visible_ocr = _ocr_text(lines)
    agreement = _token_agreement(region.native_text, visible_ocr)
    use_native = bool(region.native_text.strip()) and inspection.native_text_quality >= native_threshold
    selected = "native" if use_native else "ocr"
    warnings: list[str] = []
    if use_native and agreement is not None and agreement < 0.35:
        use_native = False
        selected = "ocr"
        warnings.append("native text disagrees with visible OCR; OCR selected")
    native_word_count = len(re.findall(r"\w+", region.native_text, re.UNICODE))
    ocr_word_count = len(re.findall(r"\w+", visible_ocr, re.UNICODE))
    if use_native and agreement is not None and agreement >= 0.35 and ocr_word_count >= native_word_count + 2:
        raw = _hybrid_text(region.native_text, visible_ocr)
        selected = "hybrid"
        use_native = False
    else:
        raw = region.native_text if use_native else visible_ocr
    normalized = clean_text(raw)
    if selected == "hybrid":
        methods = ["native PDF text", "RapidOCR", "PP-OCRv6", "Python token reconciliation", "Python layout normalization"]
    elif use_native:
        methods = ["native PDF text", "Python layout normalization"]
    else:
        methods = ["RapidOCR", "PP-OCRv6", "Python layout normalization"]
    errors = [] if normalized else ["no text recovered from region"]
    source_confidence = inspection.native_text_quality if selected in {"native", "hybrid"} else (
        sum(float(line["confidence"]) for line in lines) / len(lines) if lines else 0.0
    )
    confidence = min(source_confidence, 0.5 + agreement / 2) if agreement is not None else source_confidence
    block_type = _block_type_for_text(region, normalized)
    word_count = int(region.metadata.get("word_count", len(normalized.split())) or 0)
    if block_type == "text" and word_count <= 3:
        warnings.append("short text fragment may have been detached from an adjacent region")
    if (
        region.coordinates[2] >= 700 and inspection.possible_visual_regions >= 2 and word_count >= 20
        and region.classification_method not in {
            "visual-text-whitespace-segmentation",
            "name-role biography reconstruction",
        }
    ):
        warnings.append("wide text spans a mixed visual layout; reading order requires review")
    if "�" in normalized:
        warnings.append("text contains an invalid replacement character")
    content: dict[str, Any] = {
        "text": normalized,
        "evidence_text": {
            "selected": selected,
            "native": region.native_text or None,
            "ocr": visible_ocr or None,
            "token_agreement": agreement,
        },
    }
    structure = _text_structure(raw)
    if structure:
        content["structure"] = structure
    if block_type == "contact":
        content.update(_parse_contact_details(raw, normalized))
    return SourceBlock(
        document_id=document_id, type=block_type, page=region.page,
        block_id=f"{region.region_id}-block",
        content=content, coordinates=region.coordinates,
        extraction_method=methods, confidence=confidence,
        validation_status="passed" if normalized and not warnings else "needs_review", errors=errors, warnings=warnings,
        provenance=_provenance(source_hash, region, lines, image),
        semantic_role=_semantic_role_for_text(block_type, normalized, region.coordinates, region.page),
        heading_level=1 if block_type == "heading" and region.coordinates[1] <= 220 else (
            2 if block_type == "heading" else None
        ),
    )


def _visual_text_panel_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, page_lines: list[dict[str, Any]],
) -> list[SourceBlock]:
    """Create a structural parent and leaf blocks for mixed text embedded in one image region."""
    groups = _split_visual_text_lines(lines)
    if len(groups) <= 1:
        region.native_text = ""
        return [_text_block(document_id, source_hash, inspection, region, lines, image, 1.1)]

    parent_id = f"{region.region_id}-content-group"
    children: list[SourceBlock] = []
    for index, group_lines in enumerate(groups, 1):
        coordinates = _line_box(group_lines)
        text = clean_text(_ocr_text(group_lines))
        subregion = Region(
            region_id=f"{region.region_id}-s{index:03d}", page=region.page,
            kind="normal_text", coordinates=coordinates, reading_order=region.reading_order,
            classification_method="visual-text-whitespace-segmentation",
            confidence=sum(float(line.get("confidence", 0.0)) for line in group_lines) / len(group_lines),
            metadata={
                "word_count": len(re.findall(r"\S+", text)),
                "median_font_size": statistics.median(
                    float(line["coordinates"][3]) * inspection.height_points / 1000.0 for line in group_lines
                ),
                "source_bbox_points": [
                    coordinates[0] * inspection.width_points / 1000.0,
                    coordinates[1] * inspection.height_points / 1000.0,
                    (coordinates[0] + coordinates[2]) * inspection.width_points / 1000.0,
                    (coordinates[1] + coordinates[3]) * inspection.height_points / 1000.0,
                ],
            },
        )
        child = _text_block(document_id, source_hash, inspection, subregion, group_lines, image, 1.1)
        child.parent_block_id = parent_id
        child.hierarchy_depth = 1
        child.heading_level = 2 if child.type == "heading" else None
        child.semantic_role = (
            "brand_mark_text" if _looks_like_brand_mark(group_lines, page_lines)
            else _semantic_role_for_text(child.type, text, coordinates, region.page, nested=True)
        )
        children.append(child)

    parent = SourceBlock(
        document_id=document_id, type="group", page=region.page, block_id=parent_id,
        content={"role": "mixed_text_panel", "child_block_ids": [child.block_id for child in children]},
        coordinates=region.coordinates,
        extraction_method=["RapidOCR", "PP-OCRv6", "Python whitespace segmentation"],
        confidence=min(child.confidence for child in children), validation_status="passed",
        provenance=_provenance(source_hash, region, [], image),
        semantic_role="content_panel", child_block_ids=[child.block_id for child in children],
    )
    return [parent, *children]


def _table_blocks(
    document_id: str, source_hash: str, region: Region, lines: list[dict[str, Any]], image: Path,
    qwen_payload: dict[str, Any] | None = None, qwen_error: str | None = None,
) -> list[SourceBlock]:
    rows = region.metadata.get("rows") or []
    table_title = region.metadata.get("title")
    row_sections = region.metadata.get("row_sections") or []
    cell_coordinates = region.metadata.get("cell_coordinates") or []
    cell_evidence_ids = region.metadata.get("cell_evidence_ids") or []
    paddle_rows = region.metadata.get("rows_source") == "paddle"
    ocr_geometry_rows = region.metadata.get("rows_source") == "ocr_geometry"
    native = bool(not paddle_rows and not ocr_geometry_rows and rows
                  and max((len(row) for row in rows), default=0) >= 2)
    if not rows:
        ordered = sorted(lines, key=lambda line: (line["coordinates"][1], line["coordinates"][0]))
        grouped: list[list[dict[str, Any]]] = []
        for line in ordered:
            if not grouped or abs(line["coordinates"][1] - grouped[-1][0]["coordinates"][1]) > 16:
                grouped.append([line])
            else:
                grouped[-1].append(line)
        # Anchor wide sparse OCR rows to printed header positions. Sequential
        # padding shifts values left when earlier cells are blank; narrow prose
        # groups are too ambiguous to use as positional headers.
        header_group = next((
            group for group in grouped[:4]
            if len(group) >= 4
            and sum(bool(re.search(r"[A-Za-z]", str(item["text"]))) for item in group) >= 3
            and sum(bool(re.fullmatch(r"[$€£]?[-+]?\d[\d.,%xX]*", str(item["text"]).strip())) for item in group) <= 1
            and max(item["coordinates"][0] for item in group)
            - min(item["coordinates"][0] for item in group) >= 30
        ), None)
        if header_group:
            anchors = sorted(item["coordinates"][0] for item in header_group)
            rows = []
            for group in grouped:
                row: list[str | None] = [None] * len(anchors)
                for item in sorted(group, key=lambda item: item["coordinates"][0]):
                    column = min(range(len(anchors)), key=lambda index: abs(item["coordinates"][0] - anchors[index]))
                    row[column] = f"{row[column]}\n{item['text']}" if row[column] else item["text"]
                rows.append(row)
        else:
            rows = [[item["text"] for item in sorted(group, key=lambda item: item["coordinates"][0])] for group in grouped]
    width = max((len(row) for row in rows), default=0)
    rows = [list(row) + [None] * (width - len(row)) for row in rows]
    paddle_value_first = False
    if paddle_rows and width == 2 and len(rows) >= 3:
        first_numeric = sum(bool(NUMBER.search(str(row[0] or ""))) for row in rows)
        second_numeric = sum(bool(NUMBER.search(str(row[1] or ""))) for row in rows)
        second_labels = sum(bool(re.search(r"[A-Za-z]", str(row[1] or ""))) for row in rows)
        if (first_numeric >= max(3, math.ceil(0.6 * len(rows)))
                and second_numeric <= len(rows) // 3
                and second_labels >= math.ceil(0.7 * len(rows))):
            rows = [[row[1], row[0]] for row in rows]
            if cell_coordinates:
                cell_coordinates = [[row[1], row[0]] for row in cell_coordinates]
            paddle_value_first = True
    # A two-column label/value list often has no printed header. Preserve its
    # first observation instead of silently treating it as column names.
    inferred_key_value_header = False
    if paddle_rows and width == 2 and len(rows) >= 3:
        first_label = str(rows[0][0] or "").strip()
        first_value = str(rows[0][1] or "").strip()
        value_count = sum(bool(NUMBER.search(str(row[1] or ""))) for row in rows)
        if (first_label and not NUMBER.fullmatch(first_label)
                and bool(NUMBER.search(first_value))
                and (paddle_value_first or value_count >= len(rows) - 1)):
            rows.insert(0, ["Field", "Value"])
            if cell_coordinates:
                cell_coordinates = [[None, None], *cell_coordinates]
            inferred_key_value_header = True
    # OCR commonly places a spanning title before the actual column header.
    # Promote the next row only when it looks like a set of printed labels.
    if len(rows) >= 3 and width >= 2:
        first = [str(value or "").strip() for value in rows[0]]
        second = [str(value or "").strip() for value in rows[1]]
        first_count = sum(bool(value) for value in first)
        second_count = sum(bool(value) for value in second)
        second_numeric = sum(bool(re.fullmatch(r"[$€£]?[-+]?\d[\d.,%xX]*", value)) for value in second if value)
        if first_count == 1 and second_count >= max(2, width // 2) and second_numeric <= 1:
            table_title = table_title or next(value for value in first if value)
            rows = rows[1:]
            row_sections = row_sections[1:] if row_sections else row_sections
            cell_coordinates = cell_coordinates[1:] if cell_coordinates else cell_coordinates
    if len(rows) == 1 and width >= 2:
        layout = reconstruct_comparison_panel(region.metadata.get("native_panel_words") or [])
        if layout:
            panel_review = qwen_payload.get("panel_review") if qwen_payload else None
            review_matches = False
            comparison_warnings: list[str] = []
            if isinstance(panel_review, dict):
                expected_titles = [re.sub(r"\W+", "", title.casefold()) for title in layout["leaf_titles"]]
                actual_titles = [
                    re.sub(r"\W+", "", str(title).casefold())
                    for title in panel_review.get("leaf_titles", [])
                ]
                expected_counts = [
                    len(leaf.get("claims", []))
                    for section in layout["sections"]
                    for leaf in (section.get("subsections") or [section])
                ]
                review_matches = (
                    panel_review.get("lane_count") == layout["lane_count"]
                    and actual_titles == expected_titles
                    and panel_review.get("claim_counts") == expected_counts
                )
                if not review_matches:
                    comparison_warnings.append("Qwen comparison-layout review disagreed with positioned-word ownership")
                elif panel_review.get("structure_matches") is not True:
                    comparison_warnings.append(
                        "Qwen structure_matches flag is false despite agreement on all explicit layout fields"
                    )
            elif qwen_payload is not None:
                comparison_warnings.append("Qwen comparison-layout review returned no structured panel review")
            if qwen_error:
                comparison_warnings.append(f"Qwen comparison-layout review unavailable: {qwen_error}")
            if qwen_payload is None and not qwen_error:
                comparison_warnings.append("Qwen comparison-layout review was not configured")
            content = {
                "title": layout["title"], "sections": layout["sections"],
                "lane_count": layout["lane_count"], "claim_count": layout["claim_count"],
                "bullet_anchor_coordinates": layout["bullet_anchor_coordinates"],
                "vision_review": panel_review if isinstance(panel_review, dict) else None,
            }
            return [SourceBlock(
                document_id=document_id, type="comparison_panel", page=region.page,
                block_id=f"{region.region_id}-comparison-panel",
                content=content, coordinates=region.coordinates,
                extraction_method=[
                    "native PDF positioned words", "Python lane/heading/bullet reconstruction",
                    *(["Qwen3-VL comparison-layout review"] if qwen_payload is not None else []),
                ],
                confidence=0.90 if review_matches else 0.72,
                validation_status="passed" if review_matches else "needs_review",
                errors=[], warnings=comparison_warnings,
                provenance=_provenance(source_hash, region, lines, image),
            )]
        sections = []
        for index, value in enumerate(rows[0], 1):
            raw_text = clean_text(str(value or ""))
            first_line = str(value or "").split("\n", 1)[0].strip() if raw_text else None
            sections.append({
                "section_id": f"s{index:03d}", "title": first_line,
                "text": raw_text, "structure_complete": False,
            })
        comparison_methods = ["native PDF layout"]
        if qwen_payload is not None:
            comparison_methods.append("Qwen3-VL table structure review")
        comparison_methods.append("Python section reconstruction")
        comparison_warnings = [
            "panel columns were preserved, but nested subsections could not be separated reliably"
        ]
        if qwen_error:
            comparison_warnings.append(f"Qwen semantic table stage unavailable: {qwen_error}")
        return [SourceBlock(
            document_id=document_id, type="comparison_panel", page=region.page,
            block_id=f"{region.region_id}-comparison-panel",
            content={"title": region.metadata.get("title"), "sections": sections},
            coordinates=region.coordinates,
            extraction_method=comparison_methods,
            confidence=min(region.confidence, 0.60), validation_status="needs_review",
            errors=[], warnings=comparison_warnings,
            provenance=_provenance(source_hash, region, lines, image),
        )]
    # PDF and image table parsers sometimes split a printed currency sign into
    # its own cell. Keep the grid width but attach that sign to the adjacent
    # numeric token; a bare "$" is never a value.
    for row in rows[1:]:
        for column_index in range(1, len(row) - 1):
            current = str(row[column_index] or "").strip()
            following = str(row[column_index + 1] or "").strip()
            if current in {"$", "€", "£"} and re.fullmatch(
                r"\d[\d,]*(?:\.\d+)?\s*[$€£]?", following,
            ):
                row[column_index] = None
                row[column_index + 1] = current + following
            elif (current and current[-1] in "$€£"
                  and current not in {"$", "€", "£"}
                  and re.fullmatch(r"\d[\d,]*(?:\.\d+)?", following)):
                symbol = current[-1]
                row[column_index] = current[:-1].rstrip()
                row[column_index + 1] = symbol + following
    reconciliation = _table_total_reconciliation(rows)
    reconciliation_failed = bool(reconciliation and not reconciliation["passed"])
    parent_id = f"{region.region_id}-table"
    warnings = [] if native else (["table reconstructed from PaddleOCR structure; independently review cell values"]
                                 if paddle_rows else ["table reconstructed from OCR geometry"])
    if inferred_key_value_header:
        warnings.append("two-column key/value list has an inferred, unprinted header")
    if paddle_value_first:
        warnings.append("PaddleOCR value-first key/value rows were oriented by column content")
    paddle_review = region.metadata.get("paddle_table_review")
    if isinstance(paddle_review, dict):
        if paddle_review.get("status") == "disagrees":
            warnings.append("PaddleOCR and native PDF table cells or dimensions disagree; native cells retained")
        elif paddle_review.get("status") == "not_detected":
            warnings.append("PaddleOCR did not detect the native PDF table")
        elif paddle_review.get("status") == "layout_only":
            warnings.append("PaddleOCR detected a table but returned no usable cell grid; OCR geometry retained")
    valid_shape = len(rows) >= 2 and width >= 2
    methods = ["native PDF table parser"] if native else (["PaddleOCR PP-StructureV3"] if paddle_rows
               else ["RapidOCR", "PP-OCRv6"])
    if isinstance(paddle_review, dict):
        methods.append("PaddleOCR PP-StructureV3 table review")
    if qwen_payload is not None:
        methods.append("Qwen3-VL table structure review")
    methods.append("Python table reconstruction and validation")
    headers = ["" if value is None else str(value).strip() for value in rows[0]] if rows else []
    column_kinds: list[str] = []
    for column_index, header in enumerate(headers):
        values = [str(row[column_index] or "") for row in rows[1:] if column_index < len(row)]
        joined = " ".join(values)
        if column_index == 0:
            kind_hint = "text"
        elif "%" in header or "%" in joined:
            kind_hint = "percent"
        elif any(symbol in joined or symbol in header for symbol in "$€£"):
            kind_hint = "currency"
        elif re.search(r"\bx\b", joined, re.I):
            kind_hint = "multiple"
        else:
            kind_hint = "number"
        column_kinds.append(kind_hint)
    columns = [
        {
            "column_id": f"c{index + 1:03d}", "label": label or None, "value_kind": column_kinds[index],
            "coordinates": cell_coordinates[0][index] if cell_coordinates and index < len(cell_coordinates[0]) else None,
            "evidence_ids": [f"{region.region_id}-native-table-r000-c{index:03d}"]
            if (native and cell_coordinates and index < len(cell_coordinates[0])
                and cell_coordinates[0][index]) else [],
        }
        for index, label in enumerate(headers)
    ]
    typed_rows = []
    ambiguous_table_cells = 0
    for row_index, row in enumerate(rows[1:], 1):
        label = "" if not row or row[0] is None else str(row[0]).strip()
        cells = []
        for column_index, value in enumerate(row[1:], 1):
            raw = None if value is None or str(value).strip() == "" else str(value).strip()
            state = "blank" if raw is None else "dash" if raw in {"-", "–", "—"} else "present"
            if raw and not _TABLE_SCALAR.fullmatch(raw):
                # Any table route can merge neighboring cells (or return
                # dates, addresses, and descriptions). Keep the printed text,
                # but do not turn its first number into a trusted scalar fact.
                numeric_value = unit = normalized_value = None
                if re.search(r"\d", raw):
                    ambiguous_table_cells += 1
            else:
                numeric_value, unit, normalized_value = _numeric_value(raw or "")
            if state != "present":
                numeric_value = unit = normalized_value = None
            elif (not paddle_rows and column_index < len(column_kinds)
                  and column_kinds[column_index] == "currency" and numeric_value is not None):
                unit = "USD"
                normalized_value = numeric_value
            cells.append({
                "column_id": f"c{column_index + 1:03d}", "raw_value": raw,
                "value_state": state, "numeric_value": numeric_value,
                "unit": unit, "normalized_value": normalized_value,
                "coordinates": (
                    cell_coordinates[row_index][column_index]
                    if row_index < len(cell_coordinates) and column_index < len(cell_coordinates[row_index])
                    else None
                ),
                "evidence_ids": (
                    list(cell_evidence_ids[row_index][column_index])
                    if (ocr_geometry_rows and row_index < len(cell_evidence_ids)
                        and column_index < len(cell_evidence_ids[row_index]) and raw is not None)
                    else [f"{region.region_id}-native-table-r{row_index:03d}-c{column_index:03d}"]
                    if (native and row_index < len(cell_coordinates)
                        and column_index < len(cell_coordinates[row_index])
                        and cell_coordinates[row_index][column_index] and raw is not None)
                    else []
                ),
                "grounding_status": (
                    "ocr_value_and_row" if ocr_geometry_rows and raw is not None
                    and row_index < len(cell_evidence_ids)
                    and column_index < len(cell_evidence_ids[row_index])
                    and bool(cell_evidence_ids[row_index][0])
                    and bool(cell_evidence_ids[row_index][column_index])
                    else "native_positioned" if native and raw is not None
                    and row_index < len(cell_coordinates)
                    and column_index < len(cell_coordinates[row_index])
                    and bool(cell_coordinates[row_index][0])
                    and bool(cell_coordinates[row_index][column_index])
                    else "unverified"
                ),
            })
        typed_row = {
            "row_id": f"r{row_index:03d}",
            "section": row_sections[row_index] if row_index < len(row_sections) else None,
            "label": label or None, "cells": cells,
        }
        if ocr_geometry_rows:
            typed_row["label_evidence_ids"] = (
                list(cell_evidence_ids[row_index][0])
                if row_index < len(cell_evidence_ids) and cell_evidence_ids[row_index] else []
            )
        typed_row["label_coordinates"] = (
            cell_coordinates[row_index][0]
            if row_index < len(cell_coordinates) and cell_coordinates[row_index] else None
        )
        typed_rows.append(typed_row)
    unsupported_cells = sum(
        cell["grounding_status"] == "unverified" and cell["raw_value"] is not None
        for row in typed_rows for cell in row["cells"]
    )
    if unsupported_cells:
        warnings.append(
            f"{unsupported_cells} table cells lack value-and-owner evidence; treat their values as candidates"
        )
    if ambiguous_table_cells:
        warnings.append(
            f"{ambiguous_table_cells} table cells contain non-scalar numeric text; "
            "raw text retained without a numeric value"
        )
    structure_errors = _table_structure_errors(headers, rows)
    qwen_warnings: list[str] = []
    review = qwen_payload.get("table_review") if qwen_payload else None
    if qwen_error:
        qwen_warnings.append(f"Qwen semantic table stage unavailable: {qwen_error}")
    elif qwen_payload is not None and not isinstance(review, dict):
        qwen_warnings.append("Qwen semantic table stage returned no structured table review")
    elif isinstance(review, dict):
        model_rows = review.get("data_row_count")
        model_columns = review.get("column_count")
        if model_rows != len(typed_rows) or model_columns != width or review.get("structure_matches") is not True:
            qwen_warnings.append(
                "Qwen table structure review disagreed with deterministic row/column reconstruction"
            )
    table_passed = (valid_shape and native and not unsupported_cells
                    and not reconciliation_failed and not structure_errors
                    and not (isinstance(paddle_review, dict) and paddle_review.get("status") == "disagrees"))
    parent = SourceBlock(
        document_id=document_id, type="table", page=region.page, block_id=parent_id,
        content={
            "title": table_title, "columns": columns, "rows": typed_rows,
            "row_count": len(typed_rows), "column_count": width,
            "reconciliation": reconciliation,
        },
        coordinates=region.coordinates, extraction_method=methods, confidence=region.confidence if native else 0.55,
        validation_status="passed" if table_passed and not qwen_warnings else "needs_review",
        errors=([] if valid_shape else ["table does not contain at least two rows and two columns"])
        + (["table subtotals do not reconcile with the final total"] if reconciliation_failed else [])
        + structure_errors, warnings=warnings + qwen_warnings,
        provenance=_provenance(source_hash, region, lines, image),
    )
    return [parent]


def _table_structure_errors(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Detect malformed ownership that numeric reconciliation cannot reveal."""
    errors: list[str] = []
    if any(not header for header in headers):
        errors.append("one or more table columns have no header owner")
    if any(not row or not str(row[0] or "").strip() for row in rows[1:]):
        errors.append("one or more table rows have no label owner")
    if any(header.count("(") != header.count(")") for header in headers):
        errors.append("one or more table headers contain unmatched parentheses")
    for row in rows[1:]:
        for value in row[1:]:
            text = str(value or "").strip()
            if text in {"$", "€", "£"}:
                errors.append("standalone currency symbol has no owned numeric value")
            if re.search(r"%\s*[$€£]$", text):
                errors.append("currency symbol is attached to a percentage cell")
    return sorted(set(errors))


def _table_total_reconciliation(rows: list[list[Any]]) -> dict[str, Any] | None:
    """Reconcile generic subtotal rows against the final total wherever values are comparable."""
    if len(rows) < 4:
        return None

    total_rows = [
        row for row in rows[1:]
        if row and re.match(r"^\s*(?:grand\s+)?total\b", str(row[0] or ""), re.I)
    ]
    if len(total_rows) < 3:
        return None
    subtotal_rows, grand_total = total_rows[:-1], total_rows[-1]

    def number(value: Any) -> float | None:
        text = str(value or "").strip()
        if not text or text in {"-", "–", "—"}:
            return None
        negative = text.startswith("(") and text.endswith(")")
        match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        if not match:
            return None
        parsed = float(match.group().replace(",", ""))
        return -abs(parsed) if negative else parsed

    checks = []
    width = max(len(row) for row in rows)
    headers = rows[0]
    for column in range(1, width):
        parts = [number(row[column]) if column < len(row) else None for row in subtotal_rows]
        observed = number(grand_total[column]) if column < len(grand_total) else None
        if observed is None or any(part is None for part in parts):
            continue
        calculated = sum(part for part in parts if part is not None)
        tolerance = max(0.011, abs(observed) * 1e-8)
        passed = abs(calculated - observed) <= tolerance
        checks.append({
            "column_index": column,
            "column": str(headers[column] or "").strip() if column < len(headers) else "",
            "subtotal_values": parts,
            "observed_total": observed,
            "calculated_total": round(calculated, 5),
            "passed": passed,
        })
    if not checks:
        return None
    return {
        "passed": all(check["passed"] for check in checks),
        "subtotal_rows": [str(row[0]).strip() for row in subtotal_rows],
        "grand_total_row": str(grand_total[0]).strip(),
        "checks": checks,
    }


def _nearest_label(value_line: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    vx, vy, vw, vh = value_line["coordinates"]
    best = None
    best_distance = math.inf
    for candidate in candidates:
        cx, cy, cw, ch = candidate["coordinates"]
        distance = math.hypot((vx + vw / 2) - (cx + cw / 2), (vy + vh / 2) - (cy + ch / 2))
        if distance < best_distance:
            best, best_distance = candidate, distance
    return best


def _box_union(*boxes: list[float]) -> list[float]:
    valid = [box for box in boxes if isinstance(box, list) and len(box) == 4]
    if not valid:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = min(box[0] for box in valid)
    y0 = min(box[1] for box in valid)
    x1 = max(box[0] + box[2] for box in valid)
    y1 = max(box[1] + box[3] for box in valid)
    return [x0, y0, x1 - x0, y1 - y0]


def _line_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx, ly, lw, lh = left["coordinates"]
    rx, ry, rw, rh = right["coordinates"]
    return math.hypot((lx + lw / 2) - (rx + rw / 2), (ly + lh / 2) - (ry + rh / 2))


def _visual_title(kind: str, lines: list[dict[str, Any]], region: Region | None = None) -> str | None:
    terms = MAP_TERMS if kind == "map" else CHART_TERMS
    upper_limit = (
        float(region.coordinates[1]) + float(region.coordinates[3]) * 0.38
        if region is not None else math.inf
    )
    upper_lines = [line for line in lines if float(line["coordinates"][1]) <= upper_limit]
    matched = [line for line in upper_lines if any(term in str(line["text"]).casefold() for term in terms)]
    # Titles can legitimately contain dates, amounts, or percentages. Prefer
    # the uppermost text-bearing line without using industry-specific nouns.
    candidates = [
        line for line in upper_lines
        if len(re.findall(r"[A-Za-z]+", str(line["text"]))) >= 2
        and not NUMBER.fullmatch(str(line["text"]).strip())
    ]
    pool = matched or candidates
    if not pool:
        return None
    return str(min(pool, key=lambda line: (line["coordinates"][1], line["coordinates"][0]))["text"]).strip()


def _numeric_value(text: str) -> tuple[float | None, str | None, float | None]:
    cleaned = text.replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None, None, None
    value = float(match.group())
    if "%" in cleaned:
        return value, "percent", value / 100.0
    if "$" in cleaned:
        suffix = re.search(
            r"[-+]?\d+(?:\.\d+)?\s*(billion|million|mm|bn|[bmk])\b",
            cleaned, re.I,
        )
        scale = suffix.group(1).casefold() if suffix else ""
        multiplier = (
            1_000_000_000 if scale in {"b", "bn", "billion"}
            else 1_000_000 if scale in {"m", "mm", "million"}
            else 1_000 if scale == "k" else 1.0
        )
        return value, "USD", value * multiplier
    if cleaned.lower().endswith("x"):
        return value, "multiple", value
    return value, None, value


def _numeric_content_coverage(
    blocks: list[dict[str, Any]], lines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check whether printed numeric OCR lines survive in nearby block content.

    Region ownership alone is insufficient: one broad region may contain two
    tables while the final block serializes only one of them. This is a
    conservative content check, not a proof that each value has the right row.
    """
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [item for nested in value.values() for item in strings(nested)]
        if isinstance(value, list):
            return [item for nested in value for item in strings(nested)]
        return []

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    indexed = [
        (block.get("coordinates") or [0, 0, 0, 0],
         [normalized(item) for item in strings(block.get("content", {}))])
        for block in blocks
    ]
    counted = 0
    missing: list[dict[str, Any]] = []
    for line in lines:
        raw = str(line.get("text", ""))
        key = normalized(raw)
        if (float(line.get("confidence", 0.0)) < 0.80
                or not re.search(r"\d", raw) or len(key) < 3):
            continue
        coordinates = line.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            continue
        counted += 1
        x, y, width, height = coordinates
        center_x, center_y = x + width / 2, y + height / 2
        represented = any(
            box[0] - 10 <= center_x <= box[0] + box[2] + 10
            and box[1] - 10 <= center_y <= box[1] + box[3] + 10
            and any(key in item for item in content)
            for box, content in indexed
        )
        if not represented:
            missing.append({"evidence_id": line.get("evidence_id"), "text": raw})
    return {
        "numeric_ocr_lines": counted,
        "represented": counted - len(missing),
        "unrepresented": missing,
    }


def _preserve_ocr_lines_in_blocks(
    blocks: list[dict[str, Any]], lines: list[dict[str, Any]],
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep every nonblank OCR line inside a USB block without inventing an owner.

    Existing structured content retains its owner. An owned OCR line omitted by
    that block's content is attached as raw evidence. A line with no defensible
    owner gets its own source-text block, preserving the exact OCR and image box.
    """
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [item for nested in value.values() for item in strings(nested)]
        if isinstance(value, list):
            return [item for nested in value for item in strings(nested)]
        return []

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    def contains(block: dict[str, Any], raw: str) -> bool:
        key = normalized(raw)
        candidates = strings(block.get("content", {}))
        if not key:
            return any(raw in item for item in candidates)
        if len(key) < 3:
            return any(key == normalized(item) for item in candidates)
        return any(key in normalized(item) for item in candidates)

    def near(block: dict[str, Any], line: dict[str, Any]) -> bool:
        box = block.get("coordinates") or [0, 0, 0, 0]
        line_box = line.get("coordinates") or [0, 0, 0, 0]
        if len(box) != 4 or len(line_box) != 4:
            return False
        cx, cy = line_box[0] + line_box[2] / 2, line_box[1] + line_box[3] / 2
        return box[0] - 10 <= cx <= box[0] + box[2] + 10 and box[1] - 10 <= cy <= box[1] + box[3] + 10

    by_owner = {
        evidence_id: block
        for block in blocks
        for evidence_id in block.get("provenance", {}).get("ocr_evidence_ids", [])
    }
    existing_ids = {block["block_id"] for block in blocks}
    stats = {"ocr_lines": 0, "already_in_owner_content": 0, "owner_recovered": 0,
             "raw_lines_attached": 0, "fallback_text_blocks": 0}
    for index, line in enumerate(lines, 1):
        raw = str(line.get("text") or "")
        if not raw.strip():
            continue
        evidence_id = str(line.get("evidence_id") or "")
        if not evidence_id:
            continue
        stats["ocr_lines"] += 1
        owner = by_owner.get(evidence_id)
        if owner is None:
            candidates = [block for block in blocks if near(block, line) and contains(block, raw)]
            if len(candidates) == 1 and len(normalized(raw)) >= 3:
                owner = candidates[0]
                owner["provenance"]["ocr_evidence_ids"].append(evidence_id)
                by_owner[evidence_id] = owner
                stats["owner_recovered"] += 1
            else:
                coordinates = [float(value) for value in line.get("coordinates", [0, 0, 0, 0])]
                if len(coordinates) != 4:
                    coordinates = [0.0, 0.0, 0.0, 0.0]
                block_id = f"p{inspection.page:03d}-raw-ocr-{index:04d}-text"
                if block_id in existing_ids:
                    raise ValueError(f"duplicate raw OCR fallback block ID: {block_id}")
                existing_ids.add(block_id)
                region = Region(
                    region_id=f"p{inspection.page:03d}-raw-ocr-{index:04d}",
                    page=inspection.page, kind="normal_text", coordinates=coordinates,
                    reading_order=len(blocks) + 1,
                    classification_method="unassigned OCR line preservation",
                    confidence=float(line.get("confidence", 0.0)),
                    metadata={"source_bbox_points": [
                        coordinates[0] * inspection.width_points / 1000.0,
                        coordinates[1] * inspection.height_points / 1000.0,
                        (coordinates[0] + coordinates[2]) * inspection.width_points / 1000.0,
                        (coordinates[1] + coordinates[3]) * inspection.height_points / 1000.0,
                    ]},
                )
                fallback = SourceBlock(
                    document_id=document_id, type="text", page=inspection.page,
                    block_id=block_id, coordinates=coordinates,
                    content={"text": raw, "evidence_text": {
                        "selected": "ocr", "native": None, "ocr": raw, "token_agreement": None,
                    }},
                    extraction_method=["RapidOCR PP-OCRv6", "Python raw evidence preservation"],
                    confidence=float(line.get("confidence", 0.0)),
                    validation_status="needs_review", errors=[],
                    warnings=["OCR text retained without a verified structural owner"],
                    provenance=_provenance(source_hash, region, [line], image),
                    semantic_role="unresolved_source_text",
                ).as_dict()
                blocks.append(fallback)
                by_owner[evidence_id] = fallback
                stats["fallback_text_blocks"] += 1
                continue
        if contains(owner, raw):
            stats["already_in_owner_content"] += 1
            continue
        if any(item.get("evidence_id") == evidence_id for item in owner.get("raw_evidence_lines", [])):
            stats["raw_lines_attached"] += 1
            continue
        item = {"evidence_id": evidence_id, "text": raw,
                "confidence": max(0.0, min(1.0, float(line.get("confidence", 0.0)))),
                "coordinates": line.get("coordinates", [0, 0, 0, 0])}
        owner.setdefault("raw_evidence_lines", []).append(item)
        owner["validation"]["status"] = "needs_review"
        warning = "OCR lines are retained as raw evidence because structured content omits them"
        if warning not in owner["validation"]["warnings"]:
            owner["validation"]["warnings"].append(warning)
        stats["raw_lines_attached"] += 1
    by_id = {block["block_id"]: block for block in blocks}
    for group in sorted(
        (block for block in blocks if block.get("type") == "group"),
        key=lambda block: block.get("hierarchy", {}).get("depth", 0), reverse=True,
    ):
        if any(
            by_id[child_id]["validation"]["status"] != "passed"
            for child_id in group.get("hierarchy", {}).get("child_block_ids", [])
            if child_id in by_id
        ):
            group["validation"]["status"] = "needs_review"
            warning = "one or more child blocks require review"
            if warning not in group["validation"]["warnings"]:
                group["validation"]["warnings"].append(warning)
    return blocks, stats


def _preserve_source_observations(
    blocks: list[dict[str, Any]], observations: list[dict[str, Any]],
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
) -> dict[str, int]:
    """Retain native and Paddle observations not already covered by RapidOCR."""
    stats = {"native_words": 0, "paddle_text_blocks": 0, "matched_to_ocr": 0,
             "content_supported": 0, "verbatim_glyphs": 0,
             "raw_attached": 0, "fallback_text_blocks": 0}
    for index, observation in enumerate(observations, 1):
        source = observation["source"]
        stats["native_words" if source == "native_pdf_word" else "paddle_text_blocks"] += 1
        if observation.get("matched_ocr_evidence_ids"):
            observation["disposition"] = "matched_ocr"
            observation["owner_block_id"] = None
            stats["matched_to_ocr"] += 1
            continue
        box = observation["coordinates"]
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        candidates = [block for block in blocks
                      if block.get("type") != "group"
                      and block["coordinates"][0] - 8 <= cx <= block["coordinates"][0] + block["coordinates"][2] + 8
                      and block["coordinates"][1] - 8 <= cy <= block["coordinates"][1] + block["coordinates"][3] + 8]
        owner = min(candidates, key=lambda block: block["coordinates"][2] * block["coordinates"][3]) if candidates else None
        if owner is None:
            region = Region(
                region_id=f"p{inspection.page:03d}-source-{index:04d}", page=inspection.page,
                kind="normal_text", coordinates=box, reading_order=len(blocks) + 1,
                classification_method=f"unmatched {source} preservation",
                confidence=float(observation["confidence"]),
            )
            selected = "native" if source == "native_pdf_word" else "ocr"
            owner = SourceBlock(
                document_id=document_id, type="text", page=inspection.page,
                block_id=f"{region.region_id}-text", coordinates=box,
                content={"text": observation["text"], "evidence_text": {
                    "selected": selected,
                    "native": observation["text"] if selected == "native" else None,
                    "ocr": observation["text"] if selected == "ocr" else None,
                    "token_agreement": None,
                }},
                extraction_method=[source, "Python source observation preservation"],
                confidence=float(observation["confidence"]),
                validation_status="needs_review", errors=[],
                warnings=["source text retained without a verified structural owner"],
                provenance=_provenance(source_hash, region, [], image),
                semantic_role="unresolved_source_text",
            ).as_dict()
            blocks.append(owner)
            observation["disposition"] = "fallback_text"
            stats["fallback_text_blocks"] += 1
        elif represented_in_content(observation, owner.get("content", {})):
            observation["disposition"] = "content_supported"
            stats["content_supported"] += 1
        elif (owner.get("type") in {"text", "heading", "footnote", "contact",
                                       "brand_mark", "comparison_panel"}
              and is_layout_glyph(observation)):
            owner.setdefault("raw_evidence_lines", []).append({
                "evidence_id": observation["evidence_id"], "text": observation["text"],
                "source": source, "confidence": observation["confidence"],
                "coordinates": box,
            })
            observation["disposition"] = "verbatim_glyph"
            stats["verbatim_glyphs"] += 1
        else:
            owner.setdefault("raw_evidence_lines", []).append({
                "evidence_id": observation["evidence_id"], "source": source,
                "text": observation["text"], "confidence": observation["confidence"],
                "coordinates": box,
            })
            owner["validation"]["status"] = "needs_review"
            warning = "additional source text is retained as raw evidence"
            if warning not in owner["validation"]["warnings"]:
                owner["validation"]["warnings"].append(warning)
            observation["disposition"] = "raw_attached"
            stats["raw_attached"] += 1
        observation["owner_block_id"] = owner["block_id"]
    by_id = {block["block_id"]: block for block in blocks}
    for group in sorted((block for block in blocks if block.get("type") == "group"),
                        key=lambda block: block.get("hierarchy", {}).get("depth", 0), reverse=True):
        if any(by_id[child_id]["validation"]["status"] != "passed"
               for child_id in group.get("hierarchy", {}).get("child_block_ids", [])
               if child_id in by_id):
            group["validation"]["status"] = "needs_review"
            warning = "one or more child blocks require review"
            if warning not in group["validation"]["warnings"]:
                group["validation"]["warnings"].append(warning)
    return stats


def _ground_table_cells_from_ocr(
    blocks: list[dict[str, Any]], lines: list[dict[str, Any]],
) -> dict[str, int]:
    """Ground exact printed table text only when its row and column fit."""
    def key(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())

    def center(box: list[float]) -> tuple[float, float]:
        return box[0] + box[2] / 2, box[1] + box[3] / 2

    def inside(line: dict[str, Any], box: list[float] | None) -> bool:
        if not box or len(box) != 4:
            return True
        x, y = center(line["coordinates"])
        return box[0] - 5 <= x <= box[0] + box[2] + 5 and box[1] - 5 <= y <= box[1] + box[3] + 5

    by_id = {str(line.get("evidence_id")): line for line in lines}
    stats = {"newly_grounded": 0, "still_unverified": 0}
    for block in blocks:
        if block.get("type") != "table":
            continue
        owned = [by_id[evidence_id] for evidence_id in block.get("provenance", {}).get("ocr_evidence_ids", [])
                 if evidence_id in by_id]
        if not owned:
            continue
        rows = block.get("content", {}).get("rows", [])
        columns = block.get("content", {}).get("columns", [])
        if not rows:
            continue
        headers: dict[str, dict[str, Any]] = {}
        if len(columns) > 2:
            for column in columns[1:]:
                matches = [line for line in owned
                           if key(line.get("text")) == key(column.get("label"))
                           and inside(line, column.get("coordinates"))]
                if len(matches) == 1:
                    headers[column["column_id"]] = matches[0]
                    column["evidence_ids"] = [matches[0]["evidence_id"]]
                    column["coordinates"] = matches[0]["coordinates"]
        used_values: set[str] = set()
        for row in rows:
            label_matches = [line for line in owned
                             if key(row.get("label")) and key(line.get("text")) == key(row.get("label"))
                             and inside(line, row.get("label_coordinates"))]
            if len(label_matches) != 1:
                continue
            label_line = label_matches[0]
            label_x, label_y = center(label_line["coordinates"])
            if not row.get("label_evidence_ids"):
                row["label_evidence_ids"] = [label_line["evidence_id"]]
            if row.get("label_coordinates") is None:
                row["label_coordinates"] = label_line["coordinates"]
            for cell in row.get("cells", []):
                if cell.get("grounding_status") != "unverified" or cell.get("raw_value") is None:
                    continue
                value_matches = [line for line in owned
                                 if line["evidence_id"] not in used_values
                                 and key(line.get("text")) == key(cell.get("raw_value"))
                                 and key(cell.get("raw_value"))
                                 and inside(line, cell.get("coordinates"))]
                plausible = []
                for line in value_matches:
                    value_x, value_y = center(line["coordinates"])
                    tolerance = max(20.0, (label_line["coordinates"][3] + line["coordinates"][3]) / 2 + 12)
                    if not cell.get("coordinates") and abs(value_y - label_y) > tolerance:
                        continue
                    if len(columns) == 2:
                        if value_x <= label_x + 5:
                            continue
                    else:
                        header = headers.get(cell["column_id"])
                        if header is None:
                            continue
                        header_x, _ = center(header["coordinates"])
                        if abs(value_x - header_x) > max(35.0, block["coordinates"][2] / (len(columns) * 1.4)):
                            continue
                    plausible.append(line)
                if len(plausible) != 1:
                    continue
                match = plausible[0]
                cell["coordinates"] = match["coordinates"]
                cell["evidence_ids"] = [match["evidence_id"]]
                cell["grounding_status"] = (
                    "ocr_value_and_row" if len(columns) == 2 else "ocr_row_and_column"
                )
                used_values.add(match["evidence_id"])
                stats["newly_grounded"] += 1
        remaining = sum(cell.get("raw_value") is not None and cell.get("grounding_status") == "unverified"
                        for row in rows for cell in row.get("cells", []))
        stats["still_unverified"] += remaining
        warnings = block.get("validation", {}).get("warnings", [])
        warnings[:] = [warning for warning in warnings
                       if not warning.endswith("table cells lack value-and-owner evidence; treat their values as candidates")]
        if remaining:
            warnings.append(
                f"{remaining} table cells lack value-and-owner evidence; treat their values as candidates"
            )
            block["validation"]["status"] = "needs_review"
    return stats


def _linked_table_scalar_count(blocks: list[dict[str, Any]]) -> int:
    """Count scalar table values with a printed, text-bearing row owner."""
    return sum(
        cell.get("numeric_value") is not None
        for block in blocks if block.get("type") == "table"
        for row in block.get("content", {}).get("rows", [])
        if re.search(r"[A-Za-z]", str(row.get("label") or ""))
        for cell in row.get("cells", [])
    )


def _infer_chart_type(
    region: Region, lines: list[dict[str, Any]], features: dict[str, Any],
    qwen_payload: dict[str, Any] | None,
) -> str:
    proposed = str((qwen_payload or {}).get("chart_type") or "").strip().lower()
    if proposed:
        return proposed
    hinted = str(region.metadata.get("chart_type_hint") or "").strip().lower()
    if hinted:
        return hinted
    percentage_count = sum("%" in str(line["text"]) for line in lines)
    image_count = len(region.metadata.get("image_indices", []))
    if image_count >= 4 and percentage_count >= 3:
        return "pie"
    if (
        int(features.get("point_candidates", 0)) >= 8
        and int(features.get("horizontal_axis_candidates", 0)) >= 1
        and int(features.get("vertical_axis_candidates", 0)) >= 1
        and sum(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", str(line.get("text", "")).strip())) for line in lines) >= 3
    ):
        return "scatterplot"
    if int(features.get("bar_candidates", 0)) >= 3:
        return "bar"
    numeric_count = sum(bool(NUMBER.search(str(line.get("text", "")))) for line in lines)
    magnitude_count = sum(
        bool(re.fullmatch(r"(?:thousand|million|billion|k|m|mm|b)", str(line.get("text", "")).strip(), re.I))
        for line in lines
    )
    currency_count = sum(
        bool(re.fullmatch(r"[$€£]", str(line.get("text", "")).strip())) for line in lines
    )
    if image_count >= 1 and numeric_count >= 2 and magnitude_count >= 1 and currency_count >= 1:
        return "kpi_panel"
    return "unknown"


def _nearest_mark(coordinates: list[float], marks: list[list[float]]) -> list[float] | None:
    if not marks:
        return None
    x, y, w, h = coordinates
    cx, cy = x + w / 2, y + h / 2
    return min(
        marks,
        key=lambda mark: math.hypot(cx - (mark[0] + mark[2] / 2), cy - (mark[1] + mark[3] / 2)),
    )


def _proximity_bindings(
    lines: list[dict[str, Any]], title: str | None, mark_boxes: list[list[float]],
) -> list[dict[str, Any]]:
    values = [line for line in lines if NUMBER.fullmatch(str(line["text"]).strip())]
    labels = [
        line for line in lines
        if not NUMBER.search(str(line["text"]))
        and str(line["text"]).strip() != (title or "")
        and len(str(line["text"]).strip()) >= 2
    ]
    pair_candidates = sorted(
        ((_line_distance(value, label), value, label) for value in values for label in labels),
        key=lambda item: item[0],
    )
    used_values: set[str] = set()
    used_labels: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for distance, value, label in pair_candidates:
        value_id, label_id = value["evidence_id"], label["evidence_id"]
        if value_id in used_values or label_id in used_labels or distance > 150:
            continue
        used_values.add(value_id)
        used_labels.add(label_id)
        numeric_value, unit, normalized_value = _numeric_value(str(value["text"]))
        evidence_box = _box_union(label["coordinates"], value["coordinates"])
        bindings.append({
            "label": str(label["text"]).strip(),
            "value": str(value["text"]).strip(),
            "numeric_value": numeric_value,
            "unit": unit,
            "normalized_value": normalized_value,
            "label_evidence_id": label_id,
            "value_evidence_id": value_id,
            "label_coordinates": label["coordinates"],
            "value_coordinates": value["coordinates"],
            "visual_mark_coordinates": _nearest_mark(evidence_box, mark_boxes),
            "coordinates": evidence_box,
            "grounding_method": "unique OCR label-value proximity with raster mark ownership",
            "distance": round(distance, 5),
            "confidence": round(max(0.70, 0.96 - distance / 500), 5),
        })
    return sorted(bindings, key=lambda item: (item["value_coordinates"][1], item["value_coordinates"][0]))


def _stacked_pie_bindings(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Read printed label / amount / percent stacks without native-word fragments.

    This applies only to mixed currency-and-percent pies. It does not claim
    ownership of a slice or assume that multiple pies share one denominator.
    """
    ocr = [line for line in lines if "-ocr-" in str(line.get("evidence_id", ""))]
    percentages = [line for line in ocr if re.fullmatch(
        r"[-+]?\d[\d,.]*\s*%", str(line.get("text", "")).strip(),
    )]
    amounts = [line for line in ocr if re.fullmatch(
        r"[$€£]\s*\d[\d,.]*(?:\s*(?:k|m|mm|million|b|bn|billion))?",
        str(line.get("text", "")).strip(), re.I,
    )]
    if len(percentages) < 2 or len(amounts) < 2:
        return None
    labels = [line for line in ocr if
        re.search(r"[A-Za-z]", str(line.get("text", "")))
        and not NUMBER.search(str(line.get("text", "")))
    ]
    used_amounts: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for percent in sorted(percentages, key=lambda line: (line["coordinates"][1], line["coordinates"][0])):
        px, py = _center(percent["coordinates"])
        amount_candidates = [amount for amount in amounts if
            amount["evidence_id"] not in used_amounts
            and 8 <= py - _center(amount["coordinates"])[1] <= 100
            and abs(px - _center(amount["coordinates"])[0]) <= 110
        ]
        if not amount_candidates:
            continue
        amount = min(amount_candidates, key=lambda item: (
            py - _center(item["coordinates"])[1]
            + abs(px - _center(item["coordinates"])[0]) * 0.4
        ))
        ax, ay = _center(amount["coordinates"])
        nearby_labels = [label for label in labels if
            8 <= ay - _center(label["coordinates"])[1] <= 120
            and abs(ax - _center(label["coordinates"])[0]) <= 120
        ]
        if not nearby_labels:
            continue
        nearest = max(nearby_labels, key=lambda item: _center(item["coordinates"])[1])
        group = [nearest]
        older = [label for label in nearby_labels if
            label is not nearest
            and 8 <= _center(nearest["coordinates"])[1] - _center(label["coordinates"])[1] <= 55
            and abs(_center(nearest["coordinates"])[0] - _center(label["coordinates"])[0]) <= 65
        ]
        if older:
            group.insert(0, max(older, key=lambda item: _center(item["coordinates"])[1]))
        label_text = " ".join(str(item["text"]).strip() for item in group)
        if len(group) == 2:
            connector = next((item for item in lines if
                str(item.get("text", "")).strip() in {"&", "/"}
                and min(item_box[0] for item_box in (entry["coordinates"] for entry in group)) - 20
                <= _center(item["coordinates"])[0]
                <= max(entry["coordinates"][0] + entry["coordinates"][2] for entry in group) + 30
                and abs(_center(item["coordinates"])[1] - _center(group[0]["coordinates"])[1]) <= 35
            ), None)
            if connector:
                label_text = f"{group[0]['text'].strip()} {connector['text'].strip()} {group[1]['text'].strip()}"
        numeric_value, unit, normalized_value = _numeric_value(str(percent["text"]))
        amount_numeric, amount_unit, amount_normalized = _numeric_value(str(amount["text"]))
        bindings.append({
            "label": label_text, "value": str(percent["text"]).strip(),
            "numeric_value": numeric_value, "unit": unit,
            "normalized_value": normalized_value,
            "label_evidence_id": group[0]["evidence_id"],
            "value_evidence_id": percent["evidence_id"],
            "label_coordinates": _box_union(*(item["coordinates"] for item in group)),
            "value_coordinates": percent["coordinates"],
            "visual_mark_coordinates": None,
            "coordinates": _box_union(*(item["coordinates"] for item in [*group, amount, percent])),
            "grounding_method": "OCR label/amount/percent stack; pie-slice ownership unresolved",
            "confidence": 0.72,
            "companion_value": {
                "raw_value": str(amount["text"]).strip(),
                "numeric_value": amount_numeric,
                "normalized_value": amount_normalized,
                "unit": amount_unit,
                "evidence_id": amount["evidence_id"],
                "coordinates": amount["coordinates"],
            },
        })
        used_amounts.add(amount["evidence_id"])
    return bindings


def _map_data_value_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return map data values while excluding nearby color-scale endpoints."""
    percentage_lines = [
        line for line in lines
        if re.fullmatch(r"[-+]?\d[\d,.]*\s*%", str(line.get("text", "")).strip())
    ]
    legend_labels = [
        line for line in lines
        if "%" in str(line.get("text", ""))
        and not re.search(r"\d", str(line.get("text", "")))
    ]
    return [
        value for value in percentage_lines
        if not any(
            abs(_center(value["coordinates"])[1] - _center(label["coordinates"])[1]) <= 75
            for label in legend_labels
        )
    ]


def _map_literal_context(lines: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Preserve scale endpoints and copyright without treating them as data."""
    labels = [
        line for line in lines
        if "%" in str(line.get("text", "")) and not re.search(r"\d", str(line.get("text", "")))
    ]
    legend = None
    if labels:
        label = labels[0]
        endpoints = [
            line for line in lines
            if re.fullmatch(r"[-+]?\d[\d,.]*\s*%", str(line.get("text", "")).strip())
            and abs(_center(line["coordinates"])[1] - _center(label["coordinates"])[1]) <= 75
        ]
        endpoints.sort(key=lambda line: _center(line["coordinates"])[0])
        legend = {
            "title": str(label["text"]).strip(),
            "endpoints": [str(line["text"]).strip() for line in endpoints],
            "evidence_ids": [label["evidence_id"], *(line["evidence_id"] for line in endpoints)],
        }
    credits = [
        {"text": str(line["text"]).strip(), "evidence_id": line["evidence_id"]}
        for line in lines
        if str(line.get("text", "")).strip().lower().startswith("powered by")
        or "©" in str(line.get("text", ""))
    ]
    return legend, credits


def _map_geometry_bindings(
    lines: list[dict[str, Any]], region: Region, crop_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Accept literal map values only with printed names or registered marks."""
    values = _map_data_value_lines(lines)
    allowed_ids = {line["evidence_id"] for line in values}
    registration = register_us_state_map(crop_path)
    warnings: list[str] = []
    registration_info: dict[str, Any] = {
        "status": "accepted" if registration else "unavailable",
        "reference": ATLAS_URL if registration else None,
        "projection": registration["projection"] if registration else None,
        "silhouette_iou": registration["quality"] if registration else None,
    }
    bindings: list[dict[str, Any]] = []
    for value in values:
        resolved = (
            state_for_value(registration, value["coordinates"], region.coordinates)
            if registration else None
        )
        if resolved is None:
            continue
        name, mark_box, method = resolved
        numeric_value, unit, normalized_value = _numeric_value(str(value["text"]))
        confidence = min(0.96, 0.72 + 0.25 * registration["quality"])
        if method == "registered_state_leader_line":
            confidence -= 0.04
        bindings.append({
            "label": name,
            "value": str(value["text"]).strip(),
            "numeric_value": numeric_value,
            "unit": unit,
            "normalized_value": normalized_value,
            "label_evidence_id": None,
            "value_evidence_id": value["evidence_id"],
            "label_coordinates": None,
            "value_coordinates": value["coordinates"],
            "visual_mark_coordinates": mark_box,
            "coordinates": _box_union(value["coordinates"], mark_box),
            "grounding_method": f"OCR value + {method} + registered public US-state boundary",
            "confidence": round(confidence, 5),
            "geography_basis": "reference_geometry",
            "mark_validation": method,
            "mark_verified": True,
        })
    # Text printed on the map itself is a separate, stronger owner signal. It
    # also handles territorial insets for which a mainland atlas cannot fit.
    literal = _proximity_bindings(lines, None, [])
    used_values = {binding["value_evidence_id"] for binding in bindings}
    used_names = {str(binding["label"]).casefold() for binding in bindings}
    for candidate in literal:
        name = str(candidate["label"]).strip()
        if (
            candidate["value_evidence_id"] not in allowed_ids
            or candidate["value_evidence_id"] in used_values
            or name.casefold() not in US_GEOGRAPHIES
            or name.casefold() in used_names
        ):
            continue
        mark = colored_mark_for_value(crop_path, candidate["value_coordinates"], region.coordinates)
        if mark is None:
            continue
        candidate.update({
            "visual_mark_coordinates": mark,
            "grounding_method": "printed geography label + OCR value proximity + filled raster mark",
            "geography_basis": "printed_label",
            "mark_validation": "filled_raster_component",
            "mark_verified": True,
            "confidence": max(0.90, candidate["confidence"]),
        })
        bindings.append(candidate)
        used_values.add(candidate["value_evidence_id"])
        used_names.add(name.casefold())
    if registration:
        warnings.append(
            f"US-state reference registration: {registration['projection']} silhouette IoU {registration['quality']:.5f}"
        )
    if len(bindings) < len(values):
        warnings.append(f"{len(values) - len(bindings)} printed map value(s) remain without independently checked ownership")
    return sorted(bindings, key=lambda binding: (
        binding["value_coordinates"][1], binding["value_coordinates"][0]
    )), registration_info, warnings


def _qwen_semantic_prompt(
    kind: str, lines: list[dict[str, Any]], region: Region,
    target_value_ids: list[str] | None = None,
) -> str:
    """Build an evidence-aware semantic prompt after OCR and geometry stages complete."""
    evidence = [
        {
            "evidence_id": str(line.get("evidence_id", "")),
            "text": str(line.get("text", "")),
            "coordinates": [round(float(value), 2) for value in line.get("coordinates", [])],
        }
        for line in lines[:100]
        if str(line.get("text", "")).strip()
    ]
    if kind == "table":
        panel_layout = reconstruct_comparison_panel(region.metadata.get("native_panel_words") or [])
        if panel_layout:
            candidate = {
                "lane_count": panel_layout["lane_count"],
                "leaf_titles": panel_layout["leaf_titles"],
                "claim_counts": [
                    len(leaf.get("claims", []))
                    for section in panel_layout["sections"]
                    for leaf in (section.get("subsections") or [section])
                ],
            }
            return (
                "This is a nested comparison panel, not a row-by-column data table. Independently inspect "
                "the image and verify the number of text lanes, each offer heading, and the number of bullet "
                "claims in each lane. Do not transcribe or revise claims. Return exactly one JSON object with "
                "keys type, confidence, chart_type, bindings, panel_review. Use type table, chart_type null, "
                "bindings []. panel_review must contain lane_count, leaf_titles (left to right), claim_counts "
                "(left to right), and structure_matches. Set structure_matches false when any candidate item "
                "disagrees with the image.\nCandidate:\n"
                + json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            )
        rows = region.metadata.get("rows") or []
        width = max((len(row) for row in rows), default=0)
        candidate = {
            "data_row_count": max(0, len(rows) - 1),
            "column_count": width,
            "headers": [str(value or "") for value in rows[0]] if rows else [],
        }
        return (
            "Verify only this table structure and return immediately as one JSON object with keys type, "
            "confidence, chart_type, bindings, table_review. Use type table, chart_type null, and bindings []. "
            "table_review must contain data_row_count (excluding the header), column_count, headers, and "
            "structure_matches. Do not transcribe cells or explain. Count section labels as labels, not data "
            "rows.\nCandidate:\n"
            + json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
        )
    contract = (
        "Keep bindings compact: return only label, label_evidence_id, and value_evidence_id. Use only supplied "
        "evidence IDs. The pipeline will copy the visible OCR value from value_evidence_id. For charts, label "
        "must be the visible category and label_evidence_id is required. For KPI panels, label must be the "
        "visible metric label and label_evidence_id is required. For maps, label must be the US state or "
        "territory owning the value and label_evidence_id may be null. Do not omit clear bindings or invent values."
    )
    target_instruction = ""
    if kind == "map" and target_value_ids:
        target_instruction = (
            " Return at most one binding for each of these target value IDs and no bindings for other values: "
            + json.dumps(target_value_ids, ensure_ascii=False, separators=(",", ":"))
            + ". Omit a target when it is a legend endpoint or its geography is not visually clear."
        )
    return (
        f"This is the mandatory semantic-linking stage after OCR and OpenCV geometry for a {kind}. "
        "Return exactly one JSON object with exactly four keys: type, confidence, chart_type, bindings. "
        f"type must be {kind}; confidence must be a number from zero to one. "
        + contract
        + target_instruction
        + "\nOCR evidence (normalized page coordinates):\n"
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    )


def _ground_model_bindings(
    kind: str, payload: dict[str, Any] | None, lines: list[dict[str, Any]],
    mark_boxes: list[list[float]],
) -> list[dict[str, Any]]:
    """Ground model associations back to exact OCR evidence before accepting them."""
    if not payload:
        return []
    by_id = {str(line.get("evidence_id")): line for line in lines}

    def resolve_line(evidence_id: Any, text: str, require_unique: bool = True) -> dict[str, Any] | None:
        candidate = by_id.get(str(evidence_id)) if evidence_id else None
        if candidate is not None:
            return candidate
        folded = _fold_token(text.strip())
        matches = [line for line in lines if _fold_token(str(line.get("text", "")).strip()) == folded]
        if not matches:
            # OCR often drops spaces around ranges or uses full-width symbols.
            # Accept that typography difference only when the match is unique.
            compact = re.sub(r"\W+", "", folded, flags=re.UNICODE)
            matches = [
                line for line in lines
                if re.sub(r"\W+", "", _fold_token(str(line.get("text", "")).strip()), flags=re.UNICODE) == compact
            ] if compact else []
            if not matches and len(compact) >= 12:
                matches = [
                    line for line in lines
                    if len(part := re.sub(r"\W+", "", _fold_token(str(line.get("text", "")).strip()), flags=re.UNICODE)) >= 12
                    and compact.startswith(part)
                ]
        if len(matches) == 1 or (matches and not require_unique):
            return matches[0]
        return None

    grounded: list[dict[str, Any]] = []
    for binding in payload.get("bindings", []):
        label_text = str(
            binding.get("label") or binding.get("category") or binding.get("geography") or ""
        ).strip()
        value_text = str(binding.get("value") or binding.get("raw_value") or "").strip()
        value_id = binding.get("value_evidence_id")
        label_id = binding.get("label_evidence_id")
        if not label_text and not label_id:
            continue
        if not value_text and not value_id:
            continue
        value_line = resolve_line(value_id, value_text)
        label_line = resolve_line(label_id, label_text)
        if value_line is None or (kind in {"chart", "kpi_panel"} and label_line is None):
            continue
        if kind == "map" and label_text.casefold() not in US_GEOGRAPHIES:
            continue
        # OCR remains authoritative for the visible value and chart category.
        value_text = str(value_line.get("text", "")).strip()
        if label_line is not None and (kind in {"chart", "kpi_panel"} or not label_text):
            label_text = str(label_line.get("text", "")).strip()
        numeric_value, unit, normalized_value = _numeric_value(value_text)
        label_coordinates = label_line.get("coordinates") if label_line else None
        value_coordinates = value_line.get("coordinates")
        evidence_box = _box_union(*(
            box for box in (label_coordinates, value_coordinates) if box is not None
        ))
        grounded.append({
            "label": label_text,
            "value": value_text,
            "numeric_value": numeric_value,
            "unit": unit,
            "normalized_value": normalized_value,
            "label_evidence_id": label_line.get("evidence_id") if label_line else None,
            "value_evidence_id": value_line.get("evidence_id"),
            "label_coordinates": label_coordinates,
            "value_coordinates": value_coordinates,
            "visual_mark_coordinates": _nearest_mark(evidence_box, mark_boxes),
            "coordinates": evidence_box,
            "grounding_method": "Qwen semantic association grounded to OCR evidence and OpenCV geometry",
            "confidence": min(float(payload.get("confidence", 0.0)), float(binding.get("confidence", 1.0))),
        })
    # A model may repeat a value or assign it to multiple owners. Collapse
    # identical repeats, but reject the entire value when ownership conflicts.
    by_value: dict[str, list[dict[str, Any]]] = {}
    for binding in grounded:
        by_value.setdefault(str(binding.get("value_evidence_id") or ""), []).append(binding)
    unambiguous: list[dict[str, Any]] = []
    for candidates in by_value.values():
        labels = {_fold_token(str(candidate.get("label", ""))) for candidate in candidates}
        if len(labels) != 1:
            continue
        unambiguous.append(max(candidates, key=lambda candidate: float(candidate.get("confidence", 0.0))))
    return unambiguous


def _reconcile_visual_bindings(
    deterministic: list[dict[str, Any]], model: list[dict[str, Any]], allow_model_additions: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Union grounded model evidence with deterministic evidence without allowing replacement."""
    reconciled = list(deterministic)
    warnings: list[str] = []
    by_value_id = {
        str(binding.get("value_evidence_id")): binding
        for binding in reconciled if binding.get("value_evidence_id")
    }
    confirmed = 0
    conflicts = 0
    added = 0
    unverified = 0
    for binding in model:
        value_id = str(binding.get("value_evidence_id") or "")
        existing = by_value_id.get(value_id)
        if existing is None:
            if not allow_model_additions:
                unverified += 1
                continue
            reconciled.append(binding)
            if value_id:
                by_value_id[value_id] = binding
            added += 1
            continue
        if _fold_token(str(existing.get("label", ""))) == _fold_token(str(binding.get("label", ""))):
            existing["grounding_method"] = f"{existing.get('grounding_method')}; Qwen-confirmed"
            existing["confidence"] = max(float(existing.get("confidence", 0.0)), float(binding.get("confidence", 0.0)))
            confirmed += 1
        else:
            conflicts += 1
    if conflicts:
        warnings.append(f"Qwen disagreed with deterministic ownership for {conflicts} value(s); deterministic evidence retained")
    if model and len(model) < len(deterministic):
        warnings.append(
            f"Qwen returned {len(model)} grounded binding(s) for {len(deterministic)} deterministic binding(s); "
            "deterministic evidence retained"
        )
    if model:
        warnings.append(f"Qwen semantic reconciliation: {confirmed} confirmed, {added} added, {conflicts} conflicted")
    if unverified:
        warnings.append(
            f"Qwen proposed {unverified} semantic ownership binding(s) that Python could not independently validate; omitted"
        )
    return reconciled, warnings


def _pie_label_value_grounding(
    bindings: list[dict[str, Any]], masked_slices: list[dict[str, Any]] | None = None,
) -> bool:
    """Bind pie values to distinct PDF alpha masks only when mask areas reconcile."""
    candidates = list(masked_slices or [])
    verified = False
    if 3 <= len(bindings) == len(candidates) <= 12 and all(
        binding.get("unit") == "percent" and binding.get("numeric_value") is not None
        for binding in bindings
    ):
        total_area = sum(float(candidate.get("projected_alpha_area") or 0) for candidate in candidates)
        if total_area > 0 and len({candidate.get("smask_sha256") for candidate in candidates}) == len(candidates):
            areas = [100 * float(candidate["projected_alpha_area"]) / total_area for candidate in candidates]

            @lru_cache(maxsize=None)
            def assignment(index: int, used: int) -> tuple[float, tuple[int, ...]]:
                if index == len(bindings):
                    return 0.0, ()
                binding = bindings[index]
                evidence_box = _box_union(
                    binding.get("label_coordinates"), binding.get("value_coordinates"),
                )
                owner_center = _center(evidence_box)
                target = float(binding["numeric_value"])
                best: tuple[float, tuple[int, ...]] = (float("inf"), ())
                for candidate_index, candidate in enumerate(candidates):
                    if used & (1 << candidate_index):
                        continue
                    centroid = candidate["alpha_centroid_coordinates"]
                    spatial_cost = math.dist(owner_center, centroid) / 1000.0
                    area_cost = abs(areas[candidate_index] - target) * 20.0
                    remaining, tail = assignment(index + 1, used | (1 << candidate_index))
                    trial = (area_cost + spatial_cost + remaining, (candidate_index,) + tail)
                    if trial[0] < best[0]:
                        best = trial
                return best

            _, owners = assignment(0, 0)
            verified = len(owners) == len(bindings) and all(
                abs(areas[candidate_index] - float(binding["numeric_value"]))
                <= max(0.15, float(binding["numeric_value"]) * 0.02)
                for binding, candidate_index in zip(bindings, owners)
            )
            if verified:
                for binding, candidate_index in zip(bindings, owners):
                    candidate = candidates[candidate_index]
                    qwen_confirmed = "Qwen-confirmed" in str(binding.get("grounding_method") or "")
                    binding["visual_mark_coordinates"] = candidate["mark_coordinates"]
                    binding["visual_mark_ref"] = {
                        "source": "pdf_soft_mask",
                        "pdf_image_index": candidate["pdf_image_index"],
                        "smask_sha256": candidate["smask_sha256"],
                        "opacity_weighted_area_share_percent": round(areas[candidate_index], 5),
                    }
                    binding["grounding_method"] = (
                        "OCR label-value spatial association"
                        + ("; Qwen-confirmed label-value association" if qwen_confirmed else "")
                        + "; one-to-one PDF soft-mask opacity-weighted area/value and position match"
                    )
    if verified:
        return True
    for binding in bindings:
        qwen_confirmed = "Qwen-confirmed" in str(binding.get("grounding_method") or "")
        binding["visual_mark_coordinates"] = None
        binding["visual_mark_ref"] = None
        binding["grounding_method"] = (
            "OCR label-value spatial association"
            + ("; Qwen-confirmed label-value association" if qwen_confirmed else "")
            + "; individual pie-slice geometry unresolved"
        )
    return False


def _dedupe_spatial_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer longer OCR phrases while removing equivalent native/OCR duplicates."""
    kept: list[dict[str, Any]] = []
    for line in sorted(
        lines,
        key=lambda item: (
            0 if "-ocr-" in str(item.get("evidence_id", "")) else 1,
            -len(str(item.get("text", ""))),
        ),
    ):
        text = _fold_token(str(line.get("text", "")).strip())
        if not text:
            continue
        cx, cy = _center(line["coordinates"])
        if any(
            _fold_token(str(existing.get("text", "")).strip()) == text
            and abs(cx - _center(existing["coordinates"])[0]) <= 20
            and abs(cy - _center(existing["coordinates"])[1]) <= 35
            for existing in kept
        ):
            continue
        kept.append(line)
    return kept


def _as_of_date(lines: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for line in lines:
        text = str(line.get("text", "")).strip()
        match = re.search(r"\b(?:as\s+of\s+)?(\d{1,2})/(\d{1,2})/(\d{2,4})\b", text, re.I)
        if not match:
            continue
        month, day, year = (int(value) for value in match.groups())
        year += 2000 if year < 100 else 0
        try:
            return datetime(year, month, day).date().isoformat(), str(line.get("evidence_id"))
        except ValueError:
            return None, str(line.get("evidence_id"))
    return None, None


def _join_kpi_label(lines: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    if not lines:
        return None, None
    ordered = sorted(lines, key=lambda line: (line["coordinates"][1], line["coordinates"][0]))
    parts: list[str] = []
    used: list[dict[str, Any]] = []
    accumulated = ""
    for line in ordered:
        text = re.sub(r"\s+", " ", str(line.get("text", ""))).strip(" -")
        if not text:
            continue
        folded = _fold_token(text)
        if folded in _fold_token(accumulated):
            continue
        # Native positioned words frequently duplicate a complete OCR phrase.
        if any(folded in _fold_token(str(other.get("text", ""))) and other is not line for other in ordered):
            continue
        parts.append(text)
        used.append(line)
        accumulated = " ".join(parts)
    if not parts:
        return None, None
    return " ".join(parts), used[0]


def _kpi_bindings(
    lines: list[dict[str, Any]], mark_boxes: list[list[float]],
) -> tuple[list[dict[str, Any]], str | None, int]:
    """Reconstruct repeated amount/count KPI cards from geometry and unit ownership."""
    lines = _dedupe_spatial_lines(lines)
    as_of, _ = _as_of_date(lines)
    numeric_lines = [
        line for line in lines
        if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", str(line.get("text", "")).strip())
    ]
    unit_lines = [
        line for line in lines
        if re.fullmatch(r"(?:thousand|million|billion|k|m|mm|b)", str(line.get("text", "")).strip(), re.I)
    ]
    currency_lines = [
        line for line in lines if re.fullmatch(r"[$€£]", str(line.get("text", "")).strip())
    ]
    amount_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = []
    used_numbers: set[str] = set()
    for unit in sorted(unit_lines, key=lambda item: item["coordinates"][1]):
        ux, uy = _center(unit["coordinates"])
        candidates = []
        for number in numeric_lines:
            nx, ny = _center(number["coordinates"])
            unit_top = unit["coordinates"][1] - 25
            unit_bottom = unit["coordinates"][1] + unit["coordinates"][3] + 25
            if number["evidence_id"] not in used_numbers and nx < ux + 25 and unit_top <= ny <= unit_bottom:
                candidates.append(number)
        if not candidates:
            continue
        number = min(candidates, key=lambda item: _line_distance(item, unit))
        nx, ny = _center(number["coordinates"])
        currency = min(
            (
                item for item in currency_lines
                if _center(item["coordinates"])[0] < nx
                and abs(_center(item["coordinates"])[1] - ny) <= 90
            ),
            key=lambda item: _line_distance(item, number),
            default=None,
        )
        if currency is None:
            continue
        used_numbers.add(str(number["evidence_id"]))
        amount_rows.append((number, unit, currency))
    amount_rows.sort(key=lambda row: _center(row[0]["coordinates"])[1])
    if not amount_rows:
        return [], as_of, 0

    bindings: list[dict[str, Any]] = []
    completed_groups = 0
    centers = [_center(row[0]["coordinates"])[1] for row in amount_rows]
    for index, (amount, unit, currency) in enumerate(amount_rows):
        amount_x, amount_y = _center(amount["coordinates"])
        lower = (centers[index - 1] + amount_y) / 2 if index else amount_y - 140
        upper = (amount_y + centers[index + 1]) / 2 if index + 1 < len(centers) else amount_y + 180
        count_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for number in numeric_lines:
            if number["evidence_id"] in used_numbers:
                continue
            nx, ny = _center(number["coordinates"])
            if not lower <= ny <= upper:
                continue
            owners = [
                line for line in lines
                if re.search(r"[A-Za-z]", str(line.get("text", "")))
                and float(line["coordinates"][2]) >= 1.2 * max(1.0, float(line["coordinates"][3]))
                and not NUMBER.search(str(line.get("text", "")))
                and not re.search(r"\bas\s+of\b", str(line.get("text", "")), re.I)
                and not re.fullmatch(
                    r"(?:thousand|million|billion|k|m|mm|b)",
                    str(line.get("text", "")).strip(), re.I,
                )
                and 1 <= len(str(line.get("text", "")).split()) <= 4
                and abs(_center(line["coordinates"])[0] - nx) <= 130
                and 0 <= _center(line["coordinates"])[1] - ny <= 100
            ]
            if owners:
                owner = min(owners, key=lambda line: _line_distance(number, line))
                count_candidates.append((_line_distance(number, owner), number, owner))
        count = count_owner = None
        if count_candidates:
            _, count, count_owner = min(count_candidates, key=lambda item: item[0])

        count_x = _center(count["coordinates"])[0] if count else 1000.0
        label_lines = [
            line for line in lines
            if "-ocr-" in str(line.get("evidence_id", ""))
            and re.search(r"[A-Za-z]", str(line.get("text", "")))
            and not re.search(r"\bas\s+of\b", str(line.get("text", "")), re.I)
            and not re.fullmatch(r"(?:thousand|million|billion|k|m|mm|b)", str(line.get("text", "")).strip(), re.I)
            and lower <= _center(line["coordinates"])[1] <= upper
            and _center(line["coordinates"])[1] >= amount_y + 45
            and abs(_center(line["coordinates"])[0] - amount_x) <= 190
            and _center(line["coordinates"])[0] < count_x - 30
        ]
        label, label_line = _join_kpi_label(label_lines)
        if not label or not label_line:
            continue
        raw_amount = f"{str(currency['text']).strip()}{str(amount['text']).strip()} {str(unit['text']).strip().lower()}"
        numeric_value = float(str(amount["text"]).replace(",", ""))
        multiplier = {
            "thousand": 1_000.0, "k": 1_000.0,
            "million": 1_000_000.0, "m": 1_000_000.0, "mm": 1_000_000.0,
            "billion": 1_000_000_000.0, "b": 1_000_000_000.0,
        }[_fold_token(str(unit["text"]).strip())]
        amount_mark = _nearest_mark(_box_union(amount["coordinates"], unit["coordinates"]), mark_boxes)
        bindings.append({
            "label": "amount", "series": label, "value": raw_amount,
            "numeric_value": numeric_value, "unit": "USD", "normalized_value": numeric_value * multiplier,
            "label_evidence_id": label_line["evidence_id"], "value_evidence_id": amount["evidence_id"],
            "label_coordinates": _box_union(*(line["coordinates"] for line in label_lines)),
            "value_coordinates": _box_union(currency["coordinates"], amount["coordinates"], unit["coordinates"]),
            "visual_mark_coordinates": amount_mark,
            "grounding_method": "currency-number-unit row with vertically owned KPI label",
            "confidence": 0.94,
        })
        if count is not None and count_owner is not None:
            count_value = float(str(count["text"]).replace(",", ""))
            bindings.append({
                "label": str(count_owner["text"]).strip(), "series": label, "value": str(count["text"]).strip(),
                "numeric_value": count_value, "unit": "count", "normalized_value": count_value,
                "label_evidence_id": count_owner["evidence_id"], "value_evidence_id": count["evidence_id"],
                "label_coordinates": count_owner["coordinates"], "value_coordinates": count["coordinates"],
                "visual_mark_coordinates": _nearest_mark(_box_union(count["coordinates"], count_owner["coordinates"]), mark_boxes),
                "grounding_method": "number with vertically adjacent count label inside KPI row",
                "confidence": 0.94,
            })
            completed_groups += 1
    return bindings, as_of, completed_groups


_CHART_CATEGORY = re.compile(
    r"^(?:(?:19|20)\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)-\d{2,4})$",
    re.I,
)
_CHART_VALUE = re.compile(
    r"^(?:[$€£]\s*)?[-+]?\d[\d,.]*(?:\s*(?:%|x|m|mm|million|b|billion|k))$",
    re.I,
)


def _center(box: list[float]) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _bar_bindings(
    lines: list[dict[str, Any]], title: str | None, mark_boxes: list[list[float]],
) -> list[dict[str, Any]]:
    """Bind vertical bars to quantitative OCR and x-axis categories by x alignment."""
    marks: list[list[float]] = []
    for mark in sorted(mark_boxes, key=lambda box: _center(box)[0]):
        if not any(abs(_center(mark)[0] - _center(existing)[0]) <= 1.0 for existing in marks):
            marks.append(mark)
    if len(marks) < 3:
        return []
    baseline = statistics.median(mark[1] + mark[3] for mark in marks)
    highest_mark = min(mark[1] for mark in marks)
    categories = [
        line for line in lines
        if _CHART_CATEGORY.fullmatch(str(line["text"]).strip())
        and _center(line["coordinates"])[1] >= baseline - 15
    ]
    values = [
        line for line in lines
        if _CHART_VALUE.fullmatch(str(line["text"]).strip())
        and (_numeric_value(str(line["text"]))[1] in {"percent", "USD", "multiple"})
        and highest_mark - 35 <= _center(line["coordinates"])[1] <= baseline + 25
    ]
    min_mark_x = min(mark[0] for mark in marks)
    max_mark_x = max(mark[0] + mark[2] for mark in marks)
    series_labels = sorted([
        line for line in lines
        if _center(line["coordinates"])[1] > baseline
        and min_mark_x - 120 <= _center(line["coordinates"])[0] <= max_mark_x + 120
        and 1 <= len(str(line["text"]).split()) <= 8
        and not NUMBER.search(str(line["text"]))
        and str(line["text"]).strip() != (title or "")
        and not _CHART_CATEGORY.fullmatch(str(line["text"]).strip())
    ], key=lambda line: line["coordinates"][0])
    assignments: list[tuple[list[float], dict[str, Any]]] = []
    available = list(marks)
    for value in sorted(values, key=lambda line: _center(line["coordinates"])[0]):
        if not available:
            break
        value_x, _ = _center(value["coordinates"])
        mark = min(available, key=lambda box: abs(_center(box)[0] - value_x))
        if abs(_center(mark)[0] - value_x) > 55:
            continue
        available.remove(mark)
        assignments.append((mark, value))

    category_groups: dict[str, list[list[float]]] = {}
    for mark in marks:
        if not categories:
            continue
        mark_x, _ = _center(mark)
        category = min(categories, key=lambda line: abs(_center(line["coordinates"])[0] - mark_x))
        if abs(_center(category["coordinates"])[0] - mark_x) <= 75:
            category_groups.setdefault(category["evidence_id"], []).append(mark)

    bindings: list[dict[str, Any]] = []
    for mark, value in sorted(assignments, key=lambda item: _center(item[0])[0]):
        if not categories:
            continue
        mark_x, _ = _center(mark)
        category = min(categories, key=lambda line: abs(_center(line["coordinates"])[0] - mark_x))
        category_dx = abs(_center(category["coordinates"])[0] - mark_x)
        value_dx = abs(_center(value["coordinates"])[0] - mark_x)
        if category_dx > 75:
            continue
        group = sorted(category_groups.get(category["evidence_id"], []), key=lambda box: _center(box)[0])
        series = None
        if len(group) > 1 and series_labels:
            rank = group.index(mark)
            series = str(series_labels[min(rank, len(series_labels) - 1)]["text"]).strip()
        numeric_value, unit, normalized_value = _numeric_value(str(value["text"]))
        confidence = max(0.70, 0.98 - value_dx / 300 - category_dx / 400)
        bindings.append({
            "label": str(category["text"]).strip(),
            "series": series,
            "value": str(value["text"]).strip(),
            "numeric_value": numeric_value,
            "unit": unit,
            "normalized_value": normalized_value,
            "label_evidence_id": category["evidence_id"],
            "value_evidence_id": value["evidence_id"],
            "label_coordinates": category["coordinates"],
            "value_coordinates": value["coordinates"],
            "visual_mark_coordinates": mark,
            "coordinates": _box_union(category["coordinates"], value["coordinates"], mark),
            "grounding_method": "x-aligned OCR value/category with PDF vector bar mark",
            "confidence": round(confidence, 5),
        })
    return bindings


def _scatter_axis_evidence(
    lines: list[dict[str, Any]], marks: list[list[float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calibrate a scatterplot axis and retain point *ranges*, never exact values."""
    if not marks:
        return {"status": "unavailable", "y_ticks": [], "x_ticks": [],
                "max_tick_residual_percent": None,
                "warnings": ["no independently detected point marks"]}, []
    min_mark_x = min(mark[0] for mark in marks)
    tick_lines = [line for line in lines if
        "-ocr-" in str(line.get("evidence_id", ""))
        and re.fullmatch(r"[-+]?\d[\d,.]*\s*%", str(line.get("text", "")).strip())
        and _center(line["coordinates"])[0] < min_mark_x - 8
    ]
    ticks = []
    for line in tick_lines:
        numeric, unit, _ = _numeric_value(str(line["text"]))
        if unit == "percent" and numeric is not None:
            ticks.append({"raw_value": str(line["text"]).strip(), "percent": numeric,
                          "evidence_id": str(line["evidence_id"]),
                          "coordinates": line["coordinates"]})
    ticks.sort(key=lambda item: _center(item["coordinates"])[1])
    unique_ticks = []
    for tick in ticks:
        if not any(abs(tick["percent"] - other["percent"]) < 1e-6 for other in unique_ticks):
            unique_ticks.append(tick)
    ticks = unique_ticks
    if len(ticks) < 3:
        return {"status": "unavailable", "y_ticks": ticks, "x_ticks": [],
                "max_tick_residual_percent": None,
                "warnings": ["fewer than three distinct OCR y-axis percentage ticks"]}, [
            {"mark_id": f"p{index:03d}", "visual_mark_coordinates": mark,
             "estimated_percent_range": None, "x_interval": None,
             "validation_status": "needs_review"}
            for index, mark in enumerate(marks, 1)
        ]
    ys = [_center(tick["coordinates"])[1] for tick in ticks]
    values = [tick["percent"] for tick in ticks]
    mean_y, mean_value = statistics.mean(ys), statistics.mean(values)
    denominator = sum((y - mean_y) ** 2 for y in ys)
    slope = sum((y - mean_y) * (value - mean_value) for y, value in zip(ys, values)) / denominator if denominator else 0.0
    intercept = mean_value - slope * mean_y
    residual = max(abs((slope * y + intercept) - value) for y, value in zip(ys, values))
    accepted = slope < 0 and max(ys) - min(ys) >= 100 and residual <= 3.0
    zero_tick = min(ticks, key=lambda item: abs(item["percent"]))
    zero_y = _center(zero_tick["coordinates"])[1]
    x_ticks = [
        {"label": str(line["text"]).strip(), "evidence_id": str(line["evidence_id"]),
         "coordinates": line["coordinates"]}
        for line in lines
        if "-ocr-" in str(line.get("evidence_id", ""))
        and not re.search(r"%|[$€£]", str(line.get("text", "")))
        and 1 <= len(str(line.get("text", "")).split()) <= 3
        and abs(_center(line["coordinates"])[1] - zero_y) <= 45
        and _center(line["coordinates"])[0] > min_mark_x - 100
    ]
    x_ticks.sort(key=lambda item: _center(item["coordinates"])[0])
    warnings = ["point values are axis-interpolated ranges, not printed exact observations"]
    if not accepted:
        warnings.append("OCR y-axis ticks do not support a reliable linear calibration")
    points = []
    for index, mark in enumerate(marks, 1):
        mx, my = _center(mark)
        left = max((tick for tick in x_ticks if _center(tick["coordinates"])[0] <= mx),
                   key=lambda tick: _center(tick["coordinates"])[0], default=None)
        right = min((tick for tick in x_ticks if _center(tick["coordinates"])[0] >= mx),
                    key=lambda tick: _center(tick["coordinates"])[0], default=None)
        x_interval = [left["label"], right["label"]] if left and right else None
        if accepted:
            estimate = slope * my + intercept
            uncertainty = max(1.0, abs(slope) * (mark[3] / 2 + 5) + residual)
            value_range = [round(estimate - uncertainty, 1), round(estimate + uncertainty, 1)]
        else:
            value_range = None
        points.append({
            "mark_id": f"p{index:03d}", "visual_mark_coordinates": mark,
            "estimated_percent_range": value_range, "x_interval": x_interval,
            "validation_status": "needs_review",
        })
    return {
        "status": "accepted" if accepted else "unavailable", "y_ticks": ticks,
        "x_ticks": x_ticks, "max_tick_residual_percent": round(residual, 3),
        "warnings": warnings,
    }, points


def _visual_blocks(
    document_id: str, source_hash: str, region: Region, kind: str, classification_confidence: float,
    classification_warnings: list[str], lines: list[dict[str, Any]], features: dict[str, Any],
    image: Path, crop_path: Path, qwen_payload: dict[str, Any] | None,
    vision_features_ref: dict[str, str],
) -> list[SourceBlock]:
    parent_id = f"{region.region_id}-{kind.replace('_', '-')}"
    methods = ["RapidOCR", "PP-OCRv6", "OpenCV", "Python reconstruction"]
    if str(region.metadata.get("object_family") or "").startswith("pdf-vector"):
        methods.insert(-1, "PDF vector geometry")
    if qwen_payload is not None:
        methods.insert(-1, "Qwen3-VL semantic linking")
    labels = [str(line["text"]) for line in lines]
    title = _visual_title(kind, lines, region)
    chart_type = _infer_chart_type(region, lines, features, qwen_payload) if kind == "chart" else None
    if kind == "chart" and chart_type == "kpi_panel":
        kind = "kpi_panel"
    parent_status = "passed" if classification_confidence >= 0.75 else "needs_review"
    if kind == "chart" and chart_type not in {"bar", "pie", "scatterplot"}:
        parent_status = "needs_review"
        classification_warnings = list(classification_warnings) + [
            f"{chart_type or 'unknown'} chart family has no independently verified mark ownership"
        ]
    if kind == "unclassified_visual":
        parent_status = "needs_review"
        classification_warnings = list(classification_warnings) + [
            "unclassified visual cannot be accepted as a semantic block"
        ]
    if kind == "decoration" and int(region.metadata.get("semantic_text_overlap_count", 0)):
        parent_status = "needs_review"
        classification_warnings = list(classification_warnings) + [
            "decorative image overlaps semantic text; retain as background evidence only"
        ]
    base_visual = {
        "vision_summary": _vision_summary(features),
        "vision_features_ref": vision_features_ref,
        "region_image": f"region-images/{crop_path.name}",
    }
    if kind == "decoration":
        content: dict[str, Any] = {
            "role": "repeated_page_band" if region.metadata.get("repeated_on_pages") else "decorative_artwork",
            "repeated_on_pages": region.metadata.get("repeated_on_pages"),
            **base_visual,
        }
    elif kind == "photograph":
        content = {"caption": title, **base_visual}
    elif kind == "brand_mark":
        content = {"visible_text": labels, "evidence_mode": "visual_region", **base_visual}
    elif kind == "unclassified_visual":
        content = {"visible_text": labels, **base_visual}
    elif kind == "kpi_panel":
        content = {"title": None, "as_of": None, "metrics": [], **base_visual}
    elif kind == "map":
        legend, attribution = _map_literal_context(lines)
        content = {
            "title": title, "bindings": [], "legend": legend,
            "attribution": attribution, **base_visual,
        }
    else:
        content = {"title": title, "chart_type": chart_type, "observations": [], **base_visual}
    parent = SourceBlock(
        document_id=document_id, type=kind, page=region.page, block_id=parent_id,
        content=content,
        coordinates=region.coordinates, extraction_method=methods, confidence=classification_confidence,
        validation_status=parent_status,
        errors=[] if labels or kind in {"unclassified_visual", "photograph", "decoration"} else ["no visible labels recovered"],
        warnings=classification_warnings,
        provenance=_provenance(source_hash, region, lines, image),
        semantic_role="background_decoration" if kind == "decoration" else None,
    )
    if kind not in {"chart", "map", "kpi_panel"}:
        return [parent]
    mark_boxes = region.metadata.get("member_coordinates", [])
    if kind == "kpi_panel" and not mark_boxes:
        mark_boxes = [region.coordinates]
    deterministic_bindings: list[dict[str, Any]] = []
    completed_kpi_groups = 0
    mixed_pie_bindings = _stacked_pie_bindings(lines) if kind == "chart" and chart_type == "pie" else None
    if kind == "map":
        deterministic_bindings, registration_info, map_warnings = _map_geometry_bindings(
            lines, region, crop_path
        )
        parent.content["registration"] = registration_info
        parent.warnings.extend(map_warnings)
    elif chart_type == "bar":
        deterministic_bindings = _bar_bindings(lines, title, mark_boxes)
    elif chart_type == "scatterplot":
        calibration, points = _scatter_axis_evidence(lines, mark_boxes)
        parent.content["axis_calibration"] = calibration
        parent.content["scatter_points"] = points
        parent.validation_status = "needs_review"
        parent.warnings.extend(calibration["warnings"])
    elif chart_type == "kpi_panel" or kind == "kpi_panel":
        single_card = region.metadata.get("single_card_binding")
        if single_card:
            deterministic_bindings = [single_card]
        else:
            deterministic_bindings, as_of, completed_kpi_groups = _kpi_bindings(lines, mark_boxes)
            parent.content["as_of"] = as_of
    elif mixed_pie_bindings is not None:
        deterministic_bindings = mixed_pie_bindings
    else:
        deterministic_bindings = _proximity_bindings(lines, title, mark_boxes)

    model_bindings = _ground_model_bindings(kind, qwen_payload, lines, mark_boxes)
    model_additions = (
        bool((qwen_payload or {}).get("_vision_first"))
        and kind in {"chart", "kpi_panel"}
        and mixed_pie_bindings is None
        and chart_type != "scatterplot"
    )
    bindings, reconciliation_warnings = _reconcile_visual_bindings(
        deterministic_bindings, model_bindings, allow_model_additions=model_additions,
    )
    parent.warnings.extend(reconciliation_warnings)
    if kind == "chart" and chart_type == "pie":
        slice_verified = _pie_label_value_grounding(
            bindings, region.metadata.get("pdf_soft_mask_slices", []),
        )
        parent.content["slice_geometry_status"] = "verified" if slice_verified else "unresolved"
        if not slice_verified:
            parent.validation_status = "needs_review"
            parent.warnings.append(
                "individual pie-slice geometry was not validated; label-value associations are retained"
            )
    if qwen_payload is not None and not model_bindings:
        parent.validation_status = "needs_review"
        parent.warnings.append("Qwen semantic stage returned no evidence-grounded bindings")
    qwen_semantic_failed = any(
        warning.startswith("Qwen semantic stage") for warning in parent.warnings
    )
    if qwen_semantic_failed:
        parent.validation_status = "needs_review"

    if kind == "chart" and chart_type == "pie":
        percentage_total = sum(
            float(binding["numeric_value"])
            for binding in bindings
            if binding.get("unit") == "percent" and binding.get("numeric_value") is not None
        )
        parent.content["percentage_total"] = round(percentage_total, 5)
        percentage_lines = (
            [line for line in lines if "-ocr-" in str(line.get("evidence_id", ""))]
            if mixed_pie_bindings is not None else lines
        )
        visible_percentage_count = sum(
            bool(re.fullmatch(r"[-+]?\d[\d,.]*\s*%", str(line.get("text", "")).strip()))
            for line in percentage_lines
        )
        parent.content["expected_observation_count"] = visible_percentage_count
        parent.content["emitted_observation_count"] = len(bindings)
        parent.content["observation_completeness"] = round(
            min(1.0, len(bindings) / visible_percentage_count), 5,
        ) if visible_percentage_count else 0.0
        parent.content["percentage_total_reconciles"] = (
            98.0 <= percentage_total <= 102.0
            and len(bindings) == visible_percentage_count
        )
        if len(bindings) >= 3 and parent.content["percentage_total_reconciles"] and not qwen_semantic_failed:
            parent.confidence = max(parent.confidence, 0.90)
            if slice_verified:
                parent.validation_status = "passed"
        else:
            parent.validation_status = "needs_review"
            parent.warnings.append(
                "visible percentages do not establish one complete 100% pie; "
                "multiple or nested pies require review"
            )
    if kind == "map":
        expected_values = _map_data_value_lines(lines)
        parent.content["expected_binding_count"] = len(expected_values)
        parent.content["emitted_binding_count"] = len(bindings)
        parent.content["binding_completeness"] = round(
            len(bindings) / len(expected_values), 5,
        ) if expected_values else 0.0
        assigned_ids = {binding.get("value_evidence_id") for binding in bindings}
        parent.content["unresolved_values"] = [
            {"evidence_id": line["evidence_id"], "raw_value": str(line["text"]).strip()}
            for line in expected_values if line["evidence_id"] not in assigned_ids
        ]
        if len(bindings) != len(expected_values):
            parent.validation_status = "needs_review"
            parent.warnings.append(
                f"validated {len(bindings)} of {len(expected_values)} visible map values"
            )
    expected_marks = region.metadata.get("expected_mark_count")
    if kind == "chart" and chart_type != "pie" and isinstance(expected_marks, int) and expected_marks > 0:
        parent.content["expected_observation_count"] = expected_marks
        parent.content["emitted_observation_count"] = len(bindings)
        parent.content["observation_completeness"] = round(min(1.0, len(bindings) / expected_marks), 5)
        if len(bindings) != expected_marks:
            parent.validation_status = "needs_review"
            if chart_type == "scatterplot":
                parent.warnings.append(
                    f"{len(bindings)} exact values for {expected_marks} detected scatter mark candidates; "
                    "candidate coordinates and axis-based ranges are retained separately"
                )
            else:
                parent.warnings.append(
                    f"reconstructed {len(bindings)} of {expected_marks} detected visual marks"
                )
    nested_items: list[dict[str, Any]] = []
    for index, binding in enumerate(bindings, 1):
        label = str(binding.get("label") or binding.get("category") or binding.get("geography") or "").strip()
        value = str(binding.get("value") or "").strip()
        errors = []
        if not label:
            errors.append("binding has no owner label")
        if not value:
            errors.append("binding has no value")
        item_content = {
            "item_id": f"o{index:03d}" if kind in {"chart", "kpi_panel"} else f"b{index:03d}",
            "raw_value": value,
            "numeric_value": binding.get("numeric_value"),
            "normalized_value": binding.get("normalized_value"),
            "unit": binding.get("unit"),
            "label_evidence_id": binding.get("label_evidence_id"),
            "value_evidence_id": binding.get("value_evidence_id"),
            "label_coordinates": binding.get("label_coordinates"),
            "value_coordinates": binding.get("value_coordinates"),
            "visual_mark_coordinates": binding.get("visual_mark_coordinates"),
            "grounding_method": binding.get("grounding_method"),
        }
        if kind == "chart" and chart_type == "pie":
            item_content["visual_mark_ref"] = binding.get("visual_mark_ref")
        if binding.get("companion_value") is not None:
            item_content["companion_value"] = binding["companion_value"]
        if kind in {"chart", "kpi_panel"}:
            item_content.update({"series": binding.get("series"), "category": label})
        else:
            item_content["geography"] = label
            item_content["geography_basis"] = binding.get("geography_basis", "unverified_vision")
            item_content["mark_validation"] = binding.get("mark_validation", "unverified")
        selected_lines = [
            line for line in lines
            if line["evidence_id"] in {binding.get("label_evidence_id"), binding.get("value_evidence_id")}
        ]
        grounded = bool(selected_lines and binding.get("visual_mark_coordinates"))
        if kind == "map":
            grounded = bool(
                binding.get("mark_verified") and binding.get("value_evidence_id")
                and binding.get("visual_mark_coordinates")
            )
        if kind == "chart" and chart_type == "pie":
            grounded = bool(
                binding.get("label_evidence_id") and binding.get("value_evidence_id")
                and len(selected_lines) == 2 and slice_verified
                and binding.get("visual_mark_coordinates")
            )
        confidence = float(binding.get("confidence", 0.65 if grounded else 0.48))
        item_content["confidence"] = round(confidence, 5)
        supported_mark = kind != "chart" or chart_type in {"bar", "pie", "scatterplot"}
        item_content["validation_status"] = "passed" if grounded and supported_mark and not errors and confidence >= 0.70 else "needs_review"
        item_content["errors"] = errors
        nested_items.append(item_content)
    if kind == "chart":
        parent.content["observations"] = nested_items
    elif kind == "map":
        parent.content["bindings"] = nested_items
        if (
            len(nested_items) == parent.content["expected_binding_count"]
            and all(item["validation_status"] == "passed" for item in nested_items)
            and not qwen_semantic_failed
        ):
            parent.validation_status = "passed"
            parent.confidence = max(parent.confidence, 0.90)
        else:
            parent.validation_status = "needs_review"
    else:
        parent.content["metrics"] = nested_items
        if completed_kpi_groups >= 1 and len(bindings) == completed_kpi_groups * 2 and not qwen_semantic_failed:
            parent.validation_status = "passed"
            parent.confidence = max(parent.confidence, 0.90)
    if not bindings:
        parent.validation_status = "needs_review"
        if chart_type == "scatterplot" and parent.content.get("scatter_points"):
            parent.warnings.append("scatter mark candidates retained, but no exact values printed at the points")
        else:
            parent.warnings.append("no owned visual observations reconstructed")
    if model_additions and (len(bindings) > len(deterministic_bindings) or any(
        "disagreed with deterministic" in warning for warning in reconciliation_warnings
    )):
        parent.validation_status = "needs_review"
        parent.warnings.append("vision-first associations require review despite matching OCR text")
    if any("disagreed with deterministic" in warning for warning in reconciliation_warnings):
        parent.validation_status = "needs_review"
    return [parent]


def _attach_background_decorations(blocks: list[SourceBlock]) -> None:
    """Attach overlapping background artwork to the smallest structural group that contains it."""
    groups = [block for block in blocks if block.type == "group"]
    for decoration in (block for block in blocks if block.type == "decoration"):
        dx, dy, dw, dh = decoration.coordinates
        center = (dx + dw / 2.0, dy + dh / 2.0)
        containers = [
            group for group in groups
            if group.coordinates[0] <= center[0] <= group.coordinates[0] + group.coordinates[2]
            and group.coordinates[1] <= center[1] <= group.coordinates[1] + group.coordinates[3]
        ]
        if not containers:
            continue
        parent = min(containers, key=lambda group: group.coordinates[2] * group.coordinates[3])
        decoration.parent_block_id = parent.block_id
        decoration.hierarchy_depth = parent.hierarchy_depth + 1
        if decoration.block_id not in parent.child_block_ids:
            parent.child_block_ids.append(decoration.block_id)
        content_children = parent.content.setdefault("child_block_ids", [])
        if decoration.block_id not in content_children:
            content_children.append(decoration.block_id)


def _recover_unassigned_brand_marks(
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
    page_lines: list[dict[str, Any]], assigned: set[str], crops: Path,
    document_token_pages: Counter[str] | None = None,
) -> list[SourceBlock]:
    """Recover short logo text using page, document, and corner evidence."""
    candidates = [
        line for line in page_lines
        if line.get("evidence_id") not in assigned and float(line.get("confidence", 0.0)) >= 0.80
    ]
    if not candidates:
        return []
    groups: list[list[dict[str, Any]]] = []
    for line in sorted(candidates, key=lambda item: (item["coordinates"][1], item["coordinates"][0])):
        lx, ly, lw, lh = (float(value) for value in line["coordinates"])
        match = None
        for group in groups:
            gx, gy, gw, gh = _line_box(group)
            horizontal_overlap = max(0.0, min(lx + lw, gx + gw) - max(lx, gx))
            horizontal_related = horizontal_overlap > 0 or abs((lx + lw / 2) - (gx + gw / 2)) <= max(lw, gw)
            vertical_gap = max(0.0, ly - (gy + gh), gy - (ly + lh))
            if horizontal_related and vertical_gap <= max(35.0, 1.5 * max(lh, gh / len(group))):
                match = group
                break
        if match is None:
            groups.append([line])
        else:
            match.append(line)

    recovered: list[SourceBlock] = []
    for index, group in enumerate(groups, 1):
        if not _looks_like_brand_mark(group, page_lines, document_token_pages):
            continue
        coordinates = _line_box(group)
        region = Region(
            region_id=f"p{inspection.page:03d}-u{index:03d}", page=inspection.page,
            kind="brand_mark", coordinates=coordinates, reading_order=len(inspection.regions) + index,
            classification_method="unassigned-ocr-brand-evidence-recovery",
            confidence=sum(float(line.get("confidence", 0.0)) for line in group) / len(group),
            metadata={
                "source_bbox_points": [
                    coordinates[0] * inspection.width_points / 1000.0,
                    coordinates[1] * inspection.height_points / 1000.0,
                    (coordinates[0] + coordinates[2]) * inspection.width_points / 1000.0,
                    (coordinates[1] + coordinates[3]) * inspection.height_points / 1000.0,
                ],
            },
        )
        crop_path = crops / f"{region.region_id}.png"
        _crop(image, coordinates, crop_path)
        recovered.append(SourceBlock(
            document_id=document_id, type="brand_mark", page=inspection.page,
            block_id=f"{region.region_id}-brand-mark",
            content={
                "visible_text": _brand_visible_text(group, document_token_pages),
                "evidence_mode": "ocr_recovery",
                "vision_summary": None,
                "vision_features_ref": None,
                "region_image": f"region-images/{crop_path.name}",
            },
            coordinates=coordinates,
            extraction_method=["RapidOCR", "PP-OCRv6", "document-level brand recovery"],
            confidence=region.confidence, validation_status="passed",
            provenance=_provenance(source_hash, region, group, image), semantic_role="brand_mark",
        ))
        assigned.update(str(line["evidence_id"]) for line in group)
    return recovered


def _brand_visible_text(
    lines: list[dict[str, Any]], document_token_pages: Counter[str] | None,
) -> list[str]:
    """Split concatenated logo words only when document vocabulary supports the split."""
    vocabulary = {
        token for token, count in (document_token_pages or {}).items()
        if count >= 2 and len(token) >= 3
    }

    def segment(token: str) -> list[str] | None:
        folded = _fold_token(token)
        choices: list[list[str] | None] = [None] * (len(folded) + 1)
        choices[0] = []
        for end in range(1, len(folded) + 1):
            candidates: list[list[str]] = []
            for start in range(end):
                if start == 0 and end == len(folded):
                    continue
                if choices[start] is not None and folded[start:end] in vocabulary:
                    candidates.append([*choices[start], token[start:end]])
            if candidates:
                choices[end] = max(candidates, key=len)
        return choices[-1] if choices[-1] and len(choices[-1]) >= 2 else None

    visible = []
    for line in sorted(lines, key=lambda item: (
        float(item.get("coordinates", [0, 0])[1]), float(item.get("coordinates", [0, 0])[0])
    )):
        words = str(line.get("text", "")).split()
        repaired = []
        for word in words:
            for piece in segment(word) or [word]:
                folded_piece = _fold_token(piece)
                if folded_piece in vocabulary and piece.isupper():
                    repaired.append(folded_piece.upper())
                else:
                    repaired.append(piece)
        visible.append(" ".join(repaired))
    return visible


def _document_token_page_frequency(
    ocr_by_page: dict[int, dict[str, Any]], document_name: str = "",
) -> Counter[str]:
    """Count on how many pages each alphabetic token appears."""
    frequency: Counter[str] = Counter()
    for page in ocr_by_page.values():
        tokens = {
            _fold_token(token)
            for line in page.get("lines", [])
            for token in re.findall(r"[^\W\d_]{2,}", str(line.get("text", "")), re.UNICODE)
            if _fold_token(token)
        }
        frequency.update(tokens)
    # Filename words are useful weak evidence for logos, but requiring a visual
    # corner signature still prevents ordinary filename terms becoming brands.
    for token in re.findall(r"[A-Za-z]{3,}", document_name):
        frequency[_fold_token(token)] += 2
    return frequency


def _cluster_spatial_lines(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Cluster nearby OCR lines into independently readable footer/header elements."""
    groups: list[list[dict[str, Any]]] = []
    for line in sorted(lines, key=lambda item: (float(item["coordinates"][0]), float(item["coordinates"][1]))):
        lx, ly, lw, lh = (float(value) for value in line["coordinates"])
        match = None
        for group in groups:
            gx, gy, gw, gh = _line_box(group)
            horizontal_overlap = max(0.0, min(lx + lw, gx + gw) - max(lx, gx))
            centers_close = abs((lx + lw / 2) - (gx + gw / 2)) <= max(lw, gw) * 0.75
            vertical_gap = max(0.0, ly - (gy + gh), gy - (ly + lh))
            if (horizontal_overlap > 0 or centers_close) and vertical_gap <= max(40.0, 1.5 * max(lh, gh / len(group))):
                match = group
                break
        if match is None:
            groups.append([line])
        else:
            match.append(line)
    return groups


def _native_words_for_box(region: Region, coordinates: list[float]) -> str:
    """Return positioned native words whose centers fall inside a normalized box."""
    x, y, width, height = coordinates
    selected = []
    for word in region.metadata.get("native_visual_words", []):
        wx, wy, ww, wh = (float(value) for value in word.get("coordinates", [0, 0, 0, 0]))
        center_x, center_y = wx + ww / 2, wy + wh / 2
        if x <= center_x <= x + width and y <= center_y <= y + height:
            selected.append(word)
    return " ".join(
        str(word.get("text", ""))
        for word in sorted(selected, key=lambda item: (float(item["coordinates"][1]), float(item["coordinates"][0])))
    ).strip()


def _semantic_page_band_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], page_lines: list[dict[str, Any]], image: Path,
    crops: Path, features: dict[str, Any], vision_features_ref: dict[str, str],
    native_threshold: float, document_token_pages: Counter[str] | None,
) -> list[SourceBlock]:
    """Decompose a repeated text-bearing page band instead of discarding it as decoration."""
    if not lines or int(region.metadata.get("semantic_text_overlap_count", 0)) <= 0:
        return []
    clusters = _cluster_spatial_lines(lines)
    if not clusters:
        return []
    parent_id = f"{region.region_id}-page-band"
    children: list[SourceBlock] = []
    for index, cluster in enumerate(clusters, 1):
        coordinates = _line_box(cluster)
        child_region = Region(
            region_id=f"{region.region_id}-band-{index:03d}", page=region.page,
            kind="normal_text", coordinates=coordinates,
            reading_order=region.reading_order + index,
            classification_method="semantic repeated-page-band decomposition",
            confidence=sum(float(line.get("confidence", 0.0)) for line in cluster) / len(cluster),
            native_text=_native_words_for_box(region, coordinates),
            metadata={"source_bbox_points": region.metadata.get("source_bbox_points")},
        )
        if _looks_like_brand_mark(cluster, page_lines, document_token_pages):
            crop_path = crops / f"{child_region.region_id}.png"
            _crop(image, coordinates, crop_path)
            child = SourceBlock(
                document_id=document_id, type="brand_mark", page=region.page,
                block_id=f"{child_region.region_id}-brand-mark",
                content={
                    "visible_text": _brand_visible_text(cluster, document_token_pages),
                    "evidence_mode": "visual_region",
                    "vision_summary": _vision_summary(features),
                    "vision_features_ref": vision_features_ref,
                    "region_image": f"region-images/{crop_path.name}",
                },
                coordinates=coordinates,
                extraction_method=["RapidOCR", "PP-OCRv6", "Python repeated-band decomposition"],
                confidence=max(0.82, child_region.confidence), validation_status="passed",
                # The shared diagnostic describes the parent visual band, so its
                # provenance region must match that diagnostic's region identity.
                provenance=_provenance(source_hash, region, cluster, image),
                semantic_role="brand_mark",
            )
        else:
            child = _text_block(
                document_id, source_hash, inspection, child_region, cluster, image, native_threshold,
            )
            # A known repeated page band is navigation text, not a source
            # footnote even when it sits at the page bottom.
            if child.type == "footnote":
                child.type = "text"
            child.semantic_role = "running_footer" if coordinates[1] >= 500 else "running_header"
        child.parent_block_id = parent_id
        child.hierarchy_depth = 1
        children.append(child)
    if not children:
        return []
    parent = SourceBlock(
        document_id=document_id, type="group", page=region.page, block_id=parent_id,
        content={"role": "page_footer" if region.coordinates[1] >= 500 else "page_header",
                 "child_block_ids": [child.block_id for child in children]},
        coordinates=region.coordinates,
        extraction_method=["Python semantic repeated-page-band decomposition"],
        confidence=min(child.confidence for child in children),
        validation_status="passed" if all(child.validation_status == "passed" for child in children) else "needs_review",
        warnings=[] if all(child.validation_status == "passed" for child in children) else ["one or more child blocks require review"],
        provenance=_provenance(source_hash, region, [], image),
        semantic_role="page_footer" if region.coordinates[1] >= 500 else "page_header",
        child_block_ids=[child.block_id for child in children],
    )
    return [parent, *children]


def _group_profile_rows(
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
    blocks: list[SourceBlock],
) -> None:
    """Pair aligned photographs with adjacent structured biography blocks."""
    photographs = [
        block for block in blocks
        if block.type == "photograph" and block.parent_block_id is None
    ]
    biographies = [
        block for block in blocks
        if block.type == "group" and block.parent_block_id is None
        and block.content.get("role") == "profile_biography"
    ]
    used_biographies: set[str] = set()
    for index, photograph in enumerate(sorted(photographs, key=lambda block: block.coordinates[1]), 1):
        px, py, pw, ph = photograph.coordinates
        candidates: list[tuple[float, SourceBlock]] = []
        for biography in biographies:
            if biography.block_id in used_biographies:
                continue
            bx, by, bw, bh = biography.coordinates
            overlap = max(0.0, min(py + ph, by + bh) - max(py, by))
            overlap_ratio = overlap / max(1.0, min(ph, bh))
            horizontal_gap = bx - (px + pw)
            if overlap_ratio >= 0.45 and -20 <= horizontal_gap <= 250:
                candidates.append((overlap_ratio, biography))
        if not candidates:
            continue
        biography = max(candidates, key=lambda item: item[0])[1]
        used_biographies.add(biography.block_id)
        group_id = f"p{inspection.page:03d}-profile-{index:03d}"
        coordinates = _box_union(photograph.coordinates, biography.coordinates)
        region = Region(
            region_id=group_id, page=inspection.page, kind="group", coordinates=coordinates,
            reading_order=min(
                int(photograph.provenance.get("reading_order", index)),
                int(biography.provenance.get("reading_order", index)),
            ),
            classification_method="aligned photograph-biography row grouping",
            confidence=min(photograph.confidence, biography.confidence),
        )
        group = SourceBlock(
            document_id=document_id, type="group", page=inspection.page, block_id=group_id,
            content={"role": "profile", "child_block_ids": [photograph.block_id, biography.block_id]},
            coordinates=coordinates, extraction_method=["Python profile-row hierarchy"],
            confidence=region.confidence,
            validation_status=(
                "passed" if photograph.validation_status == biography.validation_status == "passed"
                else "needs_review"
            ),
            warnings=(
                [] if photograph.validation_status == biography.validation_status == "passed"
                else ["one or more child blocks require review"]
            ),
            provenance=_provenance(source_hash, region, [], image), semantic_role="profile",
            child_block_ids=[photograph.block_id, biography.block_id],
        )
        insertion = min(blocks.index(photograph), blocks.index(biography))
        blocks.insert(insertion, group)
        photograph.parent_block_id = group_id
        photograph.semantic_role = "profile_image"
        biography.parent_block_id = group_id
        by_id = {block.block_id: block for block in blocks}
        _set_subtree_depth(photograph, 1, by_id)
        _set_subtree_depth(biography, 1, by_id)


def _arrange_page_blocks(blocks: list[SourceBlock]) -> None:
    """Put complete left and right reading lanes in human order before grouping."""
    roots = [block for block in blocks if block.parent_block_id is None]
    if len(roots) < 3:
        return
    page_bands = {"page_header", "page_footer"}
    lane_candidates = [
        block for block in roots
        if block.type != "heading"
        and block.semantic_role not in page_bands
        and block.coordinates[2] < 700
    ]
    centers = sorted(block.coordinates[0] + block.coordinates[2] / 2 for block in lane_candidates)
    gaps = [(right - left, (left + right) / 2) for left, right in zip(centers, centers[1:])]
    if not gaps:
        return
    largest_gap, divider = max(gaps)
    if largest_gap < 120:
        return
    content_roots = [block for block in roots if block.semantic_role not in page_bands]
    left = [block for block in content_roots if block.coordinates[0] + block.coordinates[2] / 2 < divider]
    right = [block for block in content_roots if block.coordinates[0] + block.coordinates[2] / 2 >= divider]
    if not left or not right:
        return
    page_headers = [block for block in roots if block.semantic_role == "page_header"]
    page_footers = [block for block in roots if block.semantic_role == "page_footer"]
    headings = [block for block in content_roots if block.type == "heading" and block.coordinates[1] <= 220]
    remaining = [block for block in content_roots if block not in headings]
    ordered_roots = (
        sorted(page_headers, key=lambda block: (block.coordinates[1], block.coordinates[0]))
        + sorted(headings, key=lambda block: (block.coordinates[1], block.coordinates[0]))
        + sorted(
            remaining,
            key=lambda block: (
                0 if block.coordinates[0] + block.coordinates[2] / 2 < divider else 1,
                block.coordinates[1], block.coordinates[0],
            ),
        )
        + sorted(page_footers, key=lambda block: (block.coordinates[1], block.coordinates[0]))
    )
    by_parent: dict[str, list[SourceBlock]] = {}
    for block in blocks:
        if block.parent_block_id is not None:
            by_parent.setdefault(block.parent_block_id, []).append(block)
    ordered: list[SourceBlock] = []

    def append_tree(block: SourceBlock) -> None:
        ordered.append(block)
        for child in by_parent.get(block.block_id, []):
            append_tree(child)

    for root in ordered_roots:
        append_tree(root)
    blocks[:] = ordered


def _set_subtree_depth(block: SourceBlock, depth: int, by_id: dict[str, SourceBlock]) -> None:
    block.hierarchy_depth = depth
    for child_id in block.child_block_ids:
        child = by_id.get(child_id)
        if child is not None:
            _set_subtree_depth(child, depth + 1, by_id)


def _group_heading_led_sections(
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
    blocks: list[SourceBlock],
) -> None:
    """Create section parents from root headings and the root blocks that follow them."""
    roots = [block for block in blocks if block.parent_block_id is None]
    heading_positions = [index for index, block in enumerate(roots) if block.type == "heading"]
    if not heading_positions:
        return
    sections: list[tuple[int, list[SourceBlock]]] = []
    for section_index, start in enumerate(heading_positions, 1):
        end = heading_positions[section_index] if section_index < len(heading_positions) else len(roots)
        children = [
            block for block in roots[start:end]
            if block.semantic_role not in {"page_footer", "page_header"}
        ]
        if len(children) >= 2:
            sections.append((section_index, children))
    for section_index, children in sections:
        section_id = f"p{inspection.page:03d}-section-{section_index:03d}"
        coordinates = _line_box([{"coordinates": child.coordinates} for child in children])
        region = Region(
            region_id=section_id, page=inspection.page, kind="group", coordinates=coordinates,
            reading_order=children[0].provenance.get("reading_order", 1),
            classification_method="heading-led-section-grouping", confidence=min(child.confidence for child in children),
        )
        section = SourceBlock(
            document_id=document_id, type="group", page=inspection.page, block_id=section_id,
            content={"role": "section", "child_block_ids": [child.block_id for child in children]},
            coordinates=coordinates, extraction_method=["Python heading-led hierarchy"],
            confidence=region.confidence, validation_status="passed",
            provenance=_provenance(source_hash, region, [], image), semantic_role="document_section",
            child_block_ids=[child.block_id for child in children],
        )
        insertion = min(blocks.index(child) for child in children)
        blocks.insert(insertion, section)
        by_id = {block.block_id: block for block in blocks}
        for child in children:
            child.parent_block_id = section_id
            _set_subtree_depth(child, 1, by_id)


def _suppress_visual_observation_text_duplicates(blocks: list[SourceBlock]) -> None:
    """Remove native text copies of evidence already represented as visual observations.

    A narrow visual ownership box may leave an external chart label as a native
    text region. Suppress it only when its wording and location match a chart
    observation with separately owned OCR label and value evidence.
    """
    visuals = [block for block in blocks if block.type in {"chart", "map"}]
    duplicates: list[SourceBlock] = []
    for block in blocks:
        if block.type not in {"heading", "text", "footnote"} or block.parent_block_id is not None:
            continue
        if block.provenance.get("ocr_evidence_ids") or block.content.get("evidence_text", {}).get("selected") != "native":
            continue
        text = _fold_token(re.sub(r"\s+", " ", str(block.content.get("text") or "")).strip())
        if not text:
            continue
        bx, by, bw, bh = block.coordinates
        block_area = max(1.0, bw * bh)
        for visual in visuals:
            vx, vy, vw, vh = visual.coordinates
            overlap = (
                max(0.0, min(bx + bw, vx + vw) - max(bx, vx))
                * max(0.0, min(by + bh, vy + vh) - max(by, vy))
            ) / block_area
            if overlap < 0.50:
                continue
            observations = (
                visual.content.get("observations", []) if visual.type == "chart"
                else visual.content.get("bindings", [])
            )
            for observation in observations:
                label = str(observation.get("category") or observation.get("geography") or "").strip()
                value = str(observation.get("raw_value") or "").strip()
                if not label or not value or not observation.get("label_evidence_id") or not observation.get("value_evidence_id"):
                    continue
                possible_texts = {
                    _fold_token(re.sub(r"\s+", " ", candidate).strip())
                    for candidate in (label, value, f"{label} {value}")
                }
                if text not in possible_texts:
                    continue
                evidence_box = _box_union(
                    observation.get("label_coordinates"), observation.get("value_coordinates"),
                )
                if math.dist(_center(block.coordinates), _center(evidence_box)) <= max(60.0, max(bw, bh)):
                    duplicates.append(block)
                    break
            if block in duplicates:
                break
    if duplicates:
        blocks[:] = [block for block in blocks if block not in duplicates]


def _normalize_page_heading_roles(blocks: list[SourceBlock]) -> None:
    """One visual page title can coexist with subordinate numeric summaries."""
    titles = sorted(
        (block for block in blocks if block.type == "heading" and block.semantic_role == "page_title"),
        key=lambda block: (block.coordinates[1], block.coordinates[0]),
    )
    for block in titles[1:]:
        text = str(block.content.get("text") or "")
        block.semantic_role = "summary_heading" if NUMBER.search(text) and len(text.split()) <= 12 else "section_heading"
        block.heading_level = max(2, block.heading_level or 2)


def _order_overlapping_visual_headings(blocks: list[SourceBlock]) -> None:
    """Place a recovered chart/map heading before the visual crop it labels."""
    visuals = [
        block for block in blocks
        if block.parent_block_id is None and block.type in {"chart", "map", "kpi_panel"}
    ]
    headings = [
        block for block in blocks
        if block.parent_block_id is None and block.type == "heading"
        and block.semantic_role in {"section_heading", "panel_heading"}
    ]
    for visual in visuals:
        vx, vy, vw, vh = visual.coordinates
        candidates = []
        for heading in headings:
            hx, hy, hw, hh = heading.coordinates
            cx, cy = hx + hw / 2.0, hy + hh / 2.0
            if vx <= cx <= vx + vw and vy <= cy <= vy + min(vh * 0.25, 180.0):
                candidates.append(heading)
        if not candidates:
            continue
        heading = min(candidates, key=lambda block: (block.coordinates[1], block.coordinates[0]))
        heading_index = blocks.index(heading)
        visual_index = blocks.index(visual)
        if heading_index > visual_index:
            blocks.pop(heading_index)
            blocks.insert(visual_index, heading)


def _normalize_block_reading_order(blocks: list[SourceBlock]) -> None:
    """Give parents and descendants a deterministic, unique page reading order."""
    for index, block in enumerate(blocks, 1):
        block.provenance["reading_order"] = index


def _propagate_group_validation(blocks: list[SourceBlock]) -> None:
    """A structural parent cannot claim to pass while one of its direct children needs review."""
    by_id = {block.block_id: block for block in blocks}
    groups = sorted(
        (block for block in blocks if block.type == "group"),
        key=lambda block: block.hierarchy_depth, reverse=True,
    )
    for group in groups:
        children = [by_id[child_id] for child_id in group.child_block_ids if child_id in by_id]
        if any(child.validation_status != "passed" for child in children):
            group.validation_status = "needs_review"
            if "one or more child blocks require review" not in group.warnings:
                group.warnings.append("one or more child blocks require review")


def _route_and_extract_page(
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
    ocr_page: dict[str, Any], crops: Path, native_threshold: float,
    route_threshold: float, qwen: QwenVisionClient | None,
    diagnostics: Path, document_token_pages: Counter[str] | None = None,
    vision_plan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    page_lines = _augment_native_table_evidence(
        inspection, _augment_native_visual_evidence(inspection, ocr_page.get("lines", [])),
    )
    ocr_page["lines"] = page_lines
    region_results = ocr_page.get("regions", {})
    blocks: list[SourceBlock] = []
    assigned: set[str] = set()
    route_errors: list[str] = []
    ownership_decisions: list[dict[str, Any]] = []
    lines_by_region = _exclusive_region_lines(inspection.regions, page_lines, ownership_decisions)
    ocr_page["region_ownership_decisions"] = ownership_decisions
    absorbed_regions = _absorb_structured_visual_text(inspection, lines_by_region, vision_plan)
    absorbed_regions.update(_claim_single_value_cards(inspection.regions, page_lines, lines_by_region))
    visual_footnotes = _separate_visual_footnotes(inspection, lines_by_region)
    for region in inspection.regions:
        if region.region_id in absorbed_regions:
            continue
        lines = lines_by_region.get(region.region_id, [])
        assigned.update(line["evidence_id"] for line in lines)
        features = region_results.get(region.region_id, {}).get("vision_features", {})
        plan_hint = region.metadata.get("vision_plan") or {}
        plan_type = str(plan_hint.get("type") or "")
        if region.kind == "normal_text":
            structured = _inline_heading_subsection_blocks(
                document_id, source_hash, inspection, region, lines, image, native_threshold,
            )
            if not structured:
                structured = _profile_biography_blocks(
                    document_id, source_hash, inspection, region, lines, image, native_threshold,
                )
            if not structured:
                structured = _leading_heading_body_blocks(
                    document_id, source_hash, inspection, region, lines, image, native_threshold,
                )
            if structured:
                blocks.extend(structured)
            else:
                blocks.append(_text_block(document_id, source_hash, inspection, region, lines, image, native_threshold))
            continue
        plan_index = plan_hint.get("block_index")
        plan_block = (
            vision_plan["blocks"][plan_index]
            if isinstance(plan_index, int) and vision_plan and 0 <= plan_index < len(vision_plan.get("blocks", []))
            else None
        )
        if region.metadata.get("single_card_binding"):
            crop_path = crops / f"{region.region_id}.png"
            _crop(image, region.coordinates, crop_path)
            vision_features_ref = _write_vision_diagnostic(
                diagnostics, document_id, source_hash, region, features,
            )
            blocks.extend(_visual_blocks(
                document_id, source_hash, region, "kpi_panel", 0.70, [], lines, features,
                image, crop_path, None, vision_features_ref,
            ))
            continue
        numeric_density = sum(bool(NUMBER.search(str(line.get("text", "")))) for line in lines) / max(1, len(lines))
        planned_table = (
            region.kind == "visual" and plan_type == "table" and plan_block is not None
            and len(plan_block.get("rows", [])) >= 2 and numeric_density >= 0.35
        )
        if (region.kind == "table" and plan_type not in {"map", "chart", "kpi_panel"}) or planned_table:
            if planned_table and plan_block.get("title"):
                region.metadata["title"] = plan_block["title"]
            crop_path = crops / f"{region.region_id}.png"
            _crop(image, region.coordinates, crop_path)
            table_payload = None
            table_error = None
            if qwen:
                try:
                    table_payload = qwen.analyze(
                        crop_path, _qwen_semantic_prompt("table", lines, region),
                    )
                    if table_payload.get("type") != "table":
                        table_error = "Qwen returned a non-table response during mandatory table review"
                        table_payload = None
                except Exception as exc:
                    table_error = f"{type(exc).__name__}: {exc}"
            blocks.extend(_table_blocks(
                document_id, source_hash, region, lines, image, table_payload, table_error,
            ))
            continue
        if region.kind == "decoration":
            crop_path = crops / f"{region.region_id}.png"
            _crop(image, region.coordinates, crop_path)
            vision_features_ref = _write_vision_diagnostic(
                diagnostics, document_id, source_hash, region, features,
            )
            semantic_band = _semantic_page_band_blocks(
                document_id, source_hash, inspection, region, lines, page_lines, image,
                crops, features, vision_features_ref, native_threshold, document_token_pages,
            )
            if semantic_band:
                blocks.extend(semantic_band)
                continue
            blocks.extend(_visual_blocks(
                document_id, source_hash, region, "decoration", region.confidence, [], lines,
                features, image, crop_path, None, vision_features_ref,
            ))
            continue
        kind, confidence, warnings = _classify_visual(region, lines, features)
        if plan_type in {"map", "chart", "kpi_panel"}:
            # The model proposes the visual family first. OCR/OpenCV still run
            # independently, and model-selected routes start in review state.
            if kind != plan_type:
                warnings.append(
                    f"vision-first proposed {plan_type}; geometry/OCR classified {kind}"
                )
            kind = plan_type
            confidence = min(confidence, 0.70)
            region.classification_method = "vision-first proposal reconciled with OCR/OpenCV"
        routing_payload = None
        crop_path = crops / f"{region.region_id}.png"
        _crop(image, region.coordinates, crop_path)
        vision_features_ref = _write_vision_diagnostic(
            diagnostics, document_id, source_hash, region, features,
        )
        if qwen and confidence < route_threshold and not (plan_type == "kpi_panel" and kind == "kpi_panel"):
            try:
                routing_payload = qwen.analyze(crop_path, (
                    "You are a document-region classifier. Return exactly one JSON object with exactly four "
                    "keys: type, confidence, chart_type, bindings. type must be exactly one of normal_text, "
                    "table, chart, map, kpi_panel, photograph, brand_mark, unclassified_visual. confidence must be a JSON number from "
                    "zero to one. chart_type must be a short string or null. bindings must be a JSON array. "
                    "Do not return a transcription, a label-only object, Markdown, or commentary. Add bindings "
                    "only for clearly visible label and value pairs; otherwise return an empty array."
                ))
                proposed = str(routing_payload.get("type", "")).strip().lower()
                if proposed == "table" and region.kind != "table" and numeric_density < 0.35:
                    warnings.append("model table label rejected: too few printed numeric cells for a visual grid")
                    proposed = "unclassified_visual"
                place_labels = sum(
                    bool(re.search(r"\b(?:river|bay|district|street|road|hill|peninsula|lake|county)\b",
                                   str(line.get("text", "")), re.I))
                    for line in lines
                )
                if proposed == "map" and place_labels < 3 and not any(
                    term in " ".join(str(line.get("text", "")).casefold() for line in lines)
                    for term in MAP_TERMS
                ):
                    warnings.append("model map label rejected: no printed geographic context")
                    proposed = "unclassified_visual"
                if proposed in {"normal_text", "table", "chart", "map", "kpi_panel", "photograph", "brand_mark", "unclassified_visual"}:
                    kind = proposed
                    confidence = float(routing_payload.get("confidence", confidence))
                    region.classification_method = "Qwen3-VL visual classification"
                    warnings = [
                        warning for warning in warnings
                        if warning not in {
                            "visual type could not be classified confidently",
                            "visual classification is geometry-only",
                        }
                    ]
                else:
                    warnings.append("Qwen3-VL returned an unsupported region type")
            except Exception as exc:
                warnings.append(f"Qwen3-VL unavailable: {type(exc).__name__}: {exc}")
                route_errors.append(f"{region.region_id}: {warnings[-1]}")
        if kind in {"unclassified_visual", "photograph"} and _looks_like_brand_mark(
            lines, page_lines, document_token_pages,
        ):
            kind = "brand_mark"
            confidence = max(0.82, min(confidence, 0.92))
            warnings = [
                warning for warning in warnings
                if warning != "visual type could not be classified confidently"
            ]
        semantic_payload = None
        semantic_error = None
        if isinstance(plan_index, int) and vision_plan and 0 <= plan_index < len(vision_plan.get("blocks", [])):
            proposed_block = vision_plan["blocks"][plan_index]
            if proposed_block.get("type") == kind and kind in {"chart", "map", "kpi_panel"}:
                semantic_payload = {
                    "_vision_first": True, "type": kind, "confidence": 0.65,
                    "chart_type": proposed_block.get("chart_type") or None,
                    "bindings": [
                        {"label": item.get("label", ""), "value": item.get("value", ""),
                         "confidence": 0.65}
                        for item in proposed_block.get("items", [])
                        if item.get("label") and item.get("value")
                    ],
                }
        if qwen and kind in {"table", "chart", "map", "kpi_panel"}:
            try:
                if kind == "map":
                    value_ids = [str(line.get("evidence_id")) for line in _map_data_value_lines(lines)]
                    batches = [value_ids[index:index + 8] for index in range(0, len(value_ids), 8)] or [[]]
                    batch_payloads: list[dict[str, Any]] = []
                    batch_errors: list[str] = []
                    for batch_index, batch in enumerate(batches, 1):
                        try:
                            payload = qwen.analyze(
                                crop_path, _qwen_semantic_prompt(kind, lines, region, batch),
                                max_bindings=max(1, len(batch)),
                            )
                            if payload.get("type") == kind:
                                batch_payloads.append(payload)
                            else:
                                batch_errors.append(f"batch {batch_index}/{len(batches)} returned type {payload.get('type')!r}")
                        except Exception as exc:
                            batch_errors.append(f"batch {batch_index}/{len(batches)}: {type(exc).__name__}: {exc}")
                    if batch_payloads:
                        semantic_payload = {
                            "type": kind,
                            "confidence": min(float(payload.get("confidence", 0.0)) for payload in batch_payloads),
                            "chart_type": None,
                            "bindings": [
                                binding for payload in batch_payloads for binding in payload.get("bindings", [])
                            ],
                        }
                    if batch_errors:
                        semantic_error = "Qwen map semantic " + "; ".join(batch_errors)
                else:
                    semantic_payload = qwen.analyze(
                        crop_path, _qwen_semantic_prompt(kind, lines, region),
                        max_bindings=0 if kind == "table" else max(1, len(lines)),
                    )
                if semantic_payload is not None and semantic_payload.get("type") != kind:
                    semantic_error = f"Qwen returned type {semantic_payload.get('type')!r} during mandatory {kind} stage"
                    semantic_payload = None
                elif semantic_payload is None and semantic_error is None:
                    semantic_error = f"Qwen returned no usable payload during mandatory {kind} stage"
            except Exception as exc:
                semantic_error = f"{type(exc).__name__}: {exc}"
            if semantic_error:
                warnings.append(f"Qwen semantic stage unavailable: {semantic_error}")
        if kind == "kpi_panel" and plan_block and plan_block.get("type") == "kpi_panel":
            plan_bindings = [
                {"label": item["label"], "value": item["value"], "confidence": 0.65}
                for item in plan_block.get("items", []) if item.get("label") and item.get("value")
            ]
            if plan_bindings:
                if semantic_payload is None:
                    semantic_payload = {"type": kind, "confidence": 0.65, "chart_type": None, "bindings": []}
                semantic_payload["bindings"] = list(semantic_payload.get("bindings", [])) + plan_bindings
                semantic_payload["_vision_first"] = True
                semantic_payload["confidence"] = min(float(semantic_payload.get("confidence", 0.65)), 0.65)
        if kind == "normal_text":
            blocks.extend(_visual_text_panel_blocks(
                document_id, source_hash, inspection, region, lines, image, page_lines,
            ))
        elif kind == "table":
            blocks.extend(_table_blocks(
                document_id, source_hash, region, lines, image, semantic_payload, semantic_error,
            ))
        else:
            blocks.extend(_visual_blocks(
                document_id, source_hash, region, kind, confidence, warnings, lines, features,
                image, crop_path, semantic_payload, vision_features_ref,
            ))
    blocks.extend(_recover_unassigned_brand_marks(
        document_id, source_hash, inspection, image, page_lines, assigned, crops, document_token_pages,
    ))
    pending_footnotes: list[tuple[SourceBlock, str]] = []
    for owner_region_id, note_groups in visual_footnotes.items():
        for index, note_lines in enumerate(note_groups, 1):
            assigned.update(str(line["evidence_id"]) for line in note_lines)
            note_region = Region(
                region_id=f"{owner_region_id}-note-{index:03d}", page=inspection.page,
                kind="normal_text", coordinates=_line_box(note_lines),
                reading_order=10_000 + index, classification_method="marked visual footnote",
                confidence=min(float(line.get("confidence", 0.8)) for line in note_lines),
            )
            note = _text_block(
                document_id, source_hash, inspection, note_region, note_lines, image, 1.1,
            )
            separate_markers = [
                str(line.get("text", "")).strip() for line in note_lines
                if re.fullmatch(r"[*†‡]", str(line.get("text", "")).strip())
            ]
            if separate_markers:
                body = clean_text(_ocr_text([
                    line for line in note_lines
                    if str(line.get("text", "")).strip() not in separate_markers
                ]))
                note.content["text"] = f"{separate_markers[0]} {body}"
                note.content["evidence_text"]["ocr"] = note.content["text"]
            note.type = "footnote"
            note.semantic_role = "footnote"
            note.heading_level = None
            blocks.append(note)
            pending_footnotes.append((note, owner_region_id))
    _suppress_visual_observation_text_duplicates(blocks)
    _group_profile_rows(document_id, source_hash, inspection, image, blocks)
    _attach_background_decorations(blocks)
    _arrange_page_blocks(blocks)
    _order_overlapping_visual_headings(blocks)
    _group_heading_led_sections(document_id, source_hash, inspection, image, blocks)
    _normalize_page_heading_roles(blocks)
    for note, owner_region_id in pending_footnotes:
        owner = next((block for block in blocks if
            block.provenance.get("region_id") == owner_region_id
            and block.type in {"chart", "map", "kpi_panel"}
        ), None)
        if owner is not None:
            previous_parent = next((block for block in blocks if
                block.block_id == note.parent_block_id
            ), None)
            if previous_parent is not None and previous_parent is not owner:
                previous_parent.child_block_ids = [
                    child_id for child_id in previous_parent.child_block_ids
                    if child_id != note.block_id
                ]
                if isinstance(previous_parent.content.get("child_block_ids"), list):
                    previous_parent.content["child_block_ids"] = [
                        child_id for child_id in previous_parent.content["child_block_ids"]
                        if child_id != note.block_id
                    ]
            note.parent_block_id = owner.block_id
            note.hierarchy_depth = owner.hierarchy_depth + 1
            if note.block_id not in owner.child_block_ids:
                owner.child_block_ids.append(note.block_id)
            owner.validation_status = "needs_review"
            qualification_warning = "visual observations have a linked footnote that may qualify their interpretation"
            if qualification_warning not in owner.warnings:
                owner.warnings.append(qualification_warning)
    _propagate_group_validation(blocks)
    _normalize_block_reading_order(blocks)
    block_dicts = [block.as_dict() for block in blocks]
    errors = validate_source_blocks(block_dicts)
    unassigned = [
        {"evidence_id": line["evidence_id"], "reason": "outside detected regions", "evidence": line}
        for line in page_lines if line["evidence_id"] not in assigned
    ]
    return block_dicts, unassigned, errors + route_errors


def run_pipeline(
    pdf: Path, output: Path, pages_spec: str | None = None, dpi: int = 300,
    native_threshold: float = 0.78, route_threshold: float = 0.72,
    skip_ocr: bool = False, rapidocr_python: Path | None = None,
    rapidocr_models: Path | None = None, pdftoppm: Path | None = None,
    qwen_endpoint: str | None = None, qwen_model: str = "dealsynq-qwen3-vl:4b-instruct-16k",
    vision_plan_cache: Path | None = None,
    paddle_python: Path | None = None,
    paddle_device: str = "cpu",
) -> Path:
    if not qwen_endpoint or skip_ocr:
        raise ValueError("independent vision and OCR are mandatory in this pipeline")
    pdf = pdf.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output must not exist; runs are immutable: {output}")
    output.mkdir(parents=True)
    source_dir = output / "source"
    images_dir = output / "page-images"
    crops_dir = output / "region-images"
    blocks_dir = output / "source-blocks"
    inspection_dir = output / "inspection"
    diagnostics_dir = output / "diagnostics" / "vision-features"
    work_dir = output / "work"
    for path in (source_dir, images_dir, crops_dir, blocks_dir, inspection_dir, diagnostics_dir, work_dir):
        path.mkdir(parents=True)
    source_hash = _sha256(pdf)
    document_id = f"{_slug(pdf.stem)}-{source_hash[:12]}"
    page_count = len(PdfReader(pdf).pages)
    pages = parse_pages(pages_spec, page_count)
    manifest: dict[str, Any] = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "status": "running", "created_utc": _now(), "document_id": document_id,
        "source_filename": pdf.name, "source_sha256": source_hash,
        "page_count": page_count, "selected_pages": pages, "dpi": dpi,
        "coordinate_system": "top-left normalized xywh 0..1000",
        "thresholds": {"native_text_usability": native_threshold, "routing_confidence": route_threshold},
        "vision_model": {
            "enabled": bool(qwen_endpoint),
            "provider": "OpenAI-compatible" if qwen_endpoint else None,
            "endpoint": qwen_endpoint,
            "model": qwen_model if qwen_endpoint else None,
            "routing": "mandatory full-page image-only proposal; native/OCR/OpenCV extraction receives no vision hints; claim-level reconciliation after both branches",
        },
        "runtime": {
            "python": sys.version,
            "pdfplumber": importlib.metadata.version("pdfplumber"),
            "pypdf": importlib.metadata.version("pypdf"),
            "pillow": importlib.metadata.version("Pillow"),
            "pipeline": f"dealsynq-independent-vision-reconciled-pipeline/{PIPELINE_VERSION}",
        },
        "stages": [],
    }
    _write_json(output / "manifest.json", manifest)
    try:
        copied = source_dir / pdf.name
        shutil.copy2(pdf, copied)
        if _sha256(copied) != source_hash:
            raise RuntimeError("Preserved source copy hash mismatch")
        manifest["preserved_source"] = f"source/{copied.name}"
        manifest["stages"].append({"name": "ingestion", "status": "complete", "finished_utc": _now()})
        _write_json(output / "manifest.json", manifest)

        inspections, pdf_metadata = inspect_pdf(pdf)
        selected_inspections = [inspection for inspection in inspections if inspection.page in pages]
        inspection_records = []
        for page_inspection in selected_inspections:
            page_inspection_path = inspection_dir / f"page-{page_inspection.page:03d}.json"
            _write_json(page_inspection_path, {
                "schema_version": INSPECTION_SCHEMA_VERSION,
                "document_id": document_id,
                "source_sha256": source_hash,
                **page_inspection.as_dict(),
            })
            inspection_records.append({
                "page": page_inspection.page,
                "file": page_inspection_path.name,
                "sha256": _sha256(page_inspection_path),
                "region_count": len(page_inspection.regions),
            })
        inspection_path = inspection_dir / "manifest.json"
        _write_json(inspection_path, {
            "schema_version": INSPECTION_INDEX_SCHEMA_VERSION,
            "document_id": document_id, "source_sha256": source_hash,
            "page_count": page_count, "selected_pages": pages, "pdf_metadata": pdf_metadata,
            "pages": inspection_records,
        })
        manifest["inspection"] = {
            "path": str(inspection_path.relative_to(output)).replace("\\", "/"),
            "sha256": _sha256(inspection_path),
            "schema_version": INSPECTION_INDEX_SCHEMA_VERSION,
        }
        manifest["stages"].append({"name": "pdf-inspection", "status": "complete", "finished_utc": _now()})

        rendered = _render_pages(pdf, pages, images_dir, dpi, pdftoppm)
        render_executable = pdftoppm or (Path(shutil.which("pdftoppm")) if shutil.which("pdftoppm") else None)
        render_version = None
        if render_executable:
            version_result = subprocess.run(
                [str(render_executable), "-v"], capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            render_version = (version_result.stdout or version_result.stderr).strip().splitlines()[:1]
        manifest["rendering"] = {
            "engine": str(render_executable) if render_executable else None,
            "version": render_version[0] if render_version else None,
            "dpi": dpi, "format": "PNG",
        }
        manifest["stages"].append({"name": "rendering", "status": "complete", "finished_utc": _now()})
        from .vision_first import plan_pages
        plans, vision_summary = plan_pages(
            rendered, output / "vision-plans", qwen_endpoint, qwen_model,
            verified_cache=vision_plan_cache,
        )
        # Keep the model proposal out of inspection, OCR ownership, and the
        # deterministic extraction branch. Fusion happens after both finish.
        manifest["vision_plan"] = {
            "path": "vision-plans/manifest.json", **vision_summary,
        }
        manifest["stages"].append({"name": "vision-first-page-planning", "status": "complete", "finished_utc": _now()})
        _write_json(output / "manifest.json", manifest)
        jobs = [{
            "page": inspection.page, "image": str(rendered[inspection.page]),
            "regions": [{"region_id": region.region_id, "coordinates": region.coordinates} for region in inspection.regions],
        } for inspection in selected_inspections]
        if skip_ocr:
            ocr_result = {"engine": None, "pages": [{"page": page, "lines": [], "regions": {}} for page in pages]}
            manifest["ocr"] = {"enabled": False, "reason": "--skip-ocr"}
        else:
            ocr_result = run_rapidocr_worker(jobs, work_dir, rapidocr_python, rapidocr_models)
            manifest["ocr"] = {
                "enabled": True, "engine": ocr_result.get("engine"),
                "rapidocr_version": ocr_result.get("rapidocr_version"),
                "opencv_version": ocr_result.get("opencv_version"),
            }
        manifest["stages"].append({"name": "ocr-and-geometry", "status": "complete", "finished_utc": _now()})

        # PP-StructureV3 sees every likely table page, including pages where
        # the native PDF parser already found a table. It is independent of Qwen.
        from .paddle_tables import apply_paddle_tables, page_needs_table_analysis, table_candidates
        from .spatial_lanes import split_mixed_key_value_regions

        pre_paddle_inspections = {
            inspection.page: copy.deepcopy(inspection) for inspection in selected_inspections
        }
        paddle_dir = output / "paddle-tables"
        ocr_for_tables = {int(item["page"]): item for item in ocr_result.get("pages", [])}
        paddle_images = {
            inspection.page: rendered[inspection.page]
            for inspection in selected_inspections
            if page_needs_table_analysis(inspection, plans.get(inspection.page),
                                         ocr_for_tables.get(inspection.page, {"lines": []}))
        }
        if paddle_images:
            paddle_summary = run_paddle_table_worker(paddle_images, paddle_dir, paddle_python, paddle_device)
        else:
            paddle_dir.mkdir(parents=True, exist_ok=True)
            paddle_summary = {
                "engine": "PP-StructureV3", "device": paddle_device, "python": None,
                "paddleocr_version": None, "paddlepaddle_version": None, "pages": [],
            }
            _write_json(paddle_dir / "summary.json", paddle_summary)
        paddle_pages = {int(item["page"]): item for item in paddle_summary["pages"]}
        for page_inspection in selected_inspections:
            receipt = paddle_pages.get(page_inspection.page, {})
            if page_inspection.page in paddle_images and receipt.get("status") == "complete":
                payload = json.loads(Path(receipt["result"]).read_text(encoding="utf-8"))
                with Image.open(rendered[page_inspection.page]) as page_image:
                    width, height = page_image.size
                candidates = table_candidates(payload, width, height)
                apply_paddle_tables(page_inspection, candidates)
                receipt["usable_table_candidates"] = len(candidates)
            elif page_inspection.page in paddle_images:
                for region in page_inspection.regions:
                    if region.kind == "table":
                        region.metadata["paddle_table_review"] = {
                            "status": "unavailable", "error": receipt.get("error", "no result")}
            split_mixed_key_value_regions(
                page_inspection, ocr_for_tables.get(page_inspection.page, {}).get("lines", []),
            )
            page_path = inspection_dir / f"page-{page_inspection.page:03d}.json"
            _write_json(page_path, {
                "schema_version": INSPECTION_SCHEMA_VERSION,
                "document_id": document_id, "source_sha256": source_hash,
                **page_inspection.as_dict(),
            })
            next(record for record in inspection_records if record["page"] == page_inspection.page)["sha256"] = _sha256(page_path)
        _write_json(inspection_path, {
            "schema_version": INSPECTION_INDEX_SCHEMA_VERSION,
            "document_id": document_id, "source_sha256": source_hash,
            "page_count": page_count, "selected_pages": pages, "pdf_metadata": pdf_metadata,
            "pages": inspection_records,
        })
        manifest["inspection"]["sha256"] = _sha256(inspection_path)
        manifest["paddle_tables"] = {
            "path": "paddle-tables/summary.json", "engine": paddle_summary["engine"],
            "device": paddle_summary["device"], "python": paddle_summary["python"],
            "selected_pages": sorted(paddle_images),
            "paddleocr_version": paddle_summary["paddleocr_version"],
            "paddlepaddle_version": paddle_summary["paddlepaddle_version"],
            "pages": paddle_summary["pages"],
        }
        manifest["stages"].append({"name": "independent-paddle-table-analysis", "status": "complete", "finished_utc": _now()})
        _write_json(output / "manifest.json", manifest)

        # Neither branch sees the other's observations during extraction.
        qwen = None
        from .independent_reconcile import (
            accept_ocr_table_candidate, accept_registered_map_candidate,
            accept_verified_chart_candidate, has_missing_table_proposal, reconcile_page,
        )
        from .vision_first import apply_plan_hints
        reconciliation_dir = output / "reconciliation"
        reconciliation_dir.mkdir()
        deterministic_dir = output / "deterministic"
        deterministic_dir.mkdir()
        reconciliation_counts: Counter[str] = Counter()
        ocr_by_page = {int(item["page"]): item for item in ocr_result.get("pages", [])}
        from .source_observations import collect_page_observations
        source_observations = collect_page_observations(
            pdf, pages, ocr_by_page, paddle_dir, rendered,
        )
        document_token_pages = _document_token_page_frequency(ocr_by_page, pdf.stem)
        page_records = []
        type_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        collection_errors: list[str] = []
        for inspection in selected_inspections:
            raw_ocr_page = ocr_by_page.get(inspection.page, {"lines": [], "regions": {}})
            ocr_page = copy.deepcopy(raw_ocr_page)
            blocks, unassigned, errors = _route_and_extract_page(
                document_id, source_hash, inspection, rendered[inspection.page],
                ocr_page,
                crops_dir, native_threshold, route_threshold, qwen, diagnostics_dir, document_token_pages,
                None,
            )
            primary_coverage = _numeric_content_coverage(blocks, ocr_page.get("lines", []))
            primary_coverage["linked_table_scalars"] = _linked_table_scalar_count(blocks)
            coverage_comparison: dict[str, Any] = {"paddle_route": primary_coverage}
            routing_decision = "deterministic_route_retained"
            if (inspection.page in paddle_pages
                    and (len(primary_coverage["unrepresented"]) >= 2
                         or any(block["type"] == "table" for block in blocks))):
                fallback_ocr_page = copy.deepcopy(raw_ocr_page)
                fallback_blocks, fallback_unassigned, fallback_errors = _route_and_extract_page(
                    document_id, source_hash, pre_paddle_inspections[inspection.page],
                    rendered[inspection.page], fallback_ocr_page,
                    crops_dir, native_threshold, route_threshold, qwen,
                    diagnostics_dir, document_token_pages, None,
                )
                fallback_coverage = _numeric_content_coverage(
                    fallback_blocks, fallback_ocr_page.get("lines", []),
                )
                fallback_coverage["linked_table_scalars"] = _linked_table_scalar_count(fallback_blocks)
                coverage_comparison["pre_paddle_route"] = fallback_coverage
                structured_types = {"table", "chart", "map", "kpi_panel"}
                primary_structured = sum(
                    block["type"] in structured_types for block in blocks
                )
                fallback_structured = sum(
                    block["type"] in structured_types for block in fallback_blocks
                )
                more_content = (
                    fallback_coverage["represented"] >= primary_coverage["represented"] + 2
                    and fallback_structured >= primary_structured
                )
                better_links = (
                    fallback_coverage["linked_table_scalars"]
                    >= primary_coverage["linked_table_scalars"] + 3
                    and fallback_coverage["represented"] >= primary_coverage["represented"]
                )
                if (more_content or better_links) and len(fallback_errors) <= len(errors):
                    blocks, unassigned, errors = (
                        fallback_blocks, fallback_unassigned, fallback_errors
                    )
                    ocr_page = fallback_ocr_page
                    routing_decision = "pre_paddle_structured_route_preserved_more_ocr_content"
            baseline_path = deterministic_dir / f"page-{inspection.page:03d}.json"
            _write_json(baseline_path, {
                "page": inspection.page, "source_sha256": source_hash,
                "branch": "native PDF, RapidOCR, and OpenCV without vision hints",
                "blocks": blocks,
            })
            plan = plans.get(inspection.page)
            missing_proposed_visual = bool(plan) and any(
                item.get("type") in {"map", "chart"}
                and not any(block["type"] == item["type"] for block in blocks)
                for item in plan.get("blocks", [])
            )
            missing_proposed_table = has_missing_table_proposal(plan, blocks)
            if missing_proposed_visual or missing_proposed_table:
                candidate_inspection = copy.deepcopy(inspection)
                apply_plan_hints(candidate_inspection, plan)
                candidate_ocr_page = copy.deepcopy(raw_ocr_page)
                candidate_blocks, candidate_unassigned, candidate_errors = _route_and_extract_page(
                    document_id, source_hash, candidate_inspection, rendered[inspection.page],
                    candidate_ocr_page, crops_dir, native_threshold, route_threshold, None,
                    diagnostics_dir, document_token_pages, plan,
                )
                if accept_registered_map_candidate(plan, blocks, candidate_blocks, candidate_errors):
                    blocks, unassigned, errors = candidate_blocks, candidate_unassigned, candidate_errors
                    ocr_page = candidate_ocr_page
                    routing_decision = "registered_map_candidate_selected"
                elif accept_verified_chart_candidate(plan, blocks, candidate_blocks, candidate_errors):
                    blocks, unassigned, errors = candidate_blocks, candidate_unassigned, candidate_errors
                    ocr_page = candidate_ocr_page
                    routing_decision = "verified_chart_candidate_selected"
                elif missing_proposed_table and accept_ocr_table_candidate(
                    plan, blocks, candidate_blocks, candidate_errors,
                    _numeric_content_coverage(blocks, ocr_page.get("lines", [])),
                    _numeric_content_coverage(candidate_blocks, candidate_ocr_page.get("lines", [])),
                ):
                    blocks, unassigned, errors = candidate_blocks, candidate_unassigned, candidate_errors
                    ocr_page = candidate_ocr_page
                    routing_decision = "ocr_supported_table_candidate_selected"
                else:
                    routing_decision = "vision_candidate_rejected_without_independent_support"
            reconciliation = reconcile_page(
                inspection.page, plan, blocks,
                ocr_page.get("lines", []),
            )
            reconciliation["routing_decision"] = routing_decision
            reconciliation["numeric_content_coverage"] = coverage_comparison
            reconciliation["counts"][routing_decision] = 1
            receipt_path = output / "vision-plans" / f"page-{inspection.page:03d}.receipt.json"
            reconciliation["inputs"] = {
                "deterministic_snapshot": str(baseline_path.relative_to(output)).replace("\\", "/"),
                "deterministic_sha256": _sha256(baseline_path),
                "vision_receipt": str(receipt_path.relative_to(output)).replace("\\", "/"),
                "vision_receipt_sha256": _sha256(receipt_path),
            }
            structured_coverage = _numeric_content_coverage(blocks, ocr_page.get("lines", []))
            blocks, raw_capture = _preserve_ocr_lines_in_blocks(
                blocks, ocr_page.get("lines", []), document_id, source_hash,
                inspection, rendered[inspection.page],
            )
            source_capture = _preserve_source_observations(
                blocks, source_observations.get(inspection.page, []),
                document_id, source_hash, inspection, rendered[inspection.page],
            )
            table_grounding = _ground_table_cells_from_ocr(blocks, ocr_page.get("lines", []))
            owned_ids = {
                evidence_id for block in blocks
                for evidence_id in block.get("provenance", {}).get("ocr_evidence_ids", [])
            }
            unassigned = [item for item in unassigned if item.get("evidence_id") not in owned_ids]
            errors = list(dict.fromkeys(errors + validate_source_blocks(blocks)))
            reconciliation["structured_numeric_content_coverage"] = structured_coverage
            reconciliation["raw_ocr_capture"] = raw_capture
            reconciliation["source_observation_capture"] = source_capture
            reconciliation["table_grounding"] = table_grounding
            reconciliation["region_ownership_decisions"] = ocr_page.get("region_ownership_decisions", [])
            _write_json(reconciliation_dir / f"page-{inspection.page:03d}.json", reconciliation)
            reconciliation_counts.update(reconciliation["counts"])
            if inspection.page not in plans:
                for block in blocks:
                    if block["type"] in {"chart", "map", "kpi_panel", "unclassified_visual"}:
                        block["validation"]["status"] = "needs_review"
                        warning = "vision-first proposal unavailable; visual interpretation is unconfirmed"
                        if warning not in block["validation"]["warnings"]:
                            block["validation"]["warnings"].append(warning)
            for block in blocks:
                type_counts[block["type"]] += 1
                status_counts[block["validation"]["status"]] += 1
            completeness_warnings: list[str] = []
            if inspection.page not in plans:
                completeness_warnings.append("vision-first page proposal unavailable")
            high_confidence_unassigned = [
                item for item in unassigned
                if float(item.get("evidence", {}).get("confidence", 0.0)) >= 0.80
                and re.search(r"[A-Za-z0-9]", str(item.get("evidence", {}).get("text", "")))
            ]
            if high_confidence_unassigned:
                completeness_warnings.append(
                    f"{len(high_confidence_unassigned)} high-confidence OCR lines remain semantically unassigned"
                )
            if len(structured_coverage["unrepresented"]) >= 2:
                completeness_warnings.append(
                    f"{len(structured_coverage['unrepresented'])} high-confidence numeric OCR lines "
                    "are outside nearby structured content; raw OCR is retained in source blocks"
                )
            review_blocks = [
                block for block in blocks if block.get("validation", {}).get("status") != "passed"
            ]
            if review_blocks:
                completeness_warnings.append(f"{len(review_blocks)} source blocks require review")
            completeness_status = "complete" if not completeness_warnings else "needs_review"
            page_payload = {
                "schema_version": PAGE_SCHEMA_VERSION, "document_id": document_id,
                "source_sha256": source_hash, "page": inspection.page,
                "blocks": blocks,
                "evidence_ledger": {
                    "rendered_page": f"page-images/{rendered[inspection.page].name}",
                    "rendered_page_sha256": _sha256(rendered[inspection.page]),
                    "ocr_engine": ocr_result.get("engine"),
                    "ocr_lines": ocr_page.get("lines", []),
                    "source_observations": source_observations.get(inspection.page, []),
                },
                "evidence_disposition": [
                    {
                        "evidence_id": line["evidence_id"],
                        "owner_block_id": next((
                            block["block_id"] for block in blocks
                            if line["evidence_id"] in block.get("provenance", {}).get("ocr_evidence_ids", [])
                        ), None),
                        "status": "owned" if any(
                            line["evidence_id"] in block.get("provenance", {}).get("ocr_evidence_ids", [])
                            for block in blocks
                        ) else "unassigned",
                    }
                    for line in ocr_page.get("lines", [])
                ],
                "unassigned_evidence": unassigned,
                "validation": {
                    "valid": not errors, "errors": errors,
                    "completeness_status": completeness_status,
                    "warnings": completeness_warnings,
                },
            }
            page_path = blocks_dir / f"page-{inspection.page:03d}.json"
            _write_json(page_path, page_payload)
            page_records.append({
                "page": inspection.page, "file": page_path.name, "sha256": _sha256(page_path),
                "block_count": len(blocks), "unassigned_evidence_count": len(unassigned),
                "valid": not errors, "completeness_status": completeness_status,
            })
            collection_errors.extend(f"page {inspection.page}: {error}" for error in errors)
        collection_manifest = {
            "schema_version": COLLECTION_SCHEMA_VERSION, "document_id": document_id,
            "source_filename": pdf.name, "source_sha256": source_hash, "page_count": page_count,
            "selected_pages": pages, "pages": page_records, "block_counts_by_type": dict(sorted(type_counts.items())),
            "validation_counts": dict(sorted(status_counts.items())),
            "validation": {"valid": not collection_errors, "errors": collection_errors},
            "semantic_interpretation_performed": False,
        }
        _write_json(blocks_dir / "document-manifest.json", collection_manifest)
        _write_json(reconciliation_dir / "manifest.json", {
            "source_sha256": source_hash, "selected_pages": pages,
            "counts": dict(sorted(reconciliation_counts.items())),
            "policy": "image-only vision and native/OCR/OpenCV run independently; only uniquely printed, spatially owned pairs can correct a source block; uncertainty remains in review",
        })
        manifest["stages"].extend([
            {"name": "region-routing-and-extraction", "status": "complete", "finished_utc": _now()},
            {"name": "independent-claim-reconciliation", "status": "complete", "finished_utc": _now()},
            {"name": "reconstruction-and-validation", "status": "complete", "finished_utc": _now()},
            {"name": "unified-source-block-collection", "status": "complete", "finished_utc": _now()},
        ])
        manifest["status"] = "complete"
        manifest["finished_utc"] = _now()
        manifest["source_blocks_manifest"] = "source-blocks/document-manifest.json"
        manifest["validation"] = collection_manifest["validation"]
        _write_json(output / "manifest.json", manifest)
        from .validate_run import validate_run
        contract = validate_run(output)
        _write_json(output / "validation.json", contract)
        manifest["contract_validation"] = {
            "valid": contract["result"] == "valid",
            "schema_error_count": contract["schema_error_count"],
            "integrity_error_count": contract["integrity_error_count"],
            "report": "validation.json",
        }
        _write_json(output / "manifest.json", manifest)
        return output
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_utc"] = _now()
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output / "manifest.json", manifest)
        raise

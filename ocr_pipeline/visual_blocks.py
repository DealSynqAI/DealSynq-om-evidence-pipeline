"""Classify visual regions and bind chart, KPI, and map labels to values."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import math
from pathlib import Path
import re
import statistics
from typing import Any

from .common import (
    CHART_TERMS, MAP_TERMS, NUMBER, US_GEOGRAPHIES, _box_union, _center, _fold_token,
    _line_distance, _numeric_value, _provenance, _visible_lines, _vision_summary,
)
from .map_geometry import ATLAS_URL, colored_mark_for_value, register_us_state_map, state_for_value
from .models import Region, SourceBlock
from .paddle_tables import ocr_grid_rows, ocr_paired_columns

# "Price: $420,000" or "IRR: 17.3%": a printed fact, not a data label.
_FACT_LINE = re.compile(r"[^\d:]*[A-Za-z][^\d:]*:\s*\S")
_CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")


def _chart_text_support(lines: list[dict[str, Any]]) -> bool:
    """Whether a region's printed text reads like a plot.

    Chart text is at least half numbers: data labels, axis ticks, and dates, with
    category and title lines for the rest. Prose, fact-sheet lines, contact details, and a
    couple of stray numbers do not describe a plot, whatever shapes the pixels
    contain, so they cannot support a chart on their own.
    """
    texts = [str(line.get("text", "")).strip() for line in lines if str(line.get("text", "")).strip()]
    numeric = [text for text in texts if re.search(r"\d", text)]
    facts = [text for text in numeric if _FACT_LINE.match(text)]
    return len(numeric) >= 2 and len(numeric) * 2 >= len(texts) and len(facts) * 2 < len(numeric)


def _classify_visual(region: Region, lines: list[dict[str, Any]], features: dict[str, Any]) -> tuple[str, float, list[str]]:
    text = " ".join(str(line["text"]) for line in lines).lower()
    warnings: list[str] = []
    # Native words are added to visual regions one word at a time; line-level
    # evidence (prose, facts, data labels) is judged on the OCR lines.
    ocr_lines = [line for line in lines if "-ocr-" in str(line.get("evidence_id", ""))] or lines
    chart_text = _chart_text_support(ocr_lines)
    visual_hint = str(region.metadata.get("visual_hint") or "").strip().lower()
    if visual_hint == "chart" and chart_text:
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
    # Ruled lines alone also describe chart gridlines; a table needs printed
    # cells that line up in rows as well.
    if (table_grid_confidence >= 0.60 and horizontal >= 4 and vertical >= 3 and numeric_density >= 0.35
            and ocr_grid_rows(ocr_lines, min_columns=2) >= 3):
        return "table", 0.76, warnings
    if ocr_grid_rows(ocr_lines) >= 4 or ocr_paired_columns(ocr_lines):
        return "table", 0.72, warnings
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
    if numeric >= 2 and chart_text and (
            repeated_marks or legend_marks or (plot_boundary and axis_pair and axis_ticks >= 3)):
        return "chart", 0.78, warnings
    ocr_texts = [str(line.get("text", "")).strip() for line in ocr_lines if str(line.get("text", "")).strip()]
    if (sum(bool(_FACT_LINE.match(text)) for text in ocr_texts) >= 3
            or sum(bool(_CONTACT.search(text)) for text in ocr_texts) >= 2):
        return "normal_text", 0.70, ["visual region prints facts or contact details, not a data plot"]
    prose_lines = sum(len(str(line.get("text", "")).split()) >= 4 for line in lines)
    ocr_prose = sum(len(re.findall(r"[A-Za-z]{2,}", text)) >= 4 for text in ocr_texts)
    if ((len(lines) >= 10 and prose_lines >= max(8, math.ceil(0.55 * len(lines)))
            and numeric_density < 0.60 and axis_ticks < 3
            and table_grid_confidence < 0.30 and bars < 3
            and printed_place_labels < 3)
            or (len(ocr_texts) >= 6 and ocr_prose * 2 >= len(ocr_texts) and not chart_text
                and printed_place_labels < 3)):
        return "normal_text", 0.75, ["visual page has substantial OCR prose without a supported data plot"]
    if numeric >= 2 and (map_term_hits or chart_term_hits):
        warnings.append("semantic wording requires visual-model confirmation; no structural plot evidence found")
    elif (horizontal + vertical + rectangles) >= 8 and numeric >= 2:
        warnings.append("ambiguous geometry requires visual-model confirmation")
    if len(lines) >= 3 and numeric == 0:
        return "normal_text", 0.62, ["image region treated as scanned text"]
    warnings.append("visual type could not be classified confidently")
    return "unclassified_visual", 0.35, warnings


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
    # Native words that repeat an OCR phrase are rendered once, as in text blocks.
    labels = [str(line["text"]) for line in _visible_lines(lines) if str(line["text"]).strip()]
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
    # The vision model is a second opinion. Its objection stands only where an
    # item lacks its own verified evidence; verified geometry outweighs it.
    if (any("disagreed with deterministic" in warning for warning in reconciliation_warnings)
            and not (nested_items and all(item["validation_status"] == "passed" for item in nested_items))):
        parent.validation_status = "needs_review"
    return [parent]

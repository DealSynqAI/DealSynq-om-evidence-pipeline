"""Shared evidence helpers and constants for building Unified Source Blocks."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import unicodedata

from PIL import Image

from .models import Region


NUMBER = re.compile(r"(?:[$€£]\s*)?\(?\d[\d,.]*(?:\s*(?:%|x|million|billion|bn|mm|m|b|k))?\)?", re.I)
# One printed scalar, signed or in accounting parentheses: -$1,250, $-1,250,
# (1,250), ($1,250), (3.5%), −5.0% (U+2212), or 1.234.567 grouped by periods.
_SCALAR_NUMBER = r"(?:\d{1,3}(?:\.\d{3}){2,}|\d[\d,]*(?:\.\d+)?)"
_TABLE_SCALAR = re.compile(
    r"\s*(?:[+\-−–]?(?:[$€£]\s*)?[\-−–]?" + _SCALAR_NUMBER
    + r"|\(\s*(?:[$€£]\s*)?" + _SCALAR_NUMBER + r"\s*%?\s*\))\s*"
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


def _crop(image_path: Path, coordinates: list[float], output_path: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
        x, y, w, h = coordinates
        box = (
            max(0, round(x * width / 1000)), max(0, round(y * height / 1000)),
            min(width, round((x + w) * width / 1000)), min(height, round((y + h) * height / 1000)),
        )
        image.crop(box).save(output_path)


def _ocr_text(lines: list[dict[str, Any]]) -> str:
    return "\n".join(str(line["text"]) for line in _visible_lines(lines))


def _visible_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Image-backed PDF text may appear twice: RapidOCR contributes a full
    # phrase while positioned native words contribute each word separately.
    # Keep both in provenance, but render an overlapping word only once.
    def tokens(value: str) -> list[str]:
        return re.findall(r"\w+", _fold_token(value))

    phrases = [line for line in lines if
        "-native-" not in str(line.get("evidence_id", ""))
        and len(tokens(str(line.get("text", "")))) >= 2]

    def covered(native: dict[str, Any]) -> bool:
        needle = tokens(str(native.get("text", "")))
        symbol = str(native.get("text", "")).strip()
        if not needle and (len(symbol) != 1 or symbol.isalnum()):
            return False
        nx, ny = _center(native["coordinates"])
        for phrase in phrases:
            px, py, pw, ph = phrase["coordinates"]
            if not (px - 15 <= nx <= px + pw + 15 and py - 18 <= ny <= py + ph + 18):
                continue
            if not needle:
                if symbol in str(phrase.get("text", "")):
                    return True
                continue
            words = tokens(str(phrase.get("text", "")))
            if any(words[start:start + len(needle)] == needle for start in range(len(words) - len(needle) + 1)):
                return True
        return False

    visible = [line for line in lines if not (
        "-native-" in str(line.get("evidence_id", "")) and covered(line)
    )]
    return sorted(visible, key=lambda line: (line["coordinates"][1], line["coordinates"][0]))


def _token_agreement(left: str, right: str) -> float | None:
    left_tokens = set(re.findall(r"\w+", left.casefold()))
    right_tokens = set(re.findall(r"\w+", right.casefold()))
    if not left_tokens or not right_tokens:
        return None
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 5)


def _fold_token(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in value if not unicodedata.combining(character))


def _line_box(lines: list[dict[str, Any]]) -> list[float]:
    x0 = min(float(line["coordinates"][0]) for line in lines)
    y0 = min(float(line["coordinates"][1]) for line in lines)
    x1 = max(float(line["coordinates"][0]) + float(line["coordinates"][2]) for line in lines)
    y1 = max(float(line["coordinates"][1]) + float(line["coordinates"][3]) for line in lines)
    return [x0, y0, x1 - x0, y1 - y0]


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


# One printed number with the sign and parentheses that touch it. A minus may
# sit before or after the currency symbol (-$1,250, $-1,250); U+2212 and an en
# dash touching the digits are minus signs too, while a spaced dash is a range
# or separator ($450 - $550).
_SIGNED_NUMBER = re.compile(
    r"(?P<open>\(\s*)?(?P<sign>[-−–](?=[$€£]?\d))?(?P<currency>[$€£]\s*)?"
    r"(?P<inner_sign>[-−–](?=\d))?(?P<digits>\d[\d,.]*\d|\d)(?P<close>\s*%?\s*\))?"
)


def _parse_digits(digits: str) -> float:
    """Read thousands separators: 1,234.5, European 1.234,5, and grouped 1.234.567."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", digits):
        return float(digits.replace(".", "").replace(",", "."))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}", digits):
        return float(digits.replace(".", ""))
    whole = digits.replace(",", "")
    match = re.match(r"\d+(?:\.\d+)?", whole)
    return float(match.group()) if match else float("nan")


def _numeric_value(text: str) -> tuple[float | None, str | None, float | None]:
    """Parse the first printed number in ``text`` into (value, unit, normalized value).

    A value in parentheses is an accounting negative, (1,250) or ($1,250), but a
    lone parenthesized digit such as (1) is a footnote marker and is skipped. A
    single period followed by three digits stays a decimal here; only the
    surrounding table can tell whether it is a misread thousands comma.
    """
    cleaned = text.replace(",", "").strip()
    match = None
    for candidate in _SIGNED_NUMBER.finditer(text):
        footnote = (candidate.group("open") and candidate.group("close")
                    and re.fullmatch(r"\d", candidate.group("digits")))
        if not footnote:
            match = candidate
            break
    if not match:
        return None, None, None
    value = _parse_digits(match.group("digits"))
    if math.isnan(value):
        return None, None, None
    if match.group("sign") or match.group("inner_sign") or (match.group("open") and match.group("close")):
        value = -value
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


def _cluster_unassigned_lines(
    pending: list[tuple[int, dict[str, Any]]],
) -> list[list[tuple[int, dict[str, Any]]]]:
    """Make spatial text fragments into conservative raw source regions."""
    def phrase_covers(native: dict[str, Any], phrase: dict[str, Any]) -> bool:
        if "-ocr-" not in str(phrase.get("evidence_id", "")):
            return False
        needle = re.findall(r"\w+", _fold_token(str(native.get("text", ""))))
        words = re.findall(r"\w+", _fold_token(str(phrase.get("text", ""))))
        if not needle or len(needle) > len(words):
            return False
        nx, ny = _center(native["coordinates"])
        px, py, pw, ph = phrase["coordinates"]
        if not (px - 5 <= nx <= px + pw + 5 and py - 5 <= ny <= py + ph + 5):
            return False
        return any(words[start:start + len(needle)] == needle
                   for start in range(len(words) - len(needle) + 1))

    clusters: list[list[tuple[int, dict[str, Any]]]] = []
    for item in sorted(pending, key=lambda pair: (
        float(pair[1]["coordinates"][1]), float(pair[1]["coordinates"][0]),
    )):
        line = item[1]
        x, y, width, height = (float(value) for value in line["coordinates"])
        matches: list[tuple[float, list[tuple[int, dict[str, Any]]]]] = []
        for cluster in clusters:
            if line.get("evidence_source") == "native_pdf_positioned_word":
                if any(phrase_covers(line, other) for _index, other in cluster):
                    matches.append((-1.0, cluster))
                    continue
            previous = cluster[-1][1]["coordinates"]
            px, py, pw, ph = (float(value) for value in previous)
            vertical_gap = y - (py + ph)
            aligned = abs(x - px) <= 35 or abs((x + width / 2) - (px + pw / 2)) <= 60
            if aligned and -max(height, ph) <= vertical_gap <= max(30, 1.5 * max(height, ph)):
                matches.append((abs(vertical_gap) + abs(x - px) / 4, cluster))
        if matches:
            min(matches, key=lambda pair: pair[0])[1].append(item)
        else:
            clusters.append([item])
    return clusters


def _center(box: list[float]) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2

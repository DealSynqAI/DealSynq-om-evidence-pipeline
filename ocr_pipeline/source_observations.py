"""Collect independent printed-text observations for the page evidence ledger."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pdfplumber
from PIL import Image


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


_NONPRINTED_CONTENT_KEYS = {
    "column_id", "row_id", "section_id", "block_id", "child_block_ids",
    "value_kind", "value_state", "grounding_status", "validation_status",
    "unit", "selected", "token_agreement", "chart_type", "source",
}


# Separators, bullets, and box rules. Symbol-font bullets land in the Unicode
# private-use area, so that range counts too.
_LAYOUT_GLYPHS = frozenset("�|~/•→&—│▪●■◆►◦")


def is_layout_glyph(observation: dict[str, Any]) -> bool:
    """Identify short nonnumeric glyphs whose exact source text can be retained.

    Currency, percent, plus, and minus signs, hyphens, and en dashes are
    excluded because they can change a nearby numeric value or range. The
    retained glyph remains traceable to its native PDF word and page coordinates.
    """
    raw = str(observation.get("text") or "").strip()
    return (observation.get("source") == "native_pdf_word"
            and 0 < len(raw) <= 2
            and all(character in _LAYOUT_GLYPHS or 0xE000 <= ord(character) <= 0xF8FF
                    for character in raw))


def represented_in_content(observation: dict[str, Any], content: Any) -> bool:
    """Check whether an unmatched source observation is already printed in content.

    A native PDF word needs an exact text token. A Paddle layout block needs
    its complete normalized string, so a partial overlap never hides text that
    the source block omitted. Structural IDs and status fields are excluded.
    """
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for nested in value for item in strings(nested)]
        if isinstance(value, dict):
            return [item for key, nested in value.items()
                    if key not in _NONPRINTED_CONTENT_KEYS for item in strings(nested)]
        return []

    key = _normalized(str(observation.get("text") or ""))
    if len(key) < 2:
        return False
    visible = strings(content)
    if observation.get("source") == "native_pdf_word":
        return any(key in {_normalized(token) for token in re.findall(r"\S+", item)}
                   for item in visible)
    if observation.get("source") == "paddle_layout_text" and len(key) >= 5:
        return any(key in _normalized(item) for item in visible)
    return False


def _box(x0: float, y0: float, x1: float, y1: float, width: float, height: float) -> list[float]:
    x = max(0.0, min(1000.0, x0 * 1000 / width))
    y = max(0.0, min(1000.0, y0 * 1000 / height))
    return [x, y, max(0.0, min(1000.0 - x, (x1 - x0) * 1000 / width)),
            max(0.0, min(1000.0 - y, (y1 - y0) * 1000 / height))]


def _near(left: list[float], right: list[float], margin: float = 12) -> bool:
    x, y, width, height = left
    rx, ry, rw, rh = right
    cx, cy = x + width / 2, y + height / 2
    return rx - margin <= cx <= rx + rw + margin and ry - margin <= cy <= ry + rh + margin


def _match_ocr(observation: dict[str, Any], ocr_lines: list[dict[str, Any]]) -> list[str]:
    key = _normalized(observation["text"])
    if not key:
        return []
    nearby = [line for line in ocr_lines if _near(line["coordinates"], observation["coordinates"], 14)
              or _near(observation["coordinates"], line["coordinates"], 14)]
    if observation["source"] == "native_pdf_word":
        return [str(line["evidence_id"]) for line in nearby
                if (key in {_normalized(token) for token in re.findall(r"[\w]+", str(line.get("text") or ""))}
                    or (len(key) >= 4 and key in _normalized(str(line.get("text") or ""))))]
    matched = [line for line in nearby
               if (len(_normalized(str(line.get("text") or ""))) >= 3
                   and _normalized(str(line.get("text") or "")) in key)]
    matched_chars = sum(len(_normalized(str(line.get("text") or ""))) for line in matched)
    if matched_chars >= 0.6 * len(key):
        return [str(line["evidence_id"]) for line in matched]
    return []


def collect_page_observations(
    pdf: Path, pages: list[int], ocr_by_page: dict[int, dict[str, Any]],
    paddle_dir: Path, rendered: dict[int, Path],
) -> dict[int, list[dict[str, Any]]]:
    """Retain unmatched native words and Paddle text separately from RapidOCR."""
    result: dict[int, list[dict[str, Any]]] = {page: [] for page in pages}
    with pdfplumber.open(pdf) as document:
        for page_number in pages:
            page = document.pages[page_number - 1]
            words = page.extract_words(use_text_flow=False) or []
            for index, word in enumerate(words, 1):
                raw = str(word.get("text") or "").strip()
                if not raw:
                    continue
                result[page_number].append({
                    "evidence_id": f"p{page_number:03d}-pdfword-{index:04d}",
                    "source": "native_pdf_word", "text": raw, "confidence": 1.0,
                    "coordinates": _box(float(word["x0"]), float(word["top"]),
                                        float(word["x1"]), float(word["bottom"]),
                                        float(page.width), float(page.height)),
                })
            paddle_path = paddle_dir / f"page-{page_number:03d}.json"
            if paddle_path.is_file():
                payload = json.loads(paddle_path.read_text(encoding="utf-8"))
                with Image.open(rendered[page_number]) as image:
                    image_width, image_height = image.size
                for index, item in enumerate(payload.get("res", payload).get("parsing_res_list") or [], 1):
                    if str(item.get("block_label") or "").lower() == "table":
                        continue  # table candidates and their HTML are stored in paddle-tables/
                    raw = str(item.get("block_content") or "").strip()
                    bbox = item.get("block_bbox")
                    if not raw or not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    result[page_number].append({
                        "evidence_id": f"p{page_number:03d}-paddle-text-{index:04d}",
                        "source": "paddle_layout_text", "text": raw, "confidence": 0.7,
                        "coordinates": _box(*(float(v) for v in bbox), image_width, image_height),
                    })
            ocr_lines = ocr_by_page.get(page_number, {}).get("lines", [])
            for observation in result[page_number]:
                observation["matched_ocr_evidence_ids"] = _match_ocr(observation, ocr_lines)
    return result

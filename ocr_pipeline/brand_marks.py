"""Recognize and recover short logo text from repetition and placement."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

from .common import NUMBER, _crop, _fold_token, _line_box, _provenance
from .models import PageInspection, Region, SourceBlock


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

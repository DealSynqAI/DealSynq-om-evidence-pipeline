"""Late passes that keep every OCR line and source observation in some block."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .common import _cluster_unassigned_lines, _line_box, _ocr_text, _provenance, clean_text
from .models import PageInspection, Region, SourceBlock
from .source_observations import is_layout_glyph, represented_in_content

_WORD = re.compile(r"[a-z]+")
_NUMBER = re.compile(r"\d[\d,.]*\d|\d")
_STRUCTURED = {"table", "chart", "map", "kpi_panel"}
# Symbols that change a number they touch: sign, accounting parenthesis, unit.
_SIGNS = set("-+()$%~<>") | {"−", "–", "€", "£"}


def _content_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _content_strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _content_strings(nested)]
    return []


def _every_mark_valued(block: dict[str, Any]) -> bool:
    """A chart or map whose every detected mark already carries a verified printed value."""
    content = block.get("content", {})
    if block.get("type") == "map":
        items, expected = content.get("bindings") or [], content.get("expected_binding_count")
    elif block.get("type") == "chart":
        items, expected = content.get("observations") or [], content.get("expected_observation_count")
    else:
        return False
    return (bool(items) and isinstance(expected, int) and len(items) >= expected
            and all(item.get("validation_status") == "passed" for item in items))


def _beside_number(box: list[float], lines: list[dict[str, Any]]) -> bool:
    """Whether a positioned number sits on the same baseline right next to ``box``."""
    x, y, width, height = box
    center = y + height / 2
    for line in lines:
        if not re.search(r"\d", str(line.get("text", ""))):
            continue
        lx, ly, lw, lh = line["coordinates"]
        if abs(ly + lh / 2 - center) > 0.6 * max(height, lh):
            continue
        if max(lx - (x + width), x - (lx + lw)) <= 1.5 * max(height, lh):
            return True
    return False


def _immaterial_evidence(
    text: str, owner: dict[str, Any], owner_lines: list[dict[str, Any]], box: list[float] | None,
) -> bool:
    """Whether leftover source text cannot change what the owner's content says.

    It is kept as raw evidence either way; only material text sends the owner to
    review. Immaterial: a symbol that cannot be a sign or unit (a bullet, bar,
    or underscore), or a sign the content already prints or that stands apart
    from any number; a one- to
    three-letter fragment of a word the content prints; a one- or two-digit
    fragment of a number it prints; a line whose words and numbers all appear
    in the content. For tables, charts, maps, and KPI panels, text without digits
    (titles, legends, footnotes, map labels) is not a lost value, and neither is
    any number left over once every mark of a chart or map carries a verified value.
    """
    raw = str(text or "").strip()
    if not raw:
        return True
    folded = raw.casefold()
    content = " ".join(_content_strings(owner.get("content", {}))).casefold()
    words, numbers = _WORD.findall(folded), _NUMBER.findall(folded)
    if not words and not numbers:
        signs = [character for character in raw if character in _SIGNS]
        return not signs or all(character in content for character in signs) or not (
            box and _beside_number(box, owner_lines))
    content_words, content_numbers = set(_WORD.findall(content)), _NUMBER.findall(content)
    if all(word in content_words for word in words) and all(number in content_numbers for number in numbers):
        return True
    if not numbers and len(raw) <= 3 and raw.isalpha() and folded in content:
        return True
    if not words and all(len(re.sub(r"\D", "", number)) <= 2 and any(number in found for found in content_numbers)
                         for number in numbers):
        return True
    if owner.get("type") in _STRUCTURED and not numbers:
        return True
    return _every_mark_valued(owner)


def _preserve_ocr_lines_in_blocks(
    blocks: list[dict[str, Any]], lines: list[dict[str, Any]],
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep every nonblank OCR line inside a USB block without inventing an owner.

    Existing structured content retains its owner. An owned OCR line omitted by
    that block's content is attached as raw evidence. A line with no defensible
    owner joins a conservative spatial source-text block, preserving the exact
    OCR lines and their image boxes.
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
    pending_fallback: list[tuple[int, dict[str, Any]]] = []
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
                physical = [block for block in blocks
                            if "-decomposed-" in block.get("block_id", "")
                            and block.get("type") in {"text", "heading", "contact"}
                            and near(block, line)]
                if len(physical) == 1:
                    owner = physical[0]
                    owner["provenance"]["ocr_evidence_ids"].append(evidence_id)
                    by_owner[evidence_id] = owner
                    stats["owner_recovered"] += 1
            if owner is None:
                try:
                    coordinates = [float(value) for value in line.get("coordinates", [])]
                except (TypeError, ValueError):
                    coordinates = []
                if len(coordinates) != 4:
                    coordinates = [0.0, 0.0, 0.0, 0.0]
                pending_fallback.append((index, {**line, "coordinates": coordinates}))
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
        stats["raw_lines_attached"] += 1
        owned_ids = set(owner.get("provenance", {}).get("ocr_evidence_ids", []))
        if _immaterial_evidence(raw, owner, [other for other in lines if other.get("evidence_id") in owned_ids],
                                line.get("coordinates")):
            stats["immaterial_raw_lines"] = stats.get("immaterial_raw_lines", 0) + 1
            continue
        owner["validation"]["status"] = "needs_review"
        warning = "OCR lines are retained as raw evidence because structured content omits them"
        if warning not in owner["validation"]["warnings"]:
            owner["validation"]["warnings"].append(warning)
    for cluster in _cluster_unassigned_lines(pending_fallback):
        first_index = cluster[0][0]
        cluster_lines = [line for _index, line in cluster]
        coordinates = _line_box(cluster_lines)
        raw = _ocr_text(cluster_lines)
        block_id = f"p{inspection.page:03d}-raw-ocr-{first_index:04d}-text"
        if block_id in existing_ids:
            raise ValueError(f"duplicate raw OCR fallback block ID: {block_id}")
        existing_ids.add(block_id)
        region = Region(
            region_id=f"p{inspection.page:03d}-raw-ocr-{first_index:04d}",
            page=inspection.page, kind="normal_text", coordinates=coordinates,
            reading_order=len(blocks) + 1,
            classification_method="unassigned OCR spatial cluster preservation",
            confidence=min(float(line.get("confidence", 0.0)) for line in cluster_lines),
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
            content={"text": clean_text(raw), "evidence_text": {
                "selected": "ocr", "native": None, "ocr": raw, "token_agreement": None,
            }},
            extraction_method=["RapidOCR PP-OCRv6", "Python spatial evidence preservation"],
            confidence=region.confidence,
            validation_status="needs_review", errors=[],
            warnings=["OCR text retained without a verified structural owner"],
            provenance=_provenance(source_hash, region, cluster_lines, image),
            semantic_role="unresolved_source_text",
        ).as_dict()
        fallback["raw_evidence_lines"] = [{
            "evidence_id": str(line["evidence_id"]), "text": str(line["text"]),
            "confidence": max(0.0, min(1.0, float(line.get("confidence", 0.0)))),
            "coordinates": line["coordinates"],
        } for line in cluster_lines]
        blocks.append(fallback)
        for line in cluster_lines:
            by_owner[str(line["evidence_id"])] = fallback
        stats["fallback_text_blocks"] += 1
    _propagate_group_status(blocks)
    return blocks, stats


def _propagate_group_status(blocks: list[dict[str, Any]]) -> None:
    """Serialized-block form of _propagate_group_validation for passes that run after extraction."""
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


def _preserve_source_observations(
    blocks: list[dict[str, Any]], observations: list[dict[str, Any]],
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
    lines: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Retain native and Paddle observations not already covered by RapidOCR.

    ``lines`` are the page's positioned OCR lines, used to tell whether a
    leftover symbol sits beside a number.
    """
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
        elif is_layout_glyph(observation):
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
            observation["disposition"] = "raw_attached"
            stats["raw_attached"] += 1
            owned_ids = set(owner.get("provenance", {}).get("ocr_evidence_ids", []))
            if not _immaterial_evidence(observation["text"], owner,
                                        [line for line in lines or [] if line.get("evidence_id") in owned_ids], box):
                owner["validation"]["status"] = "needs_review"
                warning = "additional source text is retained as raw evidence"
                if warning not in owner["validation"]["warnings"]:
                    owner["validation"]["warnings"].append(warning)
        observation["owner_block_id"] = owner["block_id"]
    _propagate_group_status(blocks)
    return stats

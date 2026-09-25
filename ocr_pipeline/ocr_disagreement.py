"""Preserve OCR conflicts and their physical crops without guessing a winner."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image


def needs_second_ocr(ocr_page: dict[str, Any]) -> bool:
    return any(float(line.get("confidence", 1)) < 0.80
               and len(re.findall(r"[A-Za-z]", str(line.get("text", "")))) >= 8
               for line in ocr_page.get("lines", [])
               if "-ocr-" in str(line.get("evidence_id", "")))


def _paddle_lines(payload: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    result = payload.get("res", payload)
    ocr = result.get("overall_ocr_res") or {}
    found = []
    for index, (text, confidence, box) in enumerate(zip(
        ocr.get("rec_texts") or [], ocr.get("rec_scores") or [], ocr.get("rec_boxes") or [],
    )):
        if not isinstance(box, (list, tuple)) or len(box) != 4 or not str(text).strip():
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        found.append({"evidence_id": f"paddle-ocr-{index + 1:04d}", "text": str(text),
                      "confidence": float(confidence), "coordinates": [
                          x0 * 1000 / width, y0 * 1000 / height,
                          (x1 - x0) * 1000 / width, (y1 - y0) * 1000 / height,
                      ]})
    return found


def _intersection(a: list[float], b: list[float]) -> tuple[float, float]:
    return (max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])),
            max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])))


def _union(boxes: list[list[float]]) -> list[float]:
    x0 = max(0.0, min(box[0] for box in boxes))
    y0 = max(0.0, min(box[1] for box in boxes))
    x1 = min(1000.0, max(box[0] + box[2] for box in boxes))
    y1 = min(1000.0, max(box[1] + box[3] for box in boxes))
    return [x0, y0, x1 - x0, y1 - y0]


def _nearby(box: list[float], paddle: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for line in paddle:
        other = line["coordinates"]
        ix, iy = _intersection(box, other)
        if ix >= 0.35 * min(box[2], other[2]) and iy >= 0.30 * min(box[3], other[3]):
            matches.append(line)
    return sorted(matches, key=lambda item: (item["coordinates"][1] // 12,
                                             item["coordinates"][0]))


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _crop(image: Path, box: list[float], destination: Path) -> None:
    with Image.open(image) as page:
        x0 = max(0, round((box[0] - 8) * page.width / 1000))
        y0 = max(0, round((box[1] - 8) * page.height / 1000))
        x1 = min(page.width, round((box[0] + box[2] + 8) * page.width / 1000))
        y1 = min(page.height, round((box[1] + box[3] + 8) * page.height / 1000))
        page.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))).save(destination)


def record_disagreements(
    page: int, image: Path, rapid_lines: list[dict[str, Any]],
    paddle_payload: dict[str, Any] | None, blocks: list[dict[str, Any]],
    crop_dir: Path, receipt_path: Path,
) -> dict[str, Any]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image) as page_image:
        width, height = page_image.size
    paddle = _paddle_lines(paddle_payload or {}, width, height)
    by_ocr = {str(key): block for block in blocks
              for key in block.get("provenance", {}).get("ocr_evidence_ids", [])}
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(owner: dict[str, Any] | None, target: dict[str, str], box: list[float],
            primary: dict[str, Any], alternatives: list[dict[str, Any]], reason: str) -> None:
        if owner is None:
            return
        key = (owner["block_id"], json.dumps(target, sort_keys=True))
        if key in seen:
            return
        seen.add(key)
        number = len(records) + 1
        filename = f"page-{page:03d}-conflict-{number:04d}.png"
        crop_path = crop_dir / filename
        _crop(image, box, crop_path)
        record = {"disagreement_id": f"p{page:03d}-conflict-{number:04d}",
                  "status": "unresolved", "reason": reason, "target": target,
                  "coordinates": [round(float(v), 3) for v in box],
                  "crop": f"disagreement-crops/{filename}",
                  "crop_sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest(),
                  "readings": [primary, *alternatives]}
        owner.setdefault("ocr_disagreements", []).append(record)
        owner["validation"]["status"] = "needs_review"
        warning = "printed glyph or prose has unresolved OCR readings; see crop"
        if warning not in owner["validation"]["warnings"]:
            owner["validation"]["warnings"].append(warning)
        records.append({"owner_block_id": owner["block_id"], **record})

    for line in rapid_lines:
        evidence_id = str(line.get("evidence_id", ""))
        if "-ocr-" not in evidence_id or not str(line.get("text", "")).strip():
            continue
        confidence = float(line.get("confidence", 1))
        matches = _nearby(line["coordinates"], paddle)
        alternative = " ".join(item["text"] for item in matches)
        different = alternative and _normalized(alternative) != _normalized(str(line["text"]))
        meaningful = different and (difflib.SequenceMatcher(
            None, _normalized(alternative), _normalized(str(line["text"]))).ratio() >= 0.35)
        if not (confidence < 0.80 and len(re.findall(r"[A-Za-z]", str(line["text"]))) >= 8
                or meaningful and any(item["confidence"] >= 0.75 for item in matches)):
            continue
        if not different and confidence >= 0.80:
            continue
        primary = {"source": "RapidOCR", "text": str(line["text"]),
                   "confidence": confidence, "evidence_id": evidence_id}
        alternatives = [{"source": "PaddleOCR", "text": alternative,
                         "confidence": min(item["confidence"] for item in matches),
                         "evidence_id": ",".join(item["evidence_id"] for item in matches)}] if matches else []
        reason = "OCR engines disagree" if different else "low confidence; second transcript unavailable"
        add(by_ocr.get(evidence_id), {"type": "ocr_line", "evidence_id": evidence_id},
            _union([line["coordinates"], *(item["coordinates"] for item in matches)]),
            primary, alternatives, reason)

    def table_text(block: dict[str, Any], raw: str | None, box: list[float] | None,
                   target: dict[str, str], evidence_ids: list[str] | None = None) -> None:
        if not raw or not isinstance(box, list) or len(box) != 4 or box[2] <= 0 or box[3] <= 0:
            return
        matches = [item for item in _nearby(box, paddle)
                   if (box[0] - 3 <= item["coordinates"][0] + item["coordinates"][2] / 2
                       <= box[0] + box[2] + 3)
                   and item["coordinates"][2] <= max(25, box[2] * 1.5)]
        if not matches:
            return
        alternative = " ".join(item["text"] for item in matches)
        if (_normalized(str(raw)) in _normalized(alternative)
                or not any(item["confidence"] >= 0.75 for item in matches)):
            return
        rapid = [item for item in _nearby(box, rapid_lines)
                 if "-ocr-" in str(item.get("evidence_id", ""))
                 and box[0] - 3 <= item["coordinates"][0] + item["coordinates"][2] / 2
                 <= box[0] + box[2] + 3]
        readings = [{"source": "RapidOCR", "text": " ".join(item["text"] for item in rapid),
                     "confidence": min(item["confidence"] for item in rapid),
                     "evidence_id": ",".join(item["evidence_id"] for item in rapid)}] if rapid else []
        readings.append({"source": "PaddleOCR", "text": alternative,
                         "confidence": min(item["confidence"] for item in matches),
                         "evidence_id": ",".join(item["evidence_id"] for item in matches)})
        add(block, target, _union([box, *(item["coordinates"] for item in matches)]),
            {"source": "table_extraction", "text": str(raw), "confidence": None,
             "evidence_id": ",".join(evidence_ids or [])}, readings,
            "table glyphs disagree with independent OCR")

    for block in blocks:
        if block.get("type") != "table":
            continue
        for column in block.get("content", {}).get("columns", []):
            table_text(block, column.get("label"), column.get("coordinates"),
                       {"type": "table_column", "column_id": str(column["column_id"])},
                       column.get("evidence_ids"))
        for row in block.get("content", {}).get("rows", []):
            table_text(block, row.get("label"), row.get("label_coordinates"),
                       {"type": "table_row", "row_id": str(row["row_id"])},
                       row.get("label_evidence_ids"))
            for cell in row.get("cells", []):
                table_text(block, cell.get("raw_value"), cell.get("coordinates"),
                           {"type": "table_cell", "row_id": str(row["row_id"]),
                            "column_id": str(cell["column_id"])}, cell.get("evidence_ids"))
    receipt = {"page": page, "status": "complete" if paddle else "second_ocr_unavailable",
               "paddle_line_count": len(paddle), "unresolved_count": len(records),
               "records": records}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"receipt": receipt_path.relative_to(crop_dir.parent).as_posix(),
            "unresolved_count": len(records), "paddle_line_count": len(paddle)}

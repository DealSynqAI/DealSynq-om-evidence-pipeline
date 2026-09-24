"""Run Qwen3-VL on each rendered page without OCR, PDF text, or geometry hints.

The output is a model proposal, never the final USB package. Raw responses and
attempt receipts are retained even when JSON parsing fails.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image
from jsonschema import Draft202012Validator
from .proposal_geometry import xyxy_to_xywh


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["page_number", "blocks"],
    "properties": {
        "page_number": {"type": "integer", "minimum": 1},
        "blocks": {"type": "array", "maxItems": 65, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["type", "bbox", "title", "text", "chart_type", "parent_index", "items", "rows"],
            "properties": {
                "type": {"enum": ["heading", "text", "footnote", "contact", "table", "comparison_panel",
                                   "chart", "map", "kpi_panel", "photograph", "brand_mark", "decoration", "group"]},
                "bbox": {"type": "array", "minItems": 4, "maxItems": 4,
                         "description": "Normalized [x0,y0,x1,y1] corners, NOT width/height",
                         "items": {"type": "number", "minimum": 0, "maximum": 1000}},
                "title": {"type": "string"},
                "text": {"type": "string"},
                "chart_type": {"type": "string"},
                "parent_index": {"type": ["integer", "null"], "minimum": 0},
                "items": {"type": "array", "maxItems": 130, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["label", "value", "unit", "owner"],
                    "properties": {"label": {"type": "string"}, "value": {"type": "string"},
                                   "unit": {"type": "string"}, "owner": {"type": "string"}},
                }},
                "rows": {"type": "array", "maxItems": 60, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["label", "section", "cells"],
                    "properties": {
                        "label": {"type": "string"}, "section": {"type": "string"},
                        "cells": {"type": "array", "maxItems": 20, "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["column", "value"],
                            "properties": {"column": {"type": "string"}, "value": {"type": "string"}},
                        }},
                    },
                }},
            },
        }},
    },
}

PROMPT = """Read this document page image and extract its complete visible content.
Return only JSON matching the schema. Use one block per meaningful heading,
paragraph, table, chart, map, KPI cluster, comparison panel, image, or group.
Keep text in visual reading order, including multi-column pages. Use normalized
top-left [x0,y0,x1,y1] corner boxes on a 0..1000 page scale: x1 and y1 are
the bottom-right coordinates, NOT width and height. Every box must satisfy
x0 < x1 and y0 < y1. parent_index is the
zero-based index of an earlier containing block, or null. Do not copy a chart
title into a paragraph. For tables, preserve every visible row and column
heading; distinguish blank cells from printed dashes. For charts and maps,
put each visible label/value/unit association in items and name its owning
subchart or series in owner. Do not turn legend endpoints or axis ticks into
observations. If a value cannot be read or its owner is uncertain, omit it
rather than guess. Transcribe legal text in column and paragraph order.
For a KPI panel, every visible number must also be an item with its displayed
label and unit; do not leave items empty when numbers appear in text. For a
table, put each row's visible row heading in row.label, including when the
first printed column has no heading. Keep each distinct row and item once.
Do not use outside knowledge, repeat sentences, or describe invisible content.
Empty strings and arrays are valid for inapplicable fields. The image is your
only evidence; no OCR transcript, PDF metadata, or previous page is supplied."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(image: Path, page: int, endpoint: str, model: str, timeout: int,
         attempt: int = 1, prompt_override: str | None = None) -> tuple[str, dict]:
    with Image.open(image) as original:
        rgb = original.convert("RGB")
        # Both this arm and the pipeline use the same rendered 300 DPI page;
        # the model receives a bounded image just as the pipeline worker does.
        rgb.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        from io import BytesIO
        buffer = BytesIO()
        rgb.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    body = {
        "model": model, "temperature": 0, "max_tokens": 8192 if attempt == 1 else 12000,
        "enable_thinking": False, "think": False,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "vision_page_proposal", "strict": True, "schema": SCHEMA,
        }},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": f"Page number: {page}.\n\n{prompt_override or PROMPT}" + (
                "\n\nThis is a fresh retry after an invalid response. Output compact, complete JSON. "
                "Never repeat a row, value, or JSON fragment; if uncertain, omit it." if attempt > 1 else "")},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
        ]}],
    }
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        answer = json.loads(response.read().decode("utf-8"))
    content = answer["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raw = str(content).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw, answer


def validate_proposal(raw: str, page: int) -> dict:
    """Reject malformed, repetitive, or geometrically invalid model output."""
    parsed = json.loads(raw)
    Draft202012Validator(SCHEMA).validate(parsed)
    if parsed.get("page_number") != page:
        raise ValueError(f"Model page_number {parsed.get('page_number')} != {page}")
    if not isinstance(parsed.get("blocks"), list) or len(parsed["blocks"]) > 65:
        raise ValueError("Proposal blocks must be a list of at most 65 items")
    signatures = Counter()
    for index, block in enumerate(parsed["blocks"]):
        xyxy_to_xywh(block.get("bbox"))
        signature = (block.get("type"), block.get("title"), block.get("text"),
                     tuple(block["bbox"]))
        signatures[signature] += 1
        items = Counter((item.get("label"), item.get("value"), item.get("unit"),
                         item.get("owner")) for item in block.get("items", []))
        if items and max(items.values()) > 3:
            raise ValueError(f"Block {index} repeats the same structured item excessively")
        parent = block.get("parent_index")
        if parent is not None and (not isinstance(parent, int) or isinstance(parent, bool) or
                                   parent < 0 or parent >= index):
            raise ValueError(f"Block {index} has invalid parent_index {parent!r}")
    if signatures and max(signatures.values()) > 3:
        raise ValueError("Model response repeats an identical block excessively")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1/chat/completions")
    parser.add_argument("--model", default="dealsynq-qwen3-vl:4b-instruct-16k")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--attempts", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for page in range(1, args.pages + 1):
        stem = f"page-{page:03d}"
        image = args.images / f"{stem}.png"
        receipt_path = args.output / f"{stem}.receipt.json"
        if receipt_path.exists():
            print(f"{stem}: already attempted; preserving original result", flush=True)
            continue
        start = time.perf_counter()
        receipt = {"page": page, "model": args.model, "endpoint": args.endpoint,
                   "image_sha256": sha256(image), "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                   "input_policy": "rendered page image only; no OCR/PDF/OpenCV/reference",
                   "started_utc": datetime.now(timezone.utc).isoformat()}
        attempts = []
        for attempt in range(1, args.attempts + 1):
            attempt_start = time.perf_counter()
            attempt_record = {"attempt": attempt, "max_tokens": 8192 if attempt == 1 else 12000}
            try:
                raw, answer = call(image, page, args.endpoint, args.model, args.timeout, attempt)
                (args.output / f"{stem}.attempt-{attempt:02d}.raw.txt").write_text(raw, encoding="utf-8")
                attempt_record.update({"usage": answer.get("usage"),
                                       "finish_reason": answer["choices"][0].get("finish_reason"),
                                       "raw_characters": len(raw)})
                parsed = validate_proposal(raw, page)
                (args.output / f"{stem}.raw.txt").write_text(raw, encoding="utf-8")
                (args.output / f"{stem}.json").write_text(
                    json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                attempt_record["status"] = "complete"
                receipt.update({"status": "complete", "block_count": len(parsed["blocks"])})
                print(f"{stem}: {receipt['block_count']} blocks (attempt {attempt})", flush=True)
            except Exception as exc:
                attempt_record.update({"status": "failed", "error_type": type(exc).__name__,
                                       "error": str(exc)[:1000]})
                print(f"{stem}: attempt {attempt} FAILED {type(exc).__name__}: {exc}", flush=True)
            attempt_record["seconds"] = round(time.perf_counter() - attempt_start, 3)
            attempts.append(attempt_record)
            if attempt_record["status"] == "complete":
                break
        receipt["attempts"] = attempts
        if "status" not in receipt:
            receipt.update({"status": "failed", "error_type": attempts[-1]["error_type"],
                            "error": attempts[-1]["error"]})
        receipt["seconds"] = round(time.perf_counter() - start, 3)
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

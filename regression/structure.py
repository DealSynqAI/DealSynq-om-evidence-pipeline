"""Structural checks on replayed source blocks: how printed content is arranged, not just which words appear.

Two kinds of checks run on every replay:

* corpus-wide measures (structure_metrics): the share of OCR words that reach
  structured content, the share of printed numbers inside tables that land in
  cells, and counts of sign flips, misread thousands separators, empty tables,
  and charts with and without data;
* the reference set in golden.json (golden_results): table cells, chart and map
  values, review verdicts, and chart-free pages that were checked by hand
  against the page images.
"""
from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

from .metric import content_strings, tokens

GOLDEN = Path(__file__).with_name("golden.json")
_NUMBER = re.compile(r"[-(]?\$?\d[\d,]*(?:\.\d+)?%?x?")
_NEGATIVE = re.compile(r"^[-−–]\s*[$€£]?\s*\d|^[$€£]\s*[-−]\d|^\(\s*[$€£]?\s*\d[\d,.]*\s*%?\s*\)$")
_PERIOD_THOUSANDS = re.compile(r"[-−–(]?\s*[$€£]?\s*-?[1-9]\d{0,2}\.\d{3}\)?")


def _pages(run: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((run / "source-blocks").glob("page-*.json"))]


def _numbers(text: Any) -> list[str]:
    return [re.sub(r"[^\d.]", "", found) for found in _NUMBER.findall(str(text or "")) if re.search(r"\d", found)]


def _center_inside(line: dict[str, Any], box: list[float]) -> bool:
    x, y, width, height = box
    cx = line["coordinates"][0] + line["coordinates"][2] / 2
    cy = line["coordinates"][1] + line["coordinates"][3] / 2
    return x <= cx <= x + width and y <= cy <= y + height


def structure_metrics(run: Path) -> dict[str, Any]:
    counts: Counter = Counter()
    for page in _pages(run):
        ocr = [line for line in page["evidence_ledger"]["ocr_lines"] if "-ocr-" in line["evidence_id"]]
        printed = Counter(token for line in ocr for token in tokens(line["text"]))
        structured = Counter(token for block in page["blocks"] if block["type"] != "group"
                             for text in content_strings(block) for token in tokens(text))
        counts["ocr_words"] += sum(printed.values())
        counts["ocr_words_structured"] += sum((printed & structured).values())
        for block in page["blocks"]:
            content = block.get("content", {})
            if block["type"] in {"chart", "map", "kpi_panel"}:
                items = content.get("observations") or content.get("bindings") or content.get("metrics") or []
                counts["visuals"] += 1
                counts["visuals_with_data"] += bool(items)
            if block["type"] != "table":
                continue
            rows, columns = content.get("rows", []), content.get("columns", [])
            cells = [cell for row in rows for cell in row.get("cells", [])]
            counts["tables"] += 1
            counts["empty_tables"] += len(columns) < 2 or not any(cell.get("raw_value") for cell in cells)
            inside = Counter(number for line in ocr if _center_inside(line, block["coordinates"])
                             for number in _numbers(line["text"]))
            in_cells = Counter(number for cell in cells for number in _numbers(cell.get("raw_value")))
            counts["table_numbers"] += sum(inside.values())
            counts["table_numbers_in_cells"] += sum((inside & in_cells).values())
            for cell in cells:
                raw, value = str(cell.get("raw_value") or "").strip(), cell.get("numeric_value")
                if value is None:
                    continue
                counts["sign_flips"] += bool(_NEGATIVE.match(raw)) and value > 0
                counts["period_misreads"] += bool(_PERIOD_THOUSANDS.fullmatch(raw)) and abs(value) < 1000
    return {
        "ocr_words_structured": round(counts["ocr_words_structured"] / max(1, counts["ocr_words"]), 4),
        "table_numbers_in_cells": round(counts["table_numbers_in_cells"] / max(1, counts["table_numbers"]), 4),
        "ocr_words": counts["ocr_words"], "table_numbers": counts["table_numbers"],
        **{key: counts[key] for key in ("sign_flips", "period_misreads", "tables", "empty_tables",
                                         "visuals", "visuals_with_data")},
    }


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _table_cell(blocks: list[dict[str, Any]], fact: dict[str, Any]) -> dict[str, Any] | None:
    for block in blocks:
        if block["type"] != "table":
            continue
        columns = {column["column_id"]: _norm(column.get("label")) for column in block["content"].get("columns", [])}
        for row in block["content"].get("rows", []):
            if _norm(row.get("label")) != _norm(fact["row"]):
                continue
            cells = {columns.get(cell["column_id"]): cell for cell in row.get("cells", [])}
            where = fact.get("row_where")
            if where and _norm((cells.get(_norm(where["column"])) or {}).get("raw_value")) != _norm(where["value"]):
                continue
            if _norm(fact["column"]) in cells:
                return cells[_norm(fact["column"])]
    return None


def _visual_item(blocks: list[dict[str, Any]], fact: dict[str, Any]) -> dict[str, Any] | None:
    for block in blocks:
        content = block.get("content", {})
        for item in content.get("observations") or content.get("bindings") or content.get("metrics") or []:
            labels = {_norm(item.get(key)) for key in ("category", "geography", "series", "label")}
            if _norm(fact["label"]) in labels and _norm(item.get("raw_value")) == _norm(fact["value"]):
                return item
    return None


def _anchored_block(blocks: list[dict[str, Any]], check: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [block for block in blocks if block["type"] == check["type"]]
    if check.get("anchor") is None:
        return candidates[0] if len(candidates) == 1 else None
    anchor = _norm(check["anchor"])

    def texts(block: dict[str, Any]) -> set[str]:
        content = block["content"]
        items = content.get("observations") or content.get("bindings") or content.get("metrics") or []
        labels = [item.get(key) for item in items for key in ("category", "geography", "series", "label")]
        return {_norm(text) for text in [*content_strings(block), *labels] if text}

    found = [block for block in candidates if anchor in texts(block)]
    return found[0] if len(found) == 1 else None


def golden_results(run_root: Path, docs: list[str]) -> dict[str, dict[str, str]]:
    """Evaluate every reference fact for the documents replayed under ``run_root``.

    Returns {fact_id: {"expect": "pass" | "known_failure", "result": "pass" | "fail"}}.
    """
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    pages: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def blocks(doc: str, page: int) -> list[dict[str, Any]]:
        if (doc, page) not in pages:
            path = run_root / doc / "source-blocks" / f"page-{page:03d}.json"
            pages[(doc, page)] = json.loads(path.read_text(encoding="utf-8"))["blocks"] if path.is_file() else []
        return pages[(doc, page)]

    results: dict[str, dict[str, str]] = {}
    for fact in golden["table_cells"]:
        if fact["doc"] not in docs:
            continue
        cell = _table_cell(blocks(fact["doc"], fact["page"]), fact)
        passed = cell is not None and _norm(cell.get("raw_value")) == _norm(fact["value"]) and (
            "numeric" not in fact or cell.get("numeric_value") == fact["numeric"])
        results[f"cell {fact['doc']} p{fact['page']} {fact['row']} / {fact['column']}"] = {
            "expect": fact.get("expect", "pass"), "result": "pass" if passed else "fail"}
    for fact in golden["visual_values"]:
        if fact["doc"] not in docs:
            continue
        passed = _visual_item(blocks(fact["doc"], fact["page"]), fact) is not None
        results[f"visual {fact['doc']} p{fact['page']} {fact['label']} = {fact['value']}"] = {
            "expect": fact.get("expect", "pass"), "result": "pass" if passed else "fail"}
    for check in golden["review"]:
        if check["doc"] not in docs:
            continue
        block = _anchored_block(blocks(check["doc"], check["page"]), check)
        passed = block is not None and block["validation"]["status"] == check["status"]
        results[f"review {check['doc']} p{check['page']} {check['type']} {check.get('anchor')} -> {check['status']}"] = {
            "expect": check.get("expect", "pass"), "result": "pass" if passed else "fail"}
    for check in golden["no_chart_pages"]:
        if check["doc"] not in docs:
            continue
        passed = not any(block["type"] == "chart" for block in blocks(check["doc"], check["page"]))
        results[f"no chart {check['doc']} p{check['page']}"] = {
            "expect": check.get("expect", "pass"), "result": "pass" if passed else "fail"}
    return results

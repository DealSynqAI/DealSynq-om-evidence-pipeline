from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from ocr_pipeline.ocr_disagreement import needs_second_ocr, record_disagreements
from ocr_pipeline.pipeline import _preserve_ocr_lines_in_blocks
from ocr_pipeline.region_decomposition import empty_table_artifacts, propose_regions, select_regions


def _line(number: int, text: str, box: list[float], confidence: float = 0.99) -> dict:
    return {"evidence_id": f"p001-ocr-{number:04d}", "text": text,
            "coordinates": box, "confidence": confidence}


def test_mixed_region_proposals_need_independent_ocr_and_pixels(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    page = np.full((400, 400, 3), 255, dtype=np.uint8)
    page[70:350, 10:100] = rng.integers(10, 230, size=(280, 90, 3), dtype=np.uint8)
    image = tmp_path / "page.png"
    Image.fromarray(page).save(image)
    lines = [_line(index, text, [300, 180 + 55 * index, 300, 30])
             for index, text in enumerate(("Tenant A", "TOTAL SF 111,116", "Credit BBB+",
                                           "Revenue $1.09 B", "Owner biography"), 1)]
    inspection = SimpleNamespace(regions=[], page=1, width_points=400, height_points=400)
    plan = {"blocks": [
        {"type": "photograph", "bbox": [25, 175, 250, 875]},
        {"type": "photograph", "bbox": [750, 100, 975, 875]},
        {"type": "chart", "bbox": [275, 125, 625, 900]},
    ]}
    proposals = propose_regions(image, inspection, lines, plan)
    assert any(item["kind"] == "photograph" and item["accepted"]
               and item["coordinates"][0] < 100 for item in proposals)
    assert any(item["kind"] == "photograph" and not item["accepted"]
               and item["coordinates"][0] > 700 for item in proposals)
    assert any(item["kind"] == "tenant_card" and item["accepted"] for item in proposals)
    selected = select_regions(proposals, lines, [])
    assert {item["kind"] for item in selected} == {"photograph", "tenant_card"}


def test_disagreement_keeps_both_readings_and_crop(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (1000, 1000), "white").save(image)
    line = _line(1, "Thin-Market Liqvidity", [100, 100, 400, 30], 0.63)
    block = {"block_id": "p001-text", "type": "text", "content": {"text": line["text"]},
             "provenance": {"ocr_evidence_ids": [line["evidence_id"]]},
             "validation": {"status": "passed", "warnings": [], "errors": []}}
    paddle = {"res": {"overall_ocr_res": {
        "rec_texts": ["Thin-Market Liquidity"], "rec_scores": [0.99],
        "rec_boxes": [[100, 100, 500, 130]],
    }}}
    assert needs_second_ocr({"lines": [line]})
    summary = record_disagreements(1, image, [line], paddle, [block],
                                   tmp_path / "disagreement-crops",
                                   tmp_path / "diagnostics" / "page-001.json")
    assert summary["unresolved_count"] == 1
    assert block["validation"]["status"] == "needs_review"
    record = block["ocr_disagreements"][0]
    assert [item["text"] for item in record["readings"]] == [
        "Thin-Market Liqvidity", "Thin-Market Liquidity"]
    assert (tmp_path / record["crop"]).exists()
    assert json.loads((tmp_path / "diagnostics" / "page-001.json").read_text())[
        "unresolved_count"] == 1


def test_photo_keeps_separate_owner_for_a_short_printed_label() -> None:
    line = _line(1, "Shopping", [500, 600, 75, 20])
    photo = {"kind": "photograph", "coordinates": [100, 100, 800, 800],
             "origin": "pdf_image_member", "accepted": True,
             "ocr_evidence_ids": [line["evidence_id"]]}
    text = {"type": "text", "coordinates": [490, 590, 100, 40],
            "provenance": {"ocr_evidence_ids": [line["evidence_id"]]}}
    chosen = select_regions([photo], [line], [text])
    assert len(chosen) == 1
    assert chosen[0]["ocr_evidence_ids"] == []


def test_unmatched_ocr_variant_stays_with_its_unique_physical_card(tmp_path: Path) -> None:
    line = _line(1, "Deveioper", [320, 300, 90, 20])
    card = {"block_id": "p001-decomposed-001-block", "type": "text",
            "coordinates": [250, 200, 300, 600], "content": {"text": "Developer"},
            "provenance": {"ocr_evidence_ids": []},
            "validation": {"status": "passed", "warnings": [], "errors": []}}
    blocks, stats = _preserve_ocr_lines_in_blocks(
        [card], [line], "document", "0" * 64,
        SimpleNamespace(page=1, width_points=1000, height_points=1000), tmp_path / "page.png",
    )
    assert stats["fallback_text_blocks"] == 0
    assert blocks[0]["provenance"]["ocr_evidence_ids"] == [line["evidence_id"]]
    assert blocks[0]["raw_evidence_lines"][0]["text"] == "Deveioper"


def test_table_header_glyph_conflict_is_not_silently_resolved(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (1000, 1000), "white").save(image)
    block = {"block_id": "p001-table", "type": "table", "content": {
        "columns": [{"column_id": "c001", "label": "96%(Base~)",
                     "coordinates": [100, 100, 160, 30], "evidence_ids": []}],
        "rows": [],
    }, "provenance": {"ocr_evidence_ids": []},
        "validation": {"status": "passed", "warnings": [], "errors": []}}
    paddle = {"res": {"overall_ocr_res": {
        "rec_texts": ["96% (Base)"], "rec_scores": [0.95],
        "rec_boxes": [[100, 100, 260, 130]],
    }}}
    summary = record_disagreements(1, image, [], paddle, [block],
                                   tmp_path / "disagreement-crops",
                                   tmp_path / "diagnostics" / "page-001.json")
    assert summary["unresolved_count"] == 1
    record = block["ocr_disagreements"][0]
    assert record["target"] == {"type": "table_column", "column_id": "c001"}
    assert record["status"] == "unresolved"
    assert len(record["crop_sha256"]) == 64


def test_colored_card_panel_and_overhanging_logo_extend_ocr_bounds(tmp_path: Path) -> None:
    page = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((80, 78, 156, 380), fill=(216, 215, 223))
    draw.ellipse((82, 55, 137, 110), fill="white", outline=(40, 40, 60), width=2)
    image = tmp_path / "card.png"
    page.save(image)
    lines = [_line(index, text, [220, 260 + index * 75, 130, 25])
             for index, text in enumerate(("TENANT", "TOTAL SF", "111,116 SF", "CREDIT RATING",
                                           "BBB+", "REVENUE", "$1.09 B", "Owner biography"), 1)]
    inspection = SimpleNamespace(regions=[], page=1, width_points=400, height_points=400)
    proposals = propose_regions(image, inspection, lines, None)
    cards = [item for item in proposals if item["kind"] == "tenant_card" and item["accepted"]]
    assert len(cards) == 1
    assert cards[0]["coordinates"][1] < 220
    assert cards[0]["coordinates"][0] < 220


def test_blank_grid_over_photo_is_removed_only_without_printed_cells() -> None:
    photo = [100, 100, 800, 800]
    blank = {"block_id": "blank", "type": "table", "coordinates": [0, 300, 150, 300],
             "provenance": {"ocr_evidence_ids": []}, "content": {"rows": [
                 {"label": None, "cells": [{"raw_value": None}]},
             ]}}
    populated = {**blank, "block_id": "populated", "content": {"rows": [
        {"label": "Rent", "cells": [{"raw_value": "$100"}]},
    ]}}
    assert [block["block_id"] for block in empty_table_artifacts(
        [blank, populated], [photo])] == ["blank"]

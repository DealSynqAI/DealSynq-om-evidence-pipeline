from pathlib import Path

from PIL import Image
from reportlab.pdfgen import canvas

from ocr_pipeline.models import PageInspection, Region, SourceBlock
from ocr_pipeline.pipeline import _preserve_source_observations
from ocr_pipeline.source_observations import (
    _match_ocr, collect_page_observations, is_layout_glyph, represented_in_content,
)


def _inspection():
    return PageInspection(1, 600, 800, 0, False, 0, 0, 1, 0, 0, 0, 0.5, [])


def test_native_word_matches_nearby_ocr_line():
    observation = {"source": "native_pdf_word", "text": "Rent",
                   "coordinates": [100, 100, 30, 10]}
    lines = [{"evidence_id": "p001-ocr-0001", "text": "Base Rent",
              "coordinates": [98, 98, 90, 14]}]
    assert _match_ocr(observation, lines) == ["p001-ocr-0001"]
    observation["coordinates"] = [700, 700, 30, 10]
    assert _match_ocr(observation, lines) == []


def test_native_word_ids_cannot_collide_with_native_visual_ocr_ids(tmp_path):
    pdf = tmp_path / "source.pdf"
    page = canvas.Canvas(str(pdf), pagesize=(600, 800))
    page.drawString(60, 700, "Base Rent")
    page.save()
    image = tmp_path / "page-001.png"
    Image.new("RGB", (600, 800), "white").save(image)
    observations = collect_page_observations(pdf, [1], {1: {"lines": [
        {"evidence_id": "p001-native-0001", "text": "Base Rent",
         "coordinates": [95, 110, 100, 20]},
    ]}}, tmp_path / "paddle", {1: image})[1]
    assert observations
    assert all(item["evidence_id"].startswith("p001-pdfword-") for item in observations)
    assert all(item["evidence_id"] != "p001-native-0001" for item in observations)


def test_unmatched_source_text_survives_with_source_and_owner():
    region = Region("p001-r001", 1, "normal_text", [50, 50, 500, 200], 1, "native", 0.9)
    block = SourceBlock(
        document_id="doc", type="text", page=1, block_id="p001-r001-text",
        content={"text": "Known", "evidence_text": {"selected": "native", "native": "Known",
                                              "ocr": None, "token_agreement": None}},
        coordinates=region.coordinates, extraction_method=["native"], confidence=0.9,
        validation_status="passed",
        provenance={"source_sha256": "0" * 64, "region_id": region.region_id,
                    "classification_method": "native", "reading_order": 1,
                    "source_bbox_points": None, "ocr_evidence_ids": [],
                    "rendered_page": "page-images/page-001.png"},
    ).as_dict()
    observations = [
        {"evidence_id": "p001-pdfword-0002", "source": "native_pdf_word", "text": "Known",
         "confidence": 1.0, "coordinates": [110, 110, 40, 10], "matched_ocr_evidence_ids": []},
        {"evidence_id": "p001-native-0001", "source": "native_pdf_word", "text": "Missing",
         "confidence": 1.0, "coordinates": [100, 100, 40, 10], "matched_ocr_evidence_ids": []},
        {"evidence_id": "p001-pdfword-0003", "source": "native_pdf_word", "text": "|",
         "confidence": 1.0, "coordinates": [150, 100, 5, 10], "matched_ocr_evidence_ids": []},
        {"evidence_id": "p001-paddle-text-0001", "source": "paddle_layout_text", "text": "Outside",
         "confidence": 0.7, "coordinates": [700, 700, 80, 20], "matched_ocr_evidence_ids": []},
    ]
    blocks = [block]
    stats = _preserve_source_observations(blocks, observations, "doc", "0" * 64,
                                          _inspection(), Path("page-001.png"))
    assert stats["raw_attached"] == 1
    assert stats["content_supported"] == 1
    assert stats["verbatim_glyphs"] == 1
    assert stats["fallback_text_blocks"] == 1
    assert block["raw_evidence_lines"][0]["source"] == "native_pdf_word"
    assert observations[0]["disposition"] == "content_supported"
    assert observations[1]["owner_block_id"] == block["block_id"]
    assert observations[2]["disposition"] == "verbatim_glyph"
    assert block["raw_evidence_lines"][1]["text"] == "|"
    assert blocks[1]["content"]["text"] == "Outside"
    assert observations[3]["owner_block_id"] == blocks[1]["block_id"]


def test_content_support_requires_printed_text_not_structural_metadata():
    content = {"columns": [{"column_id": "c001", "label": "Annual Rent"}],
               "rows": [{"row_id": "r001", "label": "Office", "cells": []}]}
    assert represented_in_content({"source": "native_pdf_word", "text": "Rent"}, content)
    assert not represented_in_content({"source": "native_pdf_word", "text": "c001"}, content)
    assert not represented_in_content({"source": "native_pdf_word", "text": "Renter"}, content)
    assert represented_in_content(
        {"source": "paddle_layout_text", "text": "Annual Rent"}, content,
    )
    assert not represented_in_content(
        {"source": "paddle_layout_text", "text": "Annual Rent $500"}, content,
    )
    assert is_layout_glyph({"source": "native_pdf_word", "text": "&"})
    assert not is_layout_glyph({"source": "native_pdf_word", "text": "$"})
    assert not is_layout_glyph({"source": "native_pdf_word", "text": "-"})

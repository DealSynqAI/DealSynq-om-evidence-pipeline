from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ocr_pipeline.models import PageInspection, Region, SourceBlock
from ocr_pipeline.pipeline import (
    _claim_single_value_cards, _classify_visual, _linked_table_scalar_count,
    _numeric_content_coverage, _preserve_ocr_lines_in_blocks, _table_blocks,
)
from ocr_pipeline.vision_first import apply_plan_hints


def test_photo_collage_grid_requires_numeric_cell_density() -> None:
    region = Region("r1", 1, "visual", [0, 0, 1000, 1000], 1, "test", 0.7)
    lines = [{"text": f"Location narrative {index}"} for index in range(2)]
    lines += [{"text": "Investment thesis"}] * 18
    kind, _, _ = _classify_visual(region, lines, {
        "horizontal_lines": 40, "vertical_lines": 18, "table_grid_confidence": 1.0,
    })
    assert kind != "table"


def test_numeric_grid_remains_table() -> None:
    region = Region("r1", 1, "visual", [0, 0, 1000, 1000], 1, "test", 0.7)
    lines = [{"text": "Address"}, {"text": "Sale Price"}] + [{"text": str(index)} for index in range(10)]
    kind, _, _ = _classify_visual(region, lines, {
        "horizontal_lines": 10, "vertical_lines": 4, "table_grid_confidence": 0.9,
    })
    assert kind == "table"


def test_photo_with_prose_routes_to_text_without_numeric_axis_ticks() -> None:
    region = Region("r1", 1, "visual", [0, 0, 1000, 1000], 1, "test", 0.7)
    lines = [{"text": "The office property has a strong tenant roster and convenient access"}] * 12
    lines += [{"text": "2026 revenue estimate"}, {"text": "5 year term"}]
    kind, _, _ = _classify_visual(region, lines, {
        "point_candidates": 42, "horizontal_axis_candidates": 1,
        "vertical_axis_candidates": 1,
    })
    assert kind == "normal_text"


def test_spanning_title_promotes_printed_header_without_inventing_labels() -> None:
    region = Region("r1", 1, "table", [0, 0, 1000, 1000], 1, "test", 0.9, metadata={
        "rows": [["Comparable Sales", None, None], ["Address", "Sale Price", "Date"],
                 ["123 Main St", "$100,000", "2026-01-01"]],
    })
    block = _table_blocks("doc", "0" * 64, region, [], Path("page.png"))[0]
    assert block.content["title"] == "Comparable Sales"
    assert [column["label"] for column in block.content["columns"]] == ["Address", "Sale Price", "Date"]
    assert block.content["rows"][0]["label"] == "123 Main St"


def test_sparse_ocr_table_row_uses_printed_column_positions() -> None:
    region = Region("r1", 1, "table", [0, 0, 1000, 1000], 1, "test", 0.7)
    def line(text: str, x: int, y: int) -> dict:
        return {"evidence_id": f"ocr-{x}-{y}", "text": text, "coordinates": [x, y, 25, 10]}
    lines = [line("Comparable Sales", 250, 100)]
    lines += [line(text, x, 120) for text, x in [
        ("Address", 100), ("Unit", 200), ("Floor", 300),
        ("List Price", 400), ("Sale Price", 500),
    ]]
    lines += [line(text, x, 140) for text, x in [
        ("25 Main St", 105), ("3.5", 305), ("$3,904,688", 505),
    ]]
    block = _table_blocks("doc", "0" * 64, region, lines, Path("page.png"))[0]
    assert block.content["title"] == "Comparable Sales"
    assert [column["label"] for column in block.content["columns"]] == [
        "Address", "Unit", "Floor", "List Price", "Sale Price",
    ]
    assert block.content["rows"][0]["label"] == "25 Main St"
    assert [cell["raw_value"] for cell in block.content["rows"][0]["cells"]] == [
        None, "3.5", None, "$3,904,688",
    ]


def test_unowned_column_is_explicit_review_and_schema_valid() -> None:
    region = Region("r1", 1, "table", [0, 0, 1000, 1000], 1, "test", 0.9, metadata={
        "rows": [["Address", None], ["123 Main St", "$100,000"]],
    })
    block = _table_blocks("doc", "0" * 64, region, [], Path("page.png"))[0]
    assert block.validation_status == "needs_review"
    assert block.content["columns"][1]["label"] is None
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/unified-source-block.schema.json").read_text())
    assert not list(Draft202012Validator(schema).iter_errors(block.as_dict()))


def test_numeric_content_coverage_requires_nearby_serialized_value() -> None:
    blocks = [
        {"coordinates": [0, 0, 400, 400], "content": {"text": "Profit $5,313,010"}},
        {"coordinates": [600, 0, 400, 400], "content": {"rows": [{"raw_value": "$43,250,000"}]}},
    ]
    lines = [
        {"evidence_id": "left", "text": "$5,313,010", "confidence": 0.99,
         "coordinates": [100, 100, 50, 20]},
        {"evidence_id": "right-missing", "text": "$5,313,010", "confidence": 0.99,
         "coordinates": [700, 100, 50, 20]},
        {"evidence_id": "right-covered", "text": "$43,250,000", "confidence": 0.99,
         "coordinates": [700, 150, 50, 20]},
    ]
    coverage = _numeric_content_coverage(blocks, lines)
    assert coverage["represented"] == 2
    assert [item["evidence_id"] for item in coverage["unrepresented"]] == ["right-missing"]


def test_native_table_merged_label_and_amount_is_not_a_numeric_fact() -> None:
    region = Region("r1", 1, "table", [0, 0, 1000, 1000], 1, "native PDF", 0.9,
                    metadata={"rows": [["Source", "Amount"],
                                       ["Senior Debt", "$43,250,000Purchase Price"],
                                       ["Mezzanine Debt", "$3,000,000"]]})
    block = _table_blocks("doc", "0" * 64, region, [], Path("page.png"))[0]
    first, second = block.content["rows"]
    assert first["cells"][0]["raw_value"] == "$43,250,000Purchase Price"
    assert first["cells"][0]["numeric_value"] is None
    assert second["cells"][0]["numeric_value"] == 3000000
    assert any("non-scalar numeric text" in warning for warning in block.warnings)


def test_table_link_quality_excludes_unowned_numeric_cells() -> None:
    blocks = [{"type": "table", "content": {"rows": [
        {"label": "Revenue", "cells": [{"numeric_value": 100}, {"numeric_value": None}]},
        {"label": None, "cells": [{"numeric_value": 200}]},
        {"label": "77.09%", "cells": [{"numeric_value": 300}]},
    ]}}]
    assert _linked_table_scalar_count(blocks) == 1


def test_split_currency_signs_rejoin_adjacent_numeric_cells() -> None:
    region = Region("r1", 1, "table", [0, 0, 1000, 1000], 1, "native PDF", 0.9,
                    metadata={"rows": [["Item", "Sign", "Amount", "Sign", "Per Unit"],
                                       ["Senior Debt", "$", "44,000,000", "$", "382,609"],
                                       ["Construction", "$", "41,671,485 $", "362,361", ""]]})
    block = _table_blocks("doc", "0" * 64, region, [], Path("page.png"))[0]
    first, second = block.content["rows"]
    assert [cell["raw_value"] for cell in first["cells"]] == [
        None, "$44,000,000", None, "$382,609",
    ]
    assert first["cells"][1]["numeric_value"] == 44000000
    assert [cell["raw_value"] for cell in second["cells"]][:3] == [
        None, "$41,671,485", "$362,361",
    ]


def test_single_metric_vision_card_can_claim_its_full_box() -> None:
    region = Region("r1", 1, "table", [100, 100, 200, 100], 1, "test", 0.7)
    inspection = PageInspection(
        page=1, width_points=100, height_points=100, rotation=0,
        native_text_available=False, native_text_coverage=0, native_text_quality=0,
        image_coverage=1, vector_line_count=0, possible_table_regions=1,
        possible_visual_regions=0, routing_confidence=0.5, regions=[region],
    )
    apply_plan_hints(inspection, {"blocks": [{
        "type": "kpi_panel", "bbox": [90, 90, 310, 310], "title": "IRR",
        "chart_type": "none", "items": [{"label": "IRR", "value": "24-27%"}],
    }]})
    assert region.metadata["vision_plan"]["type"] == "kpi_panel"
    assert region.metadata["ownership_coordinates"] == [90, 90, 220, 220]


def test_single_value_card_claims_printed_caption_and_keeps_range() -> None:
    card = Region("card", 1, "table", [50, 300, 250, 120], 1, "test", 0.8,
                  metadata={"rows": [[None, "24-27%"], [None, None]]})
    caption_region = Region("caption", 1, "normal_text", [50, 420, 250, 70], 2, "test", 0.9)
    value = {"evidence_id": "v", "text": "24-27%", "coordinates": [80, 340, 120, 50]}
    caption = {"evidence_id": "l", "text": "IRR (5 year hold)", "coordinates": [80, 430, 180, 35]}
    owned = {"card": [value], "caption": [caption]}
    absorbed = _claim_single_value_cards([card, caption_region], [value, caption], owned)
    assert absorbed == {"caption"}
    assert card.metadata["single_card_binding"]["label"] == "IRR (5 year hold)"
    assert card.metadata["single_card_binding"]["numeric_value"] is None
    assert {line["evidence_id"] for line in owned["card"]} == {"v", "l"}


def test_kpi_hints_keep_separate_card_ownership_boxes() -> None:
    left = Region("left", 1, "table", [60, 300, 180, 120], 1, "test", 0.7)
    right = Region("right", 1, "table", [440, 300, 180, 120], 2, "test", 0.7)
    inspection = PageInspection(
        page=1, width_points=100, height_points=100, rotation=0,
        native_text_available=False, native_text_coverage=0, native_text_quality=0,
        image_coverage=1, vector_line_count=0, possible_table_regions=2,
        possible_visual_regions=0, routing_confidence=0.5, regions=[left, right],
    )
    apply_plan_hints(inspection, {"blocks": [
        {"type": "kpi_panel", "bbox": [50, 290, 300, 490], "title": "IRR", "items": [{"label": "IRR", "value": "24%"}]},
        {"type": "kpi_panel", "bbox": [430, 290, 700, 490], "title": "Preferred", "items": [{"label": "Preferred", "value": "8%"}]},
    ]})
    assert left.metadata["ownership_coordinates"] == [50, 290, 250, 200]
    assert right.metadata["ownership_coordinates"] == [430, 290, 270, 200]


def test_raw_capture_preserves_owned_value_omitted_from_structured_content() -> None:
    source_hash = "0" * 64
    region = Region("p001-r001", 1, "normal_text", [0, 0, 500, 500], 1, "test", 0.9)
    inspection = PageInspection(
        page=1, width_points=600, height_points=800, rotation=0,
        native_text_available=False, native_text_coverage=0, native_text_quality=0,
        image_coverage=1, vector_line_count=0, possible_table_regions=0,
        possible_visual_regions=0, routing_confidence=0.5, regions=[region],
    )
    line = {"evidence_id": "p001-ocr-0001", "text": "$4,167,949",
            "confidence": 0.96, "coordinates": [80, 100, 120, 15]}
    block = SourceBlock(
        document_id="doc", type="text", page=1, block_id="p001-r001-block",
        content={"text": "Returns", "evidence_text": {
            "selected": "ocr", "native": None, "ocr": "Returns", "token_agreement": None,
        }},
        coordinates=region.coordinates, extraction_method=["RapidOCR"], confidence=0.9,
        validation_status="passed", provenance={
            "source_sha256": source_hash, "region_id": region.region_id,
            "classification_method": "test", "reading_order": 1,
            "source_bbox_points": None, "ocr_evidence_ids": [line["evidence_id"]],
            "rendered_page": "page-images/page-001.png",
        },
    ).as_dict()
    blocks, stats = _preserve_ocr_lines_in_blocks(
        [block], [line], "doc", source_hash, inspection, Path("page-001.png"),
    )
    assert len(blocks) == 1
    assert blocks[0]["raw_evidence_lines"][0]["text"] == "$4,167,949"
    assert blocks[0]["validation"]["status"] == "needs_review"
    assert stats["raw_lines_attached"] == 1
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/unified-source-block.schema.json").read_text())
    assert not list(Draft202012Validator(schema).iter_errors(blocks[0]))


def test_raw_capture_creates_evidence_backed_text_block_for_unowned_line() -> None:
    source_hash = "0" * 64
    inspection = PageInspection(
        page=1, width_points=600, height_points=800, rotation=0,
        native_text_available=False, native_text_coverage=0, native_text_quality=0,
        image_coverage=1, vector_line_count=0, possible_table_regions=0,
        possible_visual_regions=0, routing_confidence=0.5, regions=[],
    )
    line = {"evidence_id": "p001-ocr-0001", "text": "Year 1 $313,197",
            "confidence": 0.91, "coordinates": [80, 100, 120, 15]}
    blocks, stats = _preserve_ocr_lines_in_blocks(
        [], [line], "doc", source_hash, inspection, Path("page-001.png"),
    )
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["content"]["text"] == line["text"]
    assert blocks[0]["provenance"]["ocr_evidence_ids"] == [line["evidence_id"]]
    assert stats["fallback_text_blocks"] == 1
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/unified-source-block.schema.json").read_text())
    assert not list(Draft202012Validator(schema).iter_errors(blocks[0]))


def test_raw_capture_groups_adjacent_lines_without_crossing_columns() -> None:
    source_hash = "0" * 64
    inspection = PageInspection(
        page=1, width_points=600, height_points=800, rotation=0,
        native_text_available=False, native_text_coverage=0, native_text_quality=0,
        image_coverage=1, vector_line_count=0, possible_table_regions=0,
        possible_visual_regions=0, routing_confidence=0.5, regions=[],
    )
    lines = [
        {"evidence_id": "p001-ocr-0001", "text": "123M+", "confidence": 0.99,
         "coordinates": [750, 200, 80, 20]},
        {"evidence_id": "p001-ocr-0002", "text": "Projects under", "confidence": 0.99,
         "coordinates": [730, 230, 120, 20]},
        {"evidence_id": "p001-ocr-0003", "text": "management", "confidence": 0.99,
         "coordinates": [735, 260, 110, 20]},
        {"evidence_id": "p001-ocr-0004", "text": "Separate column", "confidence": 0.99,
         "coordinates": [100, 230, 150, 20]},
        {"evidence_id": "p001-native-0001", "text": "under", "confidence": 1.0,
         "evidence_source": "native_pdf_positioned_word",
         "coordinates": [815, 232, 34, 16]},
    ]
    blocks, stats = _preserve_ocr_lines_in_blocks(
        [], lines, "doc", source_hash, inspection, Path("page-001.png"),
    )
    assert len(blocks) == 2
    assert stats["fallback_text_blocks"] == 2
    grouped = next(block for block in blocks if "123M+" in block["content"]["text"])
    assert grouped["content"]["text"] == "123M+ Projects under management"
    assert set(grouped["provenance"]["ocr_evidence_ids"]) == {
        "p001-ocr-0001", "p001-ocr-0002", "p001-ocr-0003", "p001-native-0001",
    }
    assert {line["evidence_id"] for line in grouped["raw_evidence_lines"]} == {
        "p001-ocr-0001", "p001-ocr-0002", "p001-ocr-0003", "p001-native-0001",
    }


def test_raw_capture_propagates_review_to_parent_group() -> None:
    source_hash = "0" * 64
    inspection = PageInspection(
        page=1, width_points=600, height_points=800, rotation=0,
        native_text_available=False, native_text_coverage=0, native_text_quality=0,
        image_coverage=1, vector_line_count=0, possible_table_regions=0,
        possible_visual_regions=0, routing_confidence=0.5, regions=[],
    )
    line = {"evidence_id": "p001-ocr-0001", "text": "$500,000",
            "confidence": 0.9, "coordinates": [80, 100, 100, 20]}
    provenance = {"source_sha256": source_hash, "region_id": "p001-r001",
                  "classification_method": "test", "reading_order": 1,
                  "source_bbox_points": None, "ocr_evidence_ids": [line["evidence_id"]],
                  "rendered_page": "page-images/page-001.png"}
    child = SourceBlock(
        document_id="doc", type="text", page=1, block_id="child",
        content={"text": "Purchase Price", "evidence_text": {
            "selected": "ocr", "native": None, "ocr": "Purchase Price", "token_agreement": None,
        }}, coordinates=[0, 0, 500, 500], extraction_method=["RapidOCR"],
        confidence=0.9, validation_status="passed", provenance=provenance,
        parent_block_id="group", hierarchy_depth=1,
    ).as_dict()
    group = SourceBlock(
        document_id="doc", type="group", page=1, block_id="group",
        content={"role": "section", "child_block_ids": ["child"]},
        coordinates=[0, 0, 500, 500], extraction_method=["Python grouping"],
        confidence=0.9, validation_status="passed",
        provenance={**provenance, "ocr_evidence_ids": []}, child_block_ids=["child"],
    ).as_dict()
    blocks, _ = _preserve_ocr_lines_in_blocks(
        [group, child], [line], "doc", source_hash, inspection, Path("page-001.png"),
    )
    assert blocks[0]["validation"]["status"] == "needs_review"
    assert blocks[1]["validation"]["status"] == "needs_review"

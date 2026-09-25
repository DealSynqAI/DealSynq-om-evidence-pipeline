from __future__ import annotations

from pathlib import Path

from PIL import Image

from ocr_pipeline.evidence_capture import _immaterial_evidence
from ocr_pipeline.ocr_disagreement import _same_reading, record_disagreements
from ocr_pipeline.table_blocks import (
    _clearly_nearest, _ground_table_cells_from_ocr, _table_review_reasons, _value_placement_errors,
)


def test_readings_agree_despite_spacing_punctuation_and_line_extent() -> None:
    assert _same_reading("972,330", "$972,330$")
    assert _same_reading("1.4%", "1.4% /")
    assert _same_reading("tenants' preferences", "tenants preferences")
    assert _same_reading("CENTER", "CENTER Ta")
    # Signs, separators, digits, and letters still count.
    assert not _same_reading("-$21,903", "$21,903")
    assert not _same_reading("(1,250)", "1,250")
    assert not _same_reading("$458,056", "$458.056")
    assert not _same_reading("$45 million", "$45 5million")
    assert not _same_reading("Deveioper", "Developer")


def test_a_low_confidence_line_confirmed_by_paddle_is_not_a_disagreement(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (1000, 1000), "white").save(image)
    line = {"evidence_id": "p001-ocr-0001", "text": "Development Pipeline Overview",
            "coordinates": [100, 100, 300, 20], "confidence": 0.62}
    block = {"block_id": "b", "type": "text", "content": {"text": line["text"]},
             "provenance": {"ocr_evidence_ids": [line["evidence_id"]]},
             "validation": {"status": "passed", "warnings": [], "errors": []}}
    paddle = {"res": {"overall_ocr_res": {"rec_texts": ["Development Pipeline Overview."], "rec_scores": [0.9],
                                          "rec_boxes": [[100, 100, 400, 120]]}}}
    summary = record_disagreements(1, image, [line], paddle, [block], tmp_path / "crops", tmp_path / "receipt.json")
    assert summary["unresolved_count"] == 0 and block["validation"]["status"] == "passed"


def _line(text: str, box: list[float]) -> dict:
    return {"evidence_id": "p001-ocr-0009", "text": text, "coordinates": box}


def test_leftover_text_is_material_only_when_it_could_change_content() -> None:
    table = {"type": "table", "content": {"rows": [{"label": "Revenue", "cells": [{"raw_value": "26,512"}]}]}}
    number = _line("26,512", [200, 100, 60, 20])
    assert _immaterial_evidence("_", table, [number], [185, 100, 10, 20])      # a rule, not a sign
    assert not _immaterial_evidence("-", table, [number], [185, 100, 10, 20])  # a minus beside a number
    assert _immaterial_evidence("-", table, [number], [600, 400, 10, 20])      # a dash far from numbers
    assert _immaterial_evidence("Annual rent roll", table, [number], None)     # words, not a value
    assert not _immaterial_evidence("$1,641,918", table, [number], None)       # a number the table lost
    text = {"type": "text", "content": {"text": "TRANSPARENT and accessible"}}
    assert _immaterial_evidence("T", text, [], None)
    assert _immaterial_evidence("and accessible", text, [], None)
    assert not _immaterial_evidence("quarterly reporting", text, [], None)
    chart = {"type": "chart", "content": {"expected_observation_count": 2, "observations": [
        {"validation_status": "passed"}, {"validation_status": "passed"}]}}
    assert _immaterial_evidence("$35M", chart, [], None)  # an axis tick once every bar is valued


def test_reading_notes_do_not_hold_a_grounded_table_back() -> None:
    rows = [{"label": "Rent", "cells": [{"raw_value": "$10", "grounding_status": "ocr_row_and_column"}]}]
    notes = ["table reconstructed from PaddleOCR structure; independently review cell values",
             "3 table cells contain non-scalar numeric text; raw text retained without a numeric value"]
    assert _table_review_reasons(notes, [], rows) == []
    rows[0]["cells"][0]["grounding_status"] = "unverified"
    assert _table_review_reasons(notes, [], rows) == ["table cells lack value-and-owner evidence"]
    assert _table_review_reasons(["PaddleOCR and native PDF table cells or dimensions disagree; native cells retained"],
                                 [], [{"label": "Rent", "cells": [{"raw_value": "$10", "grounding_status": "native_positioned"}]}]) == []


def test_misplaced_values_are_structure_errors() -> None:
    def rows(*pairs):
        return [{"label": label, "cells": [{"raw_value": value} for value in values]} for label, values in pairs]

    assert _value_placement_errors(rows(("Senior Loan", ["$32,214,751 $13,211,787"]))) != []
    assert _value_placement_errors(rows(("Effective Gross Income $2,442,407 $2,396,568", [None]))) != []
    assert _value_placement_errors(rows(("Draw", ["$1"]), ("$1,641,918", ["$2"]), ("Reserve", ["$3"]))) != []
    assert _value_placement_errors(rows(("7.50%", ["14.2%/ 1.5x"]), ("7.25%", ["15.6%/1.5x"]))) == []
    assert _value_placement_errors(rows(("Range", ["$4515 - $5208"]), ("Price", ["$21,000,000 ($125,000/unit)"]))) == []


def test_nearest_row_wins_only_when_clearly_nearest() -> None:
    near = {"coordinates": [0, 780, 50, 18]}
    far = {"coordinates": [0, 800, 50, 18]}
    assert _clearly_nearest([near, far], 789) is near
    assert _clearly_nearest([near, {"coordinates": [0, 781, 50, 18]}], 789) is None


def test_key_and_value_printed_on_one_line_ground_the_row() -> None:
    line = {"evidence_id": "p001-ocr-0003", "text": "Roof Type: Asphalt Shingles", "coordinates": [100, 200, 300, 20]}
    block = {"block_id": "t", "type": "table", "coordinates": [90, 150, 400, 200],
             "provenance": {"ocr_evidence_ids": [line["evidence_id"]]},
             "content": {"columns": [{"column_id": "c001", "label": "Field"}, {"column_id": "c002", "label": "Value"}],
                         "rows": [{"row_id": "r001", "label": "Roof Type", "label_coordinates": None, "cells": [
                             {"column_id": "c002", "raw_value": "Asphalt Shingles", "coordinates": None,
                              "evidence_ids": [], "grounding_status": "unverified"}]}]},
             "validation": {"status": "needs_review", "errors": [],
                            "warnings": ["1 table cells lack value-and-owner evidence; treat their values as candidates"]}}
    assert _ground_table_cells_from_ocr([block], [line])["newly_grounded"] == 1
    cell = block["content"]["rows"][0]["cells"][0]
    assert cell["grounding_status"] == "ocr_same_line" and cell["evidence_ids"] == [line["evidence_id"]]
    assert block["validation"]["status"] == "passed"

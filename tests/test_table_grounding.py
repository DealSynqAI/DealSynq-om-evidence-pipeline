from ocr_pipeline.table_blocks import _ground_table_cells_from_ocr


def _line(index, text, x, y):
    return {"evidence_id": f"p001-ocr-{index:04d}", "text": text,
            "coordinates": [x, y, 55, 10], "confidence": 0.95}


def _table():
    return {"type": "table", "block_id": "p001-r001-table",
            "coordinates": [0, 0, 500, 400],
            "provenance": {"ocr_evidence_ids": [f"p001-ocr-{index:04d}" for index in range(1, 5)]},
            "validation": {"status": "needs_review", "warnings": [
                "2 table cells lack value-and-owner evidence; treat their values as candidates"]},
            "content": {"columns": [{"column_id": "c001", "label": "Field", "evidence_ids": [], "coordinates": None},
                                    {"column_id": "c002", "label": "Value", "evidence_ids": [], "coordinates": None}],
                        "rows": [
                            {"label": "Rent", "label_coordinates": None, "cells": [{"column_id": "c002", "raw_value": "$100",
                                                          "coordinates": None, "evidence_ids": [],
                                                          "grounding_status": "unverified"}]},
                            {"label": "NOI", "cells": [{"column_id": "c002", "raw_value": "$200",
                                                         "coordinates": None, "evidence_ids": [],
                                                         "grounding_status": "unverified"}]},
                        ]}}


def test_unique_aligned_ocr_pairs_ground_table_cells():
    block = _table()
    lines = [_line(1, "Rent", 20, 100), _line(2, "$100", 180, 104),
             _line(3, "NOI", 20, 160), _line(4, "$200", 180, 165)]
    result = _ground_table_cells_from_ocr([block], lines)
    assert result == {"newly_grounded": 2, "still_unverified": 0}
    assert block["content"]["rows"][0]["label_evidence_ids"] == ["p001-ocr-0001"]
    assert block["content"]["rows"][0]["label_coordinates"] == lines[0]["coordinates"]
    assert block["content"]["rows"][0]["cells"][0]["evidence_ids"] == ["p001-ocr-0002"]
    assert block["content"]["rows"][0]["cells"][0]["grounding_status"] == "ocr_value_and_row"
    assert not any("lack value-and-owner" in warning for warning in block["validation"]["warnings"])


def test_distant_value_does_not_ground_to_wrong_row():
    block = _table()
    lines = [_line(1, "Rent", 20, 100), _line(2, "$100", 180, 250),
             _line(3, "NOI", 20, 160), _line(4, "$200", 180, 165)]
    result = _ground_table_cells_from_ocr([block], lines)
    assert result == {"newly_grounded": 1, "still_unverified": 1}
    assert block["content"]["rows"][0]["cells"][0]["grounding_status"] == "unverified"

from __future__ import annotations

from pathlib import Path

import pytest

from ocr_pipeline.independent_reconcile import (
    accept_ocr_table_candidate, accept_registered_map_candidate,
    accept_verified_chart_candidate, has_missing_table_proposal, reconcile_page,
)
from ocr_pipeline.pipeline import run_pipeline


def _chart() -> dict:
    return {
        "type": "chart", "block_id": "p1-chart", "coordinates": [100, 100, 500, 400],
        "content": {"observations": [{
            "item_id": "o1", "raw_value": "$100 million", "category": "million",
            "value_evidence_id": "ocr-value", "label_evidence_id": "ocr-wrong",
            "label_coordinates": [300, 200, 40, 20], "grounding_method": "OCR proximity",
            "validation_status": "passed",
        }]},
        "provenance": {"ocr_evidence_ids": ["ocr-label", "ocr-value", "ocr-wrong"]},
        "validation": {"status": "passed", "warnings": []},
    }


def test_unique_printed_pair_corrects_chart_label_but_stays_in_review() -> None:
    chart = _chart()
    proposal = {"blocks": [{
        "type": "chart", "title": "Portfolio", "bbox": [100, 100, 600, 500],
        "items": [{"label": "Example Sponsor", "value": "$100 million"}],
    }]}
    lines = [
        {"evidence_id": "ocr-label", "text": "Example Sponsor", "coordinates": [300, 200, 150, 20]},
        {"evidence_id": "ocr-value", "text": "$100 million", "coordinates": [320, 235, 100, 20]},
    ]
    result = reconcile_page(1, proposal, [chart], lines)
    assert result["counts"] == {"label_corrected_from_printed_pair": 1}
    assert chart["content"]["observations"][0]["category"] == "Example Sponsor"
    assert chart["content"]["observations"][0]["label_evidence_id"] == "ocr-label"
    assert chart["content"]["observations"][0]["validation_status"] == "needs_review"
    assert chart["validation"]["status"] == "needs_review"


def test_repeated_printed_value_cannot_correct_label() -> None:
    chart = _chart()
    proposal = {"blocks": [{
        "type": "chart", "title": "Portfolio", "bbox": [100, 100, 600, 500],
        "items": [{"label": "Example Sponsor", "value": "$100 million"}],
    }]}
    lines = [
        {"evidence_id": "ocr-label", "text": "Example Sponsor", "coordinates": [300, 200, 150, 20]},
        {"evidence_id": "ocr-value", "text": "$100 million", "coordinates": [320, 235, 100, 20]},
        {"evidence_id": "ocr-other", "text": "$100 million", "coordinates": [500, 330, 100, 20]},
    ]
    result = reconcile_page(1, proposal, [chart], lines)
    assert result["counts"] == {"ownership_disagreement_needs_review": 1}
    assert chart["content"]["observations"][0]["category"] == "million"


def test_stacked_printed_label_keeps_both_evidence_ids() -> None:
    chart = _chart()
    proposal = {"blocks": [{
        "type": "chart", "title": "Portfolio", "bbox": [100, 100, 600, 500],
        "items": [{"label": "Founders & Employees", "value": "$100 million"}],
    }]}
    lines = [
        {"evidence_id": "p001-ocr-label", "text": "Founders", "coordinates": [300, 180, 100, 20]},
        {"evidence_id": "p001-ocr-second", "text": "Employees", "coordinates": [300, 205, 110, 20]},
        {"evidence_id": "ocr-value", "text": "$100 million", "coordinates": [320, 240, 100, 20]},
    ]
    chart["provenance"]["ocr_evidence_ids"].extend(["p001-ocr-label", "p001-ocr-second"])
    result = reconcile_page(1, proposal, [chart], lines)
    assert result["counts"] == {"label_corrected_from_printed_pair": 1}
    observation = chart["content"]["observations"][0]
    assert observation["category"] == "Founders & Employees"
    assert observation["label_evidence_ids"] == ["p001-ocr-label", "p001-ocr-second"]


def test_printed_percentage_below_confirmed_amount_gets_same_owner() -> None:
    chart = _chart()
    chart["content"]["chart_type"] = "pie"
    chart["content"]["observations"].append({
        "item_id": "o2", "raw_value": "68%", "category": "Creek",
        "value_evidence_id": "p001-ocr-percent", "label_evidence_id": "ocr-wrong",
        "label_coordinates": [300, 200, 40, 20], "grounding_method": "OCR proximity",
        "validation_status": "passed",
    })
    chart["provenance"]["ocr_evidence_ids"].append("p001-ocr-percent")
    proposal = {"blocks": [{
        "type": "chart", "title": "Portfolio", "bbox": [100, 100, 600, 500],
        "items": [{"label": "Example Sponsor", "value": "$100 million"}],
    }]}
    lines = [
        {"evidence_id": "ocr-label", "text": "Example Sponsor", "coordinates": [300, 200, 150, 20]},
        {"evidence_id": "ocr-value", "text": "$100 million", "coordinates": [320, 235, 100, 20]},
        {"evidence_id": "p001-ocr-percent", "text": "68%", "coordinates": [340, 260, 50, 20]},
    ]
    result = reconcile_page(1, proposal, [chart], lines)
    assert result["counts"]["companion_percentage_label_corrected_from_printed_stack"] == 1
    assert chart["content"]["observations"][1]["category"] == "Example Sponsor"


def test_native_table_cell_is_retained_when_vision_disagrees() -> None:
    table = {
        "type": "table", "block_id": "table", "coordinates": [0, 0, 1000, 1000],
        "content": {
            "columns": [{"column_id": "c001", "label": "Address"}, {"column_id": "c002", "label": "Debt"}],
            "rows": [{"label": "Main St", "cells": [{"column_id": "c002", "raw_value": "$200"}]}],
        },
        "extraction_method": ["native PDF table parser"],
        "validation": {"status": "passed"},
    }
    proposal = {"blocks": [{
        "type": "table", "bbox": [0, 0, 1000, 1000],
        "rows": [{"label": "Main St", "cells": [{"column": "Debt", "value": "$100"}]}],
    }]}
    result = reconcile_page(1, proposal, [table], [])
    assert result["counts"] == {"native_cell_retained_over_vision_disagreement": 1}
    assert table["content"]["rows"][0]["cells"][0]["raw_value"] == "$200"
    assert table["validation"]["status"] == "passed"


def test_pipeline_requires_both_independent_branches() -> None:
    with pytest.raises(ValueError, match="mandatory"):
        run_pipeline(Path("input.pdf"), Path("unused"), qwen_endpoint=None)
    with pytest.raises(ValueError, match="mandatory"):
        run_pipeline(Path("input.pdf"), Path("unused"), qwen_endpoint="http://localhost", skip_ocr=True)


def test_map_route_requires_registered_geometry() -> None:
    proposal = {"blocks": [{"type": "map", "items": [{}, {}]}]}
    baseline = [{"type": "chart"}]
    candidate = [{"type": "map", "content": {
        "registration": {"status": "accepted", "silhouette_iou": 0.97},
        "bindings": [{}] * 10,
    }}]
    assert accept_registered_map_candidate(proposal, baseline, candidate, [])
    candidate[0]["content"]["registration"]["silhouette_iou"] = 0.50
    assert not accept_registered_map_candidate(proposal, baseline, candidate, [])
    candidate[0]["content"]["registration"]["silhouette_iou"] = 0.97
    assert not accept_registered_map_candidate(proposal, baseline, candidate, ["bad page"])


def test_chart_route_requires_verified_pie_marks_and_reconciled_total() -> None:
    proposal = {"blocks": [{"type": "chart", "items": [{}, {}, {}]}]}
    baseline = [{"type": "unclassified_visual"}]
    candidate = [{"type": "chart", "content": {
        "chart_type": "pie", "slice_geometry_status": "verified",
        "percentage_total_reconciles": True, "observations": [{}] * 10,
    }}]
    assert accept_verified_chart_candidate(proposal, baseline, candidate, [])
    candidate[0]["content"]["slice_geometry_status"] = "unresolved"
    assert not accept_verified_chart_candidate(proposal, baseline, candidate, [])


def test_missed_table_route_requires_ocr_linked_rows_and_no_coverage_loss() -> None:
    proposal = {"blocks": [{"type": "table", "bbox": [100, 100, 500, 400],
                            "rows": [{"label": "A"}, {"label": "B"}]}]}
    baseline = [{"type": "chart", "coordinates": [100, 100, 400, 300],
                 "content": {"observations": []}}]
    table = {"type": "table", "coordinates": [100, 100, 400, 300], "content": {
        "rows": [{"label": "Deal - Unlevered", "cells": [
            {"numeric_value": 4167949}, {"numeric_value": 21.5}, {"numeric_value": 1.8},
        ]}],
    }}
    candidate = [table]
    assert has_missing_table_proposal(proposal, baseline)
    assert accept_ocr_table_candidate(
        proposal, baseline, candidate, [], {"represented": 10}, {"represented": 13},
    )
    assert not accept_ocr_table_candidate(
        proposal, baseline, candidate, [], {"represented": 10}, {"represented": 9},
    )
    table["content"]["rows"][0]["cells"] = [{"numeric_value": 21.5}]
    assert not accept_ocr_table_candidate(
        proposal, baseline, candidate, [], {"represented": 10}, {"represented": 13},
    )


def test_kpi_item_label_can_name_the_metric_category() -> None:
    panel = {
        "type": "kpi_panel", "block_id": "kpi", "coordinates": [0, 0, 1000, 1000],
        "content": {"metrics": [{"item_id": "o1", "series": "Portfolio to Date",
                              "category": "Investments", "raw_value": "168"}]},
    }
    proposal = {"blocks": [{"type": "kpi_panel", "title": "Portfolio to Date",
                            "bbox": [0, 0, 1000, 1000],
                            "items": [{"label": "Investments", "value": "168"}]}]}
    result = reconcile_page(1, proposal, [panel], [])
    assert result["counts"] == {"agreement": 1}

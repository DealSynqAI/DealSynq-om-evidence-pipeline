from __future__ import annotations

from pathlib import Path
import hashlib
import json

from ocr_pipeline.models import PageInspection, Region
from ocr_pipeline.pipeline import (
    _absorb_structured_visual_text, _separate_visual_footnotes, _ocr_text, _parse_contact_details,
    _numeric_value,
)
from ocr_pipeline.vision_first import apply_plan_hints, overlap_of_smaller, plan_pages
from ocr_pipeline.vision_plan import PROMPT


def _inspection() -> PageInspection:
    return PageInspection(
        page=1, width_points=100, height_points=100, rotation=0,
        native_text_available=False, native_text_coverage=0,
        native_text_quality=0, image_coverage=1, vector_line_count=0,
        possible_table_regions=1, possible_visual_regions=1, routing_confidence=0.5,
        regions=[
            Region("p001-r001", 1, "normal_text", [0, 0, 1000, 100], 1, "test", 0.9),
            Region("p001-r002", 1, "table", [100, 100, 800, 700], 2, "test", 0.6),
        ],
    )


def test_vision_hint_never_replaces_text_or_region_coverage() -> None:
    inspection = _inspection()
    apply_plan_hints(inspection, {"blocks": [{
        "type": "map", "bbox": [100, 100, 900, 800], "title": "", "chart_type": "",
        "items": [{"label": "a", "value": "1"}, {"label": "b", "value": "2"}],
    }]})
    assert len(inspection.regions) == 2
    assert inspection.regions[0].kind == "normal_text"
    assert inspection.regions[0].metadata == {}
    assert inspection.regions[1].kind == "table"
    assert inspection.regions[1].metadata["vision_plan"]["type"] == "map"


def test_no_endpoint_still_produces_explicit_failure_receipt(tmp_path: Path) -> None:
    image = tmp_path / "page-001.png"
    image.write_bytes(b"not decoded when model disabled")
    plans, summary = plan_pages({1: image}, tmp_path / "vision", None, "test-model")
    assert plans == {}
    assert summary["failed_pages"] == [1]
    assert (tmp_path / "vision/page-001.receipt.json").is_file()


def test_overlap_of_smaller_uses_xywh() -> None:
    assert overlap_of_smaller([0, 0, 100, 100], [50, 50, 100, 100]) == 0.25


def test_large_visual_not_claimed_by_small_inset() -> None:
    inspection = _inspection()
    apply_plan_hints(inspection, {"blocks": [
        {"type": "chart", "bbox": [200, 150, 300, 250], "title": "legend",
         "items": [{"label": "a", "value": "1"}, {"label": "b", "value": "2"}]},
        {"type": "map", "bbox": [100, 100, 900, 800], "title": "map",
         "items": [{"label": "a", "value": "1"}, {"label": "b", "value": "2"}]},
    ]})
    assert inspection.regions[1].metadata["vision_plan"]["type"] == "map"


def test_empty_model_chart_cannot_retype_logo_or_text() -> None:
    inspection = _inspection()
    apply_plan_hints(inspection, {"blocks": [{
        "type": "chart", "bbox": [100, 100, 900, 800], "title": "logo", "items": [],
    }]})
    assert "vision_plan" not in inspection.regions[1].metadata


def test_cache_requires_exact_image_model_prompt_and_valid_schema(tmp_path: Path) -> None:
    image = tmp_path / "page-001.png"
    image.write_bytes(b"identical rendered page")
    cache = tmp_path / "cache"
    cache.mkdir()
    proposal = {"page_number": 1, "blocks": []}
    (cache / "page-001.json").write_text(json.dumps(proposal), encoding="utf-8")
    (cache / "page-001.receipt.json").write_text(json.dumps({
        "status": "complete", "model": "test-model",
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
    }), encoding="utf-8")
    plans, summary = plan_pages(
        {1: image}, tmp_path / "out", "http://endpoint-not-called", "test-model",
        verified_cache=cache,
    )
    assert plans[1] == proposal
    assert summary["complete_pages"] == 1
    receipt = json.loads((tmp_path / "out/page-001.receipt.json").read_text(encoding="utf-8"))
    assert receipt["verified_cache"]["source_proposal_sha256"]


def test_exact_isolated_chart_item_moves_evidence_not_general_prose() -> None:
    inspection = _inspection()
    visual = inspection.regions[1]
    visual.kind = "visual"
    visual.coordinates = [100, 100, 800, 700]
    visual.metadata["vision_plan"] = {"type": "chart", "block_index": 0}
    label_region = Region("label", 1, "normal_text", [150, 200, 120, 80], 3, "native", 1)
    prose_region = Region("prose", 1, "normal_text", [300, 300, 180, 90], 4, "native", 1)
    inspection.regions.extend([label_region, prose_region])
    lines = {
        visual.region_id: [],
        label_region.region_id: [
            {"evidence_id": "l1", "text": "Healthcare"},
            {"evidence_id": "l2", "text": "0.9%"},
        ],
        prose_region.region_id: [
            {"evidence_id": "p1", "text": "A paragraph"},
            {"evidence_id": "p2", "text": "0.9%"},
        ],
    }
    absorbed = _absorb_structured_visual_text(
        inspection, lines, {"blocks": [{"items": [{"label": "Healthcare", "value": "0.9%"}]}]},
    )
    assert absorbed == {"label"}
    assert [line["evidence_id"] for line in lines[visual.region_id]] == ["l1", "l2"]
    assert len(lines["prose"]) == 2


def test_marked_visual_note_is_separate_from_chart_values() -> None:
    inspection = _inspection()
    inspection.regions[1].kind = "visual"
    lines = {inspection.regions[1].region_id: [
        {"evidence_id": "value", "text": "37.5%", "coordinates": [300, 300, 60, 20]},
        {"evidence_id": "mark", "text": "*", "coordinates": [110, 700, 10, 20]},
        {"evidence_id": "note", "text": "This value includes an adjustment.",
         "coordinates": [125, 700, 450, 20]},
    ]}
    notes = _separate_visual_footnotes(inspection, lines)
    assert [line["evidence_id"] for line in notes[inspection.regions[1].region_id][0]] == ["note", "mark"]
    assert [line["evidence_id"] for line in lines[inspection.regions[1].region_id]] == ["value"]


def test_long_ocr_line_supersedes_same_native_words_in_text_rendering() -> None:
    lines = [
        {"evidence_id": "p001-ocr-0001", "text": "Important Note: This is a full sentence.",
         "coordinates": [100, 100, 500, 30]},
        {"evidence_id": "p001-native-0001", "text": "Important", "coordinates": [100, 102, 65, 25]},
        {"evidence_id": "p001-native-0002", "text": "Note:", "coordinates": [170, 102, 50, 25]},
        {"evidence_id": "p001-native-0003", "text": "*", "coordinates": [80, 99, 10, 25]},
    ]
    assert _ocr_text(lines) == "*\nImportant Note: This is a full sentence."


def test_contact_name_comes_from_printed_lead_segment() -> None:
    raw = "Jordan Rivera | jordan@example.com | 303-250-5898"
    details = _parse_contact_details(raw, raw)
    assert details["name"] == "Jordan Rivera"
    assert details["email"] == "jordan@example.com"
    assert details["phone"] == "303-250-5898"


def test_attached_currency_scale_is_normalized() -> None:
    assert _numeric_value("$12.5M") == (12.5, "USD", 12_500_000.0)
    assert _numeric_value("$2.1B") == (2.1, "USD", 2_100_000_000.0)
    assert _numeric_value("$900K") == (900.0, "USD", 900_000.0)

from __future__ import annotations

from ocr_pipeline.visual_blocks import _scatter_axis_evidence, _stacked_pie_bindings


def _line(evidence_id: str, text: str, x: float, y: float, width: float = 85) -> dict:
    return {"evidence_id": evidence_id, "text": text, "coordinates": [x, y, width, 18],
            "confidence": 0.99}


def test_mixed_pie_uses_complete_ocr_stacks_not_native_word_fragments() -> None:
    lines = [
        _line("p001-ocr-001", "Annual Revenue Split", 100, 10, 220),
        _line("p001-ocr-002", "Cloud Services", 100, 100, 120),
        _line("p001-ocr-003", "$80 million", 110, 130, 110),
        _line("p001-ocr-004", "80%", 135, 160, 55),
        _line("p001-ocr-005", "Hardware", 450, 100, 100),
        _line("p001-ocr-006", "$20 million", 450, 130, 110),
        _line("p001-ocr-007", "20%", 480, 160, 55),
        _line("p001-native-001", "Cloud", 100, 100, 45),
        _line("p001-native-002", "Services", 150, 100, 65),
    ]
    bindings = _stacked_pie_bindings(lines)
    assert bindings is not None
    assert [(item["label"], item["value"]) for item in bindings] == [
        ("Cloud Services", "80%"), ("Hardware", "20%"),
    ]
    assert bindings[0]["companion_value"]["normalized_value"] == 80_000_000
    assert all(item["visual_mark_coordinates"] is None for item in bindings)


def test_simple_pie_does_not_enter_mixed_currency_path() -> None:
    lines = [
        _line("p001-ocr-001", "North", 100, 100),
        _line("p001-ocr-002", "60%", 100, 130),
        _line("p001-ocr-003", "South", 400, 100),
        _line("p001-ocr-004", "40%", 400, 130),
    ]
    assert _stacked_pie_bindings(lines) is None


def test_scatter_axis_keeps_ranges_separate_from_exact_observations() -> None:
    lines = [
        _line("p001-ocr-001", "100%", 20, 100, 55),
        _line("p001-ocr-002", "50%", 20, 200, 55),
        _line("p001-ocr-003", "0%", 20, 300, 55),
        _line("p001-ocr-004", "Q1", 120, 320, 40),
        _line("p001-ocr-005", "Q2", 260, 320, 40),
    ]
    calibration, points = _scatter_axis_evidence(lines, [[190, 220, 8, 8]])
    assert calibration["status"] == "accepted"
    assert len(points) == 1
    assert points[0]["validation_status"] == "needs_review"
    assert points[0]["estimated_percent_range"][0] < 40
    assert points[0]["estimated_percent_range"][1] > 40
    assert points[0]["x_interval"] == ["Q1", "Q2"]
    assert "raw_value" not in points[0]


def test_scatter_without_reliable_ticks_keeps_coordinates_only() -> None:
    calibration, points = _scatter_axis_evidence(
        [_line("p001-ocr-001", "50%", 20, 100, 55)], [[190, 220, 8, 8]],
    )
    assert calibration["status"] == "unavailable"
    assert points[0]["estimated_percent_range"] is None

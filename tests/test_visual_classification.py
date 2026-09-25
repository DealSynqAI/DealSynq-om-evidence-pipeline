from __future__ import annotations

from ocr_pipeline.models import Region
from ocr_pipeline.paddle_tables import ocr_paired_columns
from ocr_pipeline.table_blocks import _filled_data_cells, _split_key_value_rows, _table_backdrop
from ocr_pipeline.visual_blocks import _chart_text_support, _classify_visual


def _lines(*texts: str, x: float = 100, y: float = 100, step: float = 30) -> list[dict]:
    return [{"evidence_id": f"p001-ocr-{index:04d}", "text": text, "confidence": 0.95,
             "coordinates": [x, y + index * step, 120, 18]} for index, text in enumerate(texts)]


def test_chart_text_is_mostly_numbers_and_not_printed_facts() -> None:
    assert _chart_text_support(_lines("North", "12%", "South", "18%"))
    assert _chart_text_support(_lines("2019", "2020", "2021", "11.3%", "12.9%", "16.7%", "Annual IRR"))
    assert not _chart_text_support(_lines("Price: $420,000", "Cost per Sq Ft: $358/sqft", "Built in 1994"))
    assert not _chart_text_support(_lines("Will Matheson", "Managing Partner", "will@example.com", "(803) 555-0100"))


def test_a_chart_hint_needs_chart_text() -> None:
    features = {"bar_candidates": 4, "horizontal_axis_candidates": 1, "vertical_axis_candidates": 1}
    contacts = Region("r1", 1, "visual", [0, 0, 1000, 1000], 1, "raster", 0.8, metadata={"visual_hint": "chart"})
    kind, _, _ = _classify_visual(contacts, _lines(
        "For more information, please contact our team", "Will Matheson", "will@example.com",
        "(803) 555-0100", "Evan Matheson", "evan@example.com", "(803) 555-0101"), features)
    assert kind == "normal_text"
    bars = Region("r2", 1, "visual", [0, 0, 1000, 1000], 1, "vector", 0.8, metadata={"visual_hint": "chart"})
    kind, _, _ = _classify_visual(bars, _lines("2021", "2022", "2023", "18.9%", "18.5%", "19.0%"), features)
    assert kind == "chart"


def test_paired_numeric_columns_are_a_table_but_prose_columns_are_not() -> None:
    rates = [{"evidence_id": f"p001-ocr-{index:04d}", "text": f"{7.5 - index * 0.25:.2f}%",
              "coordinates": [548, 570 + index * 20, 33, 19]} for index in range(9)]
    returns = [{"evidence_id": f"p001-ocr-{100 + index:04d}", "text": f"{14 + index:.1f}%/ 1.{index}x",
                "coordinates": [628, 571 + index * 20, 56, 17]} for index in range(9)]
    assert ocr_paired_columns(rates + returns)
    left = [{"evidence_id": f"p001-ocr-{index:04d}", "text": "Cleanliness and quality vary widely",
             "coordinates": [100, 100 + index * 30, 300, 18]} for index in range(6)]
    right = [{"evidence_id": f"p001-ocr-{50 + index:04d}", "text": "Your place is waiting for you",
              "coordinates": [520, 101 + index * 30, 300, 18]} for index in range(6)]
    assert not ocr_paired_columns(left + right)


def test_one_column_fact_list_splits_into_key_value_rows() -> None:
    rows = [["BUILDING INFORMATION"], ["Exterior Materials: Brick, Hardboard"], ["Roof Type: Asphalt"], ["Paving: Asphalt"]]
    boxes = [[[10, 10, 300, 20]], [[10, 40, 300, 20]], [[10, 70, 300, 20]], [[10, 100, 300, 20]]]
    title, split, coordinates = _split_key_value_rows(rows, boxes)
    assert title == "BUILDING INFORMATION"
    assert split == [["Field", "Value"], ["Exterior Materials", "Brick, Hardboard"],
                     ["Roof Type", "Asphalt"], ["Paving", "Asphalt"]]
    assert coordinates[1] == [[10, 40, 300, 20], [10, 40, 300, 20]]
    # Two facts on one printed line cannot be split from text alone.
    assert _split_key_value_rows([["Property: Chateau Year Built: 1969"], ["Units: 168"], ["Parking: 242"]], []) is None
    assert _filled_data_cells([["Maravida", None, ""], ["", None, None]]) == 0


def test_a_table_image_under_a_parsed_table_is_its_backdrop() -> None:
    table = Region("t", 1, "table", [0, 0, 1000, 1000], 1, "pdfplumber", 0.9,
                   metadata={"rows": [["Sources", "Uses"], ["Debt", "$3,458,000"], ["Equity", "2,039,810"]]})
    image = Region("v", 1, "visual", [100, 250, 800, 500], 2, "pdf-image-object", 0.8)
    assert _table_backdrop(image, _lines("Sources", "Uses", "Debt", "$3,458,000", "Equity", "2,039,810"), [table]) is table
    chart = _lines("Occupancy", "94%", "Rent growth", "3.1%")
    assert _table_backdrop(image, chart, [table]) is None

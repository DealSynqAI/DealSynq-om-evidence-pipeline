from ocr_pipeline.models import PageInspection, Region
from ocr_pipeline.paddle_tables import apply_paddle_tables, page_needs_table_analysis, table_candidates
from ocr_pipeline.table_blocks import _ground_table_cells_from_ocr, _table_blocks
from pathlib import Path
import pytest


def _page(*regions):
    return PageInspection(
        page=1, width_points=600, height_points=800, rotation=0,
        native_text_available=False, native_text_coverage=0, native_text_quality=0,
        image_coverage=1, vector_line_count=0, possible_table_regions=0,
        possible_visual_regions=1, routing_confidence=0.5, regions=list(regions),
    )


def test_paddle_table_can_rescue_visual_region_without_vision_values():
    payload = {"res": {
        "layout_det_res": {"boxes": [{"label": "table", "score": 0.9,
                                        "coordinate": [100, 100, 500, 400]}]},
        "table_res_list": [{"pred_html":
                            "<table><tr><th>Year</th><th>Rent</th></tr>"
                            "<tr><td>2024</td><td>$100</td></tr>"
                            "<tr><td>2025</td><td>$110</td></tr></table>"}],
    }}
    candidate = table_candidates(payload, 600, 500)
    assert len(candidate) == 1
    region = Region("p001-r001", 1, "visual", [160, 180, 680, 640], 1,
                    "OpenCV", 0.5)
    inspection = _page(region)
    apply_paddle_tables(inspection, candidate)
    assert region.kind == "table"
    assert region.metadata["rows_source"] == "paddle"
    assert region.metadata["rows"][2] == ["2025", "$110"]


def test_native_values_remain_when_paddle_dimensions_disagree():
    region = Region("p001-r001", 1, "table", [100, 100, 600, 400], 1,
                    "native PDF", 0.9, metadata={"rows": [["A", "B"], ["x", "1"]]})
    candidate = [{"index": 0, "coordinates": [100, 100, 600, 400],
                  "rows": [["A", "B", "C"], ["x", "1", "2"]], "layout_score": 0.8}]
    apply_paddle_tables(_page(region), candidate)
    assert region.metadata["rows"] == [["A", "B"], ["x", "1"]]
    assert region.metadata["paddle_table_review"]["status"] == "disagrees"
    candidate[0]["rows"] = [["A", "B"], ["x", "2"]]
    apply_paddle_tables(_page(region), candidate)
    assert region.metadata["paddle_table_review"]["cell_disagreements"] == 1
    assert region.metadata["rows"][1][1] == "1"


def test_paddle_cell_boxes_disambiguate_repeated_values_by_row():
    html = ("<table><tr><th>Item</th><th>Value</th></tr>"
            "<tr><td>Asset A</td><td>$100</td></tr>"
            "<tr><td>Asset B</td><td>$100</td></tr></table>")
    boxes = [
        [100, 100, 350, 150], [400, 100, 600, 150],
        [100, 150, 350, 200], [400, 150, 600, 200],
        [100, 200, 350, 250], [400, 200, 600, 250],
    ]
    candidate = table_candidates({"res": {
        "layout_det_res": {"boxes": [{"label": "table", "score": 0.9,
                                       "coordinate": [100, 100, 600, 250]}]},
        "table_res_list": [{"pred_html": html, "cell_box_list": boxes}],
    }}, 1000, 1000)[0]
    assert candidate["cell_coordinates"][2][1] == [400.0, 200.0, 200.0, 50.0]
    region = Region("p001-r001", 1, "table", candidate["coordinates"], 1,
                    "PaddleOCR", 0.7, metadata={
                        "rows_source": "paddle", "rows": candidate["rows"],
                        "cell_coordinates": candidate["cell_coordinates"],
                    })
    lines = [
        {"evidence_id": "label-a", "text": "Asset A", "coordinates": [120, 160, 90, 20]},
        {"evidence_id": "value-a", "text": "$100", "coordinates": [430, 160, 65, 20]},
        {"evidence_id": "label-b", "text": "Asset B", "coordinates": [120, 210, 90, 20]},
        {"evidence_id": "value-b", "text": "$100", "coordinates": [430, 210, 65, 20]},
    ]
    block = _table_blocks("doc", "hash", region, lines, Path("page-001.png"))[0].as_dict()
    stats = _ground_table_cells_from_ocr([block], lines)
    assert stats["newly_grounded"] == 2
    assert [row["cells"][0]["evidence_ids"] for row in block["content"]["rows"]] == [
        ["value-a"], ["value-b"],
    ]


def test_value_first_paddle_list_keeps_first_fact():
    region = Region("p001-r001", 1, "table", [100, 100, 600, 400], 1,
                    "PaddleOCR", 0.65, metadata={
                        "rows_source": "paddle",
                        "rows": [["6.3%", "Going-in Cap Rate"],
                                 ["$3.08", "In-Place Rent"],
                                 ["100%", "Occupancy"]],
                        "cell_coordinates": [
                            [[400, 100, 100, 30], [100, 100, 250, 30]],
                            [[400, 140, 100, 30], [100, 140, 250, 30]],
                            [[400, 180, 100, 30], [100, 180, 250, 30]],
                        ],
                        "paddle_table_review": {"status": "candidate"},
                    })
    block = _table_blocks("doc", "hash", region, [], Path("page-001.png"))[0].as_dict()
    assert block["content"]["row_count"] == 3
    assert block["content"]["rows"][0]["label"] == "Going-in Cap Rate"
    assert block["content"]["rows"][0]["cells"][0]["raw_value"] == "6.3%"
    assert block["content"]["rows"][0]["label_coordinates"] == [100, 100, 250, 30]
    assert block["content"]["rows"][0]["cells"][0]["coordinates"] == [400, 100, 100, 30]
    assert block["validation"]["status"] == "needs_review"


def test_table_pages_include_native_and_unclassified_numeric_visuals():
    native = Region("p001-r001", 1, "table", [100, 100, 600, 400], 1, "native", 0.9)
    assert page_needs_table_analysis(_page(native), {"lines": []})
    visual = Region("p001-r002", 1, "visual", [100, 100, 600, 400], 1, "OpenCV", 0.5)
    lines = [{"text": str(index), "coordinates": [150, 150 + index * 20, 20, 10]}
             for index in range(6)]
    assert page_needs_table_analysis(_page(visual), {"lines": lines})
    assert not page_needs_table_analysis(_page(visual), {"lines": lines[:2]})


def test_collapsed_paddle_html_uses_layout_without_importing_values():
    payload = {"res": {
        "layout_det_res": {"boxes": [{"label": "table", "score": 0.8,
                                        "coordinate": [100, 100, 500, 400]}]},
        "table_res_list": [{"pred_html": "<table>"
                            "<tr><td>Rent $100</td></tr>"
                            "<tr><td>Area 200</td></tr>"
                            "<tr><td>Price $300</td></tr></table>"}],
    }}
    candidate = table_candidates(payload, 600, 500)
    region = Region("p001-r001", 1, "visual", [160, 180, 680, 640], 1,
                    "OpenCV", 0.5)
    apply_paddle_tables(_page(region), candidate)
    assert region.kind == "table"
    assert "rows" not in region.metadata
    assert region.metadata["paddle_table_review"]["status"] == "layout_only"


def test_paddle_merged_cells_keep_text_without_inventing_numeric_fact():
    region = Region("p001-r001", 1, "table", [100, 100, 600, 400], 1,
                    "PaddleOCR", 0.65, metadata={
                        "rows_source": "paddle",
                        "rows": [["Item", "Value", "Amount"],
                                 ["Merged", "$6.53 3.0%", "$32,214,751 $13,211,787"],
                                 ["Date", "1/10/26", "$3.08"],
                                 ["Address", "168BeaconStU:1", "8 room, 3 bed"]],
                    })
    block = _table_blocks("doc", "hash", region, [], Path("page-001.png"))[0].as_dict()
    rows = block["content"]["rows"]
    assert rows[0]["cells"][0]["raw_value"] == "$6.53 3.0%"
    assert rows[0]["cells"][0]["numeric_value"] is None
    assert rows[0]["cells"][1]["numeric_value"] is None
    assert rows[1]["cells"][0]["numeric_value"] is None
    assert rows[1]["cells"][1]["numeric_value"] == 3.08
    assert rows[1]["cells"][1]["unit"] == "USD"
    assert rows[2]["cells"][0]["numeric_value"] is None
    assert rows[2]["cells"][1]["numeric_value"] is None
    assert any("non-scalar numeric text" in warning for warning in block["validation"]["warnings"])


def test_two_detected_tables_in_one_visual_get_separate_regions():
    visual = Region("p001-r001", 1, "visual", [100, 100, 800, 800], 1,
                    "OpenCV", 0.5)
    inspection = _page(visual)
    candidates = [
        {"index": 0, "coordinates": [120, 120, 700, 300],
         "rows": [["Item", "Value"], ["A", "$100"], ["B", "$200"], ["E", "$500"]],
         "layout_score": 0.9, "structure_status": "grid"},
        {"index": 1, "coordinates": [420, 500, 400, 300],
         "rows": [["Item", "Value"], ["C", "$300"], ["D", "$400"], ["F", "$600"]],
         "layout_score": 0.9, "structure_status": "grid"},
    ]
    apply_paddle_tables(inspection, candidates)
    tables = [region for region in inspection.regions if region.kind == "table"]
    assert len(tables) == 2
    assert {region.metadata["rows"][1][0] for region in tables} == {"A", "C"}
    assert tables[1].coordinates == [420, 500, 400, 300]


@pytest.mark.parametrize("region_kind", ["table", "visual"])
def test_broad_empty_table_splits_into_separate_numeric_grids(region_kind):
    broad = Region("p001-r001", 1, region_kind, [100, 100, 800, 600], 1,
                   "OpenCV", 0.5)
    inspection = _page(broad)
    candidates = [
        {"index": index, "coordinates": [160, 150 + index * 125, 600, 80],
         "rows": [["Metric", "Base", "Change"], [f"Grid {index}", "100", "-2.0"]],
         "layout_score": 0.6, "structure_status": "grid"}
        for index in range(3)
    ]
    candidates.append({
        "index": 3, "coordinates": [150, 140, 620, 370],
        "rows": [["Metric", "Base", "Change"], ["Grid 0", "100", "-2.0"]],
        "layout_score": 0.55, "structure_status": "grid",
    })
    apply_paddle_tables(inspection, candidates)
    tables = [region for region in inspection.regions if region.kind == "table"]
    assert len(tables) == 3
    assert {region.metadata["rows"][1][0] for region in tables} == {
        "Grid 0", "Grid 1", "Grid 2"}
    assert all(region.metadata["rows_source"] == "paddle" for region in tables)
    assert broad.kind == "normal_text"
    assert broad.metadata["paddle_split_residual"] == [0, 1, 2]


def test_small_heading_cannot_claim_whole_page_paddle_table():
    heading = Region("p001-r001", 1, "normal_text", [10, 20, 250, 40], 1,
                     "native PDF", 0.9)
    inspection = _page(heading)
    candidate = {"index": 0, "coordinates": [10, 0, 980, 900],
                 "rows": [["Item", "Amount"], ["A", "$100"], ["B", "$200"], ["C", "$300"]],
                 "layout_score": 0.95, "structure_status": "grid"}
    apply_paddle_tables(inspection, [candidate])
    assert heading.kind == "normal_text"
    table = next(region for region in inspection.regions if region.kind == "table")
    assert table.region_id != heading.region_id
    assert table.coordinates == candidate["coordinates"]


def test_aligned_numeric_ocr_grid_selects_a_page_without_a_table_region():
    lines = [{"evidence_id": f"p001-ocr-{row * 3 + column:04d}", "text": text,
              "coordinates": [100 + column * 250, 200 + row * 40, 90, 18]}
             for row in range(5)
             for column, text in enumerate([f"Unit {row}", f"{row + 1} BR", f"${row + 9},250"])]
    # The region does not contain the lines, so only the page-level grid can select it.
    prose = Region("p001-r001", 1, "normal_text", [0, 0, 50, 50], 1, "native", 0.9)
    assert page_needs_table_analysis(_page(prose), {"lines": lines})
    words = [dict(line, text="Tenant improvements") for line in lines]
    assert not page_needs_table_analysis(_page(prose), {"lines": words})

from pathlib import Path

from ocr_pipeline.models import PageInspection, Region
from ocr_pipeline.table_blocks import _table_blocks
from ocr_pipeline.spatial_lanes import split_mixed_key_value_regions


def _page(region):
    return PageInspection(1, 600, 800, 0, False, 0, 0, 1, 0, 0, 1, 0.5, [region])


def _line(index, text, x, y, width):
    return {"evidence_id": f"p001-ocr-{index:04d}", "text": text,
            "coordinates": [x, y, width, 12], "confidence": 0.95}


def test_mixed_key_value_lane_keeps_printed_pairs_and_separates_prose():
    region = Region("p001-r001", 1, "table", [0, 0, 1000, 1000], 1, "OpenCV", 0.7,
                    metadata={"paddle_table_review": {"status": "layout_only"}})
    lines = []
    for index in range(7):
        lines.extend([_line(3 * index + 1, f"FIELD {index}", 20, 180 + 50 * index, 85),
                      _line(3 * index + 2, f"{index + 10}%", 210, 186 + 50 * index, 45),
                      _line(3 * index + 3, "Long independent prose sentence about the property and its business plan.",
                            300, 180 + 25 * index, 290)])
    lines.extend(_line(100 + index, "Another long paragraph sentence that continues independently of the facts.",
                       300, 355 + 25 * index, 290) for index in range(7))
    inspection = _page(region)
    assert split_mixed_key_value_regions(inspection, lines) == 1
    table, prose = inspection.regions
    assert (table.kind, prose.kind) == ("table", "normal_text")
    assert table.coordinates[0] + table.coordinates[2] == prose.coordinates[0]
    assert table.metadata["rows"][1] == ["FIELD 0", "10%"]
    assert table.metadata["cell_evidence_ids"][1] == [["p001-ocr-0001"], ["p001-ocr-0002"]]
    block = _table_blocks("doc", "hash", table, lines[:2], Path("page-001.png"))[0].as_dict()
    assert block["content"]["rows"][0]["label_evidence_ids"] == ["p001-ocr-0001"]
    assert block["content"]["rows"][0]["cells"][0]["evidence_ids"] == ["p001-ocr-0002"]
    # Every printed value has its own OCR evidence on its label's row.
    assert block["validation"]["status"] == "passed"
    table.metadata["cell_evidence_ids"][2][1] = []
    unsupported = _table_blocks("doc", "hash", table, lines[:2], Path("page-001.png"))[0].as_dict()
    assert unsupported["validation"]["status"] == "needs_review"


def test_single_dense_table_does_not_split_without_prose_lane():
    region = Region("p001-r001", 1, "table", [0, 0, 1000, 1000], 1, "OpenCV", 0.7)
    lines = [_line(index, f"Value {index}", 20 + (index % 3) * 150,
                   100 + (index // 3) * 50, 80) for index in range(21)]
    inspection = _page(region)
    assert split_mixed_key_value_regions(inspection, lines) == 0
    assert inspection.regions == [region]

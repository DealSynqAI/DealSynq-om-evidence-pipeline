from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ocr_pipeline.models import PageInspection, Region
from ocr_pipeline.ocr_regions import add_ocr_text_regions


def _page(regions: list[Region]) -> PageInspection:
    return PageInspection(page=1, width_points=1000, height_points=1000, rotation=0,
                          native_text_available=False, native_text_coverage=0.0, native_text_quality=0.0,
                          image_coverage=0.0, vector_line_count=0, possible_table_regions=0,
                          possible_visual_regions=0, routing_confidence=1.0, regions=regions)


def _line(number: int, text: str, box: list[float]) -> dict:
    return {"evidence_id": f"p001-ocr-{number:04d}", "text": text, "coordinates": box, "confidence": 0.95}


def _white(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (1000, 1000), "white").save(path)
    return path


def test_text_lane_outside_every_region_gets_its_own_region(tmp_path: Path) -> None:
    body = Region("p001-r001", 1, "normal_text", [100, 100, 800, 600], 1, "native", 0.9)
    inspection = _page([body])
    lines = [_line(1, "Inside the body", [150, 150, 300, 20]),
             _line(2, "Proprietary and Confidential. All Rights Reserved.", [100, 900, 500, 18]),
             _line(3, "Prepared for qualified investors only", [100, 922, 480, 18])]
    assert add_ocr_text_regions(inspection, lines, _white(tmp_path)) == 1
    added = inspection.regions[-1]
    assert added.kind == "normal_text" and added.region_id == "p001-ocr-text-001"
    assert added.coordinates == [100, 900, 500, 40]


def test_scattered_short_labels_and_text_over_imagery_stay_unassigned(tmp_path: Path) -> None:
    labels = [_line(index, name, [100 + index * 150, 100 + index * 120, 90, 16])
              for index, name in enumerate(["Orlando", "Ocala", "Tampa", "Naples", "Miami"], 1)]
    assert add_ocr_text_regions(_page([]), labels, _white(tmp_path)) == 0

    noisy = tmp_path / "photo.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (1000, 1000, 3), dtype=np.uint8)).save(noisy)
    caption = [_line(1, "Neighborhood retail and shopping", [400, 500, 300, 20])]
    assert add_ocr_text_regions(_page([]), caption, noisy) == 0
    assert add_ocr_text_regions(_page([]), caption, _white(tmp_path)) == 1


def test_caller_can_reserve_a_lane_for_a_brand_mark(tmp_path: Path) -> None:
    logo = [_line(1, "Northwind Capital Partners", [700, 40, 250, 24])]
    assert add_ocr_text_regions(_page([]), logo, _white(tmp_path), skip=lambda cluster: True) == 0

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr import RapidOCR


def _native(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _xywh_to_pixels(coordinates: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = coordinates
    x0 = max(0, min(width, round(x * width / 1000)))
    y0 = max(0, min(height, round(y * height / 1000)))
    x1 = max(x0 + 1, min(width, round((x + w) * width / 1000)))
    y1 = max(y0 + 1, min(height, round((y + h) * height / 1000)))
    return x0, y0, x1, y1


def _vision_features(image: np.ndarray) -> dict[str, Any]:
    height, width = image.shape[:2]

    def box(x: int, y: int, w: int, h: int) -> list[float]:
        return [
            round(x * 1000 / width, 6), round(y * 1000 / height, 6),
            round(w * 1000 / width, 6), round(h * 1000 / height, 6),
        ]

    def segment(x1: int, y1: int, x2: int, y2: int) -> list[float]:
        return [
            round(x1 * 1000 / width, 6), round(y1 * 1000 / height, 6),
            round(x2 * 1000 / width, 6), round(y2 * 1000 / height, 6),
        ]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    minimum = max(20, min(image.shape[:2]) // 8)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=minimum, maxLineGap=8)
    horizontal_segments: list[list[float]] = []
    vertical_segments: list[list[float]] = []
    diagonal_segments: list[list[float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
            if dy <= max(2, dx * 0.12):
                horizontal_segments.append(segment(int(x1), int(y1), int(x2), int(y2)))
            elif dx <= max(2, dy * 0.12):
                vertical_segments.append(segment(int(x1), int(y1), int(x2), int(y2)))
            else:
                diagonal_segments.append(segment(int(x1), int(y1), int(x2), int(y2)))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rectangle_boxes: list[list[float]] = []
    bar_boxes: list[list[float]] = []
    point_boxes: list[list[float]] = []
    legend_boxes: list[list[float]] = []
    image_area = height * width
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = area / max(1, image_area)
        if len(polygon) == 4 and area_ratio >= 0.0005:
            rectangle_boxes.append(box(x, y, w, h))
            aspect = max(w / max(1, h), h / max(1, w))
            if 1.35 <= aspect <= 18 and 0.0007 <= area_ratio <= 0.20:
                bar_boxes.append(box(x, y, w, h))
            if max(w, h) <= max(12, min(width, height) * 0.09):
                patch = image[y:y + h, x:x + w]
                if patch.size and float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1] > 55)) >= 0.25:
                    legend_boxes.append(box(x, y, w, h))
        if 0.00005 <= area_ratio <= 0.012 and w >= 4 and h >= 4:
            circularity = 4 * np.pi * area / max(1.0, perimeter * perimeter)
            if circularity >= 0.55 and 0.45 <= w / max(1, h) <= 2.2:
                point_boxes.append(box(x, y, w, h))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_fraction = float(np.mean(hsv[:, :, 1] > 55))
    horizontal_axes = [item for item in horizontal_segments if abs(item[2] - item[0]) >= 450]
    vertical_axes = [item for item in vertical_segments if abs(item[3] - item[1]) >= 450]
    plot_boundary = None
    large_rectangles = [item for item in rectangle_boxes if item[2] >= 400 and item[3] >= 300]
    if large_rectangles:
        plot_boundary = max(large_rectangles, key=lambda item: item[2] * item[3])
    elif horizontal_axes and vertical_axes:
        xs = [value for item in vertical_axes for value in (item[0], item[2])]
        ys = [value for item in horizontal_axes for value in (item[1], item[3])]
        plot_boundary = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    grid_strength = min(len(horizontal_segments), len(vertical_segments))
    table_grid_confidence = min(1.0, grid_strength / 8.0) if len(horizontal_segments) >= 3 and len(vertical_segments) >= 3 else 0.0
    return {
        "horizontal_lines": len(horizontal_segments),
        "vertical_lines": len(vertical_segments),
        "diagonal_lines": len(diagonal_segments),
        "rectangle_candidates": len(rectangle_boxes),
        "bar_candidates": len(bar_boxes),
        "point_candidates": len(point_boxes),
        "legend_swatch_candidates": len(legend_boxes),
        "horizontal_axis_candidates": len(horizontal_axes),
        "vertical_axis_candidates": len(vertical_axes),
        "plot_boundary_detected": plot_boundary is not None,
        "table_grid_confidence": round(table_grid_confidence, 5),
        "saturated_pixel_fraction": round(saturation_fraction, 5),
        "line_segments": {
            "horizontal": horizontal_segments[:500],
            "vertical": vertical_segments[:500],
            "diagonal": diagonal_segments[:500],
        },
        "rectangle_candidate_boxes": rectangle_boxes[:500],
        "bar_candidate_boxes": bar_boxes[:500],
        "point_candidate_boxes": point_boxes[:1000],
        "legend_swatch_candidate_boxes": legend_boxes[:500],
        "horizontal_axis_candidate_segments": horizontal_axes[:100],
        "vertical_axis_candidate_segments": vertical_axes[:100],
        "plot_boundary": plot_boundary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args(argv)
    params = {"Global.return_word_box": True}
    if args.model_root:
        params["Global.model_root_dir"] = str(args.model_root.resolve(strict=True))
    engine = RapidOCR(params=params)
    request = json.loads(args.jobs.read_text(encoding="utf-8"))
    results = []
    for job in request["jobs"]:
        image_path = Path(job["image"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        result = engine(str(image_path))
        boxes = _native(result.boxes) or []
        texts = _native(result.txts) or []
        scores = _native(result.scores) or []
        lines = []
        for index, (polygon, text, score) in enumerate(zip(boxes, texts, scores), 1):
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            lines.append({
                "evidence_id": f"p{int(job['page']):03d}-ocr-{index:04d}",
                "text": str(text), "confidence": round(float(score), 5),
                "coordinates": [
                    round(x0 * 1000 / width, 6), round(y0 * 1000 / height, 6),
                    round((x1 - x0) * 1000 / width, 6), round((y1 - y0) * 1000 / height, 6),
                ],
                "polygon": [[round(float(x) * 1000 / width, 6), round(float(y) * 1000 / height, 6)] for x, y in polygon],
            })
        regions = {}
        for region in job.get("regions", []):
            x0, y0, x1, y1 = _xywh_to_pixels(region["coordinates"], width, height)
            crop = image[y0:y1, x0:x1]
            regions[region["region_id"]] = {
                "vision_features": _vision_features(crop),
                "ocr_evidence_ids": [line["evidence_id"] for line in lines if (
                    region["coordinates"][0] <= line["coordinates"][0] + line["coordinates"][2] / 2
                    <= region["coordinates"][0] + region["coordinates"][2]
                    and region["coordinates"][1] <= line["coordinates"][1] + line["coordinates"][3] / 2
                    <= region["coordinates"][1] + region["coordinates"][3]
                )],
            }
        results.append({
            "page": int(job["page"]), "image_size": [width, height],
            "lines": lines, "page_vision_features": _vision_features(image), "regions": regions,
        })
        print(f"page {job['page']}: {len(lines)} OCR lines", flush=True)
    payload = {
        "engine": "RapidOCR PP-OCRv6", "rapidocr_version": importlib.metadata.version("rapidocr"),
        "opencv_version": cv2.__version__, "pages": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

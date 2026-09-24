"""Reference-backed geometry for raster US-state choropleths.

The atlas is a public US Census cartographic boundary file, not document data.
Registration is inferred from the rendered map silhouette and is rejected when
it does not align. Other map types simply remain unresolved.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from typing import Any


ATLAS = Path(__file__).parent / "assets" / "cb_2025_us_state_5m.zip"
ATLAS_URL = "https://www2.census.gov/geo/tiger/GENZ2025/kml/cb_2025_us_state_5m.zip"
NON_MAINLAND = {
    "Alaska", "Hawaii", "Puerto Rico", "American Samoa", "Guam",
    "Northern Mariana Islands", "Commonwealth of the Northern Mariana Islands",
    "United States Virgin Islands",
}


def _mercator_y(latitude: float) -> float:
    latitude = max(-85.0, min(85.0, latitude))
    return -math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2))


def _albers_point(longitude: float, latitude: float) -> tuple[float, float]:
    """Spherical Conus Albers (29.5/45.5 N, -96 W); affine fit absorbs units."""
    phi1, phi2 = math.radians(29.5), math.radians(45.5)
    n = (math.sin(phi1) + math.sin(phi2)) / 2
    c = math.cos(phi1) ** 2 + 2 * n * math.sin(phi1)
    rho = math.sqrt(c - 2 * n * math.sin(math.radians(latitude))) / n
    theta = n * math.radians(longitude + 96)
    return rho * math.sin(theta), rho * math.cos(theta)


def _project_point(longitude: float, latitude: float, projection: str) -> tuple[float, float]:
    if projection == "conus_albers":
        return _albers_point(longitude, latitude)
    if projection == "mercator":
        return longitude, _mercator_y(latitude)
    return longitude, -latitude


@lru_cache(maxsize=1)
def _state_rings() -> dict[str, list[list[tuple[float, float]]]]:
    namespace = {"k": "http://www.opengis.net/kml/2.2"}
    with zipfile.ZipFile(ATLAS) as archive:
        root = ET.fromstring(archive.read("cb_2025_us_state_5m.kml"))
    result: dict[str, list[list[tuple[float, float]]]] = {}
    for placemark in root.findall(".//k:Placemark", namespace):
        fields = {
            node.attrib.get("name"): node.text
            for node in placemark.findall(".//k:SimpleData", namespace)
        }
        name = str(fields.get("NAME") or "")
        if not name or name in NON_MAINLAND:
            continue
        rings: list[list[tuple[float, float]]] = []
        for node in placemark.findall(".//k:outerBoundaryIs/k:LinearRing/k:coordinates", namespace):
            ring = []
            for coordinate in (node.text or "").split():
                longitude, latitude = map(float, coordinate.split(",")[:2])
                ring.append((longitude, latitude))
            if len(ring) >= 3:
                rings.append(ring)
        if rings and all(-130 <= point[0] <= -60 for ring in rings for point in ring):
            result[name] = rings
    return result


def _filled_map_mask(image: Any) -> tuple[Any, Any] | None:
    import cv2
    import numpy as np

    pixels = image.astype(np.int16)
    spread = pixels.max(axis=2) - pixels.min(axis=2)
    brightness = pixels.mean(axis=2)
    # Solid gray and colored cartographic areas; erode fine text/leader strokes.
    mask = (
        (brightness > 80) & (brightness < 245)
        & ((spread < 5) | (pixels[:, :, 0] - pixels[:, :, 2] > 15))
    ).astype(np.uint8)
    unit = max(1, round(image.shape[1] / 2000))
    mask = cv2.erode(mask, np.ones((3 * unit, 3 * unit), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((15 * unit, 15 * unit), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None
    largest = max(range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 0.04 * image.shape[0] * image.shape[1]:
        return None
    return (labels == largest).astype(np.uint8), stats[largest]


def register_us_state_map(image_path: Path, min_quality: float = 0.90) -> dict[str, Any] | None:
    """Fit Census state polygons to a rendered mainland-US silhouette."""
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None or not ATLAS.exists():
        return None
    if image.shape[1] > 2000:
        factor = 2000 / image.shape[1]
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    detected = _filled_map_mask(image)
    if detected is None:
        return None
    silhouette, bbox = detected
    x0, y0, width, height = map(float, bbox[:4])
    if width < image.shape[1] * 0.30 or height < image.shape[0] * 0.25:
        return None
    state_rings = _state_rings()
    candidates_by_projection = []
    for projection in ("conus_albers", "mercator", "equirectangular"):
        states = {
            name: [[_project_point(*point, projection) for point in ring] for ring in rings]
            for name, rings in state_rings.items()
        }
        points = [point for rings in states.values() for ring in rings for point in ring]
        min_x, max_x = min(point[0] for point in points), max(point[0] for point in points)
        min_y, max_y = min(point[1] for point in points), max(point[1] for point in points)
        scale_x = width / (max_x - min_x)
        scale_y = height / (max_y - min_y)
        offset_x = x0 - scale_x * min_x
        offset_y = y0 - scale_y * min_y

        def render(sx: float, sy: float, ox: float, oy: float) -> tuple[Any, float]:
            labels = np.zeros(image.shape[:2], np.uint8)
            for index, rings in enumerate(states.values(), start=1):
                for ring in rings:
                    vertices = np.array(
                        [[round(sx * point[0] + ox), round(sy * point[1] + oy)] for point in ring],
                        np.int32,
                    )
                    cv2.fillPoly(labels, [vertices], index)
            projected = labels > 0
            intersection = np.count_nonzero(projected & (silhouette > 0))
            union = np.count_nonzero(projected | (silhouette > 0))
            return labels, intersection / union if union else 0.0

        label_raster, quality = render(scale_x, scale_y, offset_x, offset_y)
        # Only small refinements around the inferred extent. A weak registration
        # is rejected rather than tuned to one PDF's coordinates.
        for size in (0.025, 0.012):
            best = (quality, scale_x, scale_y, offset_x, offset_y, label_raster)
            for fx in (-size, 0.0, size):
                for fy in (-size, 0.0, size):
                    for dx in (-size * width, 0.0, size * width):
                        for dy in (-size * height, 0.0, size * height):
                            sx, sy = scale_x * (1 + fx), scale_y * (1 + fy)
                            ox, oy = offset_x + dx, offset_y + dy
                            candidate_raster, candidate_quality = render(sx, sy, ox, oy)
                            if candidate_quality > best[0]:
                                best = (candidate_quality, sx, sy, ox, oy, candidate_raster)
            quality, scale_x, scale_y, offset_x, offset_y, label_raster = best
        candidates_by_projection.append((quality, projection, label_raster, states))
    quality, projection, label_raster, states = max(candidates_by_projection, key=lambda item: item[0])
    if quality < min_quality:
        return None
    names = [None, *states.keys()]
    mark_boxes = {}
    pixels = image.astype(np.int16)
    colored = (
        (pixels[:, :, 0] - pixels[:, :, 2] > 15)
        & (pixels[:, :, 0] - pixels[:, :, 1] > 7)
    )
    colored_share = {}
    for index, name in enumerate(names[1:], start=1):
        yy, xx = np.where(label_raster == index)
        if len(xx):
            mark_boxes[name] = [int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)]
            colored_share[name] = round(float(np.count_nonzero(colored & (label_raster == index))) / len(xx), 5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segments = cv2.createLineSegmentDetector().detect(gray)[0]
    return {
        "quality": round(float(quality), 5),
        "projection": projection,
        "projection_scores": {
            name: round(float(score), 5) for score, name, _, _ in candidates_by_projection
        },
        "names": names,
        "raster": label_raster,
        "silhouette": silhouette,
        "mark_boxes": mark_boxes,
        "colored_share": colored_share,
        "leader_segments": segments.reshape(-1, 4) if segments is not None else np.empty((0, 4)),
        "image": image,
        "mainland_bbox": [x0, y0, width, height],
        "atlas_url": ATLAS_URL,
    }


def _pixel_box(box: list[float], region: list[float], image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = image_shape
    return (
        (box[0] - region[0]) * width / region[2],
        (box[1] - region[1]) * height / region[3],
        box[2] * width / region[2],
        box[3] * height / region[3],
    )


def _page_box(box: list[int], region: list[float], image_shape: tuple[int, int]) -> list[float]:
    height, width = image_shape
    return [
        round(region[0] + box[0] * region[2] / width, 5),
        round(region[1] + box[1] * region[3] / height, 5),
        round(box[2] * region[2] / width, 5),
        round(box[3] * region[3] / height, 5),
    ]


def state_for_value(
    registration: dict[str, Any], coordinates: list[float], region: list[float],
) -> tuple[str, list[float], str] | None:
    """Resolve a printed value inside a registered polygon or via a leader."""
    import cv2
    import numpy as np

    raster = registration["raster"]
    shape = raster.shape
    x, y, width, height = _pixel_box(coordinates, region, shape)
    center = (round(x + width / 2), round(y + height / 2))
    if 0 <= center[0] < shape[1] and 0 <= center[1] < shape[0]:
        index = int(raster[center[1], center[0]])
        if index:
            name = registration["names"][index]
            if registration["colored_share"].get(name, 0.0) >= 0.30:
                return name, _page_box(registration["mark_boxes"][name], region, shape), "registered_state_polygon"

    # External numeric callouts require a line ending on the filled map. Do
    # not infer an owner from nearest-state distance alone.
    gray = cv2.cvtColor(registration["image"], cv2.COLOR_BGR2GRAY)
    segments = registration["leader_segments"]
    if not len(segments):
        return None
    callout_center_y = y + height / 2
    candidates = []
    for segment in segments:
        x1, y1, x2, y2 = map(float, segment)
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if not (x - 110 <= x2 <= x + 20 and abs(y2 - callout_center_y) < 45 and x2 - x1 > 15):
            continue
        middle = (round((x1 + x2) / 2), round((y1 + y2) / 2))
        ink_patch = gray[max(0, middle[1] - 2):min(shape[0], middle[1] + 3),
                         max(0, middle[0] - 2):min(shape[1], middle[0] + 3)]
        if not (0 <= middle[0] < shape[1] and 0 <= middle[1] < shape[0]) or not ink_patch.size or ink_patch.min() > 150:
            continue  # White cartographic boundaries are not callout leaders.
        if not (0 <= x1 < shape[1] and 0 <= y1 < shape[0]):
            continue
        # Leaders can stop just outside a small projected state.
        far_x, far_y = round(x1), round(y1)
        patch = raster[max(0, far_y - 10):min(shape[0], far_y + 11),
                       max(0, far_x - 10):min(shape[1], far_x + 11)]
        counts = np.bincount(patch.ravel(), minlength=len(registration["names"]))
        counts[0] = 0
        index = int(counts.argmax())
        if counts[index] < 8:
            continue
        name = registration["names"][index]
        if registration["colored_share"].get(name, 0.0) < 0.30:
            continue
        distance = math.hypot(x2 - x, y2 - callout_center_y)
        candidates.append((distance, name))
    if not candidates:
        return None
    candidates.sort()
    if len({name for distance, name in candidates if distance <= candidates[0][0] + 8}) != 1:
        return None
    name = candidates[0][1]
    return name, _page_box(registration["mark_boxes"][name], region, shape), "registered_state_leader_line"


def colored_mark_for_value(
    image_path: Path, coordinates: list[float], region: list[float],
) -> list[float] | None:
    """Find a standalone filled raster mark behind an explicitly named value."""
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    pixels = image.astype(np.int16)
    blue = (
        (pixels[:, :, 0] - pixels[:, :, 2] > 15)
        & (pixels[:, :, 0] - pixels[:, :, 1] > 7)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blue, 8)
    x, y, width, height = _pixel_box(coordinates, region, blue.shape)
    center_x, center_y = round(x + width / 2), round(y + height / 2)
    nearby = labels[max(0, center_y - 25):min(blue.shape[0], center_y + 26),
                    max(0, center_x - 25):min(blue.shape[1], center_x + 26)]
    if not nearby.size:
        return None
    counts = np.bincount(nearby.ravel(), minlength=count)
    counts[0] = 0
    index = int(counts.argmax())
    if not index or stats[index, cv2.CC_STAT_AREA] < 100:
        return None
    return _page_box(list(map(int, stats[index, :4])), region, blue.shape)

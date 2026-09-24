"""Normalize independent PP-StructureV3 table candidates and compare ownership."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
import re
from typing import Any

from .models import PageInspection, Region


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cell_index_rows: list[list[int]] = []
        self.row: list[str] | None = None
        self.row_cell_indices: list[int] | None = None
        self.cell: list[str] | None = None
        self.current_cell_index: int | None = None
        self.cell_count = 0
        self.colspan = 1
        self.rowspan = 1
        self.pending_rowspans: dict[int, int] = {}
        self.pending_cell_indices: dict[int, int] = {}

    def _fill_pending(self) -> None:
        if self.row is None:
            return
        while self.pending_rowspans.get(len(self.row), 0) > 0:
            column = len(self.row)
            self.row.append("")
            if self.row_cell_indices is not None:
                self.row_cell_indices.append(self.pending_cell_indices[column])
            self.pending_rowspans[column] -= 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
            self.row_cell_indices = []
            self._fill_pending()
        elif tag in {"td", "th"} and self.row is not None:
            self._fill_pending()
            self.cell = []
            self.current_cell_index = self.cell_count
            self.cell_count += 1
            attributes = dict(attrs)
            self.colspan = max(1, int(attributes.get("colspan") or 1))
            self.rowspan = max(1, int(attributes.get("rowspan") or 1))
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            start = len(self.row)
            self.row.append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
            self.row.extend([""] * (self.colspan - 1))
            if self.row_cell_indices is not None and self.current_cell_index is not None:
                self.row_cell_indices.extend([self.current_cell_index] * self.colspan)
            if self.rowspan > 1:
                for column in range(start, len(self.row)):
                    self.pending_rowspans[column] = self.rowspan - 1
                    if self.current_cell_index is not None:
                        self.pending_cell_indices[column] = self.current_cell_index
            self.cell = None
            self.current_cell_index = None
        elif tag == "tr" and self.row is not None:
            self._fill_pending()
            if self.row:
                self.rows.append(self.row)
                self.cell_index_rows.append(self.row_cell_indices or [])
            self.row = None
            self.row_cell_indices = None


def html_rows(html: str) -> list[list[str]]:
    parser = _TableParser()
    parser.feed(html)
    width = max((len(row) for row in parser.rows), default=0)
    return [row + [""] * (width - len(row)) for row in parser.rows]


def html_rows_with_cell_indices(html: str) -> tuple[list[list[str]], list[list[int | None]], int]:
    parser = _TableParser()
    parser.feed(html)
    width = max((len(row) for row in parser.rows), default=0)
    return (
        [row + [""] * (width - len(row)) for row in parser.rows],
        [row + [None] * (width - len(row)) for row in parser.cell_index_rows],
        parser.cell_count,
    )


def _intersection_fraction(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    area = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return area / max(1.0, min(aw * ah, bw * bh))


def _candidate_coverage(region: list[float], candidate: list[float]) -> float:
    """Fraction of the detected table box actually covered by a region."""
    ax, ay, aw, ah = region
    bx, by, bw, bh = candidate
    area = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    return area / max(1.0, bw * bh)


def _normal_box(coordinates: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = coordinates
    left = max(0.0, min(1000.0, 1000 * x0 / width))
    top = max(0.0, min(1000.0, 1000 * y0 / height))
    right = max(left, min(1000.0, 1000 * x1 / width))
    bottom = max(top, min(1000.0, 1000 * y1 / height))
    return [left, top, right - left, bottom - top]


def _cell_key(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def page_needs_table_analysis(
    inspection: PageInspection, plan: dict[str, Any] | None,
    ocr_page: dict[str, Any],
) -> bool:
    """Select likely table pages without requiring a native parser hit."""
    if any(region.kind == "table" for region in inspection.regions):
        return True
    if plan and any(block.get("type") == "table" and len(block.get("rows") or []) >= 2
                    for block in plan.get("blocks", [])):
        return True
    for region in inspection.regions:
        if region.kind not in {"visual", "normal_text"}:
            continue
        x, y, width, height = region.coordinates
        numeric = sum(
            any(char.isdigit() for char in str(line.get("text", "")))
            for line in ocr_page.get("lines", [])
            if (x <= line["coordinates"][0] + line["coordinates"][2] / 2 <= x + width
                and y <= line["coordinates"][1] + line["coordinates"][3] / 2 <= y + height)
        )
        if numeric >= (6 if region.kind == "visual" else 8):
            return True
    return False


def table_candidates(payload: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    body = payload.get("res", payload)
    layout = [box for box in (body.get("layout_det_res") or {}).get("boxes", [])
              if str(box.get("label", "")).lower() == "table"]
    results = body.get("table_res_list") or []
    candidates = []
    for index, result in enumerate(results):
        rows, cell_indices, cell_count = html_rows_with_cell_indices(
            str(result.get("pred_html") or ""),
        )
        if len(rows) < 2:
            continue
        bbox = result.get("bbox") or result.get("table_bbox")
        if bbox is None and index < len(layout):
            bbox = layout[index].get("coordinate")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            box = _normal_box([float(v) for v in bbox], width, height)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if box[2] <= 0 or box[3] <= 0:
            continue
        raw_cell_boxes = result.get("cell_box_list") or []
        cell_coordinates: list[list[list[float] | None]] = []
        if len(raw_cell_boxes) == cell_count:
            try:
                normalized_cells = [
                    _normal_box([float(value) for value in cell], width, height)
                    for cell in raw_cell_boxes
                ]
                if all(cell[2] > 0 and cell[3] > 0 for cell in normalized_cells):
                    cell_coordinates = [
                        [normalized_cells[cell_index] if cell_index is not None else None
                         for cell_index in row]
                        for row in cell_indices
                    ]
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                pass
        candidates.append({
            "index": index, "coordinates": box, "rows": rows,
            "cell_coordinates": cell_coordinates,
            "layout_score": float(layout[index].get("score", 0.0)) if index < len(layout) else None,
            "structure_status": "grid" if len(rows[0]) >= 2 else "layout_only",
        })
    return candidates


def screen_table_candidates_against_ocr(
    candidates: list[dict[str, Any]], ocr_lines: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject table transcripts contradicted by numeric OCR in their own crop.

    PP-Structure can detect the correct grid but attach HTML text from a
    different part of the page. This guard applies only when both sides have
    enough distinct numbers for a meaningful comparison.
    """
    number_pattern = re.compile(r"(?<![A-Za-z])[-+]?\$?(\d[\d,]*(?:\.\d+)?)")

    def numeric_keys(text: str) -> set[str]:
        keys: set[str] = set()
        for match in number_pattern.finditer(text):
            raw = match.group(1).replace(",", "")
            if len(raw.replace(".", "")) < 2:
                continue
            try:
                keys.add(str(Decimal(raw).normalize()))
            except InvalidOperation:
                continue
        return keys

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        x, y, width, height = (float(value) for value in candidate["coordinates"])
        nearby = [line for line in ocr_lines
                  if (x - 8 <= float(line["coordinates"][0])
                      + float(line["coordinates"][2]) / 2 <= x + width + 8)
                  and (y - 8 <= float(line["coordinates"][1])
                       + float(line["coordinates"][3]) / 2 <= y + height + 8)]
        table_numbers = numeric_keys(" ".join(
            str(cell or "") for row in candidate.get("rows", []) for cell in row
        ))
        crop_numbers = numeric_keys(" ".join(str(line.get("text", "")) for line in nearby))
        matched = table_numbers & crop_numbers
        if (len(table_numbers) >= 3 and len(crop_numbers) >= 3
                and len(matched) < 2 and len(matched) / len(table_numbers) < 0.25):
            rejected.append({
                "candidate_index": candidate["index"],
                "reason": "table transcript numbers disagree with independent OCR inside grid",
                "table_numeric_count": len(table_numbers),
                "crop_numeric_count": len(crop_numbers),
                "matched_numeric_count": len(matched),
            })
            continue
        accepted.append(candidate)
    return accepted, rejected


def _split_multi_table_regions(inspection: PageInspection, candidates: list[dict[str, Any]]) -> set[int]:
    """Split a broad empty table region around separate Paddle grid detections.

    A page detector can wrap several small grids in one visual or table box. Keep native
    rows authoritative, and split only when multiple independently detected
    grids are compact, numeric, and spatially separate. Combined detections
    covering those grids are duplicates, not additional tables.
    """
    used: set[int] = set()
    regions: list[Region] = []
    for region in inspection.regions:
        if region.kind not in {"table", "visual"} or region.metadata.get("rows"):
            regions.append(region)
            continue
        parent_area = region.coordinates[2] * region.coordinates[3]
        eligible = [item for item in candidates
                    if item["index"] not in used
                    and item.get("structure_status") == "grid"
                    and (item.get("layout_score") or 0) >= 0.5
                    and len(item.get("rows") or []) >= 2
                    and sum(bool(re.search(r"\d", str(cell)))
                            for row in item["rows"] for cell in row) >= 3
                    and _candidate_coverage(region.coordinates, item["coordinates"]) >= 0.9
                    and item["coordinates"][2] * item["coordinates"][3] <= 0.25 * parent_area]
        selected: list[dict[str, Any]] = []
        for item in sorted(eligible, key=lambda candidate: -(candidate.get("layout_score") or 0)):
            if all(_intersection_fraction(item["coordinates"], other["coordinates"]) < 0.2
                   for other in selected):
                selected.append(item)
        if len(selected) < 2:
            regions.append(region)
            continue
        # Retain the original bounds as a lower-priority text owner. Table
        # regions win overlapping OCR lines; headings and notes outside them
        # remain grouped instead of becoming isolated fallback blocks.
        region.kind = "normal_text"
        region.classification_method = "residual text around PaddleOCR tables"
        region.metadata["paddle_split_residual"] = [item["index"] for item in selected]
        regions.append(region)
        for offset, item in enumerate(sorted(selected, key=lambda candidate: (
                candidate["coordinates"][1], candidate["coordinates"][0]))):
            regions.append(Region(
                region_id=f"p{inspection.page:03d}-paddle-c{item['index']:03d}",
                page=inspection.page, kind="table", coordinates=item["coordinates"],
                reading_order=region.reading_order + offset,
                classification_method="PaddleOCR PP-StructureV3 table detection",
                confidence=min(0.75, item["layout_score"]),
                metadata={
                    "rows": item["rows"], "rows_source": "paddle",
                    "cell_coordinates": item.get("cell_coordinates", []),
                    "paddle_table_review": {
                        "status": "candidate", "candidate_index": item["index"],
                        "overlap": 1.0, "row_count": len(item["rows"]),
                        "column_count": len(item["rows"][0]),
                    },
                },
            ))
        used.update(item["index"] for item in candidates
                    if _candidate_coverage(region.coordinates, item["coordinates"]) >= 0.5
                    and any(_intersection_fraction(item["coordinates"], chosen["coordinates"]) >= 0.8
                            for chosen in selected))
    inspection.regions = regions
    return used


def apply_paddle_tables(inspection: PageInspection, candidates: list[dict[str, Any]]) -> None:
    """Run every native table through Paddle and rescue strong OCR table regions.

    Native cells remain authoritative on disagreement. Image candidates need
    both a Paddle table and an inspected region; they remain review-required.
    """
    used = _split_multi_table_regions(inspection, candidates)
    for region in inspection.regions:
        if region.kind != "table":
            continue
        matches = [(_intersection_fraction(region.coordinates, item["coordinates"]), item)
                   for item in candidates
                   if item["index"] not in used
                   and _candidate_coverage(region.coordinates, item["coordinates"]) >= 0.50]
        if not matches:
            region.metadata["paddle_table_review"] = {"status": "not_detected"}
            continue
        overlap, item = max(matches, key=lambda pair: pair[0])
        if overlap < 0.45:
            region.metadata["paddle_table_review"] = {"status": "not_detected"}
            continue
        used.add(item["index"])
        native = region.metadata.get("rows") or []
        if item.get("structure_status") == "layout_only":
            region.metadata["paddle_table_review"] = {
                "status": "layout_only", "candidate_index": item["index"],
                "overlap": round(overlap, 4),
            }
            continue
        if not native:
            region.metadata["rows"] = item["rows"]
            region.metadata["rows_source"] = "paddle"
            region.metadata["cell_coordinates"] = item.get("cell_coordinates", [])
            region.metadata["paddle_table_review"] = {
                "status": "candidate", "candidate_index": item["index"],
                "overlap": round(overlap, 4), "row_count": len(item["rows"]),
                "column_count": len(item["rows"][0]),
            }
            continue
        same_shape = (len(native) == len(item["rows"]) and
                      all(len(left) == len(right) for left, right in zip(native, item["rows"])))
        disagreements = (sum(_cell_key(left) != _cell_key(right)
                             for native_row, paddle_row in zip(native, item["rows"])
                             for left, right in zip(native_row, paddle_row))
                         if same_shape else None)
        region.metadata["paddle_table_review"] = {
            "status": "matched" if same_shape and disagreements == 0 else "disagrees",
            "candidate_index": item["index"], "overlap": round(overlap, 4),
            "row_count": len(item["rows"]), "column_count": len(item["rows"][0]),
            "cell_disagreements": disagreements,
        }
    for item in candidates:
        if item["index"] in used:
            continue
        if item["layout_score"] is not None and item["layout_score"] < 0.5:
            continue
        matches = [(_intersection_fraction(region.coordinates, item["coordinates"]), region)
                   for region in inspection.regions
                   if region.kind in {"visual", "normal_text"}
                   and _candidate_coverage(region.coordinates, item["coordinates"]) >= 0.50]
        populated_numeric = sum(bool(re.search(r"\d", cell)) for row in item["rows"] for cell in row)
        if populated_numeric < 3:
            continue
        if matches:
            overlap, region = max(matches, key=lambda pair: pair[0])
            if overlap < 0.75:
                continue
            region.kind = "table"
            region.classification_method = "PaddleOCR PP-StructureV3 table detection"
            region.confidence = min(0.75, item["layout_score"] or 0.65)
        else:
            # One inspected visual can contain several detected tables, while
            # a tiny heading can sit inside a much larger table box. Give an
            # independently detected grid its own coordinates in both cases.
            if (item.get("structure_status") == "layout_only"
                    or item["layout_score"] is None or item["layout_score"] < 0.75
                    or any(region.kind == "table" and region.metadata.get("rows_source") != "paddle"
                           and _intersection_fraction(region.coordinates, item["coordinates"]) >= 0.2
                           for region in inspection.regions)):
                continue
            region = Region(
                region_id=f"p{inspection.page:03d}-paddle-c{item['index']:03d}",
                page=inspection.page, kind="table", coordinates=item["coordinates"],
                reading_order=max((existing.reading_order for existing in inspection.regions), default=0) + 1,
                classification_method="PaddleOCR PP-StructureV3 table detection",
                confidence=min(0.75, item["layout_score"]),
            )
            inspection.regions.append(region)
            overlap = 1.0
        if item.get("structure_status") != "layout_only":
            region.metadata["rows"] = item["rows"]
            region.metadata["rows_source"] = "paddle"
            region.metadata["cell_coordinates"] = item.get("cell_coordinates", [])
        region.metadata["paddle_table_review"] = {
            "status": "layout_only" if item.get("structure_status") == "layout_only" else "candidate",
            "candidate_index": item["index"],
            "overlap": round(overlap, 4), "row_count": len(item["rows"]),
            "column_count": len(item["rows"][0]),
        }
        used.add(item["index"])

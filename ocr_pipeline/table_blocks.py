"""Build table and comparison-panel blocks and ground their cells in OCR."""

from __future__ import annotations

import math
from pathlib import Path
import re
import statistics
from typing import Any

from .common import NUMBER, _TABLE_SCALAR, _center, _numeric_value, _ocr_text, _provenance, clean_text
from .comparison_layout import reconstruct_comparison_panel
from .models import Region, SourceBlock


def _table_blocks(
    document_id: str, source_hash: str, region: Region, lines: list[dict[str, Any]], image: Path,
) -> list[SourceBlock]:
    rows = region.metadata.get("rows") or []
    table_title = region.metadata.get("title")
    row_sections = region.metadata.get("row_sections") or []
    cell_coordinates = region.metadata.get("cell_coordinates") or []
    cell_evidence_ids = region.metadata.get("cell_evidence_ids") or []
    paddle_rows = region.metadata.get("rows_source") == "paddle"
    ocr_geometry_rows = region.metadata.get("rows_source") == "ocr_geometry"
    native = bool(not paddle_rows and not ocr_geometry_rows and rows
                  and max((len(row) for row in rows), default=0) >= 2)
    if not rows:
        ordered = sorted(lines, key=lambda line: (line["coordinates"][1], line["coordinates"][0]))
        grouped: list[list[dict[str, Any]]] = []
        for line in ordered:
            if not grouped or abs(line["coordinates"][1] - grouped[-1][0]["coordinates"][1]) > 16:
                grouped.append([line])
            else:
                grouped[-1].append(line)
        # Anchor wide sparse OCR rows to printed header positions. Sequential
        # padding shifts values left when earlier cells are blank; narrow prose
        # groups are too ambiguous to use as positional headers.
        header_group = next((
            group for group in grouped[:4]
            if len(group) >= 4
            and sum(bool(re.search(r"[A-Za-z]", str(item["text"]))) for item in group) >= 3
            and sum(bool(re.fullmatch(r"[$€£]?[-+]?\d[\d.,%xX]*", str(item["text"]).strip())) for item in group) <= 1
            and max(item["coordinates"][0] for item in group)
            - min(item["coordinates"][0] for item in group) >= 30
        ), None)
        if header_group:
            anchors = sorted(item["coordinates"][0] for item in header_group)
            rows = []
            for group in grouped:
                row: list[str | None] = [None] * len(anchors)
                for item in sorted(group, key=lambda item: item["coordinates"][0]):
                    column = min(range(len(anchors)), key=lambda index: abs(item["coordinates"][0] - anchors[index]))
                    row[column] = f"{row[column]}\n{item['text']}" if row[column] else item["text"]
                rows.append(row)
        else:
            rows = [[item["text"] for item in sorted(group, key=lambda item: item["coordinates"][0])] for group in grouped]
    width = max((len(row) for row in rows), default=0)
    rows = [list(row) + [None] * (width - len(row)) for row in rows]
    paddle_value_first = False
    if paddle_rows and width == 2 and len(rows) >= 3:
        first_numeric = sum(bool(NUMBER.search(str(row[0] or ""))) for row in rows)
        second_numeric = sum(bool(NUMBER.search(str(row[1] or ""))) for row in rows)
        second_labels = sum(bool(re.search(r"[A-Za-z]", str(row[1] or ""))) for row in rows)
        if (first_numeric >= max(3, math.ceil(0.6 * len(rows)))
                and second_numeric <= len(rows) // 3
                and second_labels >= math.ceil(0.7 * len(rows))):
            rows = [[row[1], row[0]] for row in rows]
            if cell_coordinates:
                cell_coordinates = [[row[1], row[0]] for row in cell_coordinates]
            paddle_value_first = True
    # OCR commonly places a spanning title before the actual column header.
    # Promote the next row only when it looks like a set of printed labels.
    if len(rows) >= 3 and width >= 2:
        first = [str(value or "").strip() for value in rows[0]]
        second = [str(value or "").strip() for value in rows[1]]
        first_count = sum(bool(value) for value in first)
        second_count = sum(bool(value) for value in second)
        second_numeric = sum(bool(re.fullmatch(r"[$€£]?[-+]?\d[\d.,%xX]*", value)) for value in second if value)
        if first_count == 1 and second_count >= max(2, width // 2) and second_numeric <= 1:
            table_title = table_title or next(value for value in first if value)
            rows = rows[1:]
            row_sections = row_sections[1:] if row_sections else row_sections
            cell_coordinates = cell_coordinates[1:] if cell_coordinates else cell_coordinates
    # A two-column label/value list often has no printed header. Preserve its
    # first observation instead of silently treating it as column names.
    inferred_key_value_header = False
    if width == 2 and len(rows) >= 3 and not region.metadata.get("key_value_split"):
        first_label = str(rows[0][0] or "").strip()
        first_value = str(rows[0][1] or "").strip()
        value_count = sum(bool(NUMBER.search(str(row[1] or ""))) for row in rows)
        if (first_label and not NUMBER.fullmatch(first_label)
                and bool(NUMBER.search(first_value))
                and (paddle_value_first or value_count >= len(rows) - 1)):
            rows = [["Field", "Value"], *rows]
            row_sections = [None, *row_sections] if row_sections else row_sections
            if cell_coordinates:
                cell_coordinates = [[None, None], *cell_coordinates]
            inferred_key_value_header = True
    if len(rows) == 1 and width >= 2:
        # A letter-spaced banner can look like a one-row table to the PDF
        # grid finder. Positioned native glyphs preserve its actual words.
        glyphs = region.metadata.get("native_panel_words") or []
        if (region.coordinates[3] <= 45 and len(glyphs) >= 8
                and all(len(str(glyph.get("text", ""))) == 1
                        and str(glyph.get("text", "")).isalpha()
                        and isinstance(glyph.get("coordinates"), list)
                        and len(glyph["coordinates"]) == 4 for glyph in glyphs)):
            glyphs = sorted(glyphs, key=lambda glyph: float(glyph["coordinates"][0]))
            centers_y = [_center(glyph["coordinates"])[1] for glyph in glyphs]
            gaps = [float(right["coordinates"][0])
                    - (float(left["coordinates"][0]) + float(left["coordinates"][2]))
                    for left, right in zip(glyphs, glyphs[1:])]
            typical_gap = statistics.median(gap for gap in gaps if gap >= 0) if all(gap >= 0 for gap in gaps) else 0
            if (max(centers_y) - min(centers_y) <= 5 and typical_gap > 0
                    and max(gaps) >= typical_gap * 1.7):
                words = [str(glyphs[0]["text"])]
                for glyph, gap in zip(glyphs[1:], gaps):
                    if gap >= typical_gap * 1.7:
                        words.append("")
                    words[-1] += str(glyph["text"])
                if len(words) >= 2 and all(len(word) >= 2 for word in words):
                    heading_text = " ".join(words)
                    return [SourceBlock(
                        document_id=document_id, type="heading", page=region.page,
                        block_id=f"{region.region_id}-spaced-heading",
                        content={"text": heading_text, "evidence_text": {
                            "selected": "native", "native": heading_text,
                            "ocr": _ocr_text(lines), "token_agreement": None,
                        }},
                        coordinates=region.coordinates,
                        extraction_method=["native PDF positioned glyphs", "Python glyph-gap reconstruction"],
                        confidence=min(region.confidence, 0.9), validation_status="passed",
                        provenance=_provenance(source_hash, region, lines, image),
                        semantic_role="section_heading", heading_level=1,
                    )]
        layout = reconstruct_comparison_panel(region.metadata.get("native_panel_words") or [])
        if layout:
            # Positioned-word ownership has no independent structural check yet.
            comparison_warnings = ["comparison layout is not independently confirmed by a vision review"]
            content = {
                "title": layout["title"], "sections": layout["sections"],
                "lane_count": layout["lane_count"], "claim_count": layout["claim_count"],
                "bullet_anchor_coordinates": layout["bullet_anchor_coordinates"],
                "vision_review": None,
            }
            return [SourceBlock(
                document_id=document_id, type="comparison_panel", page=region.page,
                block_id=f"{region.region_id}-comparison-panel",
                content=content, coordinates=region.coordinates,
                extraction_method=["native PDF positioned words", "Python lane/heading/bullet reconstruction"],
                confidence=0.72, validation_status="needs_review",
                errors=[], warnings=comparison_warnings,
                provenance=_provenance(source_hash, region, lines, image),
            )]
        sections = []
        for index, value in enumerate(rows[0], 1):
            raw_text = clean_text(str(value or ""))
            first_line = str(value or "").split("\n", 1)[0].strip() if raw_text else None
            sections.append({
                "section_id": f"s{index:03d}", "title": first_line,
                "text": raw_text, "structure_complete": False,
            })
        comparison_methods = ["native PDF layout", "Python section reconstruction"]
        comparison_warnings = [
            "panel columns were preserved, but nested subsections could not be separated reliably"
        ]
        return [SourceBlock(
            document_id=document_id, type="comparison_panel", page=region.page,
            block_id=f"{region.region_id}-comparison-panel",
            content={"title": region.metadata.get("title"), "sections": sections},
            coordinates=region.coordinates,
            extraction_method=comparison_methods,
            confidence=min(region.confidence, 0.60), validation_status="needs_review",
            errors=[], warnings=comparison_warnings,
            provenance=_provenance(source_hash, region, lines, image),
        )]
    # Grid extraction can collapse a section label and the next data
    # row's label into one cell. Split only when separate OCR phrases exactly
    # reconstruct that native label and the final phrase aligns with the row's
    # printed numeric cells. Keep the earlier phrase as the row section.
    if cell_coordinates:
        ocr_lines = [line for line in lines if "-ocr-" in str(line.get("evidence_id", ""))]
        for row_index, row in enumerate(rows[1:], 1):
            if row_index >= len(cell_coordinates) or not cell_coordinates[row_index]:
                continue
            label_box = cell_coordinates[row_index][0]
            if not label_box or not row[0]:
                continue
            value_box = next((
                cell_coordinates[row_index][column]
                for column in range(1, min(len(row), len(cell_coordinates[row_index])))
                if re.search(r"\d", str(row[column] or ""))
                and cell_coordinates[row_index][column]
            ), None)
            if not value_box:
                continue
            lx, ly, lw, lh = (float(value) for value in label_box)
            nearby = sorted((
                line for line in ocr_lines
                if lx - 15 <= _center(line["coordinates"])[0] <= lx + lw + 20
                and ly - 15 <= _center(line["coordinates"])[1] <= ly + lh + 15
            ), key=lambda line: float(line["coordinates"][1]))
            folded = lambda value: re.sub(r"[^a-z0-9]", "", str(value).casefold())
            matches = [nearby[start:end] for start in range(len(nearby))
                       for end in range(start + 2, min(len(nearby), start + 3) + 1)
                       if folded(" ".join(str(line["text"]) for line in nearby[start:end])) == folded(row[0])]
            if len(matches) != 1:
                continue
            nearby = matches[0]
            last_y = _center(nearby[-1]["coordinates"])[1]
            previous_y = _center(nearby[-2]["coordinates"])[1]
            if (last_y - previous_y < 8
                    or abs(last_y - _center(value_box)[1]) > 10):
                continue
            section = " ".join(str(line["text"]).strip() for line in nearby[:-1])
            row[0] = str(nearby[-1]["text"]).strip()
            if len(row_sections) < len(rows):
                row_sections = list(row_sections) + [None] * (len(rows) - len(row_sections))
            row_sections[row_index] = section

        # Some PDF fonts expose a dollar glyph as native "S". Accept a
        # correction only from an independently positioned OCR symbol.
        for row_index, row in enumerate(rows[1:], 1):
            if row_index >= len(cell_coordinates):
                continue
            for column in range(1, len(row) - 1):
                if str(row[column] or "").strip() != "S":
                    continue
                if column >= len(cell_coordinates[row_index]):
                    continue
                symbol_box = cell_coordinates[row_index][column]
                if not symbol_box:
                    continue
                sx, sy, sw, sh = (float(value) for value in symbol_box)
                symbols = [line for line in ocr_lines if str(line.get("text", "")).strip() == "$"
                           and sx - 10 <= _center(line["coordinates"])[0] <= sx + sw + 10
                           and sy - 10 <= _center(line["coordinates"])[1] <= sy + sh + 10]
                if len(symbols) == 1:
                    row[column] = "$"
    # PDF and image table parsers sometimes split a printed currency sign into
    # its own cell. Keep the grid width but attach that sign to the adjacent
    # numeric token; a bare "$" is never a value.
    for row in rows[1:]:
        for column_index in range(1, len(row) - 1):
            current = str(row[column_index] or "").strip()
            following = str(row[column_index + 1] or "").strip()
            if current in {"$", "€", "£"} and re.fullmatch(
                r"\d[\d,]*(?:\.\d+)?\s*[$€£]?", following,
            ):
                row[column_index] = None
                row[column_index + 1] = current + following
            elif (current and current[-1] in "$€£"
                  and current not in {"$", "€", "£"}
                  and re.fullmatch(r"\d[\d,]*(?:\.\d+)?", following)):
                symbol = current[-1]
                row[column_index] = current[:-1].rstrip()
                row[column_index + 1] = symbol + following
    # A structure parser can prepend a stray glyph to an otherwise intact
    # scalar. Use a unique positioned OCR scalar only when its numeric token
    # exactly matches the parser's token in that same physical cell.
    if cell_coordinates:
        for row_index, row in enumerate(rows[1:], 1):
            if row_index >= len(cell_coordinates):
                continue
            for column_index in range(1, min(len(row), len(cell_coordinates[row_index]))):
                raw = str(row[column_index] or "").strip()
                box = cell_coordinates[row_index][column_index]
                if not raw or not box or _TABLE_SCALAR.fullmatch(raw):
                    continue
                numbers = re.findall(r"\d[\d,.]*", raw)
                if len(numbers) != 1:
                    continue
                x, y, width_box, height_box = (float(value) for value in box)
                candidates = [line for line in lines
                              if "-ocr-" in str(line.get("evidence_id", ""))
                              and float(line.get("confidence", 0)) >= 0.90
                              and _TABLE_SCALAR.fullmatch(str(line.get("text", "")).strip())
                              and re.findall(r"\d[\d,.]*", str(line["text"])) == numbers
                              and x - 10 <= _center(line["coordinates"])[0] <= x + width_box + 10
                              and y - 10 <= _center(line["coordinates"])[1] <= y + height_box + 10]
                if len(candidates) == 1:
                    row[column_index] = str(candidates[0]["text"]).strip()
    reconciliation = _table_total_reconciliation(rows)
    reconciliation_failed = bool(reconciliation and not reconciliation["passed"])
    parent_id = f"{region.region_id}-table"
    warnings = [] if native else (["table reconstructed from PaddleOCR structure; independently review cell values"]
                                 if paddle_rows else ["table reconstructed from OCR geometry"])
    if inferred_key_value_header:
        warnings.append("two-column key/value list has an inferred, unprinted header")
    if paddle_value_first:
        warnings.append("PaddleOCR value-first key/value rows were oriented by column content")
    if region.metadata.get("key_value_split"):
        warnings.append("one-column key/value list was split at each colon; the header is inferred")
    paddle_review = region.metadata.get("paddle_table_review")
    if isinstance(paddle_review, dict):
        if paddle_review.get("status") == "disagrees":
            warnings.append("PaddleOCR and native PDF table cells or dimensions disagree; native cells retained")
        elif paddle_review.get("status") == "not_detected":
            warnings.append("PaddleOCR did not detect the native PDF table")
        elif paddle_review.get("status") == "layout_only":
            warnings.append("PaddleOCR detected a table but returned no usable cell grid; OCR geometry retained")
    valid_shape = len(rows) >= 2 and width >= 2
    methods = ["native PDF table parser"] if native else (["PaddleOCR PP-StructureV3"] if paddle_rows
               else ["RapidOCR", "PP-OCRv6"])
    if isinstance(paddle_review, dict):
        methods.append("PaddleOCR PP-StructureV3 table review")
    methods.append("Python table reconstruction and validation")
    headers = ["" if value is None else str(value).strip() for value in rows[0]] if rows else []
    column_kinds: list[str] = []
    for column_index, header in enumerate(headers):
        values = [str(row[column_index] or "") for row in rows[1:] if column_index < len(row)]
        joined = " ".join(values)
        if column_index == 0:
            kind_hint = "text"
        elif "%" in header or "%" in joined:
            kind_hint = "percent"
        elif any(symbol in joined or symbol in header for symbol in "$€£"):
            kind_hint = "currency"
        elif re.search(r"\bx\b", joined, re.I):
            kind_hint = "multiple"
        else:
            kind_hint = "number"
        column_kinds.append(kind_hint)
    columns = [
        {
            "column_id": f"c{index + 1:03d}", "label": label or None, "value_kind": column_kinds[index],
            "coordinates": cell_coordinates[0][index] if cell_coordinates and index < len(cell_coordinates[0]) else None,
            "evidence_ids": [f"{region.region_id}-native-table-r000-c{index:03d}"]
            if (native and cell_coordinates and index < len(cell_coordinates[0])
                and cell_coordinates[0][index]) else [],
        }
        for index, label in enumerate(headers)
    ]
    typed_rows = []
    ambiguous_table_cells = 0
    for row_index, row in enumerate(rows[1:], 1):
        label = "" if not row or row[0] is None else str(row[0]).strip()
        cells = []
        for column_index, value in enumerate(row[1:], 1):
            raw = None if value is None or str(value).strip() == "" else str(value).strip()
            state = "blank" if raw is None else "dash" if raw in {"-", "–", "—"} else "present"
            if raw and not _TABLE_SCALAR.fullmatch(raw):
                # Any table route can merge neighboring cells (or return
                # dates, addresses, and descriptions). Keep the printed text,
                # but do not turn its first number into a trusted scalar fact.
                numeric_value = unit = normalized_value = None
                if re.search(r"\d", raw):
                    ambiguous_table_cells += 1
            else:
                numeric_value, unit, normalized_value = _numeric_value(raw or "")
            if state != "present":
                numeric_value = unit = normalized_value = None
            elif (not paddle_rows and column_index < len(column_kinds)
                  and column_kinds[column_index] == "currency" and numeric_value is not None):
                unit = "USD"
                normalized_value = numeric_value
            cells.append({
                "column_id": f"c{column_index + 1:03d}", "raw_value": raw,
                "value_state": state, "numeric_value": numeric_value,
                "unit": unit, "normalized_value": normalized_value,
                "coordinates": (
                    cell_coordinates[row_index][column_index]
                    if row_index < len(cell_coordinates) and column_index < len(cell_coordinates[row_index])
                    else None
                ),
                "evidence_ids": (
                    list(cell_evidence_ids[row_index][column_index])
                    if (ocr_geometry_rows and row_index < len(cell_evidence_ids)
                        and column_index < len(cell_evidence_ids[row_index]) and raw is not None)
                    else [f"{region.region_id}-native-table-r{row_index:03d}-c{column_index:03d}"]
                    if (native and row_index < len(cell_coordinates)
                        and column_index < len(cell_coordinates[row_index])
                        and cell_coordinates[row_index][column_index] and raw is not None)
                    else []
                ),
                "grounding_status": (
                    "ocr_value_and_row" if ocr_geometry_rows and raw is not None
                    and row_index < len(cell_evidence_ids)
                    and column_index < len(cell_evidence_ids[row_index])
                    and bool(cell_evidence_ids[row_index][0])
                    and bool(cell_evidence_ids[row_index][column_index])
                    else "native_positioned" if native and raw is not None
                    and row_index < len(cell_coordinates)
                    and column_index < len(cell_coordinates[row_index])
                    and bool(cell_coordinates[row_index][0])
                    and bool(cell_coordinates[row_index][column_index])
                    else "unverified"
                ),
            })
        typed_row = {
            "row_id": f"r{row_index:03d}",
            "section": row_sections[row_index] if row_index < len(row_sections) else None,
            "label": label or None, "cells": cells,
        }
        if ocr_geometry_rows:
            typed_row["label_evidence_ids"] = (
                list(cell_evidence_ids[row_index][0])
                if row_index < len(cell_evidence_ids) and cell_evidence_ids[row_index] else []
            )
        typed_row["label_coordinates"] = (
            cell_coordinates[row_index][0]
            if row_index < len(cell_coordinates) and cell_coordinates[row_index] else None
        )
        typed_rows.append(typed_row)
    misread_separators = _reinterpret_period_thousands(typed_rows)
    if misread_separators:
        warnings.append(
            f"{misread_separators} table cells print a period where their row and column use comma "
            "thousands separators; read as thousands (likely an OCR misread of ',')"
        )
    unsupported_cells = sum(
        cell["grounding_status"] == "unverified" and cell["raw_value"] is not None
        for row in typed_rows for cell in row["cells"]
    )
    if unsupported_cells:
        warnings.append(
            f"{unsupported_cells} table cells lack value-and-owner evidence; treat their values as candidates"
        )
    if ambiguous_table_cells:
        warnings.append(
            f"{ambiguous_table_cells} table cells contain non-scalar numeric text; "
            "raw text retained without a numeric value"
        )
    errors = (([] if valid_shape else ["table does not contain at least two rows and two columns"])
              + (["table subtotals do not reconcile with the final total"] if reconciliation_failed else [])
              + _table_structure_errors(headers, rows) + _value_placement_errors(typed_rows))
    parent = SourceBlock(
        document_id=document_id, type="table", page=region.page, block_id=parent_id,
        content={
            "title": table_title, "columns": columns, "rows": typed_rows,
            "row_count": len(typed_rows), "column_count": width,
            "reconciliation": reconciliation,
        },
        coordinates=region.coordinates, extraction_method=methods, confidence=region.confidence if native else 0.55,
        validation_status="needs_review" if _table_review_reasons(warnings, errors, typed_rows) else "passed",
        errors=errors, warnings=warnings,
        provenance=_provenance(source_hash, region, lines, image),
    )
    return [parent]


# Notes about how a table was read. None of them says a printed value may be
# wrong or misplaced, so none of them holds a table back on its own.
_TABLE_NOTES = (
    re.compile(r"table reconstructed from (?:PaddleOCR structure|OCR geometry)"),
    re.compile(r"\d+ table cells contain non-scalar numeric text"),
    re.compile(r"PaddleOCR did not detect the native PDF table"),
    re.compile(r"PaddleOCR detected a table but returned no usable cell grid"),
    re.compile(r"PaddleOCR value-first key/value rows were oriented"),
    re.compile(r"(?:one|two)-column key/value list"),
)


def _table_review_reasons(warnings: list[str], errors: list[str], rows: list[dict[str, Any]]) -> list[str]:
    """What still keeps a table from passing: errors and warnings that are not mere reading notes.

    A table passes when every printed value has positioned evidence for its row
    and column and nothing contradicts that. PaddleOCR disagreeing with the PDF
    text layer does not count against a table whose every cell the text layer positions.
    """
    cells = [cell for row in rows for cell in row.get("cells", []) if cell.get("raw_value") is not None]
    positioned = bool(cells) and all(cell.get("grounding_status") == "native_positioned" for cell in cells)
    reasons = list(errors)
    for warning in warnings:
        if any(note.match(warning) for note in _TABLE_NOTES):
            continue
        if positioned and warning.startswith("PaddleOCR and native PDF table cells or dimensions disagree"):
            continue
        reasons.append(warning)
    if any(cell.get("grounding_status") == "unverified" for cell in cells):
        reasons.append("table cells lack value-and-owner evidence")
    return reasons


_AMOUNT = re.compile(r"(?P<currency>[$€£])\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s?(?P<percent>%)|\d{1,3}(?:,\d{3})+(?:\.\d+)?")
# What prints between two amounts that belong together: a spaced range dash or
# word, a slash, a bar, or an opening parenthesis for an annotation.
_JOINED = re.compile(r"\s(?:to|vs|[-–—])\s|[/|(]", re.I)


def _amount_kinds(text: str) -> list[str]:
    return ["currency" if match["currency"] else "percent" if match["percent"] else "number"
            for match in _AMOUNT.finditer(str(text or ""))]


def _value_placement_errors(rows: list[dict[str, Any]]) -> list[str]:
    """Detect printed values that sit in the wrong place of an assembled table.

    Three shapes betray a broken grid: a cell holding two amounts of the same
    kind with nothing printed between them (two rows or columns merged), a row
    label holding several amounts (the row's cells collapsed into its label),
    and a row label that is itself an amount while most labels are words (a
    value moved into the label column).
    """
    errors: list[str] = []
    merged_cells = sum(
        1 for row in rows for cell in row.get("cells", [])
        if len(kinds := _amount_kinds(cell.get("raw_value"))) >= 2 and len(set(kinds)) == 1
        and not _JOINED.search(str(cell.get("raw_value")).strip().lstrip("("))
    )
    if merged_cells:
        errors.append(f"{merged_cells} table cells hold several printed amounts, so neighboring cells were merged")
    labels = [str(row.get("label") or "").strip() for row in rows]
    crowded = sum(len(_amount_kinds(label)) >= 3 or any(_amount_kinds(label).count(kind) >= 2 for kind in ("currency", "percent"))
                  for label in labels)
    if crowded:
        errors.append(f"{crowded} row labels hold several printed amounts, so the row's cells were merged into its label")
    value_labels = [label for label in labels if label and _AMOUNT.fullmatch(label.replace(" ", ""))]
    word_labels = [label for label in labels if re.search(r"[A-Za-z]{3,}", label)]
    if value_labels and len(word_labels) >= len(labels) / 2:
        errors.append(f"{len(value_labels)} row labels are printed amounts among word labels, so values moved into the label column")
    return errors


_KEY_VALUE_LINE = re.compile(r"(?P<key>[^:\n]{2,60}?):\s+(?P<value>[^:\n]+)")


def _filled_data_cells(rows: list[list[Any]] | None) -> int:
    """Count non-empty cells to the right of the row-label column."""
    return sum(1 for row in rows or [] for value in row[1:] if str(value or "").strip())


def _split_key_value_rows(
    rows: list[list[Any]], cell_coordinates: list[list[Any]],
) -> tuple[str | None, list[list[str]], list[list[Any]]] | None:
    """Split a one-column list of "Key: Value" lines into a two-column table.

    Table finders often return a bordered fact list (Roof Type: Asphalt) as a
    single column. Each printed line becomes a row whose label is the text
    before the colon and whose value is the rest, both verbatim, positioned at
    the source cell's box. A leading line without a colon is the list title.
    Returns None unless every other line is exactly one key/value pair.
    """
    if not rows or any(sum(bool(str(value or "").strip()) for value in row) > 1 for row in rows):
        return None
    entries: list[tuple[str, int]] = []
    for index, row in enumerate(rows):
        text = next((str(value) for value in row if str(value or "").strip()), "")
        entries.extend((line.strip(), index) for line in text.splitlines() if line.strip())
    title = entries.pop(0)[0] if entries and ":" not in entries[0][0] else None
    matches = [(_KEY_VALUE_LINE.fullmatch(text), index) for text, index in entries]
    if len(matches) < 2 or any(match is None for match, _index in matches):
        return None
    split_rows = [["Field", "Value"]] + [[match["key"].strip(), match["value"].strip()] for match, _index in matches]
    boxes: list[list[Any]] = [[None, None]]
    for _match, index in matches:
        box = cell_coordinates[index][0] if index < len(cell_coordinates) and cell_coordinates[index] else None
        boxes.append([box, box])
    return title, split_rows, boxes


def _table_backdrop(region: Region, lines: list[dict[str, Any]], tables: list[Region]) -> Region | None:
    """Return the parsed table whose printed image this visual region is, if any.

    Slide decks often draw a table as an image and lay its text over it. The
    image lies inside the parsed table and its OCR repeats the table's cells, so
    extracting it again would duplicate the table.
    """
    x, y, width, height = region.coordinates
    ocr_tokens = [token for line in lines if "-ocr-" in str(line.get("evidence_id", ""))
                  for token in re.findall(r"\w+", str(line.get("text", "")).casefold())]
    for table in tables:
        tx, ty, tw, th = table.coordinates
        inside = (max(0.0, min(x + width, tx + tw) - max(x, tx)) * max(0.0, min(y + height, ty + th) - max(y, ty))
                  / max(1.0, width * height))
        if inside < 0.90:
            continue
        table_tokens = {token for row in table.metadata.get("rows") or [] for value in row
                        for token in re.findall(r"\w+", str(value or "").casefold())}
        if not ocr_tokens or sum(token in table_tokens for token in ocr_tokens) >= 0.7 * len(ocr_tokens):
            return table
    return None


_PERIOD_THOUSANDS = re.compile(r"[+\-−–(]?\s*[$€£]?\s*[+\-−–]?[1-9]\d{0,2}\.\d{3}\s*\)?")
_COMMA_THOUSANDS = re.compile(r"\D*\d{1,3}(?:,\d{3})+(?:\.\d+)?\D*")


def _reinterpret_period_thousands(typed_rows: list[dict[str, Any]]) -> int:
    """Read $458.056 as $458,056 when its row and column peers print comma thousands.

    OCR sometimes returns a thousands comma as a period. The string alone cannot
    tell a misread from a real three-decimal value, so the cell's peers decide:
    at least two of them must print comma thousands, and they must outnumber
    peers that look like the same misread. The raw printed text is unchanged.
    """
    cells = [(row_index, cell) for row_index, row in enumerate(typed_rows) for cell in row["cells"]
             if cell["value_state"] == "present" and cell["raw_value"]]
    changed = 0
    for row_index, cell in cells:
        raw = str(cell["raw_value"]).strip()
        if cell["numeric_value"] is None or not _PERIOD_THOUSANDS.fullmatch(raw):
            continue
        peers = [str(other["raw_value"]).strip() for other_row, other in cells if other is not cell
                 and (other_row == row_index or other["column_id"] == cell["column_id"])]
        comma = sum(bool(_COMMA_THOUSANDS.fullmatch(peer)) for peer in peers)
        period = sum(bool(_PERIOD_THOUSANDS.fullmatch(peer)) for peer in peers)
        if comma >= 2 and comma > period:
            value = cell["numeric_value"] * 1000
            cell["numeric_value"] = round(value, 6)
            if cell["normalized_value"] is not None:
                cell["normalized_value"] = round(cell["normalized_value"] * 1000, 6)
            changed += 1
    return changed


def _table_structure_errors(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Detect malformed ownership that numeric reconciliation cannot reveal."""
    errors: list[str] = []
    if any(not header for header in headers):
        errors.append("one or more table columns have no header owner")
    if any(not row or not str(row[0] or "").strip() for row in rows[1:]):
        errors.append("one or more table rows have no label owner")
    if any(header.count("(") != header.count(")") for header in headers):
        errors.append("one or more table headers contain unmatched parentheses")
    for row in rows[1:]:
        for value in row[1:]:
            text = str(value or "").strip()
            if text in {"$", "€", "£"}:
                errors.append("standalone currency symbol has no owned numeric value")
            if re.search(r"%\s*[$€£]$", text):
                errors.append("currency symbol is attached to a percentage cell")
    return sorted(set(errors))


def _table_total_reconciliation(rows: list[list[Any]]) -> dict[str, Any] | None:
    """Reconcile generic subtotal rows against the final total wherever values are comparable."""
    if len(rows) < 4:
        return None

    total_rows = [
        row for row in rows[1:]
        if row and re.match(r"^\s*(?:grand\s+)?total\b", str(row[0] or ""), re.I)
    ]
    if len(total_rows) < 3:
        return None
    subtotal_rows, grand_total = total_rows[:-1], total_rows[-1]

    def number(value: Any) -> float | None:
        text = str(value or "").strip()
        if not text or text in {"-", "–", "—"}:
            return None
        return _numeric_value(text)[0]

    checks = []
    width = max(len(row) for row in rows)
    headers = rows[0]
    for column in range(1, width):
        parts = [number(row[column]) if column < len(row) else None for row in subtotal_rows]
        observed = number(grand_total[column]) if column < len(grand_total) else None
        if observed is None or any(part is None for part in parts):
            continue
        calculated = sum(part for part in parts if part is not None)
        tolerance = max(0.011, abs(observed) * 1e-8)
        passed = abs(calculated - observed) <= tolerance
        checks.append({
            "column_index": column,
            "column": str(headers[column] or "").strip() if column < len(headers) else "",
            "subtotal_values": parts,
            "observed_total": observed,
            "calculated_total": round(calculated, 5),
            "passed": passed,
        })
    if not checks:
        return None
    return {
        "passed": all(check["passed"] for check in checks),
        "subtotal_rows": [str(row[0]).strip() for row in subtotal_rows],
        "grand_total_row": str(grand_total[0]).strip(),
        "checks": checks,
    }


def _numeric_content_coverage(
    blocks: list[dict[str, Any]], lines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check whether printed numeric OCR lines survive in nearby block content.

    Region ownership alone is insufficient: one broad region may contain two
    tables while the final block serializes only one of them. This is a
    conservative content check, not a proof that each value has the right row.
    """
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [item for nested in value.values() for item in strings(nested)]
        if isinstance(value, list):
            return [item for nested in value for item in strings(nested)]
        return []

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    indexed = [
        (block.get("coordinates") or [0, 0, 0, 0],
         [normalized(item) for item in strings(block.get("content", {}))])
        for block in blocks
    ]
    counted = 0
    missing: list[dict[str, Any]] = []
    for line in lines:
        raw = str(line.get("text", ""))
        key = normalized(raw)
        if (float(line.get("confidence", 0.0)) < 0.80
                or not re.search(r"\d", raw) or len(key) < 3):
            continue
        coordinates = line.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            continue
        counted += 1
        x, y, width, height = coordinates
        center_x, center_y = x + width / 2, y + height / 2
        represented = any(
            box[0] - 10 <= center_x <= box[0] + box[2] + 10
            and box[1] - 10 <= center_y <= box[1] + box[3] + 10
            and any(key in item for item in content)
            for box, content in indexed
        )
        if not represented:
            missing.append({"evidence_id": line.get("evidence_id"), "text": raw})
    return {
        "numeric_ocr_lines": counted,
        "represented": counted - len(missing),
        "unrepresented": missing,
    }


def _clearly_nearest(lines: list[dict[str, Any]], target_y: float) -> dict[str, Any] | None:
    """The line whose center is nearest ``target_y``, if no other line is nearly as near."""
    def distance(line: dict[str, Any]) -> float:
        return abs(line["coordinates"][1] + line["coordinates"][3] / 2 - target_y)

    ordered = sorted(lines, key=distance)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    margin = 0.5 * statistics.median(line["coordinates"][3] for line in ordered)
    return ordered[0] if distance(ordered[1]) - distance(ordered[0]) >= margin else None


def _ground_table_cells_from_ocr(
    blocks: list[dict[str, Any]], lines: list[dict[str, Any]],
) -> dict[str, int]:
    """Ground exact printed table text only when its row and column fit."""
    def key(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())

    def center(box: list[float]) -> tuple[float, float]:
        return box[0] + box[2] / 2, box[1] + box[3] / 2

    def inside(line: dict[str, Any], box: list[float] | None) -> bool:
        if not box or len(box) != 4:
            return True
        x, y = center(line["coordinates"])
        return box[0] - 5 <= x <= box[0] + box[2] + 5 and box[1] - 5 <= y <= box[1] + box[3] + 5

    by_id = {str(line.get("evidence_id")): line for line in lines}
    stats = {"newly_grounded": 0, "still_unverified": 0}
    for block in blocks:
        if block.get("type") != "table":
            continue
        owned = [by_id[evidence_id] for evidence_id in block.get("provenance", {}).get("ocr_evidence_ids", [])
                 if evidence_id in by_id]
        if not owned:
            continue
        rows = block.get("content", {}).get("rows", [])
        columns = block.get("content", {}).get("columns", [])
        if not rows:
            continue
        headers: dict[str, dict[str, Any]] = {}
        if len(columns) > 2:
            for column in columns[1:]:
                matches = [line for line in owned
                           if key(line.get("text")) == key(column.get("label"))
                           and inside(line, column.get("coordinates"))]
                if len(matches) == 1:
                    headers[column["column_id"]] = matches[0]
                    column["evidence_ids"] = [matches[0]["evidence_id"]]
                    column["coordinates"] = matches[0]["coordinates"]
        used_values: set[str] = set()
        for row in rows:
            label_matches = [line for line in owned
                             if key(row.get("label")) and key(line.get("text")) == key(row.get("label"))
                             and inside(line, row.get("label_coordinates"))]
            if len(label_matches) > 1:
                # Rows can repeat a label (several sales at one address); the
                # right printed label is the one on this row's own baseline.
                cell_centers = [center(cell["coordinates"])[1] for cell in row.get("cells", [])
                                if cell.get("coordinates") and len(cell["coordinates"]) == 4]
                if cell_centers:
                    middle = statistics.median(cell_centers)
                    nearest = _clearly_nearest(label_matches, middle)
                    label_matches = [nearest] if nearest is not None else label_matches
            if not label_matches and len(columns) == 2:
                # A key/value list often prints "Key: Value" as one line.
                for cell in row.get("cells", []):
                    if cell.get("grounding_status") != "unverified" or not key(cell.get("raw_value")):
                        continue
                    whole = [line for line in owned if line["evidence_id"] not in used_values
                             and key(line.get("text")) == key(row.get("label")) + key(cell.get("raw_value"))]
                    if len(whole) == 1:
                        row["label_evidence_ids"] = cell["evidence_ids"] = [whole[0]["evidence_id"]]
                        row["label_coordinates"] = cell["coordinates"] = whole[0]["coordinates"]
                        cell["grounding_status"] = "ocr_same_line"
                        used_values.add(whole[0]["evidence_id"])
                        stats["newly_grounded"] += 1
                continue
            if len(label_matches) != 1:
                continue
            label_line = label_matches[0]
            label_x, label_y = center(label_line["coordinates"])
            if not row.get("label_evidence_ids"):
                row["label_evidence_ids"] = [label_line["evidence_id"]]
            if row.get("label_coordinates") is None:
                row["label_coordinates"] = label_line["coordinates"]
            for cell in row.get("cells", []):
                if cell.get("grounding_status") != "unverified" or cell.get("raw_value") is None:
                    continue
                value_matches = [line for line in owned
                                 if line["evidence_id"] not in used_values
                                 and key(line.get("text")) == key(cell.get("raw_value"))
                                 and key(cell.get("raw_value"))
                                 and inside(line, cell.get("coordinates"))]
                plausible = []
                for line in value_matches:
                    value_x, value_y = center(line["coordinates"])
                    tolerance = max(20.0, (label_line["coordinates"][3] + line["coordinates"][3]) / 2 + 12)
                    if not cell.get("coordinates") and abs(value_y - label_y) > tolerance:
                        continue
                    if len(columns) == 2:
                        if value_x <= label_x + 5:
                            continue
                    else:
                        header = headers.get(cell["column_id"])
                        if header is None:
                            continue
                        header_x, _ = center(header["coordinates"])
                        if abs(value_x - header_x) > max(35.0, block["coordinates"][2] / (len(columns) * 1.4)):
                            continue
                    plausible.append(line)
                if len(plausible) > 1 and not cell.get("coordinates"):
                    # Neighboring rows can print the same value; take this row's own.
                    nearest = _clearly_nearest(plausible, label_y)
                    plausible = [nearest] if nearest is not None else plausible
                if len(plausible) != 1:
                    continue
                match = plausible[0]
                cell["coordinates"] = match["coordinates"]
                cell["evidence_ids"] = [match["evidence_id"]]
                cell["grounding_status"] = (
                    "ocr_value_and_row" if len(columns) == 2 else "ocr_row_and_column"
                )
                used_values.add(match["evidence_id"])
                stats["newly_grounded"] += 1
        remaining = sum(cell.get("raw_value") is not None and cell.get("grounding_status") == "unverified"
                        for row in rows for cell in row.get("cells", []))
        stats["still_unverified"] += remaining
        warnings = block.get("validation", {}).get("warnings", [])
        warnings[:] = [warning for warning in warnings
                       if not warning.endswith("table cells lack value-and-owner evidence; treat their values as candidates")]
        if remaining:
            warnings.append(
                f"{remaining} table cells lack value-and-owner evidence; treat their values as candidates"
            )
        # Grounding can settle every cell; the table then passes unless a
        # remaining warning or error still says a value may be wrong.
        reasons = _table_review_reasons(warnings, block["validation"].get("errors", []), rows)
        block["validation"]["status"] = "needs_review" if reasons or block.get("ocr_disagreements") else "passed"
    return stats


def _linked_table_scalar_count(blocks: list[dict[str, Any]]) -> int:
    """Count scalar table values with a printed, text-bearing row owner."""
    return sum(
        cell.get("numeric_value") is not None
        for block in blocks if block.get("type") == "table"
        for row in block.get("content", {}).get("rows", [])
        if re.search(r"[A-Za-z]", str(row.get("label") or ""))
        for cell in row.get("cells", [])
    )

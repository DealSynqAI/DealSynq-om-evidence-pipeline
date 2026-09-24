"""Compare an image-only page proposal with independently extracted evidence.

The model never supplies OCR ownership. A correction needs a unique printed
label/value pair in the proposed region and an unambiguous target observation.
Every other disagreement remains visible in the reconciliation ledger.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import re
import unicodedata
from typing import Any


def _key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^a-z0-9%$€£.]+", "", text)


def _box_inside(line: dict[str, Any], box: list[float]) -> bool:
    x, y, w, h = line["coordinates"]
    cx, cy = x + w / 2, y + h / 2
    return box[0] - 15 <= cx <= box[2] + 15 and box[1] - 15 <= cy <= box[3] + 15


def _same_place(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax, ay, aw, ah = a["coordinates"]
    bx, by, bw, bh = b["coordinates"]
    return abs(ax + aw / 2 - bx - bw / 2) < 15 and abs(ay + ah / 2 - by - bh / 2) < 15


def _unique_printed_line(lines: list[dict[str, Any]], text: str, box: list[float]) -> dict[str, Any] | None:
    matches = [line for line in lines if _key(line.get("text")) == _key(text) and _box_inside(line, box)]
    if not matches:
        return None
    clusters: list[list[dict[str, Any]]] = []
    for line in matches:
        cluster = next((cluster for cluster in clusters if _same_place(cluster[0], line)), None)
        if cluster is None:
            clusters.append([line])
        else:
            cluster.append(line)
    if len(clusters) != 1:
        return None
    return min(clusters[0], key=lambda line: ("-ocr-" not in line["evidence_id"], -float(line.get("confidence", 0))))


def _label_owns_value(label: dict[str, Any], value: dict[str, Any]) -> bool:
    lx, ly, lw, lh = label["coordinates"]
    vx, vy, vw, vh = value["coordinates"]
    horizontal_gap = abs(lx + lw / 2 - vx - vw / 2)
    vertical_gap = vy - (ly + lh)
    return horizontal_gap <= max(90, 0.65 * (lw + vw)) and -25 <= vertical_gap <= 150


def _printed_label_lines(
    lines: list[dict[str, Any]], label: str, box: list[float], value: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    single = _unique_printed_line(lines, label, box)
    if single:
        return [single]
    if not value:
        return []
    vx, vy, vw, _ = value["coordinates"]
    center = vx + vw / 2
    pool = [
        line for line in lines
        if "-ocr-" in line["evidence_id"] and _box_inside(line, box)
        and re.search(r"[A-Za-z]", str(line.get("text", "")))
        and abs(line["coordinates"][0] + line["coordinates"][2] / 2 - center) <= 170
        and vy - 150 <= line["coordinates"][1] <= vy + 20
    ]
    matches: list[list[dict[str, Any]]] = []
    for length in (2, 3):
        for subset in combinations(pool, length):
            ordered = sorted(subset, key=lambda line: (line["coordinates"][1], line["coordinates"][0]))
            if ordered[-1]["coordinates"][1] - ordered[0]["coordinates"][1] > 100:
                continue
            if _key(" ".join(str(line["text"]) for line in ordered)) == _key(label):
                matches.append(ordered)
        if matches:
            break
    return matches[0] if len(matches) == 1 else []


def _union(lines: list[dict[str, Any]]) -> list[float]:
    x0 = min(line["coordinates"][0] for line in lines)
    y0 = min(line["coordinates"][1] for line in lines)
    x1 = max(line["coordinates"][0] + line["coordinates"][2] for line in lines)
    y1 = max(line["coordinates"][1] + line["coordinates"][3] for line in lines)
    return [x0, y0, x1 - x0, y1 - y0]


def _overlap(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bw, bh = b
    bx1, by1 = bx0 + bw, by0 + bh
    area = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return area / max(1, min((ax1 - ax0) * (ay1 - ay0), bw * bh))


def _parent(blocks: list[dict[str, Any]], proposed: dict[str, Any]) -> dict[str, Any] | None:
    kind = proposed["type"]
    matches = [block for block in blocks if block["type"] == kind and _overlap(proposed["bbox"], block["coordinates"]) >= 0.35]
    return max(matches, key=lambda block: _overlap(proposed["bbox"], block["coordinates"])) if matches else None


def has_missing_table_proposal(
    proposal: dict[str, Any] | None, blocks: list[dict[str, Any]],
) -> bool:
    return bool(proposal) and any(
        item.get("type") == "table" and len(item.get("rows", [])) >= 2
        and _parent(blocks, item) is None
        for item in proposal.get("blocks", [])
    )


def _claims(block: dict[str, Any]) -> list[dict[str, Any]]:
    content = block.get("content", {})
    return content.get({"chart": "observations", "map": "bindings", "kpi_panel": "metrics"}.get(block["type"], ""), [])


def _claim_label(claim: dict[str, Any], kind: str) -> str:
    if kind == "map":
        return str(claim.get("geography") or "")
    if kind == "kpi_panel":
        return str(claim.get("series") or claim.get("category") or "")
    return str(claim.get("category") or "")


def _companion_percentage(
    parent: dict[str, Any], lines: list[dict[str, Any]],
    value_line: dict[str, Any], label: str, label_lines: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if parent["type"] != "chart" or parent["content"].get("chart_type") != "pie":
        return None
    if not re.search(r"[$€£]", str(value_line.get("text", ""))):
        return None
    vx, vy, vw, vh = value_line["coordinates"]
    percent_lines = [
        line for line in lines
        if "-ocr-" in line["evidence_id"]
        and re.fullmatch(r"\d+(?:\.\d+)?%", str(line.get("text", "")).strip())
        and abs(line["coordinates"][0] + line["coordinates"][2] / 2 - vx - vw / 2) <= 75
        and -12 <= line["coordinates"][1] - (vy + vh) <= 75
        and line["evidence_id"] in parent.get("provenance", {}).get("ocr_evidence_ids", [])
    ]
    if len(percent_lines) != 1:
        return None
    percent = percent_lines[0]
    claims = [
        claim for claim in parent["content"].get("observations", [])
        if claim.get("value_evidence_id") == percent["evidence_id"]
        and _key(claim.get("raw_value")) == _key(percent["text"])
    ]
    if len(claims) != 1:
        return None
    claim = claims[0]
    decision = "agreement" if _key(claim.get("category")) == _key(label) else "label_corrected_from_printed_stack"
    if decision != "agreement":
        claim["category"] = label
        claim["label_evidence_id"] = label_lines[0]["evidence_id"]
        if len(label_lines) > 1:
            claim["label_evidence_ids"] = [line["evidence_id"] for line in label_lines]
        claim["label_coordinates"] = _union(label_lines)
        claim["grounding_method"] = str(claim.get("grounding_method") or "") + "; printed percentage stacked under independently confirmed amount"
        claim["validation_status"] = "needs_review"
        parent["validation"]["status"] = "needs_review"
    return {
        "decision": decision, "raw_value": percent["text"],
        "target_item_id": claim.get("item_id"), "value_evidence_id": percent["evidence_id"],
    }


def _attach_printed_amount_to_percentage(
    parent: dict[str, Any], lines: list[dict[str, Any]],
    amount: dict[str, Any], label: str, label_lines: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if parent["type"] != "chart" or parent["content"].get("chart_type") != "pie":
        return None
    if not re.search(r"[$€£]", str(amount.get("text", ""))):
        return None
    owned_ids = set(parent.get("provenance", {}).get("ocr_evidence_ids", []))
    if amount["evidence_id"] not in owned_ids or any(line["evidence_id"] not in owned_ids for line in label_lines):
        return None
    ax, ay, aw, ah = amount["coordinates"]
    percentage_lines = [
        line for line in lines
        if "-ocr-" in line["evidence_id"] and line["evidence_id"] in owned_ids
        and re.fullmatch(r"\d+(?:\.\d+)?%", str(line.get("text", "")).strip())
        and abs(line["coordinates"][0] + line["coordinates"][2] / 2 - ax - aw / 2) <= 75
        and -12 <= line["coordinates"][1] - (ay + ah) <= 75
    ]
    if len(percentage_lines) != 1:
        return None
    percent = percentage_lines[0]
    observations = [
        item for item in parent["content"].get("observations", [])
        if item.get("value_evidence_id") == percent["evidence_id"]
        and _key(item.get("raw_value")) == _key(percent["text"])
    ]
    if len(observations) != 1:
        return None
    observation = observations[0]
    existing = observation.get("companion_value")
    if existing:
        return {"decision": "agreement" if existing.get("evidence_id") == amount["evidence_id"] else "companion_conflict_needs_review"}
    from .pipeline import _numeric_value
    numeric, unit, normalized = _numeric_value(str(amount["text"]))
    if numeric is None:
        return None
    observation["companion_value"] = {
        "raw_value": amount["text"], "numeric_value": numeric,
        "normalized_value": normalized, "unit": unit,
        "evidence_id": amount["evidence_id"], "coordinates": amount["coordinates"],
    }
    if _key(observation.get("category")) != _key(label):
        observation["category"] = label
        observation["label_evidence_id"] = label_lines[0]["evidence_id"]
        if len(label_lines) > 1:
            observation["label_evidence_ids"] = [line["evidence_id"] for line in label_lines]
        observation["label_coordinates"] = _union(label_lines)
    observation["grounding_method"] = str(observation.get("grounding_method") or "") + "; amount and percentage share a printed stack confirmed by independent vision"
    observation["validation_status"] = "needs_review"
    parent["validation"]["status"] = "needs_review"
    return {
        "decision": "printed_amount_attached_to_percentage", "target_item_id": observation.get("item_id"),
        "percentage": percent["text"], "percentage_evidence_id": percent["evidence_id"],
        "amount_evidence_id": amount["evidence_id"],
    }


def _reconcile_item(
    proposed: dict[str, Any], item: dict[str, Any], index: int,
    parent: dict[str, Any] | None, lines: list[dict[str, Any]],
) -> dict[str, Any]:
    box = proposed["bbox"]
    label, value = str(item.get("label") or ""), str(item.get("value") or "")
    record: dict[str, Any] = {
        "proposal_type": proposed["type"], "proposal_title": proposed.get("title"),
        "item_index": index, "vision_label": label, "vision_value": value,
        "target_block_id": parent["block_id"] if parent else None,
    }
    value_line = _unique_printed_line(lines, value, box) if value else None
    label_lines = _printed_label_lines(lines, label, box, value_line) if label else []
    label_line = {"evidence_id": label_lines[0]["evidence_id"], "coordinates": _union(label_lines)} if label_lines else None
    record["value_evidence_id"] = value_line["evidence_id"] if value_line else None
    record["label_evidence_id"] = label_line["evidence_id"] if label_line else None
    record["label_evidence_ids"] = [line["evidence_id"] for line in label_lines]
    pair_printed = bool(value_line and label_line and _label_owns_value(label_line, value_line))
    record["printed_pair_verified"] = pair_printed
    if not parent:
        record["decision"] = "no_matching_source_block"
        return record
    kind = proposed["type"]
    candidates = [
        claim for claim in _claims(parent)
        if _key(claim.get("raw_value")) == _key(value)
        or _key((claim.get("companion_value") or {}).get("raw_value")) == _key(value)
    ]
    if len(candidates) > 1 and value_line:
        candidates = [
            claim for claim in candidates
            if claim.get("value_evidence_id") == value_line["evidence_id"]
            or (claim.get("companion_value") or {}).get("evidence_id") == value_line["evidence_id"]
        ]
    if len(candidates) != 1:
        if pair_printed and kind == "chart":
            attached = _attach_printed_amount_to_percentage(parent, lines, value_line, label, label_lines)
            if attached and attached["decision"] == "printed_amount_attached_to_percentage":
                record["decision"] = attached["decision"]
                record["target_item_id"] = attached["target_item_id"]
                record["stacked_percentage"] = attached["percentage"]
                record["percentage_evidence_id"] = attached["percentage_evidence_id"]
                return record
        record["decision"] = "printed_vision_value_missing_or_ambiguous_in_source_block" if value_line else "vision_value_not_uniquely_printed"
        return record
    claim = candidates[0]
    record["target_item_id"] = claim.get("item_id")
    record["source_label"] = _claim_label(claim, kind)
    record["source_value"] = claim.get("raw_value")
    if _key(record["source_label"]) == _key(label) or (
        kind == "kpi_panel" and (
            _key(claim.get("series")) == _key(label)
            or _key(claim.get("category")) == _key(label)
        )
    ):
        record["decision"] = "agreement"
        if pair_printed and kind == "chart":
            record["companion_percentage"] = _companion_percentage(parent, lines, value_line, label, label_lines)
        return record
    if kind != "chart" or not pair_printed:
        record["decision"] = "ownership_disagreement_needs_review"
        return record
    owned_ids = set(parent.get("provenance", {}).get("ocr_evidence_ids", []))
    if not all(line["evidence_id"] in owned_ids for line in label_lines) or value_line["evidence_id"] not in owned_ids:
        record["decision"] = "printed_pair_outside_block_ownership"
        return record
    # The printed numeric value is unchanged. Only its label owner is repaired;
    # pie-slice geometry remains unresolved and both claim and parent stay in review.
    claim["category"] = label
    claim["label_evidence_id"] = label_line["evidence_id"]
    if len(label_lines) > 1:
        claim["label_evidence_ids"] = [line["evidence_id"] for line in label_lines]
    claim["label_coordinates"] = label_line["coordinates"]
    claim["grounding_method"] = str(claim.get("grounding_method") or "") + "; independent vision proposal confirmed by unique OCR label/value geometry"
    claim["validation_status"] = "needs_review"
    parent["validation"]["status"] = "needs_review"
    note = "independent vision corrected a printed chart label; visual mark ownership still requires review"
    if note not in parent["validation"]["warnings"]:
        parent["validation"]["warnings"].append(note)
    record["decision"] = "label_corrected_from_printed_pair"
    record["companion_percentage"] = _companion_percentage(parent, lines, value_line, label, label_lines)
    return record


def _reconcile_table(proposed: dict[str, Any], parent: dict[str, Any] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in proposed.get("rows", []):
        row_label = str(row.get("label") or "")
        proposed_cells = row.get("cells") or []
        column_shift = 0
        # Some image proposals repeat the first column's header in row.label
        # and put the actual row owner in the first cell. Recognize that
        # structure by the printed header, without relying on document words.
        first_cell = proposed_cells[0] if proposed_cells else {}
        if parent and parent["content"].get("columns") and (
            _key(row_label) == _key(parent["content"]["columns"][0].get("label"))
            and _key(first_cell.get("column")) == _key(row_label)
        ):
            row_label = str(first_cell.get("value") or "")
            source_labels = {_key(item.get("label")) for item in parent["content"].get("rows", [])}
            owner_positions = [
                index for index, cell in enumerate(proposed_cells)
                if _key(cell.get("value")) in source_labels
            ]
            if len(owner_positions) == 1:
                column_shift = owner_positions[0]
                row_label = str(proposed_cells[column_shift].get("value") or "")
        for cell_index, cell in enumerate(proposed_cells):
            if parent and cell_index <= column_shift and _key(proposed_cells[0].get("column")) == _key(parent["content"]["columns"][0].get("label")):
                continue
            effective_column = proposed_cells[cell_index - column_shift].get("column") if column_shift else cell.get("column")
            record: dict[str, Any] = {
                "proposal_type": "table", "vision_row": row_label,
                "vision_column": effective_column, "vision_value": cell.get("value"),
                "model_column": cell.get("column"),
                "target_block_id": parent["block_id"] if parent else None,
            }
            if not parent:
                record["decision"] = "no_matching_source_block"
                records.append(record)
                continue
            columns = parent["content"].get("columns", [])
            matches = [column for column in columns if _key(column.get("label")) == _key(effective_column)]
            rows = [item for item in parent["content"].get("rows", []) if _key(item.get("label")) == _key(row_label)]
            if len(matches) != 1 or len(rows) != 1:
                record["decision"] = "row_or_column_owner_unresolved"
            else:
                target = next((item for item in rows[0]["cells"] if item["column_id"] == matches[0]["column_id"]), None)
                record["source_value"] = target.get("raw_value") if target else None
                if target and _key(target.get("raw_value")) == _key(cell.get("value")):
                    record["decision"] = "agreement"
                elif "native PDF table parser" in parent.get("extraction_method", []):
                    record["decision"] = "native_cell_retained_over_vision_disagreement"
                else:
                    record["decision"] = "table_cell_disagreement_needs_review"
                    parent["validation"]["status"] = "needs_review"
            records.append(record)
    return records


def accept_registered_map_candidate(
    proposal: dict[str, Any] | None,
    baseline_blocks: list[dict[str, Any]],
    candidate_blocks: list[dict[str, Any]],
    candidate_errors: list[str],
) -> bool:
    """Allow a vision-suggested map route only with independent map geometry."""
    if candidate_errors or not proposal or any(block["type"] == "map" for block in baseline_blocks):
        return False
    if not any(block.get("type") == "map" and len(block.get("items", [])) >= 2 for block in proposal.get("blocks", [])):
        return False
    return any(
        block["type"] == "map"
        and block.get("content", {}).get("registration", {}).get("status") == "accepted"
        and float(block["content"]["registration"].get("silhouette_iou", 0)) >= 0.90
        and len(block.get("content", {}).get("bindings", [])) >= 5
        for block in candidate_blocks
    )


def accept_verified_chart_candidate(
    proposal: dict[str, Any] | None,
    baseline_blocks: list[dict[str, Any]],
    candidate_blocks: list[dict[str, Any]],
    candidate_errors: list[str],
) -> bool:
    """Select a missed pie only when its PDF slices and printed totals verify."""
    if candidate_errors or not proposal or any(block["type"] == "chart" for block in baseline_blocks):
        return False
    if not any(block.get("type") == "chart" and len(block.get("items", [])) >= 3 for block in proposal.get("blocks", [])):
        return False
    return any(
        block["type"] == "chart"
        and block.get("content", {}).get("chart_type") == "pie"
        and block["content"].get("slice_geometry_status") == "verified"
        and block["content"].get("percentage_total_reconciles") is True
        and len(block["content"].get("observations", [])) >= 3
        for block in candidate_blocks
    )


def accept_ocr_table_candidate(
    proposal: dict[str, Any] | None,
    baseline_blocks: list[dict[str, Any]],
    candidate_blocks: list[dict[str, Any]],
    candidate_errors: list[str],
    baseline_numeric_coverage: dict[str, Any],
    candidate_numeric_coverage: dict[str, Any],
) -> bool:
    """Rescue a missed table only when independent OCR adds owned scalar cells.

    The image-only plan supplies a region and table hypothesis, never the cell
    values. The candidate parser must link printed OCR cells to rows without
    reducing the page's nearby numeric-text coverage.
    """
    if candidate_errors or not proposal:
        return False
    if candidate_numeric_coverage["represented"] < baseline_numeric_coverage["represented"]:
        return False
    for proposed in proposal.get("blocks", []):
        if proposed.get("type") != "table" or len(proposed.get("rows", [])) < 2:
            continue
        if _parent(baseline_blocks, proposed) is not None:
            continue
        candidate = _parent(candidate_blocks, proposed)
        if candidate is None:
            continue
        rows = candidate.get("content", {}).get("rows", [])
        linked = sum(
            cell.get("numeric_value") is not None
            for row in rows if re.search(r"[A-Za-z]", str(row.get("label") or ""))
            for cell in row.get("cells", [])
        )
        if linked < 3:
            continue
        baseline_same_area = [
            block for block in baseline_blocks
            if _overlap(proposed["bbox"], block.get("coordinates", [0, 0, 0, 0])) >= 0.35
        ]
        baseline_linked = sum(
            cell.get("numeric_value") is not None
            for block in baseline_same_area if block.get("type") == "table"
            for row in block.get("content", {}).get("rows", [])
            if re.search(r"[A-Za-z]", str(row.get("label") or ""))
            for cell in row.get("cells", [])
        )
        if linked >= baseline_linked + 3:
            return True
    return False


def reconcile_page(
    page: int, proposal: dict[str, Any] | None,
    blocks: list[dict[str, Any]], lines: list[dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if proposal:
        for proposed in proposal.get("blocks", []):
            if proposed.get("type") not in {"chart", "map", "kpi_panel", "table"}:
                continue
            parent = _parent(blocks, proposed)
            if proposed["type"] == "table":
                records.extend(_reconcile_table(proposed, parent))
            else:
                records.extend(
                    _reconcile_item(proposed, item, index, parent, lines)
                    for index, item in enumerate(proposed.get("items", []), 1)
                )
    counts = Counter(record["decision"] for record in records)
    counts.update(
        f"companion_percentage_{record['companion_percentage']['decision']}"
        for record in records if record.get("companion_percentage")
    )
    if proposal is None:
        counts["vision_proposal_unavailable"] += 1
    return {
        "page": page, "vision_proposal_available": proposal is not None,
        "independent_ocr_line_count": len(lines),
        "vision_structured_claim_count": len(records),
        "decisions": records, "counts": dict(sorted(counts.items())),
    }

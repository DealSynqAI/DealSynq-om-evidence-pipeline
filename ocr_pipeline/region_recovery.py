"""Recover regions that inspection merged or missed: photos and decomposed panels."""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any

from PIL import Image, ImageStat

from .common import _contains, _crop, _write_vision_diagnostic
from .models import PageInspection, Region, SourceBlock
from .table_blocks import _table_blocks
from .text_blocks import _text_block
from .visual_blocks import _visual_blocks


def _recover_vision_photo_regions(
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
    lines: list[dict[str, Any]], plan: dict[str, Any] | None,
    existing: list[SourceBlock | dict[str, Any]], crops: Path, diagnostics: Path,
) -> list[SourceBlock]:
    """Retain sparsely labeled, textured photo proposals on image-only pages."""
    if not plan:
        return []

    def overlap_of_smaller(left: list[float], right: list[float]) -> float:
        lx, ly, lw, lh = left
        rx, ry, rw, rh = right
        intersection = (max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
                        * max(0.0, min(ly + lh, ry + rh) - max(ly, ry)))
        return intersection / max(1.0, min(lw * lh, rw * rh))

    occupied = [
        block.coordinates if isinstance(block, SourceBlock) else block["coordinates"]
        for block in existing
        if (block.type if isinstance(block, SourceBlock) else block.get("type"))
        in {"photograph", "chart", "map", "table", "unclassified_visual"}
    ]
    recovered: list[SourceBlock] = []
    with Image.open(image) as page_image:
        for index, proposal in enumerate(plan.get("blocks", []), 1):
            label = str(proposal.get("title") or proposal.get("text") or "").strip().casefold()
            if proposal.get("type") != "photograph" and label != "image":
                continue
            raw_box = proposal.get("bbox")
            if (not isinstance(raw_box, list) or len(raw_box) != 4
                    or not all(isinstance(value, (int, float)) for value in raw_box)):
                continue
            x0, y0, x1, y1 = (float(value) for value in raw_box)
            box = [x0, y0, x1 - x0, y1 - y0]
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000
                    and 20_000 <= box[2] * box[3] <= 650_000):
                continue
            if any(overlap_of_smaller(box, other) >= 0.75 for other in occupied):
                continue
            visible = [line for line in lines
                       if "-ocr-" in str(line.get("evidence_id", ""))
                       and len(str(line.get("text", "")).strip()) >= 3
                       and _contains(box, line)]
            if len(visible) > 2:
                continue
            crop_box = (
                round(x0 * page_image.width / 1000), round(y0 * page_image.height / 1000),
                round(x1 * page_image.width / 1000), round(y1 * page_image.height / 1000),
            )
            sample = page_image.crop(crop_box).convert("RGB").resize((64, 64))
            channel_stddev = ImageStat.Stat(sample).stddev
            if statistics.median(channel_stddev) < 35:
                continue
            region = Region(
                region_id=f"p{inspection.page:03d}-vision-image-{index:03d}",
                page=inspection.page, kind="visual", coordinates=box,
                reading_order=10_000 + index,
                classification_method="vision image proposal with pixel-texture and OCR-sparsity checks",
                confidence=0.70, metadata={"visual_hint": "photograph"},
            )
            crop_path = crops / f"{region.region_id}.png"
            _crop(image, box, crop_path)
            features = {"photo_crop_channel_stddev": [round(value, 3) for value in channel_stddev],
                        "ocr_lines_inside": len(visible), "vision_proposal_index": index}
            features_ref = _write_vision_diagnostic(
                diagnostics, document_id, source_hash, region, features,
            )
            photo = _visual_blocks(
                document_id, source_hash, region, "photograph", 0.70,
                ["photo boundary comes from a vision proposal; review against the page"],
                visible, features, image, crop_path, None, features_ref,
            )[0]
            recovered.append(photo)
            occupied.append(box)
    return recovered


def _apply_region_decomposition(
    document_id: str, source_hash: str, inspection: PageInspection, image: Path, pdf: Path,
    lines: list[dict[str, Any]], plan: dict[str, Any] | None,
    blocks: list[dict[str, Any]], crops: Path, diagnostics: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace a broad owner only after distinct regions have independent support."""
    from .region_decomposition import (
        empty_table_artifacts, propose_regions, region_review_reasons, select_regions,
    )

    def _box_intersection(left: list[float], right: list[float]) -> float:
        return (max(0, min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0]))
                * max(0, min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1])))

    proposals = propose_regions(image, inspection, lines, plan, pdf)
    broad = [block for block in blocks
             if block["coordinates"][2] * block["coordinates"][3] >= 450_000
             and block["type"] in {"map", "unclassified_visual", "text"}]
    chosen = select_regions(proposals, lines, blocks)
    rapid = [line for line in lines if "-ocr-" in str(line.get("evidence_id", ""))]
    covered = {key for item in chosen for key in item["ocr_evidence_ids"]}
    other_owned = {key for block in blocks if block not in broad
                   for key in block.get("provenance", {}).get("ocr_evidence_ids", [])}
    coverage = (len((covered | other_owned) & {line["evidence_id"] for line in rapid})
                / max(1, len(rapid)))
    visual_supported = any(item["kind"] in {"photograph", "map"} for item in chosen)
    card_supported = sum(item["kind"] == "tenant_card" for item in chosen) >= 2
    decompose = bool(broad and len(chosen) >= 2 and coverage >= 0.85
                     and (visual_supported or card_supported))
    # On a page without a broad visual owner, a PDF image member can still
    # reveal a missing photograph. It may not claim existing OCR text.
    additive = [item for item in chosen if item["kind"] == "photograph"
                and item["origin"] == "pdf_image_member"
                and len(item["ocr_evidence_ids"]) <= 2]
    active = chosen if decompose else additive if not broad else []
    # A broad fallback text box can hide an independently bounded PDF photo.
    # Preserve its transcript in the photo as unresolved raw evidence.
    photo_replacement = next((item for item in additive
                              if any(block["type"] == "text"
                                     and not block["provenance"]["ocr_evidence_ids"]
                                     and _box_intersection(item["coordinates"], block["coordinates"])
                                     / max(1, block["coordinates"][2] * block["coordinates"][3]) >= 0.7
                                     for block in broad)), None)
    if broad and not decompose and photo_replacement:
        active = [photo_replacement]
    replaced = (broad if decompose else [block for block in broad
                if photo_replacement and block["type"] == "text"
                and not block["provenance"]["ocr_evidence_ids"]])
    photo_boxes = [item["coordinates"] for item in active if item["kind"] == "photograph"]
    blank_overlays = empty_table_artifacts(blocks, photo_boxes)
    receipt = {"page": inspection.page, "proposals": proposals,
               "selected": [{key: value for key, value in item.items() if key != "ocr_evidence_ids"}
                            for item in active],
               "rapidocr_line_coverage": round(coverage, 4),
               "broad_owner_replaced": [block["block_id"] for block in replaced],
               "empty_table_artifacts_removed": [block["block_id"] for block in blank_overlays],
               "status": "decomposed" if decompose else "photo_recovered" if replaced
                         else "additive_photo" if active else "retained_for_review"}
    if not active:
        return blocks, receipt
    remaining = [block for block in blocks if block not in replaced and block not in blank_overlays]
    by_id = {line["evidence_id"]: line for line in rapid}
    existing = {block["block_id"] for block in remaining}
    # A retained block keeps its OCR lines; a proposal may only claim the rest.
    kept_owned = {key for block in remaining for key in block.get("provenance", {}).get("ocr_evidence_ids", [])}
    for index, item in enumerate(active, 1):
        kind = item["kind"]
        box = item["coordinates"]
        item_lines = [by_id[key] for key in item["ocr_evidence_ids"] if key in by_id and key not in kept_owned]
        region = Region(
            region_id=f"p{inspection.page:03d}-decomposed-{index:03d}",
            page=inspection.page, kind="normal_text" if kind in {"text_panel", "tenant_card"} else "visual",
            coordinates=box, reading_order=10_000 + index,
            classification_method="independent OCR and pixel region decomposition",
            confidence=0.72, metadata={"source_bbox_points": [
                box[0] * inspection.width_points / 1000,
                box[1] * inspection.height_points / 1000,
                (box[0] + box[2]) * inspection.width_points / 1000,
                (box[1] + box[3]) * inspection.height_points / 1000,
            ]},
        )
        crop_path = crops / f"{region.region_id}.png"
        _crop(image, box, crop_path)
        review_reasons = region_review_reasons(item, item_lines)
        if kind in {"text_panel", "tenant_card"}:
            block = _text_block(document_id, source_hash, inspection, region, item_lines, image, 1.1)
            block.semantic_role = "profile_biography" if kind == "tenant_card" else "text_panel"
            block.content["region_image"] = f"region-images/{crop_path.name}"
            if review_reasons:
                block.validation_status = "needs_review"
                block.warnings.extend(review_reasons)
            new_blocks = [block]
        elif kind == "table":
            new_blocks = _table_blocks(document_id, source_hash, region, item_lines, image)
            for block in new_blocks:
                if review_reasons:
                    block.validation_status = "needs_review"
                    block.warnings.extend(review_reasons)
                block.content["region_image"] = f"region-images/{crop_path.name}"
        else:
            features = {"region_decomposition": item["support"], "boundary_origin": item["origin"]}
            features_ref = _write_vision_diagnostic(
                diagnostics, document_id, source_hash, region, features,
            )
            new_blocks = _visual_blocks(
                document_id, source_hash, region, kind,
                0.82 if not review_reasons else 0.72, review_reasons,
                item_lines, features, image, crop_path, None, features_ref,
            )
        for block in new_blocks:
            if block.block_id in existing:
                raise ValueError(f"duplicate decomposed block: {block.block_id}")
            existing.add(block.block_id)
            payload = block.as_dict()
            if photo_replacement and item is photo_replacement:
                raw = "\n".join(str(old.get("content", {}).get("text", "")) for old in replaced).strip()
                if raw:
                    payload["raw_evidence_lines"] = [{"evidence_id": f"{region.region_id}-paddle-layout",
                                                      "source": "paddle_layout_text", "text": raw,
                                                      "confidence": 0.5, "coordinates": box}]
                    payload["validation"]["status"] = "needs_review"
            remaining.append(payload)
    return remaining, receipt

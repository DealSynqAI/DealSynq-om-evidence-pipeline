"""Page-level grouping, reading order, and parent validation."""

from __future__ import annotations

from pathlib import Path

from .common import NUMBER, _box_union, _provenance
from .models import PageInspection, Region, SourceBlock


def _attach_background_decorations(blocks: list[SourceBlock]) -> None:
    """Attach overlapping background artwork to the smallest structural group that contains it."""
    groups = [block for block in blocks if block.type == "group"]
    for decoration in (block for block in blocks if block.type == "decoration"):
        dx, dy, dw, dh = decoration.coordinates
        center = (dx + dw / 2.0, dy + dh / 2.0)
        containers = [
            group for group in groups
            if group.coordinates[0] <= center[0] <= group.coordinates[0] + group.coordinates[2]
            and group.coordinates[1] <= center[1] <= group.coordinates[1] + group.coordinates[3]
        ]
        if not containers:
            continue
        parent = min(containers, key=lambda group: group.coordinates[2] * group.coordinates[3])
        decoration.parent_block_id = parent.block_id
        decoration.hierarchy_depth = parent.hierarchy_depth + 1
        if decoration.block_id not in parent.child_block_ids:
            parent.child_block_ids.append(decoration.block_id)
        content_children = parent.content.setdefault("child_block_ids", [])
        if decoration.block_id not in content_children:
            content_children.append(decoration.block_id)


def _group_profile_rows(
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
    blocks: list[SourceBlock],
) -> None:
    """Pair aligned photographs with adjacent structured biography blocks."""
    photographs = [
        block for block in blocks
        if block.type == "photograph" and block.parent_block_id is None
    ]
    biographies = [
        block for block in blocks
        if block.type == "group" and block.parent_block_id is None
        and block.content.get("role") == "profile_biography"
    ]
    used_biographies: set[str] = set()
    for index, photograph in enumerate(sorted(photographs, key=lambda block: block.coordinates[1]), 1):
        px, py, pw, ph = photograph.coordinates
        candidates: list[tuple[float, SourceBlock]] = []
        for biography in biographies:
            if biography.block_id in used_biographies:
                continue
            bx, by, bw, bh = biography.coordinates
            overlap = max(0.0, min(py + ph, by + bh) - max(py, by))
            overlap_ratio = overlap / max(1.0, min(ph, bh))
            horizontal_gap = bx - (px + pw)
            if overlap_ratio >= 0.45 and -20 <= horizontal_gap <= 250:
                candidates.append((overlap_ratio, biography))
        if not candidates:
            continue
        biography = max(candidates, key=lambda item: item[0])[1]
        used_biographies.add(biography.block_id)
        group_id = f"p{inspection.page:03d}-profile-{index:03d}"
        coordinates = _box_union(photograph.coordinates, biography.coordinates)
        region = Region(
            region_id=group_id, page=inspection.page, kind="group", coordinates=coordinates,
            reading_order=min(
                int(photograph.provenance.get("reading_order", index)),
                int(biography.provenance.get("reading_order", index)),
            ),
            classification_method="aligned photograph-biography row grouping",
            confidence=min(photograph.confidence, biography.confidence),
        )
        group = SourceBlock(
            document_id=document_id, type="group", page=inspection.page, block_id=group_id,
            content={"role": "profile", "child_block_ids": [photograph.block_id, biography.block_id]},
            coordinates=coordinates, extraction_method=["Python profile-row hierarchy"],
            confidence=region.confidence,
            validation_status=(
                "passed" if photograph.validation_status == biography.validation_status == "passed"
                else "needs_review"
            ),
            warnings=(
                [] if photograph.validation_status == biography.validation_status == "passed"
                else ["one or more child blocks require review"]
            ),
            provenance=_provenance(source_hash, region, [], image), semantic_role="profile",
            child_block_ids=[photograph.block_id, biography.block_id],
        )
        insertion = min(blocks.index(photograph), blocks.index(biography))
        blocks.insert(insertion, group)
        photograph.parent_block_id = group_id
        photograph.semantic_role = "profile_image"
        biography.parent_block_id = group_id
        by_id = {block.block_id: block for block in blocks}
        _set_subtree_depth(photograph, 1, by_id)
        _set_subtree_depth(biography, 1, by_id)


def _arrange_page_blocks(blocks: list[SourceBlock]) -> None:
    """Put complete left and right reading lanes in human order before grouping."""
    roots = [block for block in blocks if block.parent_block_id is None]
    if len(roots) < 3:
        return
    page_bands = {"page_header", "page_footer"}
    lane_candidates = [
        block for block in roots
        if block.type != "heading"
        and block.semantic_role not in page_bands
        and block.coordinates[2] < 700
    ]
    centers = sorted(block.coordinates[0] + block.coordinates[2] / 2 for block in lane_candidates)
    gaps = [(right - left, (left + right) / 2) for left, right in zip(centers, centers[1:])]
    if not gaps:
        return
    largest_gap, divider = max(gaps)
    if largest_gap < 120:
        return
    content_roots = [block for block in roots if block.semantic_role not in page_bands]
    left = [block for block in content_roots if block.coordinates[0] + block.coordinates[2] / 2 < divider]
    right = [block for block in content_roots if block.coordinates[0] + block.coordinates[2] / 2 >= divider]
    if not left or not right:
        return
    page_headers = [block for block in roots if block.semantic_role == "page_header"]
    page_footers = [block for block in roots if block.semantic_role == "page_footer"]
    headings = [block for block in content_roots if block.type == "heading" and block.coordinates[1] <= 220]
    remaining = [block for block in content_roots if block not in headings]
    ordered_roots = (
        sorted(page_headers, key=lambda block: (block.coordinates[1], block.coordinates[0]))
        + sorted(headings, key=lambda block: (block.coordinates[1], block.coordinates[0]))
        + sorted(
            remaining,
            key=lambda block: (
                0 if block.coordinates[0] + block.coordinates[2] / 2 < divider else 1,
                block.coordinates[1], block.coordinates[0],
            ),
        )
        + sorted(page_footers, key=lambda block: (block.coordinates[1], block.coordinates[0]))
    )
    by_parent: dict[str, list[SourceBlock]] = {}
    for block in blocks:
        if block.parent_block_id is not None:
            by_parent.setdefault(block.parent_block_id, []).append(block)
    ordered: list[SourceBlock] = []

    def append_tree(block: SourceBlock) -> None:
        ordered.append(block)
        for child in by_parent.get(block.block_id, []):
            append_tree(child)

    for root in ordered_roots:
        append_tree(root)
    blocks[:] = ordered


def _set_subtree_depth(block: SourceBlock, depth: int, by_id: dict[str, SourceBlock]) -> None:
    block.hierarchy_depth = depth
    for child_id in block.child_block_ids:
        child = by_id.get(child_id)
        if child is not None:
            _set_subtree_depth(child, depth + 1, by_id)


def _normalize_page_heading_roles(blocks: list[SourceBlock]) -> None:
    """One visual page title can coexist with subordinate numeric summaries."""
    titles = sorted(
        (block for block in blocks if block.type == "heading" and block.semantic_role == "page_title"),
        key=lambda block: (block.coordinates[1], block.coordinates[0]),
    )
    for block in titles[1:]:
        text = str(block.content.get("text") or "")
        block.semantic_role = "summary_heading" if NUMBER.search(text) and len(text.split()) <= 12 else "section_heading"
        block.heading_level = max(2, block.heading_level or 2)


def _order_overlapping_visual_headings(blocks: list[SourceBlock]) -> None:
    """Place a recovered chart/map heading before the visual crop it labels."""
    visuals = [
        block for block in blocks
        if block.parent_block_id is None and block.type in {"chart", "map", "kpi_panel"}
    ]
    headings = [
        block for block in blocks
        if block.parent_block_id is None and block.type == "heading"
        and block.semantic_role in {"section_heading", "panel_heading"}
    ]
    for visual in visuals:
        vx, vy, vw, vh = visual.coordinates
        candidates = []
        for heading in headings:
            hx, hy, hw, hh = heading.coordinates
            cx, cy = hx + hw / 2.0, hy + hh / 2.0
            if vx <= cx <= vx + vw and vy <= cy <= vy + min(vh * 0.25, 180.0):
                candidates.append(heading)
        if not candidates:
            continue
        heading = min(candidates, key=lambda block: (block.coordinates[1], block.coordinates[0]))
        heading_index = blocks.index(heading)
        visual_index = blocks.index(visual)
        if heading_index > visual_index:
            blocks.pop(heading_index)
            blocks.insert(visual_index, heading)


def _normalize_block_reading_order(blocks: list[SourceBlock]) -> None:
    """Give parents and descendants a deterministic, unique page reading order."""
    for index, block in enumerate(blocks, 1):
        block.provenance["reading_order"] = index


def _propagate_group_validation(blocks: list[SourceBlock]) -> None:
    """A structural parent cannot claim to pass while one of its direct children needs review."""
    by_id = {block.block_id: block for block in blocks}
    groups = sorted(
        (block for block in blocks if block.type == "group"),
        key=lambda block: block.hierarchy_depth, reverse=True,
    )
    for group in groups:
        children = [by_id[child_id] for child_id in group.child_block_ids if child_id in by_id]
        if any(child.validation_status != "passed" for child in children):
            group.validation_status = "needs_review"
            if "one or more child blocks require review" not in group.warnings:
                group.warnings.append("one or more child blocks require review")

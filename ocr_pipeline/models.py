from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ValidationStatus = Literal["passed", "needs_review", "failed"]


@dataclass
class Region:
    region_id: str
    page: int
    kind: str
    coordinates: list[float]
    reading_order: int
    classification_method: str
    confidence: float
    native_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageInspection:
    page: int
    width_points: float
    height_points: float
    rotation: int
    native_text_available: bool
    native_text_coverage: float
    native_text_quality: float
    image_coverage: float
    vector_line_count: int
    possible_table_regions: int
    possible_visual_regions: int
    routing_confidence: float
    regions: list[Region]
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regions"] = [region.as_dict() for region in self.regions]
        return payload


@dataclass
class SourceBlock:
    document_id: str
    type: str
    page: int
    block_id: str
    content: dict[str, Any]
    coordinates: list[float]
    extraction_method: list[str]
    confidence: float
    validation_status: ValidationStatus
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    semantic_role: str | None = None
    parent_block_id: str | None = None
    child_block_ids: list[str] = field(default_factory=list)
    hierarchy_depth: int = 0
    heading_level: int | None = None

    def as_dict(self) -> dict[str, Any]:
        x, y, width, height = (float(value) for value in self.coordinates)
        x = max(0.0, min(1000.0, x))
        y = max(0.0, min(1000.0, y))
        width = max(0.0, min(1000.0 - x, width))
        height = max(0.0, min(1000.0 - y, height))
        default_roles = {
            "heading": "section_heading",
            "text": "body_text",
            "footnote": "footnote",
            "contact": "contact_information",
            "table": "data_table",
            "comparison_panel": "comparison_panel",
            "chart": "data_visualization",
            "map": "geographic_visualization",
            "kpi_panel": "key_metrics",
            "photograph": "photograph",
            "brand_mark": "brand_mark",
            "decoration": "background_decoration",
            "unclassified_visual": "unresolved_visual",
            "group": "content_group",
        }
        return {
            "schema_version": "unified-source-block/3.0",
            "document_id": self.document_id,
            "type": self.type,
            "semantic_role": self.semantic_role or default_roles.get(self.type, "unspecified"),
            "page": self.page,
            "block_id": self.block_id,
            "hierarchy": {
                "parent_block_id": self.parent_block_id,
                "child_block_ids": list(self.child_block_ids),
                "depth": self.hierarchy_depth,
                "heading_level": self.heading_level,
            },
            "content": self.content,
            "coordinates": [round(value, 6) for value in (x, y, width, height)],
            "coordinate_system": "top-left normalized xywh 0..1000",
            "extraction_method": self.extraction_method,
            "confidence": round(max(0.0, min(1.0, float(self.confidence))), 5),
            "validation": {
                "status": self.validation_status,
                "errors": self.errors,
                "warnings": self.warnings,
            },
            "provenance": self.provenance,
        }


def validate_source_blocks(blocks: list[dict[str, Any]]) -> list[str]:
    """Validate root semantic blocks and their nested typed content."""
    errors: list[str] = []
    ids = [str(block.get("block_id", "")) for block in blocks]
    known = set(ids)
    if "" in known:
        errors.append("Every block must have a non-empty block_id")
    if len(ids) != len(known):
        errors.append("Block IDs must be unique within a page")
    required = {
        "schema_version", "document_id", "type", "semantic_role", "page", "block_id", "hierarchy",
        "content", "coordinates", "coordinate_system",
        "extraction_method", "confidence", "validation", "provenance",
    }
    for block in blocks:
        block_id = block.get("block_id", "<missing>")
        missing = sorted(required - set(block))
        if missing:
            errors.append(f"{block_id}: missing fields {missing}")
        coordinates = block.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            errors.append(f"{block_id}: coordinates must be normalized xywh")
        elif any(not isinstance(value, (int, float)) or value < 0 or value > 1000 for value in coordinates):
            errors.append(f"{block_id}: coordinates must be within 0..1000")
        confidence = block.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"{block_id}: confidence must be within 0..1")
        validation = block.get("validation", {})
        if validation.get("status") not in {"passed", "needs_review", "failed"}:
            errors.append(f"{block_id}: invalid validation status")
        if not isinstance(block.get("content"), dict):
            errors.append(f"{block_id}: content must be an object")
        elif "vision_features" in block["content"] or "line_segments" in block["content"]:
            errors.append(f"{block_id}: raw vision features must be stored in diagnostics")
        if block.get("type") == "decoration" and any(
            key in block.get("content", {}) for key in {"title", "labels", "text", "raw_text"}
        ):
            errors.append(f"{block_id}: decoration cannot carry semantic text fields")
        if block.get("type") == "chart" and block.get("content", {}).get("chart_type") == "pie":
            chart_content = block["content"]
            slice_status = chart_content.get("slice_geometry_status")
            observations = chart_content.get("observations", [])
            if slice_status not in {"verified", "unresolved"}:
                errors.append(f"{block_id}: pie slice_geometry_status must be explicit")
            elif slice_status == "unresolved":
                if block.get("validation", {}).get("status") == "passed":
                    errors.append(f"{block_id}: pie with unresolved slice geometry cannot be marked passed")
                if any(
                    item.get("visual_mark_coordinates") is not None or item.get("visual_mark_ref") is not None
                    for item in observations
                ):
                    errors.append(f"{block_id}: unresolved pie observations cannot claim slice mark coordinates")
            else:
                marks = [tuple(item.get("visual_mark_coordinates") or ()) for item in observations]
                refs = [item.get("visual_mark_ref") for item in observations]
                indices = [ref.get("pdf_image_index") if isinstance(ref, dict) else None for ref in refs]
                if (
                    any(not mark for mark in marks) or len(marks) != len(set(marks))
                    or any(index is None for index in indices) or len(indices) != len(set(indices))
                ):
                    errors.append(f"{block_id}: verified pie slices need distinct soft-mask references and mark coordinates")
        validation = block.get("validation", {})
        if validation.get("status") == "passed" and validation.get("errors"):
            errors.append(f"{block_id}: passed block cannot contain errors")
        hierarchy = block.get("hierarchy", {})
        if not isinstance(hierarchy, dict):
            errors.append(f"{block_id}: hierarchy must be an object")
            continue
        parent_id = hierarchy.get("parent_block_id")
        child_ids = hierarchy.get("child_block_ids")
        depth = hierarchy.get("depth")
        if parent_id is not None and parent_id not in known:
            errors.append(f"{block_id}: unknown parent_block_id {parent_id}")
        if not isinstance(child_ids, list) or any(child_id not in known for child_id in child_ids):
            errors.append(f"{block_id}: child_block_ids must reference blocks on the same page")
        if block_id in (child_ids or []):
            errors.append(f"{block_id}: block cannot be its own child")
        if not isinstance(depth, int) or depth < 0:
            errors.append(f"{block_id}: hierarchy depth must be a non-negative integer")
        elif parent_id is None and depth != 0:
            errors.append(f"{block_id}: root blocks must have hierarchy depth 0")
        elif parent_id is not None and depth == 0:
            errors.append(f"{block_id}: child blocks must have hierarchy depth greater than 0")
        heading_level = hierarchy.get("heading_level")
        if heading_level is not None and (block.get("type") != "heading" or heading_level not in {1, 2, 3, 4, 5, 6}):
            errors.append(f"{block_id}: heading_level is valid only for heading blocks and must be 1..6")
    by_id = {str(block.get("block_id")): block for block in blocks}
    for block_id, block in by_id.items():
        hierarchy = block.get("hierarchy", {})
        if not isinstance(hierarchy, dict):
            continue
        parent_id = hierarchy.get("parent_block_id")
        if parent_id in by_id and block_id not in by_id[parent_id].get("hierarchy", {}).get("child_block_ids", []):
            errors.append(f"{block_id}: parent/child hierarchy is not reciprocal")
        for child_id in hierarchy.get("child_block_ids", []):
            if child_id in by_id and by_id[child_id].get("hierarchy", {}).get("parent_block_id") != block_id:
                errors.append(f"{block_id}: child/parent hierarchy is not reciprocal for {child_id}")
    return errors

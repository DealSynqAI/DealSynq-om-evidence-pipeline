from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from jsonschema import Draft202012Validator
from .comparison_layout import _bullet_candidates
from .source_observations import is_layout_glyph, represented_in_content


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Validation report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_run_path(run: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else run / path


def validate_run(run: Path, schema_path: Path | None = None) -> dict[str, Any]:
    run = run.resolve(strict=True)
    schema_path = (schema_path or Path(__file__).resolve().parents[1] / "schemas/unified-source-block.schema.json").resolve(strict=True)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    schema_dir = schema_path.parent
    page_schema = json.loads((schema_dir / "unified-source-page.schema.json").read_text(encoding="utf-8"))
    inspection_index_schema = json.loads((schema_dir / "pdf-inspection-index.schema.json").read_text(encoding="utf-8"))
    page_inspection_schema = json.loads((schema_dir / "pdf-page-inspection.schema.json").read_text(encoding="utf-8"))
    for contract in (page_schema, inspection_index_schema, page_inspection_schema):
        Draft202012Validator.check_schema(contract)
    page_validator = Draft202012Validator(page_schema)
    inspection_index_validator = Draft202012Validator(inspection_index_schema)
    page_inspection_validator = Draft202012Validator(page_inspection_schema)

    run_manifest_path = run / "manifest.json"
    collection_path = run / "source-blocks/document-manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    schema_errors: list[dict[str, Any]] = []
    integrity_errors: list[str] = []
    status_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    review_by_page: Counter[int] = Counter()
    unassigned_by_page: dict[str, int] = {}
    completeness_by_page: dict[str, str] = {}
    page_inspections: dict[int, dict[str, Any]] = {}
    total_blocks = 0

    if run_manifest.get("status") != "complete":
        integrity_errors.append(f"run status is {run_manifest.get('status')!r}, not 'complete'")
    if run_manifest.get("schema_version") != "unified-source-collection/3.0":
        integrity_errors.append("run manifest is not unified-source-collection/3.0")
    if collection.get("schema_version") != "unified-source-collection/3.0":
        integrity_errors.append("collection manifest is not unified-source-collection/3.0")
    if collection.get("semantic_interpretation_performed") is not False:
        integrity_errors.append("collection does not explicitly stop before semantic interpretation")
    if run_manifest.get("source_sha256") != collection.get("source_sha256"):
        integrity_errors.append("run and collection source hashes differ")

    inspection_ref = run_manifest.get("inspection", {})
    inspection_path = run / str(inspection_ref.get("path", ""))
    if not inspection_path.is_file():
        integrity_errors.append("separate inspection file is missing")
        inspection = {}
    else:
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        for error in inspection_index_validator.iter_errors(inspection):
            integrity_errors.append(f"inspection index schema: {error.message}")
        if _sha256(inspection_path) != inspection_ref.get("sha256"):
            integrity_errors.append("inspection file hash mismatch")
        if inspection.get("schema_version") != "pdf-inspection-index/1.0":
            integrity_errors.append("inspection index schema is not pdf-inspection-index/1.0")
        if inspection.get("document_id") != collection.get("document_id"):
            integrity_errors.append("inspection document_id mismatch")
        if inspection.get("source_sha256") != collection.get("source_sha256"):
            integrity_errors.append("inspection source hash mismatch")
        if inspection.get("page_count") != collection.get("page_count"):
            integrity_errors.append("inspection page_count mismatch")
        if inspection.get("selected_pages") != collection.get("selected_pages"):
            integrity_errors.append("inspection selected_pages mismatch")
        inspected_pages = [int(page.get("page")) for page in inspection.get("pages", [])]
        if inspected_pages != collection.get("selected_pages"):
            integrity_errors.append("inspection page records do not match selected pages")
        for record in inspection.get("pages", []):
            page_number = int(record.get("page", 0))
            page_inspection_path = inspection_path.parent / str(record.get("file", ""))
            if not page_inspection_path.is_file():
                integrity_errors.append(f"page {page_number}: page inspection file is missing")
                continue
            if _sha256(page_inspection_path) != record.get("sha256"):
                integrity_errors.append(f"page {page_number}: page inspection hash mismatch")
            page_inspection = json.loads(page_inspection_path.read_text(encoding="utf-8"))
            page_inspections[page_number] = page_inspection
            for error in page_inspection_validator.iter_errors(page_inspection):
                integrity_errors.append(f"page {page_number}: inspection schema: {error.message}")
            if page_inspection.get("schema_version") != "pdf-page-inspection/1.0":
                integrity_errors.append(f"page {page_number}: page inspection schema mismatch")
            if page_inspection.get("page") != page_number:
                integrity_errors.append(f"page {page_number}: inspection page number mismatch")
            if page_inspection.get("document_id") != collection.get("document_id"):
                integrity_errors.append(f"page {page_number}: inspection document_id mismatch")
            if page_inspection.get("source_sha256") != collection.get("source_sha256"):
                integrity_errors.append(f"page {page_number}: inspection source hash mismatch")
    preserved_source = _resolve_run_path(run, run_manifest.get("preserved_source"))
    if not preserved_source.is_file():
        integrity_errors.append("preserved source file is missing")
    elif _sha256(preserved_source) != collection.get("source_sha256"):
        integrity_errors.append("preserved source hash differs from collection source hash")

    expected_pages = list(collection.get("selected_pages", []))
    listed_pages = [int(record["page"]) for record in collection.get("pages", [])]
    if listed_pages != expected_pages:
        integrity_errors.append(f"page listing {listed_pages} does not match selected pages {expected_pages}")

    for page_record in collection.get("pages", []):
        page_number = int(page_record["page"])
        page_path = run / "source-blocks" / page_record["file"]
        if not page_path.is_file():
            integrity_errors.append(f"page {page_number}: missing {page_record['file']}")
            continue
        actual_hash = _sha256(page_path)
        if actual_hash != page_record.get("sha256"):
            integrity_errors.append(f"page {page_number}: page JSON hash mismatch")
        page_payload = json.loads(page_path.read_text(encoding="utf-8"))
        for error in page_validator.iter_errors(page_payload):
            integrity_errors.append(f"page {page_number}: page schema: {error.message}")
        if page_payload.get("schema_version") != "unified-source-page/4.0":
            integrity_errors.append(f"page {page_number}: page schema is not unified-source-page/4.0")
        if "inspection" in page_payload:
            integrity_errors.append(f"page {page_number}: v2 page JSON embeds inspection")
        if page_payload.get("page") != page_number:
            integrity_errors.append(f"page {page_number}: page number mismatch inside JSON")
        if page_payload.get("document_id") != collection.get("document_id"):
            integrity_errors.append(f"page {page_number}: document_id mismatch")
        if page_payload.get("source_sha256") != collection.get("source_sha256"):
            integrity_errors.append(f"page {page_number}: source hash mismatch")
        rendered = _resolve_run_path(run, page_payload.get("evidence_ledger", {}).get("rendered_page"))
        rendered_hash = page_payload.get("evidence_ledger", {}).get("rendered_page_sha256")
        if not rendered.is_file():
            integrity_errors.append(f"page {page_number}: rendered evidence image is missing")
        elif _sha256(rendered) != rendered_hash:
            integrity_errors.append(f"page {page_number}: rendered evidence hash mismatch")

        blocks = page_payload.get("blocks", [])
        page_ledger_ids = {
            str(item.get("evidence_id"))
            for item in page_payload.get("evidence_ledger", {}).get("ocr_lines", [])
        }
        total_blocks += len(blocks)
        known_ids = {block.get("block_id") for block in blocks}
        blocks_by_id = {block.get("block_id"): block for block in blocks}
        if len(known_ids) != len(blocks):
            integrity_errors.append(f"page {page_number}: block IDs are not unique")
        for block_index, block in enumerate(blocks):
            for error in validator.iter_errors(block):
                schema_errors.append({
                    "page": page_number,
                    "block_id": block.get("block_id"),
                    "block_index": block_index,
                    "json_path": "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path),
                    "message": error.message,
                })
            if block.get("page") != page_number:
                integrity_errors.append(f"page {page_number}: {block.get('block_id')} has incorrect page field")
            if block.get("document_id") != collection.get("document_id"):
                integrity_errors.append(f"page {page_number}: {block.get('block_id')} has incorrect document_id")
            content = block.get("content", {})
            region_image = content.get("region_image")
            if region_image and not _resolve_run_path(run, region_image).is_file():
                integrity_errors.append(f"page {page_number}: {block.get('block_id')} region crop is missing")
            for disagreement in block.get("ocr_disagreements", []):
                crop = _resolve_run_path(run, disagreement.get("crop"))
                if not crop.is_file():
                    integrity_errors.append(
                        f"page {page_number}: {block.get('block_id')} OCR disagreement crop is missing"
                    )
                elif disagreement.get("crop_sha256") and _sha256(crop) != disagreement["crop_sha256"]:
                    integrity_errors.append(
                        f"page {page_number}: {block.get('block_id')} OCR disagreement crop hash mismatch"
                    )
                if block.get("validation", {}).get("status") == "passed":
                    integrity_errors.append(
                        f"page {page_number}: {block.get('block_id')} has unresolved OCR disagreement but passed"
                    )
            if "vision_features" in content or "line_segments" in content:
                integrity_errors.append(f"page {page_number}: {block.get('block_id')} embeds raw vision features")
            requires_visual_diagnostic = block.get("type") in {
                "chart", "map", "kpi_panel", "photograph", "decoration", "unclassified_visual",
            } or (
                block.get("type") == "brand_mark" and content.get("evidence_mode") == "visual_region"
            )
            if requires_visual_diagnostic:
                feature_ref = content.get("vision_features_ref")
                if not isinstance(feature_ref, dict) or not feature_ref.get("path") or not feature_ref.get("sha256"):
                    integrity_errors.append(f"page {page_number}: {block.get('block_id')} has no complete vision_features_ref")
                else:
                    diagnostic_path = run / str(feature_ref["path"])
                    if not diagnostic_path.is_file():
                        integrity_errors.append(f"page {page_number}: {block.get('block_id')} vision diagnostic is missing")
                    else:
                        if _sha256(diagnostic_path) != feature_ref["sha256"]:
                            integrity_errors.append(f"page {page_number}: {block.get('block_id')} vision diagnostic hash mismatch")
                        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
                        if diagnostic.get("schema_version") != "opencv-region-diagnostic/1.0":
                            integrity_errors.append(f"page {page_number}: {block.get('block_id')} vision diagnostic schema mismatch")
                        if diagnostic.get("document_id") != collection.get("document_id"):
                            integrity_errors.append(f"page {page_number}: {block.get('block_id')} diagnostic document_id mismatch")
                        if diagnostic.get("source_sha256") != collection.get("source_sha256"):
                            integrity_errors.append(f"page {page_number}: {block.get('block_id')} diagnostic source hash mismatch")
                        if diagnostic.get("page") != page_number:
                            integrity_errors.append(f"page {page_number}: {block.get('block_id')} diagnostic page mismatch")
                        if diagnostic.get("region_id") != block.get("provenance", {}).get("region_id"):
                            integrity_errors.append(f"page {page_number}: {block.get('block_id')} diagnostic region mismatch")
            status = str(block.get("validation", {}).get("status"))
            if status == "passed" and block.get("validation", {}).get("errors"):
                integrity_errors.append(f"page {page_number}: {block.get('block_id')} is passed but contains errors")
            if block.get("type") == "decoration" and any(key in content for key in {"title", "labels", "text", "raw_text"}):
                integrity_errors.append(f"page {page_number}: decoration {block.get('block_id')} carries semantic text")
            hierarchy = block.get("hierarchy", {})
            parent_id = hierarchy.get("parent_block_id")
            child_ids = hierarchy.get("child_block_ids", [])
            if parent_id is not None:
                parent = blocks_by_id.get(parent_id)
                if parent is None:
                    integrity_errors.append(f"page {page_number}: {block.get('block_id')} references missing parent {parent_id}")
                elif block.get("block_id") not in parent.get("hierarchy", {}).get("child_block_ids", []):
                    integrity_errors.append(f"page {page_number}: {block.get('block_id')} parent link is not reciprocal")
            for child_id in child_ids:
                child = blocks_by_id.get(child_id)
                if child is None:
                    integrity_errors.append(f"page {page_number}: {block.get('block_id')} references missing child {child_id}")
                elif child.get("hierarchy", {}).get("parent_block_id") != block.get("block_id"):
                    integrity_errors.append(f"page {page_number}: {block.get('block_id')} child link is not reciprocal for {child_id}")
            if block.get("type") == "group":
                if block.get("provenance", {}).get("ocr_evidence_ids"):
                    integrity_errors.append(f"page {page_number}: structural group {block.get('block_id')} owns OCR evidence")
                child_statuses = [
                    blocks_by_id[child_id].get("validation", {}).get("status")
                    for child_id in child_ids if child_id in blocks_by_id
                ]
                if any(child_status != "passed" for child_status in child_statuses) and status == "passed":
                    integrity_errors.append(f"page {page_number}: group {block.get('block_id')} passes while a child needs review")
            if block.get("type") == "table":
                columns = content.get("columns", [])
                rows = content.get("rows", [])
                column_ids = [column.get("column_id") for column in columns]
                # A review-status table may retain columns whose printed
                # headers could not be owned. Missing labels are an explicit
                # uncertainty, not a corrupt file; passed tables cannot have them.
                if len(column_ids) != len(set(column_ids)) or (
                    status == "passed" and any(not column.get("label") for column in columns)
                ):
                    integrity_errors.append(f"page {page_number}: {block.get('block_id')} has invalid column ownership")
                if status == "passed" and any(not row.get("label") for row in rows):
                    integrity_errors.append(f"page {page_number}: {block.get('block_id')} has an unowned row")
                expected_cell_ids = set(column_ids[1:])
                for row in rows:
                    actual_cell_ids = {cell.get("column_id") for cell in row.get("cells", [])}
                    if actual_cell_ids != expected_cell_ids:
                        integrity_errors.append(f"page {page_number}: {block.get('block_id')} row {row.get('row_id')} has incomplete cells")
                    for cell in row.get("cells", []):
                        raw = str(cell.get("raw_value") or "").strip()
                        if raw in {"$", "€", "£"} or ("%" in raw and raw.endswith(("$", "€", "£"))):
                            integrity_errors.append(f"page {page_number}: {block.get('block_id')} has structurally malformed cell {raw!r}")
            if block.get("type") == "comparison_panel" and "claim_count" in content:
                claims: list[dict[str, Any]] = []
                leaf_sections: list[dict[str, Any]] = []

                def collect(section: dict[str, Any]) -> None:
                    children = section.get("subsections") or []
                    if children:
                        for child in children:
                            collect(child)
                    else:
                        leaf_sections.append(section)
                        claims.extend(section.get("claims") or [])

                for section in content.get("sections", []):
                    collect(section)
                claim_ids = [str(claim.get("claim_id")) for claim in claims]
                if (content.get("claim_count") != len(claims)
                        or len(claim_ids) != len(set(claim_ids))
                        or content.get("lane_count") != len(leaf_sections)
                        or content.get("lane_count") != len(content.get("bullet_anchor_coordinates", []))):
                    integrity_errors.append(
                        f"page {page_number}: {block.get('block_id')} comparison lane/claim counts disagree with its tree"
                    )
                claim_evidence = [str(evidence_id) for claim in claims for evidence_id in claim.get("evidence_ids", [])]
                if len(claim_evidence) != len(set(claim_evidence)) or not set(claim_evidence) <= page_ledger_ids:
                    integrity_errors.append(
                        f"page {page_number}: {block.get('block_id')} comparison claims have duplicate or untraceable evidence"
                    )
                for claim in claims:
                    for mention in claim.get("numeric_mentions", []):
                        if str(mention.get("raw")) not in str(claim.get("text")):
                            integrity_errors.append(
                                f"page {page_number}: {block.get('block_id')} has a numeric mention absent from its owning claim"
                            )
                if status == "passed":
                    review = content.get("vision_review") or {}
                    expected_titles = [
                        re.sub(r"\W+", "", str(section.get("title") or "").casefold())
                        for section in leaf_sections
                    ]
                    actual_titles = [
                        re.sub(r"\W+", "", str(title).casefold())
                        for title in review.get("leaf_titles", [])
                    ]
                    review_agrees = (
                        review.get("lane_count") == content.get("lane_count")
                        and actual_titles == expected_titles
                        and review.get("claim_counts") == [len(section.get("claims") or []) for section in leaf_sections]
                    )
                    if not review_agrees or any(
                        section.get("structure_complete") is not True for section in content.get("sections", [])
                    ):
                        integrity_errors.append(
                            f"page {page_number}: {block.get('block_id')} passes without complete comparison structure and vision review"
                        )
                    region_id = block.get("provenance", {}).get("region_id")
                    matching_regions = [
                        region for region in page_inspections.get(page_number, {}).get("regions", [])
                        if region.get("region_id") == region_id
                    ]
                    native_words = (matching_regions[0].get("metadata", {}).get("native_panel_words", [])
                                    if matching_regions else [])
                    anchors = content.get("bullet_anchor_coordinates", [])
                    expected_bullets = sum(
                        any(abs(float(word.get("coordinates", [0])[0]) - float(anchor)) <= 12 for anchor in anchors)
                        for word in _bullet_candidates(native_words)
                    )
                    if expected_bullets and expected_bullets != len(claims):
                        integrity_errors.append(
                            f"page {page_number}: {block.get('block_id')} passed with {len(claims)} claims for {expected_bullets} printed bullets"
                        )
            status_counts[status] += 1
            type_counts[str(block.get("type"))] += 1
            if status == "needs_review":
                review_by_page[page_number] += 1
        for block in blocks:
            if block.get("type") != "chart":
                continue
            if (block.get("validation", {}).get("status") == "passed"
                    and block.get("content", {}).get("chart_type") not in {"bar", "pie", "scatterplot"}):
                integrity_errors.append(
                    f"page {page_number}: {block.get('block_id')} passes without supported chart mark ownership"
                )
            content = block.get("content", {})
            owned_evidence = set(block.get("provenance", {}).get("ocr_evidence_ids", []))
            for observation in content.get("observations", []):
                references = [observation.get("label_evidence_id"), observation.get("value_evidence_id")]
                references.extend(observation.get("label_evidence_ids", []))
                companion = observation.get("companion_value")
                if companion:
                    references.append(companion.get("evidence_id"))
                for evidence_id in references:
                    if evidence_id and (evidence_id not in page_ledger_ids or evidence_id not in owned_evidence):
                        integrity_errors.append(
                            f"page {page_number}: {block.get('block_id')} observation references unowned evidence {evidence_id}"
                        )
            calibration = content.get("axis_calibration") or {}
            for tick in calibration.get("y_ticks", []) + calibration.get("x_ticks", []):
                evidence_id = tick.get("evidence_id")
                if evidence_id not in page_ledger_ids or evidence_id not in owned_evidence:
                    integrity_errors.append(
                        f"page {page_number}: {block.get('block_id')} axis references unowned evidence {evidence_id}"
                    )
            if content.get("chart_type") == "scatterplot" and content.get("scatter_points"):
                if block.get("validation", {}).get("status") == "passed":
                    integrity_errors.append(
                        f"page {page_number}: {block.get('block_id')} has only candidate scatter points but is marked passed"
                    )
            if content.get("chart_type") == "pie" and content.get("slice_geometry_status") != "verified":
                if any(item.get("validation_status") == "passed" for item in content.get("observations", [])):
                    integrity_errors.append(
                        f"page {page_number}: {block.get('block_id')} has passed pie items without verified slices"
                    )
            if block.get("content", {}).get("chart_type") == "pie":
                slice_status = block.get("content", {}).get("slice_geometry_status")
                observations = block.get("content", {}).get("observations", [])
                if slice_status == "verified":
                    region_id = block.get("provenance", {}).get("region_id")
                    matching_regions = [
                        region for region in page_inspections.get(page_number, {}).get("regions", [])
                        if region.get("region_id") == region_id
                    ]
                    candidates = (
                        matching_regions[0].get("metadata", {}).get("pdf_soft_mask_slices", [])
                        if matching_regions else []
                    )
                    by_index = {candidate.get("pdf_image_index"): candidate for candidate in candidates}
                    total_area = sum(float(candidate.get("projected_alpha_area") or 0) for candidate in candidates)
                    image_indices: list[int] = []
                    for observation in observations:
                        ref = observation.get("visual_mark_ref") or {}
                        image_index = ref.get("pdf_image_index")
                        image_indices.append(image_index)
                        candidate = by_index.get(image_index)
                        if candidate is None or candidate.get("smask_sha256") != ref.get("smask_sha256"):
                            integrity_errors.append(
                                f"page {page_number}: {block.get('block_id')} has an untraceable pie mask reference"
                            )
                            continue
                        estimated_percent = (
                            100 * float(candidate.get("projected_alpha_area") or 0) / total_area
                            if total_area > 0 else -1
                        )
                        if abs(estimated_percent - float(ref.get("opacity_weighted_area_share_percent") or 0)) > 0.01:
                            integrity_errors.append(
                                f"page {page_number}: {block.get('block_id')} pie mask area reference disagrees with inspection"
                            )
                    if len(image_indices) != len(set(image_indices)) or len(image_indices) != len(candidates):
                        integrity_errors.append(
                            f"page {page_number}: {block.get('block_id')} pie mask ownership is not one-to-one"
                        )
            expected = block.get("content", {}).get("expected_observation_count")
            emitted = block.get("content", {}).get("emitted_observation_count")
            actual = len(block.get("content", {}).get("observations", []))
            if isinstance(emitted, int) and emitted != actual:
                integrity_errors.append(
                    f"page {page_number}: {block.get('block_id')} emitted count {emitted} differs from {actual} observations"
                )
            if isinstance(expected, int) and expected > 0 and actual != expected and block.get("validation", {}).get("status") == "passed":
                integrity_errors.append(
                    f"page {page_number}: incomplete chart {block.get('block_id')} is marked passed"
                )
        evidence_owners: dict[str, list[str]] = {}
        for block in blocks:
            for evidence_id in block.get("provenance", {}).get("ocr_evidence_ids", []):
                evidence_owners.setdefault(str(evidence_id), []).append(str(block.get("block_id")))
        duplicates = {key: owners for key, owners in evidence_owners.items() if len(owners) > 1}
        if duplicates:
            integrity_errors.append(f"page {page_number}: OCR evidence has multiple block owners: {duplicates}")
        reconciliation_path = run / "reconciliation" / f"page-{page_number:03d}.json"
        if reconciliation_path.is_file():
            reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
            if "raw_ocr_capture" in reconciliation:
                def text_strings(value: Any) -> list[str]:
                    if isinstance(value, str):
                        return [value]
                    if isinstance(value, dict):
                        return [item for nested in value.values() for item in text_strings(nested)]
                    if isinstance(value, list):
                        return [item for nested in value for item in text_strings(nested)]
                    return []

                def normalized(value: str) -> str:
                    return re.sub(r"[^a-z0-9]", "", value.casefold())

                by_id = {
                    str(line.get("evidence_id")): line
                    for line in page_payload.get("evidence_ledger", {}).get("ocr_lines", [])
                }
                source_by_id = {
                    str(item.get("evidence_id")): item
                    for item in page_payload.get("evidence_ledger", {}).get("source_observations", [])
                }
                for block in blocks:
                    owned = set(block.get("provenance", {}).get("ocr_evidence_ids", []))
                    for item in block.get("raw_evidence_lines", []):
                        evidence_id = str(item.get("evidence_id"))
                        source_line = by_id.get(evidence_id)
                        if source_line is None:
                            observation = source_by_id.get(evidence_id)
                            if (observation is None or item.get("text") != observation.get("text")
                                    or observation.get("owner_block_id") != block.get("block_id")):
                                integrity_errors.append(
                                    f"page {page_number}: {block.get('block_id')} has untraceable source observation {evidence_id}"
                                )
                            continue
                        if evidence_id not in owned or source_line is None or item.get("text") != source_line.get("text"):
                            integrity_errors.append(
                                f"page {page_number}: {block.get('block_id')} has untraceable raw OCR evidence {evidence_id}"
                            )
                block_by_id = {str(block.get("block_id")): block for block in blocks}
                for evidence_id, line in by_id.items():
                    raw = str(line.get("text") or "")
                    if not raw.strip():
                        continue
                    owners = evidence_owners.get(evidence_id, [])
                    if len(owners) != 1:
                        integrity_errors.append(f"page {page_number}: OCR line {evidence_id} has no unique USB owner")
                        continue
                    owner = block_by_id[owners[0]]
                    key = normalized(raw)
                    content_strings = text_strings(owner.get("content", {}))
                    if not key:
                        in_content = any(raw in item for item in content_strings)
                    elif len(key) < 3:
                        in_content = any(key == normalized(item) for item in content_strings)
                    else:
                        in_content = any(key in normalized(item) for item in content_strings)
                    in_raw = any(item.get("evidence_id") == evidence_id for item in owner.get("raw_evidence_lines", []))
                    if not in_content and not in_raw:
                        integrity_errors.append(
                            f"page {page_number}: OCR line {evidence_id} is owned but absent from USB content and raw evidence"
                        )
                for evidence_id, observation in source_by_id.items():
                    disposition = observation.get("disposition")
                    if disposition == "matched_ocr":
                        matched = observation.get("matched_ocr_evidence_ids") or []
                        if not matched or any(item not in by_id for item in matched):
                            integrity_errors.append(
                                f"page {page_number}: source observation {evidence_id} has an invalid OCR match"
                            )
                        continue
                    owner = block_by_id.get(str(observation.get("owner_block_id")))
                    if owner is None:
                        integrity_errors.append(f"page {page_number}: source observation {evidence_id} has no USB owner")
                    elif disposition == "raw_attached":
                        if not any(item.get("evidence_id") == evidence_id
                                   for item in owner.get("raw_evidence_lines", [])):
                            integrity_errors.append(f"page {page_number}: source observation {evidence_id} is absent from raw evidence")
                    elif disposition == "content_supported":
                        if not represented_in_content(observation, owner.get("content", {})):
                            integrity_errors.append(
                                f"page {page_number}: source observation {evidence_id} is not in owner content"
                            )
                        box = observation.get("coordinates") or []
                        owner_box = owner.get("coordinates") or []
                        if len(box) != 4 or len(owner_box) != 4 or not (
                            owner_box[0] - 8 <= box[0] + box[2] / 2 <= owner_box[0] + owner_box[2] + 8
                            and owner_box[1] - 8 <= box[1] + box[3] / 2 <= owner_box[1] + owner_box[3] + 8
                        ):
                            integrity_errors.append(
                                f"page {page_number}: source observation {evidence_id} is outside owner coordinates"
                            )
                    elif disposition == "verbatim_glyph":
                        if (not is_layout_glyph(observation)
                                or owner.get("type") not in {"text", "heading", "footnote", "contact",
                                                            "brand_mark", "comparison_panel"}
                                or not any(
                                    item.get("evidence_id") == evidence_id
                                    and item.get("text") == observation.get("text")
                                    and item.get("coordinates") == observation.get("coordinates")
                                    for item in owner.get("raw_evidence_lines", [])
                                )):
                            integrity_errors.append(
                                f"page {page_number}: source observation {evidence_id} has invalid glyph capture"
                            )
                    elif disposition == "fallback_text":
                        if observation.get("text") != owner.get("content", {}).get("text"):
                            integrity_errors.append(f"page {page_number}: source observation {evidence_id} has incorrect fallback text")
                    else:
                        integrity_errors.append(f"page {page_number}: source observation {evidence_id} has no valid disposition")
        dispositions = page_payload.get("evidence_disposition", [])
        disposition_ids = [str(item.get("evidence_id")) for item in dispositions]
        ledger_ids = [str(item.get("evidence_id")) for item in page_payload.get("evidence_ledger", {}).get("ocr_lines", [])]
        if len(disposition_ids) != len(set(disposition_ids)) or set(disposition_ids) != set(ledger_ids):
            integrity_errors.append(f"page {page_number}: evidence disposition is not a one-to-one ledger partition")
        for block in blocks:
            if block.get("type") != "table":
                continue
            referenced = []
            referenced.extend(
                evidence_id for column in block.get("content", {}).get("columns", [])
                for evidence_id in column.get("evidence_ids", [])
            )
            referenced.extend(
                evidence_id for row in block.get("content", {}).get("rows", [])
                for cell in row.get("cells", []) for evidence_id in cell.get("evidence_ids", [])
            )
            referenced.extend(
                evidence_id for row in block.get("content", {}).get("rows", [])
                for evidence_id in row.get("label_evidence_ids", [])
            )
            for evidence_id in referenced:
                if block.get("block_id") not in evidence_owners.get(str(evidence_id), []):
                    integrity_errors.append(
                        f"page {page_number}: table cell evidence {evidence_id} is not owned by {block.get('block_id')}"
                    )
            ocr_by_id = {
                str(line.get("evidence_id")): line
                for line in page_payload.get("evidence_ledger", {}).get("ocr_lines", [])
            }
            for row in block.get("content", {}).get("rows", []):
                for cell in row.get("cells", []):
                    grounding = cell.get("grounding_status")
                    if grounding in {"ocr_value_and_row", "ocr_row_and_column"}:
                        labels = row.get("label_evidence_ids") or []
                        values = cell.get("evidence_ids") or []
                        printed_label = "".join(str(ocr_by_id.get(item, {}).get("text") or "") for item in labels)
                        printed_value = "".join(str(ocr_by_id.get(item, {}).get("text") or "") for item in values)
                        compact = lambda value: re.sub(r"[^a-z0-9]", "", str(value or "").casefold())
                        if (not labels or not values
                                or compact(printed_label) != compact(row.get("label"))
                                or compact(printed_value) != compact(cell.get("raw_value"))
                                or row.get("label_coordinates") is None or cell.get("coordinates") is None):
                            integrity_errors.append(
                                f"page {page_number}: {block.get('block_id')} has an unsupported OCR table cell"
                            )
                        else:
                            label_box = row["label_coordinates"]
                            value_box = cell["coordinates"]
                            label_center = label_box[1] + label_box[3] / 2
                            value_center = value_box[1] + value_box[3] / 2
                            if abs(label_center - value_center) > max(
                                25.0, (label_box[3] + value_box[3]) / 2 + 12.0,
                            ):
                                integrity_errors.append(
                                    f"page {page_number}: {block.get('block_id')} has a distant OCR row/value pair"
                                )
                            if grounding == "ocr_row_and_column":
                                column = next((item for item in block["content"]["columns"]
                                               if item.get("column_id") == cell.get("column_id")), None)
                                if not column or not column.get("evidence_ids") or column.get("coordinates") is None:
                                    integrity_errors.append(
                                        f"page {page_number}: {block.get('block_id')} has an ungrounded OCR column"
                                    )
                    elif grounding == "native_positioned":
                        if row.get("label_coordinates") is None or cell.get("coordinates") is None:
                            integrity_errors.append(
                                f"page {page_number}: {block.get('block_id')} has an unpositioned native table cell"
                            )
                    elif grounding == "unverified" and cell.get("raw_value") is not None:
                        if block.get("validation", {}).get("status") == "passed":
                            integrity_errors.append(
                                f"page {page_number}: {block.get('block_id')} passes with an unverified table cell"
                            )
        for item in dispositions:
            evidence_id = str(item.get("evidence_id"))
            owners = evidence_owners.get(evidence_id, [])
            if item.get("status") == "owned" and item.get("owner_block_id") not in owners:
                integrity_errors.append(f"page {page_number}: evidence {evidence_id} has an incorrect owner disposition")
            if item.get("status") == "unassigned" and owners:
                integrity_errors.append(f"page {page_number}: owned evidence {evidence_id} is marked unassigned")
        unassigned_by_page[str(page_number)] = sum(item.get("status") == "unassigned" for item in dispositions)
        completeness = str(page_payload.get("validation", {}).get("completeness_status"))
        completeness_by_page[str(page_number)] = completeness
        high_confidence_unassigned = [
            item for item in page_payload.get("unassigned_evidence", [])
            if float(item.get("evidence", {}).get("confidence", 0.0)) >= 0.80
            and any(character.isalnum() for character in str(item.get("evidence", {}).get("text", "")))
        ]
        if completeness == "complete" and (high_confidence_unassigned or review_by_page[page_number]):
            integrity_errors.append(
                f"page {page_number}: completeness is marked complete despite unresolved semantic evidence"
            )

    if total_blocks != sum(int(value) for value in collection.get("block_counts_by_type", {}).values()):
        integrity_errors.append("total blocks do not match collection type counts")
    if dict(sorted(type_counts.items())) != collection.get("block_counts_by_type"):
        integrity_errors.append("recomputed block type counts do not match collection manifest")
    if dict(sorted(status_counts.items())) != collection.get("validation_counts"):
        integrity_errors.append("recomputed validation counts do not match collection manifest")

    return {
        "schema_version": "unified-source-collection-validation/3.0",
        "run": str(run),
        "source_sha256": collection.get("source_sha256"),
        "schema": str(schema_path),
        "schema_sha256": _sha256(schema_path),
        "json_schema_draft": "2020-12",
        "validator": "jsonschema.Draft202012Validator",
        "pages_validated": len(listed_pages),
        "blocks_validated": total_blocks,
        "schema_valid": not schema_errors,
        "schema_error_count": len(schema_errors),
        "schema_errors": schema_errors,
        "integrity_valid": not integrity_errors,
        "integrity_error_count": len(integrity_errors),
        "integrity_errors": integrity_errors,
        "block_counts_by_type": dict(sorted(type_counts.items())),
        "validation_counts": dict(sorted(status_counts.items())),
        "needs_review_by_page": {str(key): value for key, value in sorted(review_by_page.items())},
        "unassigned_evidence_by_page": unassigned_by_page,
        "completeness_by_page": completeness_by_page,
        "content_approval_claimed": False,
        "result": "valid" if not schema_errors and not integrity_errors else "invalid",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an immutable Unified Source Block run")
    parser.add_argument("run", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_run(args.run, args.schema)
    if args.output:
        _write_json_new(args.output.resolve(), report)
        print(args.output.resolve())
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["result"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())

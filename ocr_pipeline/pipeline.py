"""End-to-end run: render, OCR, table analysis, vision planning, per-page routing,
and the validated Unified Source Block collection."""

from __future__ import annotations

from collections import Counter
import copy
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from PIL import Image
from pypdf import PdfReader

from .brand_marks import (
    _document_token_page_frequency, _looks_like_brand_mark, _recover_unassigned_brand_marks,
)
from .common import (
    NUMBER, _crop, _line_box, _now, _ocr_text, _sha256, _write_json, _write_vision_diagnostic,
    clean_text,
)
from .evidence_capture import (
    _preserve_ocr_lines_in_blocks, _preserve_source_observations, _propagate_group_status,
)
from .inspection import inspect_pdf
from .line_ownership import (
    _augment_native_table_evidence, _augment_native_visual_evidence, _exclusive_region_lines,
    _separate_visual_footnotes,
)
from .models import PageInspection, Region, SourceBlock, validate_source_blocks
from .ocr_regions import add_ocr_text_regions
from .page_layout import (
    _arrange_page_blocks, _attach_background_decorations, _group_profile_rows,
    _normalize_block_reading_order, _normalize_page_heading_roles,
    _order_overlapping_visual_headings, _propagate_group_validation,
)
from .region_recovery import _apply_region_decomposition, _recover_vision_photo_regions
from .table_blocks import (
    _filled_data_cells, _ground_table_cells_from_ocr, _linked_table_scalar_count, _numeric_content_coverage,
    _split_key_value_rows, _table_backdrop, _table_blocks,
)
from .text_blocks import (
    _inline_heading_subsection_blocks, _leading_heading_body_blocks, _profile_biography_blocks,
    _semantic_page_band_blocks, _spatial_residual_text_blocks, _text_block,
    _visual_text_panel_blocks,
)
from .visual_blocks import _classify_visual, _visual_blocks
from .workers import run_paddle_table_worker, run_rapidocr_worker


PIPELINE_VERSION = "0.1.0"
PAGE_SCHEMA_VERSION = "unified-source-page/4.0"
COLLECTION_SCHEMA_VERSION = "unified-source-collection/3.0"
INSPECTION_SCHEMA_VERSION = "pdf-page-inspection/1.0"
INSPECTION_INDEX_SCHEMA_VERSION = "pdf-inspection-index/1.0"


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "document"


def parse_pages(spec: str | None, page_count: int) -> list[int]:
    if not spec:
        return list(range(1, page_count + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    invalid = sorted(page for page in pages if page < 1 or page > page_count)
    if invalid:
        raise ValueError(f"Pages outside 1..{page_count}: {invalid}")
    return sorted(pages)


def _render_pages(pdf: Path, pages: list[int], output: Path, dpi: int, pdftoppm: Path | None) -> dict[int, Path]:
    executable = pdftoppm or (Path(shutil.which("pdftoppm")) if shutil.which("pdftoppm") else None)
    if executable is None or not executable.exists():
        raise RuntimeError("pdftoppm was not found; pass --pdftoppm with a Poppler executable")
    rendered = {}
    for page in pages:
        target = output / f"page-{page:03d}"
        subprocess.run([
            str(executable), "-f", str(page), "-l", str(page), "-singlefile",
            "-png", "-r", str(dpi), str(pdf), str(target),
        ], check=True, capture_output=True)
        rendered[page] = target.with_suffix(".png")
    return rendered


def _route_and_extract_page(
    document_id: str, source_hash: str, inspection: PageInspection, image: Path,
    ocr_page: dict[str, Any], crops: Path, native_threshold: float,
    diagnostics: Path, document_token_pages: Counter[str] | None = None,
    vision_plan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    page_lines = _augment_native_table_evidence(
        inspection, _augment_native_visual_evidence(inspection, ocr_page.get("lines", [])),
    )
    ocr_page["lines"] = page_lines
    region_results = ocr_page.get("regions", {})
    blocks: list[SourceBlock] = []
    assigned: set[str] = set()
    route_errors: list[str] = []
    ownership_decisions: list[dict[str, Any]] = []
    lines_by_region = _exclusive_region_lines(inspection.regions, page_lines, ownership_decisions)
    ocr_page["region_ownership_decisions"] = ownership_decisions
    visual_footnotes = _separate_visual_footnotes(inspection, lines_by_region)
    for region in inspection.regions:
        split = (_split_key_value_rows(region.metadata["rows"], region.metadata.get("cell_coordinates") or [])
                 if region.kind == "table" and region.metadata.get("rows") else None)
        if split:
            title, rows, boxes = split
            region.metadata["rows"] = rows
            if region.metadata.get("cell_coordinates"):
                region.metadata["cell_coordinates"] = boxes
            region.metadata["title"] = region.metadata.get("title") or title
            region.metadata["key_value_split"] = True
    # A comparison layout is populated by its positioned words, not by cells.
    populated_tables = [region for region in inspection.regions
                        if region.kind == "table" and (_filled_data_cells(region.metadata.get("rows")) >= 2
                                                       or region.metadata.get("native_panel_words"))]
    # A table image under a parsed table is not a separate region: its OCR
    # lines are more evidence for the table's own cells.
    backdrops: set[str] = set()
    for region in inspection.regions:
        if region.kind == "visual" and str(region.metadata.get("visual_hint") or "") != "photograph":
            table = _table_backdrop(region, lines_by_region.get(region.region_id, []), populated_tables)
            if table is not None:
                backdrops.add(region.region_id)
                lines_by_region.setdefault(table.region_id, []).extend(lines_by_region.pop(region.region_id, []))
    for region in inspection.regions:
        lines = lines_by_region.get(region.region_id, [])
        assigned.update(line["evidence_id"] for line in lines)
        features = region_results.get(region.region_id, {}).get("vision_features", {})
        plan_hint = region.metadata.get("vision_plan") or {}
        plan_type = str(plan_hint.get("type") or "")
        not_a_table = region.kind == "table" and bool(region.metadata.get("rows")) and region not in populated_tables
        if not_a_table:
            # Table finders also return chart gridlines, text-box borders, and
            # page frames as tables with no filled cells. Classify those by content.
            for key in ("rows", "cell_coordinates", "row_sections", "cell_evidence_ids"):
                region.metadata.pop(key, None)
            region.metadata["table_rows_rejected"] = "fewer than two filled data cells"
        if region.region_id in backdrops:
            continue
        if region.kind == "normal_text":
            if (region.metadata.get("paddle_split_residual")
                    or region.classification_method == "RapidOCR spatial prose lane split") and lines:
                blocks.extend(_spatial_residual_text_blocks(
                    document_id, source_hash, inspection, region, lines, image, native_threshold,
                ))
                continue
            structured = _inline_heading_subsection_blocks(
                document_id, source_hash, inspection, region, lines, image, native_threshold,
            )
            if not structured:
                structured = _profile_biography_blocks(
                    document_id, source_hash, inspection, region, lines, image, native_threshold,
                )
            if not structured:
                structured = _leading_heading_body_blocks(
                    document_id, source_hash, inspection, region, lines, image, native_threshold,
                )
            if structured:
                blocks.extend(structured)
            else:
                blocks.append(_text_block(document_id, source_hash, inspection, region, lines, image, native_threshold))
            continue
        plan_index = plan_hint.get("block_index")
        plan_block = (
            vision_plan["blocks"][plan_index]
            if isinstance(plan_index, int) and vision_plan and 0 <= plan_index < len(vision_plan.get("blocks", []))
            else None
        )
        numeric_density = sum(bool(NUMBER.search(str(line.get("text", "")))) for line in lines) / max(1, len(lines))
        planned_table = (
            region.kind == "visual" and plan_type == "table" and plan_block is not None
            and len(plan_block.get("rows", [])) >= 2 and numeric_density >= 0.35
        )
        if (region.kind == "table" and not not_a_table and plan_type not in {"map", "chart", "kpi_panel"}) or planned_table:
            if planned_table and plan_block.get("title"):
                region.metadata["title"] = plan_block["title"]
            crop_path = crops / f"{region.region_id}.png"
            _crop(image, region.coordinates, crop_path)
            blocks.extend(_table_blocks(document_id, source_hash, region, lines, image))
            continue
        if region.kind == "decoration":
            crop_path = crops / f"{region.region_id}.png"
            _crop(image, region.coordinates, crop_path)
            vision_features_ref = _write_vision_diagnostic(
                diagnostics, document_id, source_hash, region, features,
            )
            semantic_band = _semantic_page_band_blocks(
                document_id, source_hash, inspection, region, lines, page_lines, image,
                crops, features, vision_features_ref, native_threshold, document_token_pages,
            )
            if semantic_band:
                blocks.extend(semantic_band)
                continue
            blocks.extend(_visual_blocks(
                document_id, source_hash, region, "decoration", region.confidence, [], lines,
                features, image, crop_path, None, vision_features_ref,
            ))
            continue
        kind, confidence, warnings = _classify_visual(region, lines, features)
        if plan_type in {"map", "chart", "kpi_panel"}:
            # The model proposes the visual family first. OCR/OpenCV still run
            # independently, and model-selected routes start in review state.
            if kind != plan_type:
                warnings.append(
                    f"vision-first proposed {plan_type}; geometry/OCR classified {kind}"
                )
            kind = plan_type
            confidence = min(confidence, 0.70)
            region.classification_method = "vision-first proposal reconciled with OCR/OpenCV"
        crop_path = crops / f"{region.region_id}.png"
        _crop(image, region.coordinates, crop_path)
        vision_features_ref = _write_vision_diagnostic(
            diagnostics, document_id, source_hash, region, features,
        )
        if kind in {"unclassified_visual", "photograph"} and _looks_like_brand_mark(
            lines, page_lines, document_token_pages,
        ):
            kind = "brand_mark"
            confidence = max(0.82, min(confidence, 0.92))
            warnings = [
                warning for warning in warnings
                if warning != "visual type could not be classified confidently"
            ]
        semantic_payload = None
        if isinstance(plan_index, int) and vision_plan and 0 <= plan_index < len(vision_plan.get("blocks", [])):
            proposed_block = vision_plan["blocks"][plan_index]
            if proposed_block.get("type") == kind and kind in {"chart", "map", "kpi_panel"}:
                semantic_payload = {
                    "_vision_first": True, "type": kind, "confidence": 0.65,
                    "chart_type": proposed_block.get("chart_type") or None,
                    "bindings": [
                        {"label": item.get("label", ""), "value": item.get("value", ""),
                         "confidence": 0.65}
                        for item in proposed_block.get("items", [])
                        if item.get("label") and item.get("value")
                    ],
                }
        if kind == "kpi_panel" and plan_block and plan_block.get("type") == "kpi_panel":
            plan_bindings = [
                {"label": item["label"], "value": item["value"], "confidence": 0.65}
                for item in plan_block.get("items", []) if item.get("label") and item.get("value")
            ]
            if plan_bindings:
                if semantic_payload is None:
                    semantic_payload = {"type": kind, "confidence": 0.65, "chart_type": None, "bindings": []}
                semantic_payload["bindings"] = list(semantic_payload.get("bindings", [])) + plan_bindings
                semantic_payload["_vision_first"] = True
                semantic_payload["confidence"] = min(float(semantic_payload.get("confidence", 0.65)), 0.65)
        if kind == "normal_text":
            blocks.extend(_visual_text_panel_blocks(
                document_id, source_hash, inspection, region, lines, image, page_lines,
            ))
        elif kind == "table":
            blocks.extend(_table_blocks(document_id, source_hash, region, lines, image))
        else:
            blocks.extend(_visual_blocks(
                document_id, source_hash, region, kind, confidence, warnings, lines, features,
                image, crop_path, semantic_payload, vision_features_ref,
            ))
    blocks.extend(_recover_unassigned_brand_marks(
        document_id, source_hash, inspection, image, page_lines, assigned, crops, document_token_pages,
    ))
    pending_footnotes: list[tuple[SourceBlock, str]] = []
    for owner_region_id, note_groups in visual_footnotes.items():
        for index, note_lines in enumerate(note_groups, 1):
            assigned.update(str(line["evidence_id"]) for line in note_lines)
            note_region = Region(
                region_id=f"{owner_region_id}-note-{index:03d}", page=inspection.page,
                kind="normal_text", coordinates=_line_box(note_lines),
                reading_order=10_000 + index, classification_method="marked visual footnote",
                confidence=min(float(line.get("confidence", 0.8)) for line in note_lines),
            )
            note = _text_block(
                document_id, source_hash, inspection, note_region, note_lines, image, 1.1,
            )
            separate_markers = [
                str(line.get("text", "")).strip() for line in note_lines
                if re.fullmatch(r"[*†‡]", str(line.get("text", "")).strip())
            ]
            if separate_markers:
                body = clean_text(_ocr_text([
                    line for line in note_lines
                    if str(line.get("text", "")).strip() not in separate_markers
                ]))
                note.content["text"] = f"{separate_markers[0]} {body}"
                note.content["evidence_text"]["ocr"] = note.content["text"]
            note.type = "footnote"
            note.semantic_role = "footnote"
            note.heading_level = None
            blocks.append(note)
            pending_footnotes.append((note, owner_region_id))
    _group_profile_rows(document_id, source_hash, inspection, image, blocks)
    _attach_background_decorations(blocks)
    _arrange_page_blocks(blocks)
    _order_overlapping_visual_headings(blocks)
    # Heading-order sections are a semantic convenience, not physical source
    # regions. A mixed page can put an unrelated sidebar or header after a
    # heading in reading order, so evidence USBs keep only spatially supported
    # parents here. Semantic sections belong in the downstream context stage.
    _normalize_page_heading_roles(blocks)
    for note, owner_region_id in pending_footnotes:
        owner = next((block for block in blocks if
            block.provenance.get("region_id") == owner_region_id
            and block.type in {"chart", "map", "kpi_panel"}
        ), None)
        if owner is not None:
            previous_parent = next((block for block in blocks if
                block.block_id == note.parent_block_id
            ), None)
            if previous_parent is not None and previous_parent is not owner:
                previous_parent.child_block_ids = [
                    child_id for child_id in previous_parent.child_block_ids
                    if child_id != note.block_id
                ]
                if isinstance(previous_parent.content.get("child_block_ids"), list):
                    previous_parent.content["child_block_ids"] = [
                        child_id for child_id in previous_parent.content["child_block_ids"]
                        if child_id != note.block_id
                    ]
            note.parent_block_id = owner.block_id
            note.hierarchy_depth = owner.hierarchy_depth + 1
            if note.block_id not in owner.child_block_ids:
                owner.child_block_ids.append(note.block_id)
            # The footnote qualifies how to read the values; it does not make
            # any of them uncertain, so it is noted rather than reviewed.
            qualification_warning = "visual observations have a linked footnote that may qualify their interpretation"
            if qualification_warning not in owner.warnings:
                owner.warnings.append(qualification_warning)
    _propagate_group_validation(blocks)
    _normalize_block_reading_order(blocks)
    block_dicts = [block.as_dict() for block in blocks]
    errors = validate_source_blocks(block_dicts)
    unassigned = [
        {"evidence_id": line["evidence_id"], "reason": "outside detected regions", "evidence": line}
        for line in page_lines if line["evidence_id"] not in assigned
    ]
    return block_dicts, unassigned, errors + route_errors


def run_pipeline(
    pdf: Path, output: Path, pages_spec: str | None = None, dpi: int = 300,
    native_threshold: float = 0.78, rapidocr_python: Path | None = None,
    rapidocr_models: Path | None = None, pdftoppm: Path | None = None,
    qwen_endpoint: str | None = None, qwen_model: str = "dealsynq-qwen3-vl:4b-instruct-16k",
    vision_plan_cache: Path | None = None,
    paddle_python: Path | None = None,
    paddle_device: str = "cpu",
) -> Path:
    if not qwen_endpoint:
        raise ValueError("independent vision and OCR are mandatory in this pipeline")
    pdf = pdf.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output must not exist; runs are immutable: {output}")
    output.mkdir(parents=True)
    source_dir = output / "source"
    images_dir = output / "page-images"
    crops_dir = output / "region-images"
    blocks_dir = output / "source-blocks"
    inspection_dir = output / "inspection"
    diagnostics_dir = output / "diagnostics" / "vision-features"
    work_dir = output / "work"
    for path in (source_dir, images_dir, crops_dir, blocks_dir, inspection_dir, diagnostics_dir, work_dir):
        path.mkdir(parents=True)
    source_hash = _sha256(pdf)
    document_id = f"{_slug(pdf.stem)}-{source_hash[:12]}"
    page_count = len(PdfReader(pdf).pages)
    pages = parse_pages(pages_spec, page_count)
    manifest: dict[str, Any] = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "status": "running", "created_utc": _now(), "document_id": document_id,
        "source_filename": pdf.name, "source_sha256": source_hash,
        "page_count": page_count, "selected_pages": pages, "dpi": dpi,
        "coordinate_system": "top-left normalized xywh 0..1000",
        "thresholds": {"native_text_usability": native_threshold},
        "vision_model": {
            "enabled": bool(qwen_endpoint),
            "provider": "OpenAI-compatible" if qwen_endpoint else None,
            "endpoint": qwen_endpoint,
            "model": qwen_model if qwen_endpoint else None,
            "routing": "mandatory full-page image-only proposal; native/OCR/OpenCV extraction receives no vision hints; claim-level reconciliation after both branches",
        },
        "runtime": {
            "python": sys.version,
            "pdfplumber": importlib.metadata.version("pdfplumber"),
            "pypdf": importlib.metadata.version("pypdf"),
            "pillow": importlib.metadata.version("Pillow"),
            "pipeline": f"dealsynq-independent-vision-reconciled-pipeline/{PIPELINE_VERSION}",
        },
        "stages": [],
    }
    _write_json(output / "manifest.json", manifest)
    try:
        copied = source_dir / pdf.name
        shutil.copy2(pdf, copied)
        if _sha256(copied) != source_hash:
            raise RuntimeError("Preserved source copy hash mismatch")
        manifest["preserved_source"] = f"source/{copied.name}"
        manifest["stages"].append({"name": "ingestion", "status": "complete", "finished_utc": _now()})
        _write_json(output / "manifest.json", manifest)

        inspections, pdf_metadata = inspect_pdf(pdf)
        selected_inspections = [inspection for inspection in inspections if inspection.page in pages]
        inspection_records = []
        for page_inspection in selected_inspections:
            page_inspection_path = inspection_dir / f"page-{page_inspection.page:03d}.json"
            _write_json(page_inspection_path, {
                "schema_version": INSPECTION_SCHEMA_VERSION,
                "document_id": document_id,
                "source_sha256": source_hash,
                **page_inspection.as_dict(),
            })
            inspection_records.append({
                "page": page_inspection.page,
                "file": page_inspection_path.name,
                "sha256": _sha256(page_inspection_path),
                "region_count": len(page_inspection.regions),
            })
        inspection_path = inspection_dir / "manifest.json"
        _write_json(inspection_path, {
            "schema_version": INSPECTION_INDEX_SCHEMA_VERSION,
            "document_id": document_id, "source_sha256": source_hash,
            "page_count": page_count, "selected_pages": pages, "pdf_metadata": pdf_metadata,
            "pages": inspection_records,
        })
        manifest["inspection"] = {
            "path": str(inspection_path.relative_to(output)).replace("\\", "/"),
            "sha256": _sha256(inspection_path),
            "schema_version": INSPECTION_INDEX_SCHEMA_VERSION,
        }
        manifest["stages"].append({"name": "pdf-inspection", "status": "complete", "finished_utc": _now()})

        rendered = _render_pages(pdf, pages, images_dir, dpi, pdftoppm)
        render_executable = pdftoppm or (Path(shutil.which("pdftoppm")) if shutil.which("pdftoppm") else None)
        render_version = None
        if render_executable:
            version_result = subprocess.run(
                [str(render_executable), "-v"], capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            render_version = (version_result.stdout or version_result.stderr).strip().splitlines()[:1]
        manifest["rendering"] = {
            "engine": str(render_executable) if render_executable else None,
            "version": render_version[0] if render_version else None,
            "dpi": dpi, "format": "PNG",
        }
        manifest["stages"].append({"name": "rendering", "status": "complete", "finished_utc": _now()})
        from .vision_first import plan_pages
        plans, vision_summary = plan_pages(
            rendered, output / "vision-plans", qwen_endpoint, qwen_model,
            verified_cache=vision_plan_cache,
        )
        # Keep the model proposal out of inspection, OCR ownership, and the
        # deterministic extraction branch. Fusion happens after both finish.
        manifest["vision_plan"] = {
            "path": "vision-plans/manifest.json", **vision_summary,
        }
        manifest["stages"].append({"name": "vision-first-page-planning", "status": "complete", "finished_utc": _now()})
        _write_json(output / "manifest.json", manifest)
        jobs = [{
            "page": inspection.page, "image": str(rendered[inspection.page]),
            "regions": [{"region_id": region.region_id, "coordinates": region.coordinates} for region in inspection.regions],
        } for inspection in selected_inspections]
        ocr_result = run_rapidocr_worker(jobs, work_dir, rapidocr_python, rapidocr_models)
        manifest["ocr"] = {
            "enabled": True, "engine": ocr_result.get("engine"),
            "rapidocr_version": ocr_result.get("rapidocr_version"),
            "opencv_version": ocr_result.get("opencv_version"),
        }
        manifest["stages"].append({"name": "ocr-and-geometry", "status": "complete", "finished_utc": _now()})

        # PP-StructureV3 sees every likely table page, including pages where
        # the native PDF parser already found a table. It is independent of Qwen.
        from .paddle_tables import (
            apply_paddle_tables, page_needs_table_analysis,
            screen_table_candidates_against_ocr, table_candidates,
        )
        from .ocr_disagreement import needs_second_ocr, record_disagreements
        from .spatial_lanes import split_mixed_key_value_regions

        paddle_dir = output / "paddle-tables"
        ocr_for_tables = {int(item["page"]): item for item in ocr_result.get("pages", [])}
        document_token_pages = _document_token_page_frequency(ocr_for_tables, pdf.stem)
        paddle_images = {
            inspection.page: rendered[inspection.page]
            for inspection in selected_inspections
            if page_needs_table_analysis(inspection, ocr_for_tables.get(inspection.page, {"lines": []}))
            or needs_second_ocr(ocr_for_tables.get(inspection.page, {"lines": []}))
        }
        if paddle_images:
            paddle_summary = run_paddle_table_worker(paddle_images, paddle_dir, paddle_python, paddle_device)
        else:
            paddle_dir.mkdir(parents=True, exist_ok=True)
            paddle_summary = {
                "engine": "PP-StructureV3", "device": paddle_device, "python": None,
                "paddleocr_version": None, "paddlepaddle_version": None, "pages": [],
            }
            _write_json(paddle_dir / "summary.json", paddle_summary)
        paddle_pages = {int(item["page"]): item for item in paddle_summary["pages"]}
        paddle_payloads: dict[int, dict[str, Any]] = {}
        for page_inspection in selected_inspections:
            receipt = paddle_pages.get(page_inspection.page, {})
            if page_inspection.page in paddle_images and receipt.get("status") == "complete":
                payload = json.loads(Path(receipt["result"]).read_text(encoding="utf-8"))
                paddle_payloads[page_inspection.page] = payload
                with Image.open(rendered[page_inspection.page]) as page_image:
                    width, height = page_image.size
                candidates, rejected = screen_table_candidates_against_ocr(
                    table_candidates(payload, width, height),
                    ocr_for_tables.get(page_inspection.page, {}).get("lines", []),
                )
                apply_paddle_tables(page_inspection, candidates)
                receipt["usable_table_candidates"] = len(candidates)
                receipt["rejected_table_candidates"] = rejected
            elif page_inspection.page in paddle_images:
                for region in page_inspection.regions:
                    if region.kind == "table":
                        region.metadata["paddle_table_review"] = {
                            "status": "unavailable", "error": receipt.get("error", "no result")}
            split_mixed_key_value_regions(
                page_inspection, ocr_for_tables.get(page_inspection.page, {}).get("lines", []),
            )
            page_ocr_lines = ocr_for_tables.get(page_inspection.page, {}).get("lines", [])
            # Logo text stays unassigned so brand-mark recovery can claim it.
            add_ocr_text_regions(page_inspection, page_ocr_lines, rendered[page_inspection.page],
                                 lambda cluster, lines=page_ocr_lines:
                                 _looks_like_brand_mark(cluster, lines, document_token_pages))
            page_path = inspection_dir / f"page-{page_inspection.page:03d}.json"
            _write_json(page_path, {
                "schema_version": INSPECTION_SCHEMA_VERSION,
                "document_id": document_id, "source_sha256": source_hash,
                **page_inspection.as_dict(),
            })
            next(record for record in inspection_records if record["page"] == page_inspection.page)["sha256"] = _sha256(page_path)
        _write_json(inspection_path, {
            "schema_version": INSPECTION_INDEX_SCHEMA_VERSION,
            "document_id": document_id, "source_sha256": source_hash,
            "page_count": page_count, "selected_pages": pages, "pdf_metadata": pdf_metadata,
            "pages": inspection_records,
        })
        manifest["inspection"]["sha256"] = _sha256(inspection_path)
        manifest["paddle_tables"] = {
            "path": "paddle-tables/summary.json", "engine": paddle_summary["engine"],
            "device": paddle_summary["device"], "python": paddle_summary["python"],
            "selected_pages": sorted(paddle_images),
            "paddleocr_version": paddle_summary["paddleocr_version"],
            "paddlepaddle_version": paddle_summary["paddlepaddle_version"],
            "pages": paddle_summary["pages"],
        }
        manifest["stages"].append({"name": "independent-paddle-table-analysis", "status": "complete", "finished_utc": _now()})
        _write_json(output / "manifest.json", manifest)

        # Neither branch sees the other's observations during extraction.
        from .independent_reconcile import (
            accept_ocr_table_candidate, accept_registered_map_candidate,
            accept_verified_chart_candidate, has_missing_table_proposal, reconcile_page,
        )
        from .vision_first import apply_plan_hints
        reconciliation_dir = output / "reconciliation"
        reconciliation_dir.mkdir()
        deterministic_dir = output / "deterministic"
        deterministic_dir.mkdir()
        reconciliation_counts: Counter[str] = Counter()
        ocr_by_page = {int(item["page"]): item for item in ocr_result.get("pages", [])}
        from .source_observations import collect_page_observations
        source_observations = collect_page_observations(
            pdf, pages, ocr_by_page, paddle_dir, rendered,
        )
        page_records = []
        type_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        decomposition_counts: Counter[str] = Counter()
        disagreement_total = 0
        collection_errors: list[str] = []
        for inspection in selected_inspections:
            raw_ocr_page = ocr_by_page.get(inspection.page, {"lines": [], "regions": {}})
            ocr_page = copy.deepcopy(raw_ocr_page)
            blocks, unassigned, errors = _route_and_extract_page(
                document_id, source_hash, inspection, rendered[inspection.page],
                ocr_page, crops_dir, native_threshold, diagnostics_dir, document_token_pages, None,
            )
            primary_coverage = _numeric_content_coverage(blocks, ocr_page.get("lines", []))
            primary_coverage["linked_table_scalars"] = _linked_table_scalar_count(blocks)
            coverage_comparison: dict[str, Any] = {"paddle_route": primary_coverage}
            routing_decision = "deterministic_route_retained"
            baseline_path = deterministic_dir / f"page-{inspection.page:03d}.json"
            _write_json(baseline_path, {
                "page": inspection.page, "source_sha256": source_hash,
                "branch": "native PDF, RapidOCR, and OpenCV without vision hints",
                "blocks": blocks,
            })
            plan = plans.get(inspection.page)
            missing_proposed_visual = bool(plan) and any(
                item.get("type") in {"map", "chart"}
                and not any(block["type"] == item["type"] for block in blocks)
                for item in plan.get("blocks", [])
            )
            missing_proposed_table = has_missing_table_proposal(plan, blocks)
            if missing_proposed_visual or missing_proposed_table:
                candidate_inspection = copy.deepcopy(inspection)
                apply_plan_hints(candidate_inspection, plan)
                candidate_ocr_page = copy.deepcopy(raw_ocr_page)
                candidate_blocks, candidate_unassigned, candidate_errors = _route_and_extract_page(
                    document_id, source_hash, candidate_inspection, rendered[inspection.page],
                    candidate_ocr_page, crops_dir, native_threshold,
                    diagnostics_dir, document_token_pages, plan,
                )
                if accept_registered_map_candidate(plan, blocks, candidate_blocks, candidate_errors):
                    blocks, unassigned, errors = candidate_blocks, candidate_unassigned, candidate_errors
                    ocr_page = candidate_ocr_page
                    routing_decision = "registered_map_candidate_selected"
                elif accept_verified_chart_candidate(plan, blocks, candidate_blocks, candidate_errors):
                    blocks, unassigned, errors = candidate_blocks, candidate_unassigned, candidate_errors
                    ocr_page = candidate_ocr_page
                    routing_decision = "verified_chart_candidate_selected"
                elif missing_proposed_table and accept_ocr_table_candidate(
                    plan, blocks, candidate_blocks, candidate_errors,
                    _numeric_content_coverage(blocks, ocr_page.get("lines", [])),
                    _numeric_content_coverage(candidate_blocks, candidate_ocr_page.get("lines", [])),
                ):
                    blocks, unassigned, errors = candidate_blocks, candidate_unassigned, candidate_errors
                    ocr_page = candidate_ocr_page
                    routing_decision = "ocr_supported_table_candidate_selected"
                else:
                    routing_decision = "vision_candidate_rejected_without_independent_support"
            if plan:
                blocks.extend(block.as_dict() for block in _recover_vision_photo_regions(
                    document_id, source_hash, inspection, rendered[inspection.page],
                    ocr_page.get("lines", []), plan, blocks, crops_dir, diagnostics_dir,
                ))
            blocks, region_decomposition = _apply_region_decomposition(
                document_id, source_hash, inspection, rendered[inspection.page], pdf,
                ocr_page.get("lines", []), plan, blocks, crops_dir, diagnostics_dir,
            )
            reconciliation = reconcile_page(
                inspection.page, plan, blocks,
                ocr_page.get("lines", []),
            )
            reconciliation["routing_decision"] = routing_decision
            reconciliation["numeric_content_coverage"] = coverage_comparison
            reconciliation["counts"][routing_decision] = 1
            receipt_path = output / "vision-plans" / f"page-{inspection.page:03d}.receipt.json"
            reconciliation["inputs"] = {
                "deterministic_snapshot": str(baseline_path.relative_to(output)).replace("\\", "/"),
                "deterministic_sha256": _sha256(baseline_path),
                "vision_receipt": str(receipt_path.relative_to(output)).replace("\\", "/"),
                "vision_receipt_sha256": _sha256(receipt_path),
            }
            structured_coverage = _numeric_content_coverage(blocks, ocr_page.get("lines", []))
            blocks, raw_capture = _preserve_ocr_lines_in_blocks(
                blocks, ocr_page.get("lines", []), document_id, source_hash,
                inspection, rendered[inspection.page],
            )
            source_capture = _preserve_source_observations(
                blocks, source_observations.get(inspection.page, []),
                document_id, source_hash, inspection, rendered[inspection.page], ocr_page.get("lines", []),
            )
            table_grounding = _ground_table_cells_from_ocr(blocks, ocr_page.get("lines", []))
            disagreement = record_disagreements(
                inspection.page, rendered[inspection.page], ocr_page.get("lines", []),
                paddle_payloads.get(inspection.page), blocks,
                output / "disagreement-crops",
                output / "diagnostics" / "ocr-disagreements" / f"page-{inspection.page:03d}.json",
            )
            decomposition_counts[region_decomposition["status"]] += 1
            disagreement_total += disagreement["unresolved_count"]
            owned_ids = {
                evidence_id for block in blocks
                for evidence_id in block.get("provenance", {}).get("ocr_evidence_ids", [])
            }
            unassigned = [item for item in unassigned if item.get("evidence_id") not in owned_ids]
            errors = list(dict.fromkeys(errors + validate_source_blocks(blocks)))
            reconciliation["structured_numeric_content_coverage"] = structured_coverage
            reconciliation["raw_ocr_capture"] = raw_capture
            reconciliation["source_observation_capture"] = source_capture
            reconciliation["table_grounding"] = table_grounding
            reconciliation["ocr_disagreement"] = disagreement
            reconciliation["region_decomposition"] = region_decomposition
            reconciliation["region_ownership_decisions"] = ocr_page.get("region_ownership_decisions", [])
            _write_json(reconciliation_dir / f"page-{inspection.page:03d}.json", reconciliation)
            reconciliation_counts.update(reconciliation["counts"])
            if inspection.page not in plans:
                for block in blocks:
                    if block["type"] in {"chart", "map", "kpi_panel", "unclassified_visual"}:
                        block["validation"]["status"] = "needs_review"
                        warning = "vision-first proposal unavailable; visual interpretation is unconfirmed"
                        if warning not in block["validation"]["warnings"]:
                            block["validation"]["warnings"].append(warning)
            # Every pass above may flag a child after extraction propagated
            # review status, so parents are settled once, last.
            _propagate_group_status(blocks)
            for block in blocks:
                type_counts[block["type"]] += 1
                status_counts[block["validation"]["status"]] += 1
            completeness_warnings: list[str] = []
            if inspection.page not in plans:
                completeness_warnings.append("vision-first page proposal unavailable")
            high_confidence_unassigned = [
                item for item in unassigned
                if float(item.get("evidence", {}).get("confidence", 0.0)) >= 0.80
                and re.search(r"[A-Za-z0-9]", str(item.get("evidence", {}).get("text", "")))
            ]
            if high_confidence_unassigned:
                completeness_warnings.append(
                    f"{len(high_confidence_unassigned)} high-confidence OCR lines remain semantically unassigned"
                )
            if len(structured_coverage["unrepresented"]) >= 2:
                completeness_warnings.append(
                    f"{len(structured_coverage['unrepresented'])} high-confidence numeric OCR lines "
                    "are outside nearby structured content; raw OCR is retained in source blocks"
                )
            review_blocks = [
                block for block in blocks if block.get("validation", {}).get("status") != "passed"
            ]
            if review_blocks:
                completeness_warnings.append(f"{len(review_blocks)} source blocks require review")
            completeness_status = "complete" if not completeness_warnings else "needs_review"
            page_payload = {
                "schema_version": PAGE_SCHEMA_VERSION, "document_id": document_id,
                "source_sha256": source_hash, "page": inspection.page,
                "blocks": blocks,
                "evidence_ledger": {
                    "rendered_page": f"page-images/{rendered[inspection.page].name}",
                    "rendered_page_sha256": _sha256(rendered[inspection.page]),
                    "ocr_engine": ocr_result.get("engine"),
                    "ocr_lines": ocr_page.get("lines", []),
                    "source_observations": source_observations.get(inspection.page, []),
                },
                "evidence_disposition": [
                    {
                        "evidence_id": line["evidence_id"],
                        "owner_block_id": next((
                            block["block_id"] for block in blocks
                            if line["evidence_id"] in block.get("provenance", {}).get("ocr_evidence_ids", [])
                        ), None),
                        "status": "owned" if any(
                            line["evidence_id"] in block.get("provenance", {}).get("ocr_evidence_ids", [])
                            for block in blocks
                        ) else "unassigned",
                    }
                    for line in ocr_page.get("lines", [])
                ],
                "unassigned_evidence": unassigned,
                "validation": {
                    "valid": not errors, "errors": errors,
                    "completeness_status": completeness_status,
                    "warnings": completeness_warnings,
                },
            }
            page_path = blocks_dir / f"page-{inspection.page:03d}.json"
            _write_json(page_path, page_payload)
            page_records.append({
                "page": inspection.page, "file": page_path.name, "sha256": _sha256(page_path),
                "block_count": len(blocks), "unassigned_evidence_count": len(unassigned),
                "valid": not errors, "completeness_status": completeness_status,
            })
            collection_errors.extend(f"page {inspection.page}: {error}" for error in errors)
        collection_manifest = {
            "schema_version": COLLECTION_SCHEMA_VERSION, "document_id": document_id,
            "source_filename": pdf.name, "source_sha256": source_hash, "page_count": page_count,
            "selected_pages": pages, "pages": page_records, "block_counts_by_type": dict(sorted(type_counts.items())),
            "validation_counts": dict(sorted(status_counts.items())),
            "validation": {"valid": not collection_errors, "errors": collection_errors},
            "semantic_interpretation_performed": False,
        }
        _write_json(blocks_dir / "document-manifest.json", collection_manifest)
        _write_json(reconciliation_dir / "manifest.json", {
            "source_sha256": source_hash, "selected_pages": pages,
            "counts": dict(sorted(reconciliation_counts.items())),
            "policy": "image-only vision and native/OCR/OpenCV run independently; only uniquely printed, spatially owned pairs can correct a source block; uncertainty remains in review",
        })
        manifest["region_decomposition"] = {
            "page_status_counts": dict(sorted(decomposition_counts.items())),
            "receipts": "reconciliation/page-NNN.json#region_decomposition",
        }
        manifest["ocr_disagreement"] = {
            "unresolved_count": disagreement_total,
            "receipts": "diagnostics/ocr-disagreements/page-NNN.json",
            "crops": "disagreement-crops/",
        }
        manifest["stages"].extend([
            {"name": "region-routing-and-extraction", "status": "complete", "finished_utc": _now()},
            {"name": "evidence-gated-page-region-decomposition", "status": "complete", "finished_utc": _now()},
            {"name": "independent-claim-reconciliation", "status": "complete", "finished_utc": _now()},
            {"name": "ocr-disagreement-preservation", "status": "complete", "finished_utc": _now()},
            {"name": "reconstruction-and-validation", "status": "complete", "finished_utc": _now()},
            {"name": "unified-source-block-collection", "status": "complete", "finished_utc": _now()},
        ])
        manifest["status"] = "complete"
        manifest["finished_utc"] = _now()
        manifest["source_blocks_manifest"] = "source-blocks/document-manifest.json"
        manifest["validation"] = collection_manifest["validation"]
        _write_json(output / "manifest.json", manifest)
        from .validate_run import validate_run
        contract = validate_run(output)
        _write_json(output / "validation.json", contract)
        manifest["contract_validation"] = {
            "valid": contract["result"] == "valid",
            "schema_error_count": contract["schema_error_count"],
            "integrity_error_count": contract["integrity_error_count"],
            "report": "validation.json",
        }
        _write_json(output / "manifest.json", manifest)
        return output
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_utc"] = _now()
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output / "manifest.json", manifest)
        raise

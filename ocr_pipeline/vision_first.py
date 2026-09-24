"""Image-first proposals for the production extraction path.

The proposal is advisory. It never removes PDF regions or OCR jobs, and a
failed model call cannot prevent Unified Source Block output.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from .models import PageInspection
from .vision_plan import PROMPT, call, validate_proposal
from .vision_tiles import recover


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _corners(box: list[float]) -> tuple[float, float, float, float]:
    return box[0], box[1], box[0] + box[2], box[1] + box[3]


def overlap_of_smaller(left: list[float], right: list[float]) -> float:
    a = _corners(left)
    b = _corners(right)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = min(left[2] * left[3], right[2] * right[3])
    return intersection / area if area > 0 else 0.0


def plan_pages(
    rendered: dict[int, Path], output: Path, endpoint: str | None, model: str,
    attempts: int = 2, timeout: int = 360, verified_cache: Path | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Call vision before OCR; persist every response and failure receipt."""
    output.mkdir(parents=True, exist_ok=True)
    plans: dict[int, dict[str, Any]] = {}
    summary: dict[str, Any] = {"enabled": bool(endpoint), "pages": []}
    for page, image in sorted(rendered.items()):
        stem = f"page-{page:03d}"
        started = time.perf_counter()
        receipt: dict[str, Any] = {
            "page": page, "model": model if endpoint else None,
            "endpoint": endpoint, "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "input_policy": "rendered page image only; before OCR or geometry workers",
            "started_utc": datetime.now(timezone.utc).isoformat(), "attempts": [],
        }
        if verified_cache is not None:
            cached_receipt_path = verified_cache / f"{stem}.receipt.json"
            cached_proposal_path = verified_cache / f"{stem}.json"
            if cached_receipt_path.is_file() and cached_proposal_path.is_file():
                cached_receipt = json.loads(cached_receipt_path.read_text(encoding="utf-8"))
                if (
                    cached_receipt.get("status") == "complete"
                    and cached_receipt.get("image_sha256") == receipt["image_sha256"]
                    and cached_receipt.get("prompt_sha256") == receipt["prompt_sha256"]
                    and cached_receipt.get("model") == model
                ):
                    cached_plan = validate_proposal(cached_proposal_path.read_text(encoding="utf-8"), page)
                    plans[page] = cached_plan
                    for source in verified_cache.glob(f"{stem}.*"):
                        if source.is_file() and source.name != f"{stem}.receipt.json":
                            shutil.copy2(source, output / source.name)
                    receipt.update({
                        "status": "complete", "block_count": len(cached_plan["blocks"]),
                        "verified_cache": {
                            "source_receipt": str(cached_receipt_path),
                            "source_receipt_sha256": hashlib.sha256(cached_receipt_path.read_bytes()).hexdigest(),
                            "source_proposal_sha256": hashlib.sha256(cached_proposal_path.read_bytes()).hexdigest(),
                            "verification": "model, image SHA-256, prompt SHA-256, and proposal schema",
                        },
                    })
                    receipt["seconds"] = round(time.perf_counter() - started, 3)
                    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    _save(output / f"{stem}.receipt.json", receipt)
                    summary["pages"].append({"page": page, "status": "complete",
                                             "block_count": len(cached_plan["blocks"]),
                                             "seconds": receipt["seconds"], "verified_cache": True})
                    continue
        if endpoint:
            for attempt in range(1, attempts + 1):
                attempt_start = time.perf_counter()
                record: dict[str, Any] = {"attempt": attempt}
                try:
                    raw, answer = call(image, page, endpoint, model, timeout, attempt)
                    (output / f"{stem}.attempt-{attempt:02d}.raw.txt").write_text(raw, encoding="utf-8")
                    proposal = validate_proposal(raw, page)
                    plans[page] = proposal
                    _save(output / f"{stem}.json", proposal)
                    record.update({"status": "complete", "usage": answer.get("usage"),
                                   "finish_reason": answer["choices"][0].get("finish_reason")})
                    receipt.update({"status": "complete", "block_count": len(proposal["blocks"])})
                except Exception as exc:
                    record.update({"status": "failed", "error_type": type(exc).__name__,
                                   "error": str(exc)[:1000]})
                record["seconds"] = round(time.perf_counter() - attempt_start, 3)
                receipt["attempts"].append(record)
                if record["status"] == "complete":
                    break
            if page not in plans:
                receipt["status"] = "failed"
                _save(output / f"{stem}.receipt.json", receipt)
                # A second image-only view is useful for pages that overflow a
                # single model response; it cannot mask failed tiles.
                try:
                    receipt = recover(image, output, page, endpoint, model, timeout)
                    if receipt["status"] == "complete":
                        plans[page] = json.loads((output / f"{stem}.json").read_text(encoding="utf-8"))
                except Exception as exc:
                    receipt["tile_recovery_status"] = f"failed: {type(exc).__name__}: {exc}"
        else:
            receipt.update({"status": "disabled", "reason": "no vision endpoint supplied"})
        receipt["seconds"] = round(time.perf_counter() - started, 3)
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _save(output / f"{stem}.receipt.json", receipt)
        summary["pages"].append({"page": page, "status": receipt["status"],
                                 "block_count": receipt.get("block_count", 0),
                                 "seconds": receipt["seconds"]})
    summary["complete_pages"] = len(plans)
    summary["failed_pages"] = [page for page in rendered if page not in plans]
    _save(output / "manifest.json", summary)
    return plans, summary


def apply_plan_hints(inspection: PageInspection, plan: dict[str, Any] | None) -> None:
    """Annotate existing visual candidates; never replace page-wide coverage."""
    if not plan:
        return
    eligible = {"table", "chart", "map", "kpi_panel", "comparison_panel"}
    for region in inspection.regions:
        if region.kind not in {"visual", "table"}:
            continue
        candidates = []
        for index, block in enumerate(plan.get("blocks", [])):
            if block.get("type") not in eligible:
                continue
            # A first-pass model can call a logo or a text panel a "chart".
            # Require actual structured observations before it may reroute an
            # independently inspected visual. Geometry-only detection can
            # still classify a chart without any model observations.
            minimum_items = 1 if block["type"] == "kpi_panel" else 2
            if block["type"] in {"chart", "map", "kpi_panel"} and len(block.get("items", [])) < minimum_items:
                continue
            x0, y0, x1, y1 = block["bbox"]
            plan_box = [x0, y0, x1 - x0, y1 - y0]
            score = overlap_of_smaller(region.coordinates, plan_box)
            if score >= 0.55:
                a = _corners(region.coordinates)
                b = _corners(plan_box)
                intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0.0, min(a[3], b[3]) - max(a[1], b[1])
                )
                region_coverage = intersection / (region.coordinates[2] * region.coordinates[3])
                candidates.append((score, region_coverage, index, block))
        if not candidates:
            continue
        score, coverage, index, block = max(candidates, key=lambda item: (item[0], item[1]))
        region.metadata["vision_plan"] = {
            "block_index": index, "type": block["type"],
            "overlap_of_smaller": round(score, 5), "region_coverage": round(coverage, 5),
            "title": block.get("title", ""),
            "chart_type": block.get("chart_type", ""),
        }
        if block["type"] == "kpi_panel":
            # Native PDF table detection often isolates the large value while
            # leaving its printed caption in a neighboring text region.
            # The proposed card box lets the OCR ownership pass consider both.
            x0, y0, x1, y1 = block["bbox"]
            region.metadata["ownership_coordinates"] = [x0, y0, x1 - x0, y1 - y0]

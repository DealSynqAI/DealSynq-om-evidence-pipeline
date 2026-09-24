"""Image-only, auditable tile fallback for failed full-page model proposals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from PIL import Image

from .proposal_geometry import xyxy_to_xywh
from .vision_plan import PROMPT, call, validate_proposal


def reproject_box(box: list[float], tile: tuple[float, float, float, float]) -> list[float]:
    xyxy_to_xywh(box)
    left, top, right, bottom = tile
    return [round(left + box[0] * (right - left) / 1000, 3),
            round(top + box[1] * (bottom - top) / 1000, 3),
            round(left + box[2] * (right - left) / 1000, 3),
            round(top + box[3] * (bottom - top) / 1000, 3)]


def overlap_of_smaller(left: list[float], right: list[float]) -> float:
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * \
        max(0, min(left[3], right[3]) - max(left[1], right[1]))
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (left, right)]
    return intersection / min(areas) if min(areas) > 0 else 0.0


def merge_tiles(parts: list[tuple[tuple[float, float, float, float], dict]], page: int) -> dict:
    blocks = []
    for tile, proposal in parts:
        for original in proposal["blocks"]:
            block = dict(original)
            block["bbox"] = reproject_box(original["bbox"], tile)
            # A parent may be in another crop or removed as an overlap duplicate.
            # Do not fabricate cross-tile hierarchy.
            block["parent_index"] = None
            same_index = next((index for index, other in enumerate(blocks) if
                other["type"] == block["type"] and
                other["title"].strip().casefold() == block["title"].strip().casefold() and
                other["text"].strip().casefold() == block["text"].strip().casefold() and
                overlap_of_smaller(other["bbox"], block["bbox"]) >= 0.6
                ), None)
            if same_index is None:
                blocks.append(block)
            elif len(block["items"]) > len(blocks[same_index]["items"]):
                blocks[same_index] = block
    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return {"page_number": page, "blocks": blocks}


def recover(image: Path, output: Path, page: int, endpoint: str, model: str, timeout: int) -> dict:
    receipt_path = output / f"page-{page:03d}.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "failed":
        raise ValueError(f"Page {page} was not a failed page")
    prompt = PROMPT + (
        "\n\nThis image is a crop of the page, not the whole page. Extract ONLY what is "
        "visible inside this crop. Return at most one block for each visual; never repeat "
        "a chart block. For a chart, put each readable category/value pairing into its "
        "items array. If a value is unreadable, omit it rather than guessing. Boxes "
        "must be xyxy coordinates relative to THIS CROP."
    )
    tiles = [("upper", (0.0, 0.0, 1000.0, 340.0)),
             ("lower", (0.0, 280.0, 1000.0, 1000.0))]
    parts = []
    tile_records = []
    with Image.open(image) as page_image:
        width, height = page_image.size
        for name, bounds in tiles:
            start = time.perf_counter()
            tile_path = output / f"page-{page:03d}.tile-{name}.png"
            page_image.crop((round(bounds[0] * width / 1000), round(bounds[1] * height / 1000),
                             round(bounds[2] * width / 1000), round(bounds[3] * height / 1000))).save(tile_path)
            raw_path = output / f"page-{page:03d}.tile-{name}.raw.txt"
            try:
                raw, answer = call(tile_path, page, endpoint, model, timeout, 2, prompt)
                raw_path.write_text(raw, encoding="utf-8")
                parsed = validate_proposal(raw, page)
                parts.append((bounds, parsed))
                record = {"tile": name, "bbox_xyxy": bounds, "status": "complete",
                          "blocks": len(parsed["blocks"]), "usage": answer.get("usage")}
            except Exception as exc:
                record = {"tile": name, "bbox_xyxy": bounds, "status": "failed",
                          "error_type": type(exc).__name__, "error": str(exc)[:1000]}
            record["seconds"] = round(time.perf_counter() - start, 3)
            tile_records.append(record)
    receipt["tile_recovery"] = tile_records
    if len(parts) == len(tiles):
        merged = merge_tiles(parts, page)
        if len(merged["blocks"]) > 65:
            receipt["tile_recovery_status"] = "failed: merged blocks exceed schema limit"
        else:
            proposal_path = output / f"page-{page:03d}.json"
            proposal_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            receipt.update({"status": "complete", "block_count": len(merged["blocks"]),
                            "recovery_method": "two overlapping page-image tiles; no OCR/PDF/OpenCV input",
                            "tile_recovery_status": "complete",
                            "seconds": round(float(receipt.get("seconds", 0)) +
                                             sum(item["seconds"] for item in tile_records), 3)})
    else:
        receipt["tile_recovery_status"] = "failed: one or more tile calls failed"
        receipt["seconds"] = round(float(receipt.get("seconds", 0)) +
                                   sum(item["seconds"] for item in tile_records), 3)
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1/chat/completions")
    parser.add_argument("--model", default="dealsynq-qwen3-vl:4b-instruct-16k")
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args()
    result = recover(args.image, args.output, args.page, args.endpoint, args.model, args.timeout)
    print(json.dumps({"page": args.page, "status": result["status"],
                      "tile_recovery_status": result["tile_recovery_status"]}, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""PP-StructureV3 page analysis in its own supported Python runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--device", default="cpu", help="Paddle inference device, e.g. cpu or gpu:0")
    args = parser.parse_args()

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if args.device.startswith("gpu"):
        import paddle
        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("GPU device requested, but this PaddlePaddle runtime has no CUDA support")
    from paddleocr import PPStructureV3

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    if isinstance(jobs, dict):
        jobs = [jobs]
    engine = PPStructureV3(
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_seal_recognition=False,
        use_formula_recognition=False,
        use_table_recognition=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pages = []
    for job in jobs:
        page = int(job["page"])
        destination = args.output / f"page-{page:03d}.json"
        started = time.perf_counter()
        try:
            results = list(engine.predict(input=str(job["image"])))
            if len(results) != 1:
                raise RuntimeError(f"Expected one page result, got {len(results)}")
            payload = results[0].json
            destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            body = payload.get("res", payload)
            pages.append({
                "page": page, "status": "complete", "result": str(destination),
                "table_count": len(body.get("table_res_list") or []),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
        except Exception as exc:
            pages.append({
                "page": page, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
    args.summary.write_text(json.dumps({
        "engine": "PP-StructureV3", "device": args.device,
        "paddleocr_version": importlib.metadata.version("paddleocr"),
        "paddlepaddle_version": importlib.metadata.version(
            "paddlepaddle-gpu" if args.device.startswith("gpu") else "paddlepaddle"
        ),
        "pages": pages,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

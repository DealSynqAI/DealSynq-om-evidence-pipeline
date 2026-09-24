from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF ingestion through a page-organized Unified Source Block collection"
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, help="New immutable run directory")
    parser.add_argument("--pages", help="Page selection such as 1-3,8")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--native-threshold", type=float, default=0.78)
    parser.add_argument("--route-threshold", type=float, default=0.72)
    parser.add_argument("--skip-ocr", action="store_true", help="Native-PDF smoke tests only")
    parser.add_argument("--rapidocr-python", type=Path)
    parser.add_argument("--rapidocr-models", type=Path)
    parser.add_argument("--paddle-python", type=Path,
                        help="Python 3.9–3.13 with PaddleOCR PP-StructureV3; defaults to .venv-paddle")
    parser.add_argument("--paddle-device", default="cpu",
                        help="PaddleOCR device, e.g. cpu or gpu:0 (requires a CUDA Paddle runtime)")
    parser.add_argument("--pdftoppm", type=Path)
    parser.add_argument("--qwen-endpoint", default="http://127.0.0.1:11434/v1/chat/completions",
                        help="OpenAI-compatible full-page vision endpoint")
    parser.add_argument("--qwen-model", default="dealsynq-qwen3-vl:4b-instruct-16k")
    parser.add_argument("--vision-plan-cache", type=Path,
                        help="Reuse same-image, same-model proposals with verified SHA-256 and schema")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.qwen_endpoint or args.skip_ocr:
        parser.error("this pipeline requires an independent full-page vision attempt and OCR; remove --skip-ocr and supply --qwen-endpoint")
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(__file__).resolve().parents[1] / "runs" / f"{args.pdf.stem}-{stamp}"
    result = run_pipeline(
        args.pdf, output, pages_spec=args.pages, dpi=args.dpi,
        native_threshold=args.native_threshold, route_threshold=args.route_threshold,
        skip_ocr=args.skip_ocr, rapidocr_python=args.rapidocr_python,
        rapidocr_models=args.rapidocr_models, pdftoppm=args.pdftoppm,
        qwen_endpoint=args.qwen_endpoint, qwen_model=args.qwen_model,
        vision_plan_cache=args.vision_plan_cache,
        paddle_python=args.paddle_python,
        paddle_device=args.paddle_device,
    )
    print(f"Unified Source Block collection: {result / 'source-blocks' / 'document-manifest.json'}")
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("contract_validation", {}).get("valid", False):
        print(f"Independent validation failed: {result / 'validation.json'}")
        return 1
    return 0

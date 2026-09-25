"""Replay the pipeline's in-process logic against one stored run.

python -m regression.replay <stored_run> <output>

Page rendering, RapidOCR, and PaddleOCR are replaced by the stored run's
artifacts, so only Python pipeline logic reruns and the result is
deterministic. Pages the current Paddle selector wants but the stored run never
sent to Paddle are run once through the real worker and cached under
runs/.replay-cache/paddle/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import ocr_pipeline.pipeline as P

CACHE = Path(__file__).resolve().parents[1] / "runs" / ".replay-cache" / "paddle"


def replay(stored: Path, output: Path, paddle_device: str = "gpu:0") -> Path:
    stored = stored.resolve()
    real_paddle = P.run_paddle_table_worker

    def fake_render(pdf, pages, target_dir, dpi, pdftoppm):
        rendered = {}
        for page in pages:
            source = stored / "page-images" / f"page-{page:03d}.png"
            target = target_dir / source.name
            shutil.copy2(source, target)
            rendered[page] = target
        return rendered

    def fake_rapid(jobs, work_dir, python_executable=None, model_root=None):
        data = json.loads((stored / "work" / "rapidocr-results.json").read_text(encoding="utf-8"))
        wanted = {job["page"] for job in jobs}
        data["pages"] = [page for page in data["pages"] if page["page"] in wanted]
        (work_dir / "rapidocr-results.json").write_text(json.dumps(data), encoding="utf-8")
        return data

    def fake_paddle(rendered, target_dir, python_executable=None, device="cpu"):
        target_dir.mkdir(parents=True, exist_ok=True)
        summary = json.loads((stored / "paddle-tables" / "summary.json").read_text(encoding="utf-8"))
        pages = []
        for item in summary["pages"]:
            if item["page"] not in rendered:
                continue
            item = dict(item)
            if item.get("status") == "complete":
                source = stored / "paddle-tables" / Path(item["result"]).name
                target = target_dir / source.name
                shutil.copy2(source, target)
                item["result"] = str(target)
            pages.append(item)
        missing = sorted(set(rendered) - {item["page"] for item in pages})
        if missing:
            cache = CACHE / stored.name
            cache.mkdir(parents=True, exist_ok=True)
            todo = [page for page in missing if not (cache / f"page-{page:03d}.json").is_file()]
            if todo:
                extra = real_paddle({page: rendered[page] for page in todo}, cache / "work", None, paddle_device)
                for item in extra["pages"]:
                    if item.get("status") != "complete":
                        raise RuntimeError(f"real Paddle failed: {item}")
                    shutil.copy2(item["result"], cache / Path(item["result"]).name)
            for page in missing:
                source = cache / f"page-{page:03d}.json"
                target = target_dir / source.name
                shutil.copy2(source, target)
                pages.append({"page": page, "status": "complete", "result": str(target), "replay_filled": True})
            pages.sort(key=lambda item: item["page"])
        summary["pages"] = pages
        summary["python"] = "replay"
        (target_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    P._render_pages = fake_render
    P.run_rapidocr_worker = fake_rapid
    P.run_paddle_table_worker = fake_paddle
    manifest = json.loads((stored / "manifest.json").read_text(encoding="utf-8"))
    pdf = next((stored / "source").glob("*.pdf"))
    return P.run_pipeline(
        pdf, output, pages_spec=None, dpi=manifest["dpi"],
        qwen_endpoint=manifest["vision_model"]["endpoint"], qwen_model=manifest["vision_model"]["model"],
        vision_plan_cache=stored / "vision-plans", paddle_device=manifest["paddle_tables"]["device"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stored", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = replay(args.stored, args.output)
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    print(args.output.name, manifest["contract_validation"])


if __name__ == "__main__":
    main()

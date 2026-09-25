"""Compare the final source blocks of two gate labels, page by page.

python -m regression.diff <label-a> <label-b> [--docs isabella ...] [--verbose]
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs" / "regression"
IGNORED_CONTENT = ("vision_features_ref",)


def _pages(run: Path) -> dict[int, list[dict]]:
    return {json.loads(path.read_text(encoding="utf-8"))["page"]:
            json.loads(path.read_text(encoding="utf-8"))["blocks"]
            for path in sorted((run / "source-blocks").glob("page-*.json"))}


def _strip(block: dict) -> dict:
    block = json.loads(json.dumps(block))
    for key in IGNORED_CONTENT:
        block.get("content", {}).pop(key, None)
    for item in block.get("ocr_disagreements", []):
        item.pop("crop_sha256", None)
    return block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--docs", nargs="*")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    docs = args.docs or sorted(path.name for path in (RUNS / args.left).iterdir() if path.is_dir())
    for doc in docs:
        left, right = _pages(RUNS / args.left / doc), _pages(RUNS / args.right / doc)
        for page in sorted(set(left) | set(right)):
            a = {block["block_id"]: _strip(block) for block in left.get(page, [])}
            b = {block["block_id"]: _strip(block) for block in right.get(page, [])}
            removed, added = sorted(set(a) - set(b)), sorted(set(b) - set(a))
            changed = sorted(key for key in set(a) & set(b) if a[key] != b[key])
            if not (removed or added or changed):
                continue
            types_a = Counter(block["type"] for block in a.values())
            types_b = Counter(block["type"] for block in b.values())
            shape = "" if types_a == types_b else f"  types {dict(types_a)} -> {dict(types_b)}"
            print(f"{doc} p{page:03d}: -{len(removed)} +{len(added)} ~{len(changed)}{shape}")
            if args.verbose:
                for key in changed:
                    fields = [name for name in a[key] if a[key].get(name) != b[key].get(name)]
                    print(f"    ~ {key} {fields}")
                for key in removed:
                    print(f"    - {key} {a[key]['type']}")
                for key in added:
                    print(f"    + {key} {b[key]['type']}")


if __name__ == "__main__":
    main()

"""Replay the regression corpus with the current code and compare it to the baseline.

python -m regression.gate                    replay every document, score, compare
python -m regression.gate --docs isabella    replay a subset (compared per document)
python -m regression.gate --update-baseline  accept the current scores as the new baseline

Outputs land in runs/regression/<label>/<doc>. Only the final source blocks,
manifest and validation are kept, so a full corpus replay uses about 20 MB.

The gate fails when any document crashes or fails validation, when text
fidelity or structure (words reaching structured content, table numbers
landing in cells) drops beyond the tolerances below, when a sign flip or a
misread thousands separator appears, or when a hand-checked reference fact in
golden.json stops holding. Review counts are reported, not gated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .metric import score_run
from .structure import golden_results, structure_metrics

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CORPUS = Path(__file__).with_name("corpus.json")
BASELINE = Path(__file__).with_name("baseline.json")
HEAVY_OUTPUTS = ("page-images", "region-images", "disagreement-crops", "diagnostics", "paddle-tables",
                 "source", "work", "vision-plans", "inspection", "deterministic", "reconciliation")
# Largest tolerated drop, corpus-wide and for any single document.
TOLERANCE = {"recall_owner": (0.002, 0.01), "capture_any": (0.002, 0.01), "precision": (0.002, 0.01)}
FIDELITY = ("recall_owner", "capture_any", "precision", "borrowed", "misread", "duplicated")
STRUCTURE_TOLERANCE = {"ocr_words_structured": (0.002, 0.005), "table_numbers_in_cells": (0.005, 0.02)}
STRUCTURE_WEIGHTS = {"ocr_words_structured": "ocr_words", "table_numbers_in_cells": "table_numbers"}
NEVER_MORE = ("sign_flips", "period_misreads")


def replay(doc: str, stored: str, out_root: Path) -> tuple[str, str | None]:
    out = out_root / doc
    if out.exists():
        shutil.rmtree(out)
    proc = subprocess.run([sys.executable, "-m", "regression.replay", str(RUNS / stored), str(out)],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for sub in HEAVY_OUTPUTS:
        shutil.rmtree(out / sub, ignore_errors=True)
    if proc.returncode:
        return doc, (proc.stderr.strip().splitlines() or ["?"])[-1][:400]
    return doc, None


def structure(run: Path) -> Counter:
    counts = Counter()
    for path in (run / "source-blocks").glob("page-*.json"):
        for block in json.loads(path.read_text(encoding="utf-8"))["blocks"]:
            counts["blocks"] += 1
            counts["review"] += block["validation"]["status"] != "passed"
            counts["fallback"] += block.get("semantic_role") == "unresolved_source_text"
            counts["raw_lines"] += len(block.get("raw_evidence_lines", []))
    validation = json.loads((run / "validation.json").read_text(encoding="utf-8"))
    counts["invalid"] += validation["result"] != "valid"
    return counts


def score(doc: str, stored: str, run: Path) -> dict[str, Any]:
    pdf = next((RUNS / stored / "source").glob("*.pdf"))
    result = score_run(run, pdf)
    pages = result["pages"]
    row: dict[str, Any] = {key: result[key] for key in FIDELITY}
    row["native_tokens"] = sum(page["native_tokens"] for page in pages)
    row["produced"] = sum(page["produced"] for page in pages)
    row.update(structure(run))
    row["structure"] = structure_metrics(run)
    return row


def combine(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    native = sum(row["native_tokens"] for row in rows.values())
    produced = sum(row["produced"] for row in rows.values())
    total: dict[str, Any] = {}
    for key in ("recall_owner", "capture_any"):
        total[key] = round(sum(row[key] * row["native_tokens"] for row in rows.values()) / max(1, native), 4)
    for key in ("precision", "borrowed", "misread", "duplicated"):
        total[key] = round(sum(row[key] * row["produced"] for row in rows.values()) / max(1, produced), 4)
    for key in ("blocks", "review", "fallback", "raw_lines", "invalid"):
        total[key] = sum(row.get(key, 0) for row in rows.values())
    parts = [row["structure"] for row in rows.values() if "structure" in row]
    if parts:
        shape: dict[str, Any] = {}
        for key, weight in STRUCTURE_WEIGHTS.items():
            shape[key] = round(sum(part[key] * part[weight] for part in parts) / max(1, sum(part[weight] for part in parts)), 4)
        for key in parts[0]:
            if key not in shape:
                shape[key] = sum(part[key] for part in parts)
        total["structure"] = shape
    return total


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    problems = []
    for doc, row in current["documents"].items():
        if row.get("invalid"):
            problems.append(f"{doc}: run failed contract validation")
        base = baseline.get("documents", {}).get(doc)
        if not base:
            continue
        for key, (_overall, per_doc) in TOLERANCE.items():
            if row[key] < base[key] - per_doc:
                problems.append(f"{doc}: {key} {base[key]:.4f} -> {row[key]:.4f}")
        shape, base_shape = row.get("structure", {}), base.get("structure")
        if base_shape:
            for key, (_overall, per_doc) in STRUCTURE_TOLERANCE.items():
                if shape[key] < base_shape[key] - per_doc:
                    problems.append(f"{doc}: {key} {base_shape[key]:.4f} -> {shape[key]:.4f}")
            for key in NEVER_MORE:
                if shape[key] > base_shape[key]:
                    problems.append(f"{doc}: {key} {base_shape[key]} -> {shape[key]}")
    if set(current["documents"]) == set(baseline.get("documents", {})):
        for key, (overall, _per_doc) in TOLERANCE.items():
            if current["total"][key] < baseline["total"][key] - overall:
                problems.append(f"corpus: {key} {baseline['total'][key]:.4f} -> {current['total'][key]:.4f}")
        base_shape = baseline["total"].get("structure")
        if base_shape:
            for key, (overall, _per_doc) in STRUCTURE_TOLERANCE.items():
                if current["total"]["structure"][key] < base_shape[key] - overall:
                    problems.append(f"corpus: {key} {base_shape[key]:.4f} -> {current['total']['structure'][key]:.4f}")
    for fact, result in current.get("golden", {}).items():
        if result["expect"] == "pass" and result["result"] != "pass":
            problems.append(f"reference fact no longer holds: {fact}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", nargs="*", help="subset of corpus documents")
    parser.add_argument("--label", default="current", help="output folder under runs/regression/")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))["documents"]
    docs = {doc: corpus[doc] for doc in (args.docs or corpus)}
    out_root = RUNS / "regression" / args.label
    out_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failures = {doc: error for doc, error in pool.map(lambda item: replay(item[0], item[1], out_root),
                                                        docs.items()) if error}
    rows = {doc: score(doc, stored, out_root / doc) for doc, stored in docs.items() if doc not in failures}
    current = {"documents": rows, "total": combine(rows), "golden": golden_results(out_root, list(rows))}
    (out_root / "scores.json").write_text(json.dumps(current, indent=1), encoding="utf-8")

    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.is_file() else {}
    header = f"{'document':14s} {'recall':>7s} {'capture':>7s} {'precis':>7s} {'dup':>6s} {'blocks':>6s} {'review':>6s}"
    print(header)
    for doc, row in list(rows.items()) + [("TOTAL", current["total"])]:
        base = baseline.get("documents", {}).get(doc) if doc != "TOTAL" else baseline.get("total")
        delta = (f"  vs baseline: recall {row['recall_owner'] - base['recall_owner']:+.4f} "
                 f"precision {row['precision'] - base['precision']:+.4f} review {row['review'] - base['review']:+d}"
                 if base else "")
        print(f"{doc:14s} {row['recall_owner']:7.4f} {row['capture_any']:7.4f} {row['precision']:7.4f} "
              f"{row['duplicated']:6.4f} {row['blocks']:6d} {row['review']:6d}{delta}")
    print(f"\n{'document':14s} {'structured':>10s} {'in cells':>8s} {'tables':>6s} {'empty':>5s} "
          f"{'visuals':>7s} {'w/ data':>7s} {'signs':>5s} {'seps':>4s}")
    for doc, row in list(rows.items()) + [("TOTAL", current["total"])]:
        shape = row.get("structure")
        if shape:
            print(f"{doc:14s} {shape['ocr_words_structured']:10.1%} {shape['table_numbers_in_cells']:8.1%} "
                  f"{shape['tables']:6d} {shape['empty_tables']:5d} {shape['visuals']:7d} "
                  f"{shape['visuals_with_data']:7d} {shape['sign_flips']:5d} {shape['period_misreads']:4d}")
    golden = current["golden"]
    held = sum(item["result"] == "pass" for item in golden.values() if item["expect"] == "pass")
    expected = sum(item["expect"] == "pass" for item in golden.values())
    known = [fact for fact, item in golden.items() if item["expect"] == "known_failure"]
    fixed = [fact for fact in known if golden[fact]["result"] == "pass"]
    print(f"\nreference facts holding: {held}/{expected}; known failures: {len(known) - len(fixed)} still failing, "
          f"{len(fixed)} now pass")
    for fact in fixed:
        print("  now passes (move it to expect=pass in golden.json):", fact)
    problems = [f"{doc}: replay crashed: {error}" for doc, error in failures.items()]
    problems += compare(current, baseline) if baseline else []
    if args.update_baseline:
        if failures:
            print("refusing to update the baseline while documents crash")
            return 1
        merged = {"documents": {**baseline.get("documents", {}), **rows}}
        merged["total"] = combine(merged["documents"])
        merged["golden"] = {**baseline.get("golden", {}), **golden}
        BASELINE.write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
        print(f"baseline updated: {BASELINE}")
        return 0
    for problem in problems:
        print("REGRESSION", problem)
    print("gate:", "FAIL" if problems else "pass")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

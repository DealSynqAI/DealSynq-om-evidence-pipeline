"""Independent text-fidelity score for a USB run using the PDF text layer as reference.

Only pages whose embedded text layer independently agrees with RapidOCR are scored,
so image-only print is not mistaken for hallucination.

Per scored page:
  recall_any    printed words (native tokens) present anywhere in structured block content
  recall_owner  printed words present in the content of the leaf block whose box contains them
  capture_any   like recall_any but also counting raw_evidence_lines (loss vs. "kept somewhere")
  precision     content tokens that match print inside their own block box
  borrowed      content tokens absent from own box but printed elsewhere on page (contamination/dupes)
  misread       content tokens printed nowhere on the page (OCR errors, invented labels)
  duplicated    content tokens printed in the box but emitted more often than printed
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import pickle
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pdfplumber


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokens(text: str) -> list[str]:
    # Words compare as whole tokens. Text layers often split numbers into
    # fragments ("4 00 000"), so digits compare as individual characters.
    folded = fold(text)
    return re.findall(r"[a-z]+", folded) + ["#" + d for d in re.findall(r"\d", folded)]


def content_strings(block: dict[str, Any]) -> list[str]:
    kind, c = block["type"], block.get("content", {})
    out: list[str] = []
    if kind in {"text", "heading", "footnote", "contact"}:
        out.append(c.get("text") or "")
    elif kind == "table":
        out.append(c.get("title") or "")
        out += [col.get("label") or "" for col in c.get("columns", [])
                if col.get("label") not in {"Field", "Value"}]
        for row in c.get("rows", []):
            out += [row.get("section") or "", row.get("label") or ""]
            out += [cell.get("raw_value") or "" for cell in row.get("cells", [])]
    elif kind == "chart":
        out.append(c.get("title") or "")
        for o in c.get("observations", []):
            out += [o.get("category") or "", o.get("series") or "", o.get("raw_value") or ""]
            out.append((o.get("companion_value") or {}).get("raw_value") or "")
    elif kind == "kpi_panel":
        out.append(c.get("title") or "")
        for o in c.get("metrics", []):
            out += [o.get("category") or "", o.get("series") or "", o.get("raw_value") or ""]
    elif kind == "map":
        out.append(c.get("title") or "")
        for o in c.get("bindings", []):
            if o.get("geography_basis") != "reference_geometry":
                out.append(o.get("geography") or "")
            out.append(o.get("raw_value") or "")
        legend = c.get("legend") or {}
        out += [legend.get("title") or ""] + list(legend.get("endpoints") or [])
        out += [a.get("text") or "" for a in c.get("attribution") or []]
    elif kind == "comparison_panel":
        out.append(c.get("title") or "")

        def walk(section):
            out.append(section.get("title") or "")
            if section.get("subsections"):
                for child in section["subsections"]:
                    walk(child)
            else:
                out.append(section.get("text") or "")
        for s in c.get("sections", []):
            walk(s)
    elif kind == "photograph":
        out.append(c.get("caption") or "")
    elif kind in {"brand_mark", "unclassified_visual"}:
        out += list(c.get("visible_text") or [])
    return out


def inside(box, point, margin=0.0):
    x, y, w, h = box
    return x - margin <= point[0] <= x + w + margin and y - margin <= point[1] <= y + h + margin


_CACHE = Path(__file__).resolve().parents[1] / "runs" / ".replay-cache" / "native"


def native_words(pdf: Path) -> dict[int, list[tuple[str, tuple[float, float]]]]:
    key = _CACHE / (hashlib.sha256(pdf.read_bytes()).hexdigest()[:20] + ".pkl")
    if key.is_file():
        return pickle.loads(key.read_bytes())
    result = _native_words(pdf)
    _CACHE.mkdir(parents=True, exist_ok=True)
    key.write_bytes(pickle.dumps(result))
    return result


def _native_words(pdf: Path) -> dict[int, list[tuple[str, tuple[float, float]]]]:
    result: dict[int, list] = {}
    with pdfplumber.open(pdf) as doc:
        for number, page in enumerate(doc.pages, 1):
            words = []
            for w in page.extract_words(use_text_flow=False) or []:
                cx = (float(w["x0"]) + float(w["x1"])) / 2 * 1000 / float(page.width)
                cy = (float(w["top"]) + float(w["bottom"])) / 2 * 1000 / float(page.height)
                for t in tokens(w["text"]):
                    words.append((t, (cx, cy)))
            result[number] = words
    return result


def score_page(page: dict[str, Any], words: list) -> dict[str, Any] | None:
    ocr = Counter(t for line in page["evidence_ledger"]["ocr_lines"]
                  if "-ocr-" in str(line.get("evidence_id", "")) for t in tokens(line.get("text", "")))
    native = Counter(t for t, _ in words)
    if sum(native.values()) < 30 or not ocr:
        return None
    agreement = sum((ocr & native).values()) / sum(ocr.values())
    if agreement < 0.80:
        return None
    leaves = [b for b in page["blocks"] if b["type"] != "group"]
    block_tokens = {b["block_id"]: Counter(t for s in content_strings(b) for t in tokens(s)) for b in leaves}
    raw_tokens = Counter(t for b in page["blocks"] for r in b.get("raw_evidence_lines", [])
                         for t in tokens(r.get("text", "")))
    all_content = Counter()
    for c in block_tokens.values():
        all_content += c
    total = len(words)
    recall_any = sum((native & all_content).values()) / total
    capture_any = sum((native & (all_content + raw_tokens)).values()) / total
    # owner recall: native word must appear in a leaf block containing it
    remaining = {k: Counter(v) for k, v in block_tokens.items()}
    owned_hit = 0
    no_owner = 0
    for t, point in words:
        owners = [b for b in leaves if inside(b["coordinates"], point, 3)]
        if not owners:
            no_owner += 1
            continue
        owners.sort(key=lambda b: b["coordinates"][2] * b["coordinates"][3])
        for b in owners:
            if remaining[b["block_id"]][t] > 0:
                remaining[b["block_id"]][t] -= 1
                owned_hit += 1
                break
    # precision / contamination per block
    in_box = borrowed = misread = duplicated = 0
    by_type: dict[str, Counter] = {}
    worst: list[tuple[int, str, str, list[str]]] = []
    for b in leaves:
        bt = block_tokens[b["block_id"]]
        if not bt:
            continue
        local = Counter(t for t, p in words if inside(b["coordinates"], p, 10))
        if sum(local.values()) < 0.5 * sum(bt.values()):
            # Mostly image-rendered print: the text layer cannot referee it.
            s = by_type.setdefault(b["type"], Counter())
            s["unscorable"] += sum(bt.values())
            continue
        hit = bt & local
        rest = bt - hit
        repeated = Counter({t: n for t, n in rest.items() if t in local})
        rest = rest - repeated
        elsewhere = rest & native
        wrong = rest - elsewhere
        in_box += sum(hit.values()); borrowed += sum(elsewhere.values()); misread += sum(wrong.values())
        duplicated += sum(repeated.values())
        s = by_type.setdefault(b["type"], Counter())
        s["in_box"] += sum(hit.values()); s["borrowed"] += sum(elsewhere.values())
        s["misread"] += sum(wrong.values()); s["duplicated"] += sum(repeated.values())
        bad = sum(elsewhere.values()) + sum(wrong.values()) + sum(repeated.values())
        if bad:
            worst.append((bad, b["block_id"], b["type"], sorted((elsewhere + wrong + repeated).elements())[:12]))
    produced = in_box + borrowed + misread + duplicated
    return {
        "page": page["page"], "native_tokens": total, "ocr_native_agreement": round(agreement, 3),
        "recall_any": recall_any, "recall_owner": owned_hit / total, "capture_any": capture_any,
        "no_owner_region": no_owner / total,
        "produced": produced, "precision": in_box / max(1, produced),
        "borrowed": borrowed / max(1, produced), "misread": misread / max(1, produced),
        "duplicated": duplicated / max(1, produced),
        "by_type": {k: dict(v) for k, v in by_type.items()},
        "worst": sorted(worst, reverse=True)[:6],
    }


def score_run(run: Path, pdf: Path) -> dict[str, Any]:
    words = native_words(pdf)
    pages = []
    for path in sorted((run / "source-blocks").glob("page-*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        scored = score_page(page, words.get(page["page"], []))
        if scored:
            pages.append(scored)
    agg: dict[str, Any] = {"run": run.name, "scored_pages": len(pages)}
    weight = sum(p["native_tokens"] for p in pages)
    produced = sum(p["produced"] for p in pages)
    for key in ("recall_any", "recall_owner", "capture_any", "no_owner_region"):
        agg[key] = round(sum(p[key] * p["native_tokens"] for p in pages) / max(1, weight), 4)
    for key in ("precision", "borrowed", "misread", "duplicated"):
        agg[key] = round(sum(p[key] * p["produced"] for p in pages) / max(1, produced), 4)
    agg["pages"] = pages
    return agg


if __name__ == "__main__":
    # python -m regression.metric <run> <source.pdf>
    r = score_run(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({k: v for k, v in r.items() if k != "pages"}))

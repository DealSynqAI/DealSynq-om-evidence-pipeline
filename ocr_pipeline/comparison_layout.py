"""Reconstruct nested text lanes in comparison panels from positioned PDF words.

The outer PDF table cells are only visual containers. Ownership is inferred from
repeated bullet anchors, heading spans, and each word's page coordinates, never
from a document title, page number, or a particular financial value.
"""

from __future__ import annotations

from bisect import bisect_right
import re
import statistics
import unicodedata
from typing import Any


_PERCENT_RANGE = re.compile(
    r"(?<![\w.])(?P<low>\d+(?:\.\d+)?)\s*%?\s*(?:-|–|—|to)\s*"
    r"(?P<high>\d+(?:\.\d+)?)\s*%(?:\+)?", re.I,
)
_PERCENT_VALUE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*%(?:\+)?")
_BULLETS = {"•", "●", "▪", "◦", "‣", "►", "▸", "➢", "➤", "", "-", "–", "*"}


def _is_bullet(text: str) -> bool:
    value = text.strip()
    return value in _BULLETS or (len(value) == 1 and unicodedata.category(value) == "Co")


def _bullet_candidates(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Discard punctuation inside a sentence before clustering bullet anchors."""
    candidates = []
    for word in words:
        value = str(word.get("text", "")).strip()
        if not _is_bullet(value):
            continue
        if value in {"-", "–", "*"}:
            x, top = (float(item) for item in word["coordinates"][:2])
            if any(
                other is not word and not _is_bullet(str(other.get("text", "")))
                and abs(float(other["coordinates"][1]) - top) <= 3.5
                and 0 <= x - (float(other["coordinates"][0]) + float(other["coordinates"][2])) <= 25
                for other in words
            ):
                continue
        candidates.append(word)
    return candidates


def _box(words: list[dict[str, Any]]) -> list[float]:
    left = min(float(word["coordinates"][0]) for word in words)
    top = min(float(word["coordinates"][1]) for word in words)
    right = max(float(word["coordinates"][0]) + float(word["coordinates"][2]) for word in words)
    bottom = max(float(word["coordinates"][1]) + float(word["coordinates"][3]) for word in words)
    return [round(left, 5), round(top, 5), round(right - left, 5), round(bottom - top, 5)]


def _line_text(words: list[dict[str, Any]]) -> str:
    text = " ".join(str(word["text"]).strip() for word in sorted(words, key=lambda item: item["coordinates"][0])).strip()
    text = re.sub(r"\s+([:;,.)])", r"\1", text)
    return re.sub(r"(?<=\d)\s+(?=(?:st|nd|rd|th)\b)", "", text)


def _lines(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["coordinates"][1], item["coordinates"][0])):
        top = float(word["coordinates"][1])
        group = next((line for line in grouped if abs(float(line[0]["coordinates"][1]) - top) <= 3.5), None)
        if group is None:
            grouped.append([word])
        else:
            group.append(word)
    return sorted(grouped, key=lambda line: min(word["coordinates"][1] for word in line))


def _percent_mentions(text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for match in _PERCENT_RANGE.finditer(text):
        mentions.append({
            "raw": match.group(0), "values": [float(match.group("low")), float(match.group("high"))],
            "unit": "percent", "kind": "range",
        })
        occupied.append(match.span())
    for match in _PERCENT_VALUE.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        mentions.append({
            "raw": match.group(0), "values": [float(match.group(0).split("%", 1)[0].strip())],
            "unit": "percent", "kind": "value",
        })
    return mentions


def _claims_for_lane(
    lane_words: list[dict[str, Any]], markers: list[dict[str, Any]],
    first_claim_number: int,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    ordered = sorted(markers, key=lambda item: item["coordinates"][1])
    for offset, marker in enumerate(ordered):
        start = float(marker["coordinates"][1]) - 2.0
        end = float(ordered[offset + 1]["coordinates"][1]) - 2.0 if offset + 1 < len(ordered) else float("inf")
        members = [
            word for word in lane_words
            if start <= float(word["coordinates"][1]) < end
        ]
        body_words = [word for word in members if word is not marker]
        text = " ".join(_line_text(line) for line in _lines(body_words)).strip()
        label = text.split(":", 1)[0].strip() if ":" in text and text.index(":") <= 70 else None
        claims.append({
            "claim_id": f"c{first_claim_number + offset:03d}",
            "label": label, "text": text,
            "coordinates": _box(members),
            "evidence_ids": [word["evidence_id"] for word in members],
            "numeric_mentions": _percent_mentions(text),
        })
    return claims


def reconstruct_comparison_panel(words: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a grounded category/offer/claim tree, or decline weak layouts."""
    if not words:
        return None
    bullet_words = _bullet_candidates(words)
    if len(bullet_words) < 4:
        return None
    clusters: list[list[dict[str, Any]]] = []
    for word in sorted(bullet_words, key=lambda item: item["coordinates"][0]):
        x = float(word["coordinates"][0])
        cluster = next((items for items in clusters if abs(statistics.median(
            float(item["coordinates"][0]) for item in items
        ) - x) <= 12), None)
        if cluster is None:
            clusters.append([word])
        else:
            cluster.append(word)
    clusters = [cluster for cluster in clusters if len(cluster) >= 2]
    anchors = [statistics.median(float(word["coordinates"][0]) for word in cluster) for cluster in clusters]
    if len(anchors) < 2 or any(right - left < 40 for left, right in zip(anchors, anchors[1:])):
        return None
    boundaries = [anchor - 8 for anchor in anchors[1:]]
    lanes = [
        [word for word in words if bisect_right(boundaries, float(word["coordinates"][0])) == index]
        for index in range(len(anchors))
    ]
    markers = [
        sorted(cluster, key=lambda word: word["coordinates"][1])
        for cluster in clusters
    ]
    first_bullets = [float(items[0]["coordinates"][1]) for items in markers]
    body_sizes = [
        float(word.get("font_size_points") or 0)
        for index, lane in enumerate(lanes) for word in lane
        if float(word["coordinates"][1]) >= first_bullets[index]
        and not _is_bullet(str(word["text"]))
        and float(word.get("font_size_points") or 0) > 0
    ]
    if not body_sizes:
        return None
    body_size = statistics.median(body_sizes)
    pre_words = [
        word for index, lane in enumerate(lanes) for word in lane
        if float(word["coordinates"][1]) < first_bullets[index] - 3
        and not _is_bullet(str(word["text"]))
    ]
    headings: list[dict[str, Any]] = []
    for baseline in _lines(pre_words):
        current: list[dict[str, Any]] = []
        for word in sorted(baseline, key=lambda item: item["coordinates"][0]):
            if current and float(word["coordinates"][0]) - max(
                float(item["coordinates"][0]) + float(item["coordinates"][2]) for item in current
            ) > 45:
                if current:
                    headings.append({"words": current})
                current = []
            current.append(word)
        if current:
            headings.append({"words": current})
    headings = [
        heading for heading in headings
        if statistics.median(float(word.get("font_size_points") or 0) for word in heading["words"])
        >= body_size * 1.17
    ]
    for heading in headings:
        heading["title"] = _line_text(heading["words"])
        heading["coordinates"] = _box(heading["words"])
        box = heading["coordinates"]
        heading["first_lane"] = bisect_right(boundaries, box[0])
        heading["last_lane"] = bisect_right(boundaries, box[0] + box[2] - 1)
        heading["top"] = box[1]
    headings.sort(key=lambda heading: (heading["top"], heading["coordinates"][0]))
    lane_heads: dict[int, dict[str, Any]] = {}
    categories: list[dict[str, Any]] = []
    panel_heading: dict[str, Any] | None = None
    for heading in headings:
        first, last = heading["first_lane"], heading["last_lane"]
        if first == 0 and last == len(lanes) - 1 and len(lanes) >= 3:
            panel_heading = heading
        elif first < last:
            categories.append(heading)
        else:
            lane_heads[first] = heading
    if any(index not in lane_heads and not any(
        category["first_lane"] <= index <= category["last_lane"] for category in categories
    ) for index in range(len(lanes))):
        return None

    lane_claims: list[list[dict[str, Any]]] = []
    claim_number = 1
    for lane, lane_markers in zip(lanes, markers):
        claims = _claims_for_lane(lane, lane_markers, claim_number)
        if len(claims) != len(lane_markers) or any(not claim["text"] for claim in claims):
            return None
        lane_claims.append(claims)
        claim_number += len(claims)

    top_sections: list[dict[str, Any]] = []
    consumed_lanes: set[int] = set()
    section_number = 1
    for lane_index in range(len(lanes)):
        if lane_index in consumed_lanes:
            continue
        category = next((heading for heading in categories if heading["first_lane"] == lane_index), None)
        if category:
            children = []
            for child_index in range(category["first_lane"], category["last_lane"] + 1):
                heading = lane_heads.get(child_index)
                if heading is None:
                    return None
                children.append({
                    "section_id": f"s{section_number:03d}-{child_index - category['first_lane'] + 1:02d}",
                    "title": heading["title"],
                    "text": "\n".join(claim["text"] for claim in lane_claims[child_index]),
                    "structure_complete": True,
                    "coordinates": _box([*heading["words"], *lanes[child_index]]),
                    "evidence_ids": [word["evidence_id"] for word in heading["words"]],
                    "claims": lane_claims[child_index],
                })
                consumed_lanes.add(child_index)
            top_sections.append({
                "section_id": f"s{section_number:03d}", "title": category["title"],
                "text": category["title"], "structure_complete": True,
                "coordinates": _box([*category["words"], *(word for index in range(
                    category["first_lane"], category["last_lane"] + 1
                ) for word in lanes[index])]),
                "evidence_ids": [word["evidence_id"] for word in category["words"]],
                "subsections": children,
            })
        else:
            heading = lane_heads.get(lane_index)
            if heading is None:
                return None
            top_sections.append({
                "section_id": f"s{section_number:03d}", "title": heading["title"],
                "text": "\n".join(claim["text"] for claim in lane_claims[lane_index]),
                "structure_complete": True,
                "coordinates": _box(lanes[lane_index]),
                "evidence_ids": [word["evidence_id"] for word in heading["words"]],
                "claims": lane_claims[lane_index],
            })
            consumed_lanes.add(lane_index)
        section_number += 1
    if len(top_sections) < 2 or len(consumed_lanes) != len(lanes):
        return None
    return {
        "title": panel_heading["title"] if panel_heading else None,
        "sections": top_sections, "lane_count": len(lanes),
        "claim_count": sum(map(len, lane_claims)),
        "leaf_titles": [lane_heads[index]["title"] for index in range(len(lanes))],
        "bullet_anchor_coordinates": [round(anchor, 5) for anchor in anchors],
    }

"""Build text, heading, contact, and page-band blocks from native text and OCR."""

from __future__ import annotations

from collections import Counter
import difflib
from pathlib import Path
import re
import statistics
from typing import Any

from .brand_marks import _brand_visible_text, _looks_like_brand_mark
from .common import (
    _cluster_unassigned_lines, _crop, _fold_token, _line_box, _ocr_text, _provenance,
    _token_agreement, _vision_summary, clean_text,
)
from .line_ownership import _split_visual_text_lines
from .models import PageInspection, Region, SourceBlock


def _hybrid_text(native: str, ocr: str) -> str:
    """Use complete OCR wording while repairing OCR spelling from native PDF tokens."""
    native_words = re.findall(r"[\w’'-]+", native, re.UNICODE)
    by_folded: dict[str, list[str]] = {}
    for word in native_words:
        by_folded.setdefault(_fold_token(word), []).append(word)

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        exact = by_folded.get(_fold_token(word))
        if exact:
            return exact[0]
        candidates = difflib.get_close_matches(_fold_token(word), by_folded, n=1, cutoff=0.90)
        return by_folded[candidates[0]][0] if candidates else word

    return re.sub(r"[\w’'-]+", replace, ocr, flags=re.UNICODE)


def _merge_owned_native_word_insertions(
    native: str, ocr: str, lines: list[dict[str, Any]],
) -> str:
    """Fill native-text omissions only with OCR words also printed as owned PDF words."""
    native_matches = list(re.finditer(r"\w+", native, re.UNICODE))
    ocr_matches = list(re.finditer(r"\w+", ocr, re.UNICODE))
    native_words = [_fold_token(match.group()) for match in native_matches]
    ocr_words = [_fold_token(match.group()) for match in ocr_matches]
    native_counts = Counter(native_words)
    ocr_counts = Counter(ocr_words)
    positioned = Counter(
        _fold_token(word)
        for line in lines if line.get("evidence_source") == "native_pdf_positioned_word"
        for word in re.findall(r"\w+", str(line.get("text", "")), re.UNICODE)
    )
    inserted: Counter[str] = Counter()
    edits: list[tuple[int, str]] = []
    for op, i1, _i2, j1, j2 in difflib.SequenceMatcher(
        None, native_words, ocr_words, autojunk=False,
    ).get_opcodes():
        if op != "insert" or not 1 <= j2 - j1 <= 4:
            continue
        words = ocr_words[j1:j2]
        if not all(
            positioned[word] >= inserted[word] + words.count(word)
            and native_counts[word] + inserted[word] + words.count(word) <= ocr_counts[word]
            for word in set(words)
        ):
            continue
        insertion = " ".join(match.group() for match in ocr_matches[j1:j2])
        offset = native_matches[i1].start() if i1 < len(native_matches) else len(native)
        edits.append((offset, insertion))
        inserted.update(words)
    merged = native
    for offset, insertion in sorted(edits, reverse=True):
        merged = merged[:offset] + (insertion + " " if offset < len(merged) else " " + insertion) + merged[offset:]
    return merged


def _append_distinct_ocr_identifiers(
    raw: str, lines: list[dict[str, Any]],
) -> str:
    """Retain separately printed OCR URLs or emails omitted by native PDF text."""
    identifier = re.compile(r"(?:https?://)?(?:www\.)?(?:[\w-]+\.)+[A-Za-z]{2,}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.I)
    existing = {token.strip(".,;:()[]{}<>|").casefold()
                for token in re.split(r"\s+", raw) if token.strip()}
    additions: list[str] = []
    for line in sorted(lines, key=lambda item: (
        float(item["coordinates"][1]), float(item["coordinates"][0]),
    )):
        if "-ocr-" not in str(line.get("evidence_id", "")) or float(line.get("confidence", 0.0)) < 0.90:
            continue
        candidate = str(line.get("text", "")).strip(" \t\n|,;")
        if not identifier.fullmatch(candidate):
            continue
        folded = candidate.casefold()
        if folded in existing:
            continue
        additions.append(candidate)
        existing.add(folded)
    return raw + ("\n" + "\n".join(additions) if additions else "")


def _text_structure(raw: str) -> dict[str, Any] | None:
    """Recover paragraph and bullet semantics from line breaks without document vocabulary."""
    normalized_paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", raw)
        if paragraph.strip()
    ]
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines() if line.strip()]
    # Symbol fonts commonly expose bullets/arrows as Unicode private-use
    # characters. At the start of a line they carry list structure, not text.
    lines = [re.sub(r"^[\uF000-\uF8FF]\s*", "• ", line) for line in lines]
    bullet_indexes = [index for index, line in enumerate(lines) if re.match(r"^(?:[-•▪‣]|\d+[.)])\s+", line)]
    if not bullet_indexes:
        if len(normalized_paragraphs) <= 1:
            return None
        return {"paragraphs": normalized_paragraphs, "lists": []}
    first = bullet_indexes[0]
    intro_index = first - 1 if first and lines[first - 1].endswith(":") else None
    prose_lines = lines[:intro_index if intro_index is not None else first]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in prose_lines:
        current.append(line)
        if re.search(r"[.!?][\"'’)]?$", line):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    items = [
        re.sub(r"^(?:[-•▪‣]|\d+[.)])\s+", "", lines[index]).strip()
        for index in bullet_indexes
    ]
    return {
        "paragraphs": paragraphs,
        "lists": [{
            "intro": lines[intro_index] if intro_index is not None else None,
            "ordered": bool(re.match(r"^\d+[.)]\s+", lines[first])),
            "items": items,
        }],
    }


def _block_type_for_text(region: Region, candidate_text: str | None = None) -> str:
    text = clean_text(region.native_text if candidate_text is None else candidate_text)
    font = float(region.metadata.get("median_font_size", 0) or 0)
    word_count = int(region.metadata.get("word_count", len(text.split())) or 0)
    y = region.coordinates[1]
    if y >= 890 or (font and font <= 8 and y >= 820):
        return "footnote"
    if "@" in text or re.search(r"\b\d{3}[-.) ]\d{3}[- ]\d{4}\b", text):
        return "contact"
    if word_count <= 14 and not text.endswith((".", ",", ";", ":")) and (font >= 14 or text.isupper() or text.istitle()):
        return "heading"
    return "text"


def _semantic_role_for_text(
    block_type: str, text: str, coordinates: list[float], page_number: int, nested: bool = False,
) -> str:
    folded = text.casefold()
    legal_signals = {
        "confidential", "important note", "legal notice", "disclaimer", "offer to sell",
        "solicitation", "securities", "offering materials", "terms and conditions",
    }
    if sum(signal in folded for signal in legal_signals) >= 2:
        return "legal_notice"
    if block_type == "heading":
        if nested:
            return "panel_heading"
        if coordinates[1] <= 220:
            return "document_title" if page_number == 1 else "page_title"
        return "section_heading"
    if block_type == "footnote":
        return "footnote"
    if block_type == "contact":
        return "contact_information"
    return "body_text"


def _inline_heading_subsection_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, native_threshold: float,
) -> list[SourceBlock] | None:
    """Split repeated inline all-caps labels into semantic subsection trees."""
    native_lines = [line.strip() for line in region.native_text.splitlines() if line.strip()]
    heading_pattern = re.compile(r"^([A-Z][A-Z0-9 &/'-]{2,}?):\s*(.*)$")
    headings = [
        (index, match.group(1).strip(), match.group(2).strip())
        for index, line in enumerate(native_lines)
        if (match := heading_pattern.match(line))
    ]
    # Repetition is the structural evidence. A lone colon-led line can be ordinary prose.
    if len(headings) < 2:
        return None
    ordered_ocr = sorted(lines, key=lambda line: (float(line["coordinates"][1]), float(line["coordinates"][0])))
    if len(ordered_ocr) != len(native_lines):
        return None

    positive_steps = [
        float(current["coordinates"][1]) - float(previous["coordinates"][1])
        for previous, current in zip(ordered_ocr, ordered_ocr[1:])
        if float(current["coordinates"][1]) > float(previous["coordinates"][1])
    ]
    typical_step = statistics.median(positive_steps) if positive_steps else 0.0
    blocks: list[SourceBlock] = []
    for position, (start, heading_text, first_body_text) in enumerate(headings, 1):
        end = headings[position][0] if position < len(headings) else len(native_lines)
        section_ocr = [dict(line) for line in ordered_ocr[start:end]]
        if not section_ocr:
            return None
        body_lines = [first_body_text, *native_lines[start + 1:end]]
        body_parts: list[str] = []
        for line_index, body_line in enumerate(body_lines):
            if line_index and typical_step:
                previous = section_ocr[line_index - 1]
                current = section_ocr[line_index]
                step = float(current["coordinates"][1]) - float(previous["coordinates"][1])
                if step > typical_step * 1.35:
                    body_parts.append("")
            body_parts.append(body_line)
        body_native = "\n".join(body_parts).strip()
        if not body_native:
            return None

        first_ocr = str(section_ocr[0].get("text", ""))
        ocr_match = heading_pattern.match(first_ocr)
        ocr_heading = ocr_match.group(1).strip() if ocr_match else heading_text
        section_ocr[0]["text"] = ocr_match.group(2).strip() if ocr_match else first_ocr
        section_box = _line_box(section_ocr)
        first_box = [float(value) for value in ordered_ocr[start]["coordinates"]]
        heading_fraction = min(0.72, max(0.12, (len(heading_text) + 1) / max(1, len(first_ocr))))
        heading_box = [first_box[0], first_box[1], first_box[2] * heading_fraction, first_box[3]]
        group_id = f"{region.region_id}-subsection-{position:03d}"
        heading_id = f"{group_id}-heading"
        body_region = Region(
            region_id=f"{group_id}-body", page=region.page, kind="normal_text",
            coordinates=section_box, reading_order=region.reading_order + position,
            classification_method="inline-heading subsection reconstruction",
            confidence=region.confidence, native_text=body_native,
            metadata={
                "word_count": len(re.findall(r"\w+", body_native, re.UNICODE)),
                "source_bbox_points": region.metadata.get("source_bbox_points"),
            },
        )
        body = _text_block(
            document_id, source_hash, inspection, body_region, section_ocr, image, native_threshold,
        )
        body.parent_block_id = group_id
        body.hierarchy_depth = 1
        heading_agreement = _token_agreement(heading_text, ocr_heading)
        heading_confidence = min(
            inspection.native_text_quality,
            0.5 + heading_agreement / 2 if heading_agreement is not None else region.confidence,
        )
        heading = SourceBlock(
            document_id=document_id, type="heading", page=region.page, block_id=heading_id,
            content={
                "text": heading_text,
                "evidence_text": {
                    "selected": "native", "native": heading_text, "ocr": ocr_heading,
                    "token_agreement": heading_agreement,
                },
            },
            coordinates=heading_box,
            extraction_method=["native PDF text", "Python inline-heading reconstruction"],
            confidence=heading_confidence, validation_status="passed",
            provenance=_provenance(source_hash, body_region, [], image),
            semantic_role="section_heading", parent_block_id=group_id,
            hierarchy_depth=1, heading_level=2,
        )
        group = SourceBlock(
            document_id=document_id, type="group", page=region.page, block_id=group_id,
            content={"role": "subsection", "child_block_ids": [heading_id, body.block_id]},
            coordinates=section_box,
            extraction_method=["Python inline-heading hierarchy"],
            confidence=min(heading.confidence, body.confidence),
            validation_status="passed" if body.validation_status == "passed" else "needs_review",
            warnings=[] if body.validation_status == "passed" else ["one or more child blocks require review"],
            provenance=_provenance(source_hash, body_region, [], image),
            semantic_role="document_subsection", child_block_ids=[heading_id, body.block_id],
        )
        blocks.extend([group, heading, body])
    return blocks


def _profile_biography_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, native_threshold: float,
) -> list[SourceBlock] | None:
    """Split a name-and-role lead line from the biography that follows it."""
    native_lines = [line.strip() for line in region.native_text.splitlines() if line.strip()]
    if len(native_lines) < 2 or len(" ".join(native_lines[1:]).split()) < 20:
        return None
    rx, ry, rw, rh = region.coordinates
    adjacent_portrait = any(
        candidate.kind == "visual"
        and str(candidate.metadata.get("visual_hint") or "").casefold() == "photograph"
        and (
            max(0.0, min(ry + rh, candidate.coordinates[1] + candidate.coordinates[3])
                - max(ry, candidate.coordinates[1]))
            / max(1.0, min(rh, candidate.coordinates[3]))
        ) >= 0.45
        and -20 <= rx - (candidate.coordinates[0] + candidate.coordinates[2]) <= 250
        for candidate in inspection.regions
    )
    if not adjacent_portrait:
        return None
    heading_text = native_lines[0]
    if not re.fullmatch(r"[^.!?:]{2,80}\s+[–—-]\s+[^.!?:]{2,60}", heading_text):
        return None
    if not 2 <= len(heading_text.split()) <= 12:
        return None
    ordered_ocr = sorted(lines, key=lambda line: (float(line["coordinates"][1]), float(line["coordinates"][0])))
    if len(ordered_ocr) < 2:
        return None
    heading_ocr = ordered_ocr[0]
    body_ocr = ordered_ocr[1:]
    body_native = "\n".join(native_lines[1:])
    group_id = f"{region.region_id}-profile-biography"
    heading_id = f"{group_id}-heading"
    body_region = Region(
        region_id=f"{group_id}-body", page=region.page, kind="normal_text",
        coordinates=_line_box(body_ocr), reading_order=region.reading_order + 1,
        classification_method="name-role biography reconstruction",
        confidence=region.confidence, native_text=body_native,
        metadata={
            "word_count": len(re.findall(r"\w+", body_native, re.UNICODE)),
            "source_bbox_points": region.metadata.get("source_bbox_points"),
        },
    )
    body = _text_block(
        document_id, source_hash, inspection, body_region, body_ocr, image, native_threshold,
    )
    body.parent_block_id = group_id
    body.hierarchy_depth = 1
    heading_ocr_text = str(heading_ocr.get("text", "")).strip()
    agreement = _token_agreement(heading_text, heading_ocr_text)
    heading = SourceBlock(
        document_id=document_id, type="heading", page=region.page, block_id=heading_id,
        content={
            "text": heading_text,
            "evidence_text": {
                "selected": "native", "native": heading_text, "ocr": heading_ocr_text or None,
                "token_agreement": agreement,
            },
        },
        coordinates=[float(value) for value in heading_ocr["coordinates"]],
        extraction_method=["native PDF text", "RapidOCR", "Python name-role heading reconstruction"],
        confidence=min(inspection.native_text_quality, 0.5 + agreement / 2 if agreement is not None else region.confidence),
        validation_status="passed", provenance=_provenance(source_hash, region, [heading_ocr], image),
        semantic_role="profile_name_and_role", parent_block_id=group_id,
        hierarchy_depth=1, heading_level=2,
    )
    group = SourceBlock(
        document_id=document_id, type="group", page=region.page, block_id=group_id,
        content={"role": "profile_biography", "child_block_ids": [heading_id, body.block_id]},
        coordinates=region.coordinates, extraction_method=["Python profile-biography hierarchy"],
        confidence=min(heading.confidence, body.confidence),
        validation_status="passed" if body.validation_status == "passed" else "needs_review",
        warnings=[] if body.validation_status == "passed" else ["one or more child blocks require review"],
        provenance=_provenance(source_hash, region, [], image), semantic_role="document_subsection",
        child_block_ids=[heading_id, body.block_id],
    )
    return [group, heading, body]


def _leading_heading_body_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, native_threshold: float,
) -> list[SourceBlock] | None:
    """Split a visually separated lead heading from the text that follows."""
    native_lines = [line.strip() for line in region.native_text.splitlines() if line.strip()]
    if len(native_lines) < 2:
        return None
    heading_text = native_lines[0]
    if (
        not 1 <= len(heading_text.split()) <= 14
        or heading_text.endswith((".", ",", ";", ":"))
    ):
        return None
    ordered_ocr = sorted(lines, key=lambda line: (float(line["coordinates"][1]), float(line["coordinates"][0])))
    if len(ordered_ocr) < 2:
        return None
    first_box = [float(value) for value in ordered_ocr[0]["coordinates"]]
    second_box = [float(value) for value in ordered_ocr[1]["coordinates"]]
    gap = second_box[1] - (first_box[1] + first_box[3])
    median_height = statistics.median(float(line["coordinates"][3]) for line in ordered_ocr)
    if gap < max(12.0, median_height * 0.60):
        return None

    body_lines = ordered_ocr[1:]
    body_native = "\n".join(native_lines[1:])
    group_id = f"{region.region_id}-heading-body"
    heading_id = f"{group_id}-heading"
    body_region = Region(
        region_id=f"{group_id}-body", page=region.page, kind="normal_text",
        coordinates=_line_box(body_lines), reading_order=region.reading_order + 1,
        classification_method="leading-heading whitespace reconstruction",
        confidence=region.confidence, native_text=body_native,
        metadata={
            "word_count": len(re.findall(r"\w+", body_native, re.UNICODE)),
            "source_bbox_points": region.metadata.get("source_bbox_points"),
        },
    )
    body = _text_block(
        document_id, source_hash, inspection, body_region, body_lines, image, native_threshold,
    )
    body.parent_block_id = group_id
    body.hierarchy_depth = 1
    ocr_heading = str(ordered_ocr[0].get("text", "")).strip()
    agreement = _token_agreement(heading_text, ocr_heading)
    heading_role = _semantic_role_for_text("heading", heading_text, first_box, region.page)
    heading_level = 1 if heading_role in {"document_title", "page_title"} else 2
    heading = SourceBlock(
        document_id=document_id, type="heading", page=region.page, block_id=heading_id,
        content={
            "text": heading_text,
            "evidence_text": {
                "selected": "native", "native": heading_text, "ocr": ocr_heading or None,
                "token_agreement": agreement,
            },
        },
        coordinates=first_box,
        extraction_method=["native PDF text", "RapidOCR", "Python leading-heading reconstruction"],
        confidence=min(
            inspection.native_text_quality,
            0.5 + agreement / 2 if agreement is not None else region.confidence,
        ),
        validation_status="passed", provenance=_provenance(source_hash, region, [ordered_ocr[0]], image),
        semantic_role=heading_role, parent_block_id=group_id, hierarchy_depth=1,
        heading_level=heading_level,
    )
    group = SourceBlock(
        document_id=document_id, type="group", page=region.page, block_id=group_id,
        content={"role": "section", "child_block_ids": [heading_id, body.block_id]},
        coordinates=region.coordinates,
        extraction_method=["Python leading-heading hierarchy"],
        confidence=min(heading.confidence, body.confidence),
        validation_status="passed" if body.validation_status == "passed" else "needs_review",
        warnings=[] if body.validation_status == "passed" else ["one or more child blocks require review"],
        provenance=_provenance(source_hash, region, [], image), semantic_role="document_section",
        child_block_ids=[heading_id, body.block_id],
    )
    return [group, heading, body]


def _parse_contact_details(raw_text: str, normalized: str) -> dict[str, Any]:
    """Recover common contact fields while preserving the original text as evidence."""
    lines = [re.sub(r"\s+", " ", line).strip(" |│") for line in raw_text.splitlines() if line.strip()]
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", normalized)
    phone_match = re.search(r"\b(?:\+?1[-. )]*)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b", normalized)
    website_match = re.search(
        r"(?<![@\w])(?:https?://|www\.)?[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}(?:/[^\s|│]*)?",
        normalized, re.I,
    )
    organization = None
    suffix = re.compile(r"\b(?:LLC|L\.L\.C\.|INC\.?|CORP\.?|LTD\.?|LLP|LP|PLC)\b", re.I)
    for line in lines[:3]:
        if not any(character.isdigit() for character in line) and (
            suffix.search(line) or (line.isupper() and 1 <= len(line.split()) <= 10)
        ):
            organization = line.rstrip(",")
            break

    name = None
    if lines and (email_match or phone_match):
        # Contact cards commonly print a person's name before a separator,
        # followed by email/phone. Do not infer a person from an organization,
        # address, or unstructured sentence.
        first = re.split(r"\s*[|│]\s*", lines[0], maxsplit=1)[0].strip()
        pieces = first.split()
        person_token = re.compile(r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*$", re.UNICODE)
        if (
            2 <= len(pieces) <= 4
            and all(person_token.fullmatch(piece) for piece in pieces)
            and not first.isupper()
            and not suffix.search(first)
            and first != organization
        ):
            name = first

    street = city = state = postal_code = None
    street_suffix = re.compile(
        r"\b(?:STREET|ST|AVENUE|AVE|ROAD|RD|DRIVE|DR|LANE|LN|BOULEVARD|BLVD|"
        r"PARKWAY|PKWY|HIGHWAY|HWY|COURT|CT|CIRCLE|CIR|TRAIL|TRL|WAY|SUITE|STE)\b",
        re.I,
    )
    for index, line in enumerate(lines):
        if re.search(r"\d", line) and street_suffix.search(line):
            street = line
            if index + 1 < len(lines):
                locality = re.match(r"^(.*?)(?:,\s*|\s+)([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$", lines[index + 1])
                if locality:
                    city, state, postal_code = locality.group(1).strip(), locality.group(2), locality.group(3)
            break
    address = None
    if any((street, city, state, postal_code)):
        address = {"street": street, "city": city, "state": state, "postal_code": postal_code}
    website = website_match.group(0) if website_match else None
    if website and not re.match(r"https?://", website, re.I):
        website = f"https://{website}"
    return {
        "name": name,
        "organization": organization,
        "address": address,
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
        "website": website,
    }


def _text_block(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, native_threshold: float,
) -> SourceBlock:
    visible_ocr = _ocr_text(lines)
    agreement = _token_agreement(region.native_text, visible_ocr)
    use_native = bool(region.native_text.strip()) and inspection.native_text_quality >= native_threshold
    selected = "native" if use_native else "ocr"
    warnings: list[str] = []
    if use_native and agreement is not None and agreement < 0.35:
        use_native = False
        selected = "ocr"
        warnings.append("native text disagrees with visible OCR; OCR selected")
    native_word_count = len(re.findall(r"\w+", region.native_text, re.UNICODE))
    ocr_word_count = len(re.findall(r"\w+", visible_ocr, re.UNICODE))
    repaired_native = (
        _merge_owned_native_word_insertions(region.native_text, visible_ocr, lines)
        if use_native and agreement is not None and agreement >= 0.65 else region.native_text
    )
    if use_native and repaired_native != region.native_text:
        raw = repaired_native
        selected = "hybrid"
        use_native = False
    elif use_native and agreement is not None and agreement >= 0.35 and ocr_word_count >= native_word_count + 2:
        raw = _hybrid_text(region.native_text, visible_ocr)
        selected = "hybrid"
        use_native = False
    else:
        raw = region.native_text if use_native else visible_ocr
    if selected in {"native", "hybrid"}:
        supplemented = _append_distinct_ocr_identifiers(raw, lines)
        if supplemented != raw:
            raw = supplemented
            selected = "hybrid"
            use_native = False
    normalized = clean_text(raw)
    if selected == "hybrid":
        methods = ["native PDF text", "RapidOCR", "PP-OCRv6", "Python token reconciliation", "Python layout normalization"]
    elif use_native:
        methods = ["native PDF text", "Python layout normalization"]
    else:
        methods = ["RapidOCR", "PP-OCRv6", "Python layout normalization"]
    errors = [] if normalized else ["no text recovered from region"]
    source_confidence = inspection.native_text_quality if selected in {"native", "hybrid"} else (
        sum(float(line["confidence"]) for line in lines) / len(lines) if lines else 0.0
    )
    confidence = min(source_confidence, 0.5 + agreement / 2) if agreement is not None else source_confidence
    block_type = _block_type_for_text(region, normalized)
    word_count = int(region.metadata.get("word_count", len(normalized.split())) or 0)
    if block_type == "text" and word_count <= 3:
        warnings.append("short text fragment may have been detached from an adjacent region")
    if (
        region.coordinates[2] >= 700 and inspection.possible_visual_regions >= 2 and word_count >= 20
        and region.classification_method not in {
            "visual-text-whitespace-segmentation",
            "name-role biography reconstruction",
        }
    ):
        warnings.append("wide text spans a mixed visual layout; reading order requires review")
    if "�" in normalized:
        warnings.append("text contains an invalid replacement character")
    content: dict[str, Any] = {
        "text": normalized,
        "evidence_text": {
            "selected": selected,
            "native": region.native_text or None,
            "ocr": visible_ocr or None,
            "token_agreement": agreement,
        },
    }
    structure = _text_structure(raw)
    if structure:
        content["structure"] = structure
    if block_type == "contact":
        content.update(_parse_contact_details(raw, normalized))
    return SourceBlock(
        document_id=document_id, type=block_type, page=region.page,
        block_id=f"{region.region_id}-block",
        content=content, coordinates=region.coordinates,
        extraction_method=methods, confidence=confidence,
        validation_status="passed" if normalized and not warnings else "needs_review", errors=errors, warnings=warnings,
        provenance=_provenance(source_hash, region, lines, image),
        semantic_role=_semantic_role_for_text(block_type, normalized, region.coordinates, region.page),
        heading_level=1 if block_type == "heading" and region.coordinates[1] <= 220 else (
            2 if block_type == "heading" else None
        ),
    )


def _visual_text_panel_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, page_lines: list[dict[str, Any]],
) -> list[SourceBlock]:
    """Keep whitespace-separated image text in independent physical regions."""
    groups = _split_visual_text_lines(lines)
    if len(groups) <= 1:
        region.native_text = ""
        return [_text_block(document_id, source_hash, inspection, region, lines, image, 1.1)]

    blocks: list[SourceBlock] = []
    for index, group_lines in enumerate(groups, 1):
        coordinates = _line_box(group_lines)
        text = clean_text(_ocr_text(group_lines))
        subregion = Region(
            region_id=f"{region.region_id}-s{index:03d}", page=region.page,
            kind="normal_text", coordinates=coordinates, reading_order=region.reading_order,
            classification_method="visual-text-whitespace-segmentation",
            confidence=sum(float(line.get("confidence", 0.0)) for line in group_lines) / len(group_lines),
            metadata={
                "word_count": len(re.findall(r"\S+", text)),
                "median_font_size": statistics.median(
                    float(line["coordinates"][3]) * inspection.height_points / 1000.0 for line in group_lines
                ),
                "source_bbox_points": [
                    coordinates[0] * inspection.width_points / 1000.0,
                    coordinates[1] * inspection.height_points / 1000.0,
                    (coordinates[0] + coordinates[2]) * inspection.width_points / 1000.0,
                    (coordinates[1] + coordinates[3]) * inspection.height_points / 1000.0,
                ],
            },
        )
        block = _text_block(document_id, source_hash, inspection, subregion, group_lines, image, 1.1)
        block.semantic_role = (
            "brand_mark_text" if _looks_like_brand_mark(group_lines, page_lines)
            else _semantic_role_for_text(block.type, text, coordinates, region.page)
        )
        blocks.append(block)
    return blocks


def _spatial_residual_text_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], image: Path, native_threshold: float,
) -> list[SourceBlock]:
    """Keep text around separate tables as independent printed regions."""
    clusters = _cluster_unassigned_lines(list(enumerate(lines)))
    groups = [segment for cluster in clusters
              for segment in _split_visual_text_lines([line for _position, line in cluster])]
    if len(groups) <= 1:
        return [_text_block(document_id, source_hash, inspection, region, lines, image, native_threshold)]
    blocks: list[SourceBlock] = []
    for index, group_lines in enumerate(groups, 1):
        box = _line_box(group_lines)
        subregion = Region(
            region_id=f"{region.region_id}-s{index:03d}", page=region.page,
            kind="normal_text", coordinates=box, reading_order=region.reading_order + index,
            classification_method="spatial text grouping around table regions",
            confidence=region.confidence, metadata={
                "word_count": len(re.findall(r"\w+", _ocr_text(group_lines), re.UNICODE)),
                "source_bbox_points": [
                    box[0] * inspection.width_points / 1000,
                    box[1] * inspection.height_points / 1000,
                    (box[0] + box[2]) * inspection.width_points / 1000,
                    (box[1] + box[3]) * inspection.height_points / 1000,
                ],
            },
        )
        block = _text_block(
            document_id, source_hash, inspection, subregion, group_lines, image, native_threshold,
        )
        blocks.append(block)
    return blocks


def _cluster_spatial_lines(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Cluster nearby OCR lines into independently readable footer/header elements."""
    groups: list[list[dict[str, Any]]] = []
    for line in sorted(lines, key=lambda item: (float(item["coordinates"][0]), float(item["coordinates"][1]))):
        lx, ly, lw, lh = (float(value) for value in line["coordinates"])
        match = None
        for group in groups:
            gx, gy, gw, gh = _line_box(group)
            horizontal_overlap = max(0.0, min(lx + lw, gx + gw) - max(lx, gx))
            centers_close = abs((lx + lw / 2) - (gx + gw / 2)) <= max(lw, gw) * 0.75
            vertical_gap = max(0.0, ly - (gy + gh), gy - (ly + lh))
            if (horizontal_overlap > 0 or centers_close) and vertical_gap <= max(40.0, 1.5 * max(lh, gh / len(group))):
                match = group
                break
        if match is None:
            groups.append([line])
        else:
            match.append(line)
    return groups


def _native_words_for_box(region: Region, coordinates: list[float]) -> str:
    """Return positioned native words whose centers fall inside a normalized box."""
    x, y, width, height = coordinates
    selected = []
    for word in region.metadata.get("native_visual_words", []):
        wx, wy, ww, wh = (float(value) for value in word.get("coordinates", [0, 0, 0, 0]))
        center_x, center_y = wx + ww / 2, wy + wh / 2
        if x <= center_x <= x + width and y <= center_y <= y + height:
            selected.append(word)
    return " ".join(
        str(word.get("text", ""))
        for word in sorted(selected, key=lambda item: (float(item["coordinates"][1]), float(item["coordinates"][0])))
    ).strip()


def _semantic_page_band_blocks(
    document_id: str, source_hash: str, inspection: PageInspection, region: Region,
    lines: list[dict[str, Any]], page_lines: list[dict[str, Any]], image: Path,
    crops: Path, features: dict[str, Any], vision_features_ref: dict[str, str],
    native_threshold: float, document_token_pages: Counter[str] | None,
) -> list[SourceBlock]:
    """Decompose a repeated text-bearing page band instead of discarding it as decoration."""
    if not lines or int(region.metadata.get("semantic_text_overlap_count", 0)) <= 0:
        return []
    clusters = _cluster_spatial_lines(lines)
    if not clusters:
        return []
    parent_id = f"{region.region_id}-page-band"
    children: list[SourceBlock] = []
    for index, cluster in enumerate(clusters, 1):
        coordinates = _line_box(cluster)
        child_region = Region(
            region_id=f"{region.region_id}-band-{index:03d}", page=region.page,
            kind="normal_text", coordinates=coordinates,
            reading_order=region.reading_order + index,
            classification_method="semantic repeated-page-band decomposition",
            confidence=sum(float(line.get("confidence", 0.0)) for line in cluster) / len(cluster),
            native_text=_native_words_for_box(region, coordinates),
            metadata={"source_bbox_points": region.metadata.get("source_bbox_points")},
        )
        if _looks_like_brand_mark(cluster, page_lines, document_token_pages):
            crop_path = crops / f"{child_region.region_id}.png"
            _crop(image, coordinates, crop_path)
            child = SourceBlock(
                document_id=document_id, type="brand_mark", page=region.page,
                block_id=f"{child_region.region_id}-brand-mark",
                content={
                    "visible_text": _brand_visible_text(cluster, document_token_pages),
                    "evidence_mode": "visual_region",
                    "vision_summary": _vision_summary(features),
                    "vision_features_ref": vision_features_ref,
                    "region_image": f"region-images/{crop_path.name}",
                },
                coordinates=coordinates,
                extraction_method=["RapidOCR", "PP-OCRv6", "Python repeated-band decomposition"],
                confidence=max(0.82, child_region.confidence), validation_status="passed",
                # The shared diagnostic describes the parent visual band, so its
                # provenance region must match that diagnostic's region identity.
                provenance=_provenance(source_hash, region, cluster, image),
                semantic_role="brand_mark",
            )
        else:
            child = _text_block(
                document_id, source_hash, inspection, child_region, cluster, image, native_threshold,
            )
            # A known repeated page band is navigation text, not a source
            # footnote even when it sits at the page bottom.
            if child.type == "footnote":
                child.type = "text"
            child.semantic_role = "running_footer" if coordinates[1] >= 500 else "running_header"
        child.parent_block_id = parent_id
        child.hierarchy_depth = 1
        children.append(child)
    if not children:
        return []
    parent = SourceBlock(
        document_id=document_id, type="group", page=region.page, block_id=parent_id,
        content={"role": "page_footer" if region.coordinates[1] >= 500 else "page_header",
                 "child_block_ids": [child.block_id for child in children]},
        coordinates=region.coordinates,
        extraction_method=["Python semantic repeated-page-band decomposition"],
        confidence=min(child.confidence for child in children),
        validation_status="passed" if all(child.validation_status == "passed" for child in children) else "needs_review",
        warnings=[] if all(child.validation_status == "passed" for child in children) else ["one or more child blocks require review"],
        provenance=_provenance(source_hash, region, [], image),
        semantic_role="page_footer" if region.coordinates[1] >= 500 else "page_header",
        child_block_ids=[child.block_id for child in children],
    )
    return [parent, *children]

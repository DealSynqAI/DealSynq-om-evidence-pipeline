from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile
from PIL import Image
from jsonschema import Draft202012Validator

from ocr_pipeline import __version__
from ocr_pipeline.comparison_layout import reconstruct_comparison_panel
from ocr_pipeline.inspection import (
    detect_aligned_financial_table,
    detect_small_raster_chart,
    detect_vector_bar_chart,
    mark_repeated_decorations,
    merge_visual_objects,
    _sparse_page_table,
    _partition_words_at_filled_bands,
    _merge_numeric_fragments,
    _mark_repeated_portrait_visuals,
    _group_words,
    _normalized_xywh,
)
from ocr_pipeline.models import PageInspection, Region, SourceBlock, validate_source_blocks
from ocr_pipeline.workers import normalize_vision_payload
from ocr_pipeline.pipeline import (
    COLLECTION_SCHEMA_VERSION,
    INSPECTION_SCHEMA_VERSION,
    INSPECTION_INDEX_SCHEMA_VERSION,
    PAGE_SCHEMA_VERSION,
    PIPELINE_VERSION,
    VISION_DIAGNOSTIC_SCHEMA_VERSION,
    _bar_bindings,
    _classify_visual,
    _brand_visible_text,
    _table_total_reconciliation,
    _table_structure_errors,
    _table_blocks,
    _exclusive_region_lines,
    _hybrid_text,
    _merge_owned_native_word_insertions,
    _append_distinct_ocr_identifiers,
    _ocr_text,
    _kpi_bindings,
    _map_data_value_lines,
    _reconstruct_vertical_values,
    _reconstruct_vertical_words,
    _infer_chart_type,
    _inline_heading_subsection_blocks,
    _ground_model_bindings,
    _numeric_value,
    _provenance,
    _proximity_bindings,
    _qwen_semantic_prompt,
    _reconcile_visual_bindings,
    _visual_title,
    _vision_summary,
    _write_vision_diagnostic,
    _looks_like_brand_mark,
    _leading_heading_body_blocks,
    _parse_contact_details,
    _profile_biography_blocks,
    _recover_unassigned_brand_marks,
    _semantic_role_for_text,
    _normalize_page_heading_roles,
    _order_overlapping_visual_headings,
    _pie_label_value_grounding,
    _suppress_visual_observation_text_duplicates,
    _semantic_page_band_blocks,
    _arrange_page_blocks,
    _split_visual_text_lines,
    _text_structure,
    clean_text,
    parse_pages,
)
from ocr_pipeline.validate_run import _resolve_run_path
from ocr_pipeline.map_geometry import _albers_point, _state_rings, register_us_state_map, state_for_value


class PipelineUnitTests(unittest.TestCase):
    @staticmethod
    def _synthetic_panel_words() -> list[dict[str, object]]:
        words: list[dict[str, object]] = []

        def word(text: str, x: float, y: float, width: float = 80, size: float = 11) -> None:
            words.append({
                "evidence_id": f"synthetic-w{len(words) + 1:03d}", "text": text,
                "coordinates": [x, y, width, 18], "font_size_points": size,
            })

        word("Evergreen", 150, 100, 190, 18)
        word("Investments", 345, 100, 130, 18)
        word("Direct SPVs", 700, 100, 170, 14)
        word("Senior Units", 50, 150, 130, 14)
        word("Class B Shares", 380, 150, 145, 14)
        for anchor, label, value in (
            (50, "Fixed Returns: 6.5% - 8.0%", "Liquidity: 10%"),
            (380, "Preferred Return: 8%", "Targeted Returns: 10% to 15%"),
            (700, "Cash Returns: 9-13%+", "LP Total Returns: 14.2%"),
        ):
            word("•", anchor, 200, 8)
            word(label, anchor + 18, 201, 190)
            word("•", anchor, 250, 8)
            word(value, anchor + 18, 251, 190)
        return words

    def test_general_comparison_panel_recovers_nested_offers_and_metric_ownership(self) -> None:
        words = self._synthetic_panel_words()
        layout = reconstruct_comparison_panel(words)
        self.assertIsNotNone(layout)
        self.assertEqual(layout["lane_count"], 3)
        self.assertEqual(layout["claim_count"], 6)
        self.assertEqual([section["title"] for section in layout["sections"]], [
            "Evergreen Investments", "Direct SPVs",
        ])
        evergreen, direct = layout["sections"]
        self.assertEqual([offer["title"] for offer in evergreen["subsections"]], [
            "Senior Units", "Class B Shares",
        ])
        self.assertEqual(evergreen["subsections"][0]["claims"][0]["numeric_mentions"], [{
            "raw": "6.5% - 8.0%", "values": [6.5, 8.0], "unit": "percent", "kind": "range",
        }])
        self.assertEqual(direct["claims"][0]["numeric_mentions"][0]["values"], [9.0, 13.0])
        self.assertNotIn("Preferred Return", evergreen["subsections"][0]["text"])

        region = Region("p001-r001", 1, "table", [30, 80, 940, 600], 1, "synthetic", 0.68,
                        metadata={"rows": [["left background", "right background"]],
                                  "native_panel_words": words})
        review = {
            "type": "table", "panel_review": {"lane_count": 3,
                "leaf_titles": ["Senior Units", "Class B Shares", "Direct SPVs"],
                "claim_counts": [2, 2, 2], "structure_matches": True},
        }
        block = _table_blocks("synthetic-doc", "0" * 64, region, [], Path("page.png"), review)[0]
        self.assertEqual(block.validation_status, "passed")
        schema = json.loads((Path(__file__).resolve().parents[1] /
                             "schemas/unified-source-block.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(block.as_dict())), [])

        review["panel_review"]["structure_matches"] = False
        flag_only = _table_blocks("synthetic-doc", "0" * 64, region, [], Path("page.png"), review)[0]
        self.assertEqual(flag_only.validation_status, "passed")
        self.assertIn("flag is false", " ".join(flag_only.warnings))
        review["panel_review"]["claim_counts"] = [2, 1, 2]
        conflicted = _table_blocks("synthetic-doc", "0" * 64, region, [], Path("page.png"), review)[0]
        self.assertEqual(conflicted.validation_status, "needs_review")

    def test_comparison_layout_declines_non_bulleted_two_cell_visual(self) -> None:
        words = [
            {"evidence_id": "w1", "text": "Left", "coordinates": [50, 100, 80, 18], "font_size_points": 14},
            {"evidence_id": "w2", "text": "Right", "coordinates": [550, 100, 80, 18], "font_size_points": 14},
        ]
        self.assertIsNone(reconstruct_comparison_panel(words))
        region = Region("p001-r001", 1, "table", [30, 80, 940, 600], 1, "synthetic", 0.68,
                        metadata={"rows": [["Left\nbody", "Right\nbody"]],
                                  "native_panel_words": words})
        block = _table_blocks("synthetic-doc", "0" * 64, region, [], Path("page.png"))[0]
        self.assertEqual(block.validation_status, "needs_review")
        self.assertEqual([section["title"] for section in block.content["sections"]], ["Left", "Right"])

    def test_vision_first_versions(self) -> None:
        self.assertEqual(__version__, "0.1.0")
        self.assertEqual(PIPELINE_VERSION, "0.1.0")
        self.assertEqual(PAGE_SCHEMA_VERSION, "unified-source-page/4.0")
        self.assertEqual(COLLECTION_SCHEMA_VERSION, "unified-source-collection/3.0")
        self.assertEqual(INSPECTION_SCHEMA_VERSION, "pdf-page-inspection/1.0")
        self.assertEqual(INSPECTION_INDEX_SCHEMA_VERSION, "pdf-inspection-index/1.0")
        self.assertEqual(VISION_DIAGNOSTIC_SCHEMA_VERSION, "opencv-region-diagnostic/1.0")

    def test_page_parser(self) -> None:
        self.assertEqual(parse_pages("1-3,5,3", 6), [1, 2, 3, 5])
        with self.assertRaises(ValueError):
            parse_pages("7", 6)

    def test_run_artifact_paths_are_portable(self) -> None:
        region = Region("p001-r001", 1, "visual", [0, 0, 10, 10], 1, "test", 0.8)
        provenance = _provenance("0" * 64, region, [], Path("C:/render/page-001.png"))
        self.assertEqual(provenance["rendered_page"], "page-images/page-001.png")
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.assertEqual(_resolve_run_path(run, "source/input.pdf"), run / "source/input.pdf")

    def test_ocr_cleanup_dehyphenates_and_joins(self) -> None:
        raw = "The firm tar-\ngets income produc-\ning real estate.\n\nNext paragraph."
        self.assertEqual(
            clean_text(raw),
            "The firm targets income producing real estate.\n\nNext paragraph.",
        )

    def test_hybrid_text_keeps_ocr_omissions_and_repairs_native_spelling(self) -> None:
        native = "offer investors to a portfolio of seasoned assets"
        ocr = "offer investors access to a portfolío of seasoned assèts"
        self.assertEqual(
            _hybrid_text(native, ocr),
            "offer investors access to a portfolio of seasoned assets",
        )

    def test_bullet_structure_is_preserved(self) -> None:
        structure = _text_structure(
            "First paragraph.\nSecond paragraph.\nCapital stack:\n- senior debt\n- preferred equity"
        )
        self.assertEqual(structure["paragraphs"], ["First paragraph.", "Second paragraph."])
        self.assertEqual(structure["lists"][0], {
            "intro": "Capital stack:", "ordered": False,
            "items": ["senior debt", "preferred equity"],
        })

    def test_inline_all_caps_labels_create_subsection_hierarchy(self) -> None:
        native = (
            "OPERATING MODEL: First paragraph begins here.\n"
            "It continues on this line.\n"
            "A second paragraph starts after whitespace.\n"
            "It also continues.\n"
            "RISK CONTROLS: Another section starts here.\n"
            "It finishes here."
        )
        lines = [
            {"evidence_id": f"e{index}", "text": text, "confidence": 0.99,
             "coordinates": [50, y, 800, 25]}
            for index, (text, y) in enumerate([
                ("OPERATING MODEL: First paragraph begins here.", 180),
                ("It continues on this line.", 215),
                ("A second paragraph starts after whitespace.", 270),
                ("It also continues.", 305),
                ("RISK CONTROLS: Another section starts here.", 360),
                ("It finishes here.", 395),
            ], 1)
        ]
        inspection = PageInspection(2, 720, 540, 0, True, 0.9, 1.0, 0.0, 0, 0, 0, 0.95, [], [])
        region = Region("p002-r001", 2, "normal_text", [50, 180, 800, 240], 1, "native", 0.95, native)
        blocks = _inline_heading_subsection_blocks(
            "doc", "0" * 64, inspection, region, lines, Path("page-002.png"), 0.78,
        )
        self.assertIsNotNone(blocks)
        self.assertEqual([block.type for block in blocks], ["group", "heading", "text"] * 2)
        self.assertEqual(blocks[0].content["role"], "subsection")
        self.assertEqual(blocks[1].content["text"], "OPERATING MODEL")
        self.assertEqual(blocks[2].content["structure"]["paragraphs"], [
            "First paragraph begins here. It continues on this line.",
            "A second paragraph starts after whitespace. It also continues.",
        ])
        self.assertEqual(validate_source_blocks([block.as_dict() for block in blocks]), [])

    def test_text_bearing_footer_decomposes_into_brand_and_running_text(self) -> None:
        inspection = PageInspection(2, 720, 540, 0, True, 0.2, 1.0, 0.2, 0, 0, 1, 0.9, [], [])
        region = Region(
            "p002-r001", 2, "decoration", [0, 870, 1000, 130], 3, "repeated image", 0.98,
            metadata={
                "semantic_text_overlap_count": 4,
                "native_visual_words": [
                    {"text": "Quarterly", "coordinates": [750, 930, 80, 25]},
                    {"text": "Review", "coordinates": [835, 930, 60, 25]},
                    {"text": "2026", "coordinates": [900, 930, 45, 25]},
                ],
            },
        )
        lines = [
            {"evidence_id": "brand-1", "text": "ACMEHARBOR", "confidence": 0.99, "coordinates": [50, 910, 180, 25]},
            {"evidence_id": "brand-2", "text": "ČAPITAL", "confidence": 0.99, "coordinates": [50, 940, 100, 25]},
            {"evidence_id": "footer", "text": "Quarterly Review 2026", "confidence": 0.99, "coordinates": [750, 930, 195, 25]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "page-002.png"
            crops = root / "regions"
            crops.mkdir()
            Image.new("RGB", (1000, 1000), "white").save(image)
            blocks = _semantic_page_band_blocks(
                "doc", "0" * 64, inspection, region, lines, lines, image, crops, {},
                {"path": "diagnostics/vision.json", "sha256": "0" * 64}, 0.78,
                __import__("collections").Counter({"acme": 2, "harbor": 2, "capital": 2}),
            )
        self.assertEqual(blocks[0].content["role"], "page_footer")
        self.assertEqual([block.type for block in blocks[1:]], ["brand_mark", "text"])
        self.assertEqual(blocks[1].content["visible_text"], ["ACME HARBOR", "CAPITAL"])
        self.assertEqual(blocks[2].semantic_role, "running_footer")
        self.assertEqual(validate_source_blocks([block.as_dict() for block in blocks]), [])

    def test_common_contract_carries_explicit_hierarchy(self) -> None:
        block = SourceBlock(
            document_id="doc", type="chart", page=1, block_id="chart",
            content={"title": "Allocation", "chart_type": "pie", "slice_geometry_status": "unresolved", "observations": []},
            coordinates=[0, 0, 1000, 1000], extraction_method=["test"], confidence=0.8,
            validation_status="needs_review", provenance={"source_sha256": "0" * 64, "region_id": "r1"},
        ).as_dict()
        self.assertEqual(block["hierarchy"], {
            "parent_block_id": None, "child_block_ids": [], "depth": 0, "heading_level": None,
        })
        self.assertEqual(block["semantic_role"], "data_visualization")
        self.assertEqual(validate_source_blocks([block]), [])

    def test_hierarchy_requires_reciprocal_parent_child_links(self) -> None:
        parent = SourceBlock(
            document_id="doc", type="group", page=1, block_id="parent",
            content={"role": "mixed_text_panel", "child_block_ids": ["child"]},
            coordinates=[0, 0, 100, 100], extraction_method=["test"], confidence=0.8,
            validation_status="passed", child_block_ids=["child"],
        ).as_dict()
        child = SourceBlock(
            document_id="doc", type="text", page=1, block_id="child",
            content={"text": "Body", "evidence_text": {"selected": "ocr", "native": None, "ocr": "Body", "token_agreement": None}},
            coordinates=[10, 10, 20, 20], extraction_method=["test"], confidence=0.8,
            validation_status="passed", parent_block_id="parent", hierarchy_depth=1,
        ).as_dict()
        self.assertEqual(validate_source_blocks([parent, child]), [])
        child["hierarchy"]["parent_block_id"] = None
        self.assertIn("not reciprocal", " ".join(validate_source_blocks([parent, child])))

    def test_image_backed_text_splits_on_meaningful_vertical_whitespace(self) -> None:
        lines = [
            {"evidence_id": "a", "text": "BRAND", "coordinates": [100, 100, 100, 20]},
            {"evidence_id": "b", "text": "NAME", "coordinates": [100, 120, 100, 20]},
            {"evidence_id": "c", "text": "LEGAL HEADING", "coordinates": [100, 180, 200, 20]},
            {"evidence_id": "d", "text": "Body line one", "coordinates": [100, 230, 500, 16]},
            {"evidence_id": "e", "text": "Body line two", "coordinates": [100, 247, 500, 16]},
        ]
        self.assertEqual([len(group) for group in _split_visual_text_lines(lines)], [2, 1, 2])

    def test_image_backed_text_keeps_parallel_columns_separate(self) -> None:
        lines = [
            {"evidence_id": f"left-{i}", "text": f"Left line {i}",
             "coordinates": [110, 300 + i * 25, 330, 22]}
            for i in range(4)
        ] + [
            {"evidence_id": f"right-{i}", "text": f"Right line {i}",
             "coordinates": [550, 300 + i * 25, 330, 22]}
            for i in range(4)
        ]
        groups = _split_visual_text_lines(lines)
        self.assertEqual(len(groups), 2)
        self.assertEqual([[line["evidence_id"] for line in group] for group in groups], [
            [f"left-{i}" for i in range(4)],
            [f"right-{i}" for i in range(4)],
        ])

    def test_private_use_symbol_font_markers_become_list_structure(self) -> None:
        structure = _text_structure(
            "\uf0d8 First general claim\n\uf0d8 Second general claim\n\uf0d8 Third general claim"
        )

        self.assertIsNotNone(structure)
        self.assertEqual(
            structure["lists"][0]["items"],
            ["First general claim", "Second general claim", "Third general claim"],
        )

    def test_name_role_hierarchy_requires_adjacent_portrait_geometry(self) -> None:
        region = Region(
            "text", 1, "normal_text", [300, 200, 600, 300], 1, "native", 0.99,
            native_text=(
                "Jordan Lee — Operations Lead\n"
                "Jordan leads planning delivery controls reporting analysis coordination quality review "
                "training documentation scheduling communication governance testing support research service "
                "maintenance and improvement."
            ),
        )
        inspection = PageInspection(
            1, 720, 540, 0, True, 0.8, 0.99, 0.2, 0, 0, 1, 0.9, [region], [],
        )
        lines = [
            {"evidence_id": "h", "text": "Jordan Lee - Operations Lead", "coordinates": [300, 200, 300, 30], "confidence": 0.99},
            {"evidence_id": "b", "text": "Jordan leads planning delivery controls reporting analysis coordination quality review training documentation scheduling communication governance testing support research service maintenance and improvement.", "coordinates": [300, 250, 600, 180], "confidence": 0.99},
        ]

        self.assertIsNone(
            _profile_biography_blocks("doc", "0" * 64, inspection, region, lines, Path("page.png"), 0.78)
        )
        inspection.regions.append(Region(
            "photo", 1, "visual", [80, 205, 170, 250], 2, "image", 0.8,
            metadata={"visual_hint": "photograph"},
        ))
        blocks = _profile_biography_blocks(
            "doc", "0" * 64, inspection, region, lines, Path("page.png"), 0.78,
        )
        self.assertIsNotNone(blocks)
        self.assertEqual(blocks[0].content["role"], "profile_biography")

    def test_whitespace_separated_page_heading_uses_general_section_hierarchy(self) -> None:
        region = Region(
            "text", 2, "normal_text", [60, 70, 850, 260], 1, "native", 0.99,
            native_text="PERFORMANCE SUMMARY\n\uf0d8 First claim\n\uf0d8 Second claim",
        )
        inspection = PageInspection(
            2, 720, 540, 0, True, 0.8, 0.99, 0.2, 0, 0, 1, 0.9, [region], [],
        )
        lines = [
            {"evidence_id": "h", "text": "PERFORMANCE SUMMARY", "coordinates": [60, 70, 400, 35], "confidence": 0.99},
            {"evidence_id": "b1", "text": "First claim", "coordinates": [100, 145, 300, 35], "confidence": 0.99},
            {"evidence_id": "b2", "text": "Second claim", "coordinates": [100, 185, 300, 35], "confidence": 0.99},
        ]

        blocks = _leading_heading_body_blocks(
            "doc", "0" * 64, inspection, region, lines, Path("page.png"), 0.78,
        )

        self.assertIsNotNone(blocks)
        self.assertEqual(blocks[0].content["role"], "section")
        self.assertEqual(blocks[1].semantic_role, "page_title")
        self.assertEqual(blocks[2].content["structure"]["lists"][0]["items"], ["First claim", "Second claim"])

    def test_brand_mark_requires_short_repeated_page_text(self) -> None:
        logo = [
            {"evidence_id": "logo-1", "text": "ACME", "coordinates": [100, 100, 50, 20]},
            {"evidence_id": "logo-2", "text": "CAPITAL", "coordinates": [100, 120, 70, 20]},
        ]
        page = logo + [{"evidence_id": "title", "text": "ACME CAPITAL", "coordinates": [20, 20, 200, 30]}]
        self.assertTrue(_looks_like_brand_mark(logo, page))
        self.assertFalse(_looks_like_brand_mark(
            [{"evidence_id": "chart", "text": "PORTFOLIO 2025", "coordinates": [100, 100, 100, 20]}], page,
        ))

    def test_brand_mark_can_use_strong_footer_corner_evidence(self) -> None:
        logo = [
            {"evidence_id": "logo-1", "text": "ACME CREEK", "coordinates": [710, 870, 190, 28]},
            {"evidence_id": "logo-2", "text": "CAPITAL", "coordinates": [710, 902, 100, 25]},
        ]
        self.assertTrue(_looks_like_brand_mark(logo, logo))

    def test_brand_text_uses_document_vocabulary_to_split_concatenation(self) -> None:
        visible = _brand_visible_text(
            [{"text": "ACMECREEK"}, {"text": "CAPITAL"}],
            __import__("collections").Counter({"acme": 2, "creek": 2, "capital": 2}),
        )
        self.assertEqual(visible, ["ACME CREEK", "CAPITAL"])

    def test_brand_text_follows_spatial_reading_order(self) -> None:
        visible = _brand_visible_text([
            {"text": "CAPITAL", "coordinates": [100, 930, 100, 25]},
            {"text": "ACME CREEK", "coordinates": [100, 900, 180, 25]},
        ], None)
        self.assertEqual(visible, ["ACME CREEK", "CAPITAL"])

    def test_later_page_prominent_heading_is_page_title(self) -> None:
        self.assertEqual(_semantic_role_for_text("heading", "OVERVIEW", [50, 60, 200, 40], 2), "page_title")
        self.assertEqual(_semantic_role_for_text("heading", "REPORT", [50, 60, 200, 40], 1), "document_title")

    def test_contact_parser_recovers_organization_address_phone_and_website(self) -> None:
        raw = (
            "ACME CAPITAL, LLC\n5613 DTC PARKWAY, SUITE 830\n"
            "GREENWOOD VILLAGE, CO 80111\n720-502-1149 | ACMECAPITAL.COM"
        )
        result = _parse_contact_details(raw, raw.replace("\n", " "))
        self.assertEqual(result["organization"], "ACME CAPITAL, LLC")
        self.assertEqual(result["address"], {
            "street": "5613 DTC PARKWAY, SUITE 830", "city": "GREENWOOD VILLAGE",
            "state": "CO", "postal_code": "80111",
        })
        self.assertEqual(result["phone"], "720-502-1149")
        self.assertEqual(result["website"], "https://ACMECAPITAL.COM")
        self.assertIsNone(result["email"])

    def test_unassigned_repeated_logo_text_is_recovered(self) -> None:
        inspection = PageInspection(2, 720, 540, 0, True, 0.1, 1.0, 0.2, 0, 0, 1, 0.58, [], [])
        lines = [
            {"evidence_id": "title", "text": "ACME CAPITAL", "confidence": 1.0, "coordinates": [50, 50, 200, 30]},
            {"evidence_id": "logo-1", "text": "ACME", "confidence": 0.99, "coordinates": [700, 850, 100, 25]},
            {"evidence_id": "logo-2", "text": "CAPITAL", "confidence": 0.99, "coordinates": [700, 878, 120, 25]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "page-002.png"
            crops = root / "region-images"
            crops.mkdir()
            Image.new("RGB", (1000, 1000), "white").save(image)
            assigned = {"title"}
            recovered = _recover_unassigned_brand_marks(
                "doc", "0" * 64, inspection, image, lines, assigned, crops,
            )
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].type, "brand_mark")
        self.assertEqual(recovered[0].content["evidence_mode"], "ocr_recovery")
        self.assertEqual(assigned, {"title", "logo-1", "logo-2"})

    def test_normalized_coordinates_are_clipped_to_page_edges(self) -> None:
        x, y, width, height = _normalized_xywh((-4, 470, 724, 542), 720, 540)
        self.assertEqual(x, 0.0)
        self.assertEqual(y + height, 1000.0)
        self.assertEqual(x + width, 1000.0)

    def test_common_contract_rejects_embedded_raw_vision(self) -> None:
        block = SourceBlock(
            document_id="doc", type="chart", page=1, block_id="chart",
            content={"vision_features": {"line_segments": []}}, coordinates=[0, 0, 10, 10],
            extraction_method=["test"], confidence=0.8, validation_status="passed",
            provenance={"source_sha256": "0" * 64, "region_id": "r1"},
        ).as_dict()
        self.assertIn("raw vision features", " ".join(validate_source_blocks([block])))

    def test_decoration_rejects_semantic_labels(self) -> None:
        block = SourceBlock(
            document_id="doc", type="decoration", page=1, block_id="decoration",
            content={"title": "giant footer label"}, coordinates=[0, 0, 10, 10],
            extraction_method=["test"], confidence=0.8, validation_status="passed",
        ).as_dict()
        self.assertIn("decoration cannot carry semantic text", " ".join(validate_source_blocks([block])))

    def test_vision_summary_contains_counts_not_arrays(self) -> None:
        summary = _vision_summary({
            "horizontal_lines": 4, "vertical_lines": 2, "bar_candidates": 3,
            "line_segments": {"horizontal": [[0, 0, 1, 1]]},
        })
        self.assertEqual(summary["horizontal_line_count"], 4)
        self.assertEqual(summary["bar_candidate_count"], 3)
        self.assertNotIn("line_segments", summary)

    def test_vision_payload_is_normalized_to_routing_contract(self) -> None:
        payload = normalize_vision_payload({
            "type": " Chart ", "confidence": "1.4", "chart_type": " BAR ",
            "bindings": [{"label": "A", "value": "1"}, "invalid"], "extra": "discarded",
            "table_review": {"data_row_count": 3, "column_count": 2},
        })
        self.assertEqual(payload["type"], "chart")
        self.assertEqual(payload["confidence"], 1.0)
        self.assertEqual(payload["chart_type"], "bar")
        self.assertEqual(payload["bindings"], [{"label": "A", "value": "1"}])
        self.assertEqual(payload["table_review"], {"data_row_count": 3, "column_count": 2})
        self.assertNotIn("extra", payload)

    def test_semantic_vision_prompt_uses_ocr_evidence_after_geometry(self) -> None:
        region = Region("p001-r001", 1, "visual", [0, 0, 1000, 800], 1, "geometry", 0.9)
        prompt = _qwen_semantic_prompt("chart", [
            {"evidence_id": "label-1", "text": "Office", "coordinates": [100, 100, 60, 20]},
            {"evidence_id": "value-1", "text": "12.5%", "coordinates": [100, 130, 50, 20]},
        ], region)
        self.assertIn("after OCR and OpenCV geometry", prompt)
        self.assertIn("label-1", prompt)
        self.assertIn("value_evidence_id", prompt)

    def test_qwen_bindings_are_grounded_and_reconciled_without_replacement(self) -> None:
        lines = [
            {"evidence_id": "label-1", "text": "Office", "coordinates": [100, 100, 60, 20]},
            {"evidence_id": "value-1", "text": "12.5%", "coordinates": [100, 130, 50, 20]},
            {"evidence_id": "label-2", "text": "Retail", "coordinates": [300, 100, 60, 20]},
            {"evidence_id": "value-2", "text": "7.5%", "coordinates": [300, 130, 50, 20]},
        ]
        deterministic = _proximity_bindings(lines[:2], None, [[80, 80, 120, 120]])
        payload = {
            "type": "chart", "confidence": 0.92, "chart_type": "pie",
            "bindings": [
                {"label": "Office", "value": "12.5%", "label_evidence_id": "label-1", "value_evidence_id": "value-1", "confidence": 0.9},
                {"label": "Retail", "value": "7.5%", "label_evidence_id": "label-2", "value_evidence_id": "value-2", "confidence": 0.9},
            ],
        }
        model = _ground_model_bindings("chart", payload, lines, [[80, 80, 120, 120], [280, 80, 120, 120]])
        reconciled, warnings = _reconcile_visual_bindings(deterministic, model)
        self.assertEqual(len(reconciled), 2)
        self.assertIn("Qwen-confirmed", reconciled[0]["grounding_method"])
        self.assertIn("1 confirmed, 1 added, 0 conflicted", " ".join(warnings))

    def test_conflicting_qwen_owners_for_one_value_are_rejected(self) -> None:
        lines = [{"evidence_id": "value-1", "text": "12.5%", "coordinates": [100, 130, 50, 20]}]
        payload = {
            "type": "map", "confidence": 0.9, "chart_type": None,
            "bindings": [
                {"label": "Texas", "label_evidence_id": None, "value_evidence_id": "value-1"},
                {"label": "Florida", "label_evidence_id": None, "value_evidence_id": "value-1"},
            ],
        }
        self.assertEqual(_ground_model_bindings("map", payload, lines, []), [])

    def test_map_legend_endpoints_are_not_data_values(self) -> None:
        lines = [
            {"evidence_id": "legend", "text": "% of Portfolio", "coordinates": [400, 100, 100, 20]},
            {"evidence_id": "low", "text": "0%", "coordinates": [450, 130, 30, 20]},
            {"evidence_id": "high", "text": "41%", "coordinates": [550, 130, 40, 20]},
            {"evidence_id": "state", "text": "5.4%", "coordinates": [200, 400, 50, 20]},
        ]
        self.assertEqual([line["evidence_id"] for line in _map_data_value_lines(lines)], ["state"])

    def test_registered_map_geometry_owns_only_colored_state_marks(self) -> None:
        import cv2
        import numpy as np

        rings = {
            name: [[_albers_point(*point) for point in ring] for ring in members]
            for name, members in _state_rings().items()
        }
        points = [point for members in rings.values() for ring in members for point in ring]
        minimum_x, maximum_x = min(p[0] for p in points), max(p[0] for p in points)
        minimum_y, maximum_y = min(p[1] for p in points), max(p[1] for p in points)
        sx, sy = 720 / (maximum_x - minimum_x), 300 / (maximum_y - minimum_y)
        image = np.full((440, 900, 3), 255, np.uint8)
        colorado_mask = np.zeros(image.shape[:2], np.uint8)
        for name, members in rings.items():
            for ring in members:
                vertices = np.array([
                    [round(80 + sx * (x - minimum_x)), round(70 + sy * (y - minimum_y))]
                    for x, y in ring
                ], np.int32)
                cv2.fillPoly(image, [vertices], (225, 200, 185) if name == "Colorado" else (224, 224, 224))
                cv2.polylines(image, [vertices], True, (255, 255, 255), 2)
                if name == "Colorado":
                    cv2.fillPoly(colorado_mask, [vertices], 1)
        moments = cv2.moments(colorado_mask)
        center = [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        coordinates = [center[0] / 900 * 1000 - 10, center[1] / 440 * 1000 - 10, 20, 20]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic-us-map.png"
            cv2.imwrite(str(path), image)
            registration = register_us_state_map(path)
            self.assertIsNotNone(registration)
            self.assertGreater(registration["quality"], 0.90)
            self.assertEqual(registration["projection"], "conus_albers")
            owner = state_for_value(registration, coordinates, [0, 0, 1000, 1000])
            self.assertIsNotNone(owner)
            self.assertEqual(owner[0], "Colorado")
            self.assertEqual(owner[2], "registered_state_polygon")

    def test_us_reference_rejects_unrelated_rectangular_visual(self) -> None:
        import cv2
        import numpy as np

        image = np.full((440, 900, 3), 255, np.uint8)
        cv2.rectangle(image, (80, 70), (800, 370), (225, 200, 185), -1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-a-us-state-map.png"
            cv2.imwrite(str(path), image)
            self.assertIsNone(register_us_state_map(path))

    def test_diagnostic_explains_coordinate_formats_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            diagnostics = run / "diagnostics" / "vision-features"
            diagnostics.mkdir(parents=True)
            region = Region("p001-r001", 1, "visual", [10, 20, 30, 40], 1, "test", 0.8)
            ref = _write_vision_diagnostic(
                diagnostics, "doc", "0" * 64, region,
                {"line_segments": {"horizontal": [[0, 1, 2, 3]]}},
            )
            payload = __import__("json").loads((run / ref["path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], VISION_DIAGNOSTIC_SCHEMA_VERSION)
            self.assertIn("page-relative", payload["coordinate_formats"]["region_coordinates"])
            self.assertIn("region-relative", payload["coordinate_formats"]["line_segments"])
            self.assertEqual(len(ref["sha256"]), 64)

    def test_overlapping_visual_objects_merge(self) -> None:
        merged = merge_visual_objects(
            [(100, 100, 300, 300), (200, 150, 400, 350), (700, 100, 800, 200)], 1000, 1000,
        )
        self.assertEqual(sorted(len(item["image_indices"]) for item in merged), [1, 2])

    def test_footer_band_does_not_merge_with_chart(self) -> None:
        merged = merge_visual_objects(
            [(200, 300, 700, 850), (0, 855, 1000, 1000)], 1000, 1000,
        )
        self.assertEqual(len(merged), 2)

    def test_page_wide_background_band_does_not_merge_with_chart(self) -> None:
        merged = merge_visual_objects(
            [(0, 190, 960, 410), (80, 130, 570, 475)], 960, 540,
        )
        self.assertEqual(len(merged), 2)

    def test_sparse_page_frame_is_not_a_data_table(self) -> None:
        sparse = [["Long page heading", None, None, None]] + [[None] * 4 for _ in range(7)]
        sparse[3][1] = "A paragraph spanning the central panel"
        self.assertTrue(_sparse_page_table((0, 0, 960, 540), sparse, 960, 540))
        actual_grid = [["Property", "Price", "Year"]] + [
            [f"Property {index}", f"${index * 100}", str(2020 + index)] for index in range(1, 8)
        ]
        self.assertFalse(_sparse_page_table((0, 0, 960, 540), actual_grid, 960, 540))
        scattered_panels = [[None] * 13 for _ in range(17)]
        for index in (2, 7, 16):
            scattered_panels[index][0] = "Panel heading"
            scattered_panels[index][8] = "Unrelated text"
        self.assertTrue(_sparse_page_table((0, 0, 960, 540), scattered_panels, 960, 540))

    def test_filled_page_band_separates_adjacent_paragraphs(self) -> None:
        words = [
            {"text": "biography", "top": 710, "bottom": 725},
            {"text": "footer", "top": 740, "bottom": 755},
        ]
        band = [{"x0": 0, "x1": 600, "top": 730, "bottom": 760,
                 "fill": True, "non_stroking_color": (1.0, 0.8, 0.2)}]
        self.assertEqual(
            [[word["text"] for word in group] for group in
             _partition_words_at_filled_bands(words, band, 600, 800)],
            [["biography"], ["footer"]],
        )

    def test_native_words_inside_short_ocr_phrase_render_once(self) -> None:
        lines = [
            {"evidence_id": "p001-ocr-0001", "text": "North Harbor", "coordinates": [100, 100, 120, 20]},
            {"evidence_id": "p001-native-0001", "text": "North", "coordinates": [102, 102, 50, 16]},
            {"evidence_id": "p001-native-0002", "text": "Harbor", "coordinates": [156, 102, 55, 16]},
            {"evidence_id": "p001-native-0003", "text": "Marina", "coordinates": [300, 100, 60, 20]},
        ]
        self.assertEqual(_ocr_text(lines), "North Harbor\nMarina")

    def test_native_symbol_inside_ocr_phrase_renders_once(self) -> None:
        lines = [
            {"evidence_id": "p001-ocr-0001", "text": "North Harbor & Co",
             "coordinates": [100, 100, 200, 20]},
            {"evidence_id": "p001-native-0001", "text": "&",
             "coordinates": [230, 102, 15, 16]},
        ]
        self.assertEqual(_ocr_text(lines), "North Harbor & Co")

    def test_native_transcript_inserts_only_owned_printed_ocr_words(self) -> None:
        native = "The team offers a quality service. Visit example.com."
        ocr = "The team offers a full quality service. Visit example.com."
        supported = [{"text": "full", "evidence_source": "native_pdf_positioned_word"}]
        self.assertEqual(
            _merge_owned_native_word_insertions(native, ocr, supported),
            "The team offers a full quality service. Visit example.com.",
        )
        self.assertEqual(_merge_owned_native_word_insertions(native, ocr, []), native)

    def test_distinct_ocr_website_is_retained_beside_email(self) -> None:
        raw = "Contact: agent@example.com"
        lines = [
            {"evidence_id": "p001-ocr-0001", "text": "example.com", "confidence": 0.99,
             "coordinates": [100, 100, 90, 16]},
            {"evidence_id": "p001-ocr-0002", "text": "agent@example.com", "confidence": 0.99,
             "coordinates": [100, 120, 150, 16]},
        ]
        self.assertEqual(_append_distinct_ocr_identifiers(raw, lines),
                         "Contact: agent@example.com\nexample.com")

    def test_adjacent_visual_objects_merge(self) -> None:
        merged = merge_visual_objects(
            [(100, 100, 300, 300), (305, 120, 500, 310)], 1000, 1000,
        )
        self.assertEqual(len(merged), 1)

    def test_merged_visual_retains_exact_content_box_for_text_exclusion(self) -> None:
        merged = merge_visual_objects(
            [(360, 80, 605, 256), (384, 230, 611, 407)], 720, 540,
        )[0]
        self.assertEqual(merged["content_bbox"], (360, 80, 611, 407))
        self.assertLess(merged["bbox"][0], merged["content_bbox"][0])

    def test_visual_title_prefers_chart_heading(self) -> None:
        lines = [
            {"text": "Office", "coordinates": [100, 200, 40, 20]},
            {"text": "Portfolio Allocation", "coordinates": [100, 100, 180, 30]},
        ]
        self.assertEqual(_visual_title("chart", lines), "Portfolio Allocation")

    def test_visual_title_allows_numeric_portfolio_title(self) -> None:
        lines = [
            {"text": "$146 Million Portfolio", "coordinates": [100, 100, 180, 30]},
            {"text": "Founders", "coordinates": [400, 200, 80, 20]},
        ]
        self.assertEqual(_visual_title("chart", lines), "$146 Million Portfolio")

    def test_visual_title_rejects_compact_financial_axis_value(self) -> None:
        lines = [
            {"text": "$40M", "coordinates": [40, 100, 50, 20]},
            {"text": "Performance Overview", "coordinates": [300, 130, 250, 30]},
            {"text": "2019", "coordinates": [200, 700, 60, 20]},
        ]

        self.assertEqual(_visual_title("chart", lines), "Performance Overview")

    def test_visual_title_ignores_axis_dates_and_lower_footnotes(self) -> None:
        region = Region("generic", 1, "visual", [50, 300, 900, 550], 1, "geometry", 0.9)
        lines = [
            {"text": "Sep-20", "coordinates": [200, 700, 60, 30]},
            {"text": "*Pro forma values for prior periods", "coordinates": [100, 830, 500, 25]},
        ]

        self.assertIsNone(_visual_title("chart", lines, region))

    def test_overlapping_visual_heading_is_ordered_before_visual(self) -> None:
        chart = SourceBlock(
            "doc", "chart", 1, "chart", {"title": None}, [50, 300, 900, 550], ["test"], 0.9, "passed",
        )
        heading = SourceBlock(
            "doc", "heading", 1, "heading", {"text": "Quarterly Results"},
            [300, 320, 300, 40], ["test"], 0.99, "passed", semantic_role="section_heading",
        )
        blocks = [chart, heading]

        _order_overlapping_visual_headings(blocks)

        self.assertEqual([block.block_id for block in blocks], ["heading", "chart"])

    def test_business_vocabulary_alone_does_not_classify_a_chart(self) -> None:
        region = Region(
            "generic-r001", 1, "visual", [100, 100, 700, 500], 1, "pdf-image-object", 0.58,
            metadata={"image_indices": []},
        )
        lines = [
            {"text": "Asset Portfolio"},
            {"text": "12"},
            {"text": "34"},
        ]

        kind, confidence, warnings = _classify_visual(region, lines, {})

        self.assertNotEqual(kind, "chart")
        self.assertLess(confidence, 0.72)
        self.assertTrue(any("visual type" in warning for warning in warnings))

    def test_metric_cards_are_classified_by_structure_not_business_terms(self) -> None:
        region = Region(
            "generic-r002", 1, "visual", [100, 100, 700, 500], 1, "pdf-image-object-merge", 0.72,
            metadata={"image_indices": [1, 2]},
        )
        lines = [
            {"text": "Operating Metrics"},
            {"text": "$"},
            {"text": "42"},
            {"text": "MILLION"},
            {"text": "73"},
            {"text": "Projects"},
        ]

        kind, confidence, warnings = _classify_visual(
            region, lines, {"point_candidates": 30, "vertical_axis_candidates": 1},
        )

        self.assertEqual(kind, "kpi_panel")
        self.assertGreaterEqual(confidence, 0.80)
        self.assertEqual(warnings, [])

    def test_chart_classification_requires_repeated_marks_or_axes(self) -> None:
        region = Region("generic-r003", 1, "visual", [100, 100, 700, 500], 1, "image", 0.58)
        lines = [{"text": "North"}, {"text": "12%"}, {"text": "South"}, {"text": "18%"}]

        kind, confidence, _ = _classify_visual(
            region, lines, {"bar_candidates": 4, "horizontal_axis_candidates": 1},
        )

        self.assertEqual(kind, "chart")
        self.assertGreaterEqual(confidence, 0.75)

    def test_numeric_percentage_is_normalized(self) -> None:
        self.assertEqual(_numeric_value("15.9%"), (15.9, "percent", 0.159))

    def test_pie_type_from_multiple_raster_marks(self) -> None:
        region = Region(
            "p007-r001", 7, "visual", [100, 100, 700, 700], 1, "merge", 0.8,
            metadata={"image_indices": [1, 2, 3, 4]},
        )
        lines = [{"text": value} for value in ("6.1%", "15.9%", "37.5%")]
        self.assertEqual(_infer_chart_type(region, lines, {}, None), "pie")

    def test_proximity_binding_is_unique_and_evidence_linked(self) -> None:
        lines = [
            {"evidence_id": "label-a", "text": "Office", "coordinates": [100, 100, 60, 20]},
            {"evidence_id": "value-a", "text": "6.1%", "coordinates": [105, 125, 50, 20]},
            {"evidence_id": "label-b", "text": "Land", "coordinates": [400, 100, 60, 20]},
            {"evidence_id": "value-b", "text": "7.7%", "coordinates": [405, 125, 50, 20]},
        ]
        bindings = _proximity_bindings(lines, None, [[80, 80, 120, 120], [380, 80, 120, 120]])
        self.assertEqual({item["label"] for item in bindings}, {"Office", "Land"})
        self.assertEqual(len({item["label_evidence_id"] for item in bindings}), 2)
        self.assertTrue(all(item["visual_mark_coordinates"] for item in bindings))

    def test_synthetic_pie_proximity_regression(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures/synthetic-portfolio-pie.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        marks = [[150, 250, 700, 600]] * len(fixture["expected_observations"])
        bindings = _proximity_bindings(fixture["ocr_lines"], fixture["chart_title"], marks)
        actual = {item["label"]: item["value"] for item in bindings}
        self.assertEqual(actual, fixture["expected_observations"])
        self.assertAlmostEqual(
            sum(item["numeric_value"] for item in bindings if item["unit"] == "percent"),
            fixture["expected_percentage_total"], places=5,
        )
        self.assertEqual(len({item["value_evidence_id"] for item in bindings}), 10)

    def test_pie_fragment_boxes_do_not_claim_slice_ownership(self) -> None:
        bindings = [{
            "visual_mark_coordinates": [200, 200, 400, 400],
            "grounding_method": "unique OCR label-value proximity with raster mark ownership; Qwen-confirmed",
        }]
        self.assertFalse(_pie_label_value_grounding(bindings))
        self.assertIsNone(bindings[0]["visual_mark_coordinates"])
        self.assertIn("Qwen-confirmed label-value association", bindings[0]["grounding_method"])
        self.assertIn("slice geometry unresolved", bindings[0]["grounding_method"])
        self.assertNotIn("raster mark ownership", bindings[0]["grounding_method"])

    def test_percent_concordant_pdf_masks_own_distinct_pie_slices(self) -> None:
        bindings = [{
            "unit": "percent", "numeric_value": value,
            "label_coordinates": [x, 200, 50, 20],
            "value_coordinates": [x, 225, 40, 20],
            "grounding_method": "OCR label-value proximity; Qwen-confirmed",
        } for value, x in ((10.0, 100), (20.0, 300), (70.0, 500))]
        candidates = [{
            "pdf_image_index": index,
            "smask_sha256": f"{index + 1:064x}",
            "projected_alpha_area": value,
            "mark_coordinates": [x, 250, 100, 100],
            "alpha_centroid_coordinates": [x + 50, 300],
        } for index, (value, x) in enumerate(((10.0, 100), (20.0, 300), (70.0, 500)))]
        self.assertTrue(_pie_label_value_grounding(bindings, candidates))
        self.assertEqual([item["visual_mark_ref"]["pdf_image_index"] for item in bindings], [0, 1, 2])
        self.assertEqual(bindings[0]["visual_mark_ref"]["opacity_weighted_area_share_percent"], 10.0)
        self.assertEqual(len({tuple(item["visual_mark_coordinates"]) for item in bindings}), 3)

    def test_unresolved_pie_cannot_claim_passed_or_slice_marks(self) -> None:
        chart = SourceBlock(
            "doc", "chart", 1, "chart",
            {"chart_type": "pie", "slice_geometry_status": "unresolved", "observations": [
                {"visual_mark_coordinates": [200, 200, 400, 400]}
            ]},
            [150, 200, 600, 600], ["OCR"], 0.9, "passed",
        ).as_dict()
        errors = validate_source_blocks([chart])
        self.assertTrue(any("cannot be marked passed" in error for error in errors))
        self.assertTrue(any("cannot claim slice mark coordinates" in error for error in errors))

    def test_native_copy_of_owned_chart_label_value_is_suppressed(self) -> None:
        observation = {
            "category": "Education", "raw_value": "12.5%",
            "label_evidence_id": "label-1", "value_evidence_id": "value-1",
            "label_coordinates": [178, 448, 96, 28],
            "value_coordinates": [199, 480, 48, 29],
        }
        chart = SourceBlock(
            "doc", "chart", 1, "chart", {"observations": [observation]},
            [167, 200, 676, 661], ["OCR"], 0.9, "passed",
        )
        duplicate = SourceBlock(
            "doc", "heading", 1, "duplicate",
            {"text": "Education 12.5%", "evidence_text": {"selected": "native"}},
            [181, 454, 90, 59], ["native"], 1.0, "passed",
        )
        separate = SourceBlock(
            "doc", "heading", 1, "separate",
            {"text": "Education 12.5%", "evidence_text": {"selected": "native"}},
            [40, 80, 90, 59], ["native"], 1.0, "passed",
        )
        blocks = [chart, duplicate, separate]
        _suppress_visual_observation_text_duplicates(blocks)
        self.assertEqual([block.block_id for block in blocks], ["chart", "separate"])

    def test_numeric_summary_is_not_a_second_page_title(self) -> None:
        title = SourceBlock(
            "doc", "heading", 2, "title", {"text": "PORTFOLIO SUMMARY"},
            [50, 80, 600, 40], ["native"], 1.0, "passed", semantic_role="page_title", heading_level=1,
        )
        summary = SourceBlock(
            "doc", "heading", 2, "summary", {"text": "$250 million / 80 Assets"},
            [250, 165, 500, 50], ["native"], 1.0, "passed", semantic_role="page_title", heading_level=1,
        )
        _normalize_page_heading_roles([summary, title])
        self.assertEqual(title.semantic_role, "page_title")
        self.assertEqual(summary.semantic_role, "summary_heading")
        self.assertEqual(summary.heading_level, 2)

    def test_repeated_footer_is_classified_as_decoration(self) -> None:
        inspections = []
        for page in range(1, 4):
            region = Region(f"p{page:03d}-r001", page, "visual", [0, 870, 1000, 130], 1, "image", 0.58)
            inspections.append(PageInspection(
                page, 720, 540, 0, True, 0.1, 0.9, 0.1, 0, 0, 1, 0.58, [region], [],
            ))
        mark_repeated_decorations(inspections)
        self.assertTrue(all(page.regions[0].kind == "decoration" for page in inspections))
        self.assertTrue(all(page.regions[0].confidence == 0.98 for page in inspections))

    def test_vector_bars_create_one_chart_region(self) -> None:
        rectangles = [
            {"x0": x, "top": top, "x1": x + 20, "bottom": 400, "fill": True}
            for x, top in [(100, 300), (160, 250), (220, 280), (280, 210)]
        ]
        rectangles.append({"x0": 0, "top": 0, "x1": 720, "bottom": 540, "fill": False})
        visual = detect_vector_bar_chart(rectangles, 720, 540)
        self.assertIsNotNone(visual)
        self.assertEqual(visual["chart_type_hint"], "bar")
        self.assertEqual(visual["expected_mark_count"], 4)

    def test_small_square_rasters_create_scatterplot_region(self) -> None:
        images = [(100 + index * 12, 200 + index % 3 * 10, 110 + index * 12, 210 + index % 3 * 10) for index in range(10)]
        visual = detect_small_raster_chart(images, 720, 540)
        self.assertIsNotNone(visual)
        self.assertEqual(visual["chart_type_hint"], "scatterplot")
        self.assertEqual(visual["expected_mark_count"], 10)

    def test_aligned_financial_table_infers_columns_without_document_labels(self) -> None:
        def word(text: str, x0: float, top: float, width: float = 30) -> dict[str, object]:
            return {"text": text, "x0": x0, "x1": x0 + width, "top": top, "bottom": top + 10}

        words = [
            word("Asset", 250, 90), word("Schedule", 310, 90),
            word("Asset", 30, 140), word("Cost", 240, 140),
            word("Share", 440, 140), word("Debt", 625, 140),
        ]
        for index in range(6):
            top = 190 + index * 20
            words.extend([
                word(f"Holding-{index + 1}", 30, top, 80),
                word(f"{index + 1}", 260, top, 5),
                word("00,000", 264, top, 46),
                word(f"{index + 1}.0%", 445, top, 35),
                word(f"{index + 2}50,000", 630, top, 50),
            ])
        rectangles = [
            {"x0": 10, "top": 124, "x1": 710, "bottom": 175, "fill": True},
            {"x0": 10, "top": 390, "x1": 710, "bottom": 408, "fill": True},
        ]
        table = detect_aligned_financial_table(words, rectangles, 720, 540)
        self.assertIsNotNone(table)
        self.assertEqual(len(table["rows"][0]), 4)
        self.assertEqual(table["rows"][1][1], "100,000")
        self.assertEqual(table["title"], "Asset Schedule")
        self.assertEqual(table["classification_method"], "native-positioned-aligned-financial-table")

    def test_aligned_financial_table_rejects_layout_without_numeric_anchors(self) -> None:
        words = [
            {"text": "Narrative", "x0": 30, "x1": 90, "top": 140, "bottom": 150},
            {"text": "copy", "x0": 30, "x1": 60, "top": 200, "bottom": 210},
        ]
        rectangles = [{"x0": 10, "top": 124, "x1": 710, "bottom": 175, "fill": True}]
        self.assertIsNone(detect_aligned_financial_table(words, rectangles, 720, 540))

    def test_adjacent_numeric_fragments_are_joined_before_column_assignment(self) -> None:
        fragments = [
            {"text": "3", "x0": 517.5, "x1": 523.2, "top": 362.9, "bottom": 372.9},
            {"text": "1,031,365", "x0": 522.9, "x1": 567.1, "top": 362.9, "bottom": 372.9},
        ]
        merged = _merge_numeric_fragments(fragments, 720)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "31,031,365")

    def test_currency_symbol_joins_the_number_on_its_right(self) -> None:
        fragments = [
            {"text": "53.4%", "x0": 300, "x1": 340, "top": 100, "bottom": 110},
            {"text": "$", "x0": 361, "x1": 366, "top": 100, "bottom": 110},
            {"text": "972,330", "x0": 369, "x1": 413, "top": 100, "bottom": 110},
        ]
        merged = _merge_numeric_fragments(fragments, 720)
        self.assertEqual([item["text"] for item in merged], ["53.4%", "$972,330"])

    def test_table_structure_rejects_shifted_currency_and_bad_headers(self) -> None:
        errors = _table_structure_errors(
            ["Asset", "Return of Capital (CCC", "Leverage Debt)"],
            [["Asset", "Return", "Leverage"], ["Total", "53.4% $", "$"]],
        )
        self.assertIn("one or more table headers contain unmatched parentheses", errors)
        self.assertIn("standalone currency symbol has no owned numeric value", errors)

    def test_ocr_evidence_has_one_region_owner(self) -> None:
        regions = [
            Region("text", 1, "normal_text", [0, 0, 100, 100], 1, "test", 0.8),
            Region("decoration", 1, "decoration", [0, 0, 100, 100], 2, "test", 0.98),
        ]
        lines = [{"evidence_id": "e1", "coordinates": [10, 10, 20, 10], "text": "visible"}]
        owners = _exclusive_region_lines(regions, lines)
        self.assertEqual([item["evidence_id"] for item in owners["text"]], ["e1"])
        self.assertEqual(owners["decoration"], [])

    def test_visual_owns_text_drawn_inside_it(self) -> None:
        regions = [
            Region("visual", 1, "visual", [0, 0, 100, 100], 1, "test", 0.8),
            Region("text", 1, "normal_text", [10, 10, 30, 30], 2, "test", 0.98),
        ]
        lines = [{"evidence_id": "e1", "coordinates": [15, 15, 10, 10], "text": "Category A 0.9%"}]
        owners = _exclusive_region_lines(regions, lines)
        self.assertEqual([item["evidence_id"] for item in owners["visual"]], ["e1"])
        self.assertEqual(owners["text"], [])

    def test_visual_padding_does_not_steal_neighboring_text(self) -> None:
        regions = [
            Region("visual", 1, "visual", [0, 0, 300, 100], 1, "test", 0.8,
                   metadata={"member_coordinates": [[200, 0, 100, 100]]}),
            Region("text", 1, "normal_text", [0, 0, 150, 100], 2, "test", 0.9),
        ]
        line = {"evidence_id": "e1", "coordinates": [40, 20, 50, 10], "text": "Neighboring text"}
        owners = _exclusive_region_lines(regions, [line])
        self.assertEqual(owners["visual"], [])
        self.assertEqual([item["evidence_id"] for item in owners["text"]], ["e1"])

    def test_native_word_outside_visual_ocr_lane_remains_unassigned(self) -> None:
        regions = [Region(
            "visual", 1, "visual", [400, 100, 600, 800], 1, "test", 0.8,
            metadata={"member_coordinates": [[650, 150, 350, 750]]},
        )]
        lines = [
            {"evidence_id": f"p001-ocr-{index:04d}", "text": f"Sidebar line {index}",
             "coordinates": [760, 200 + index * 30, 170, 20]}
            for index in range(1, 5)
        ]
        lines.extend([
            {"evidence_id": "p001-native-0001", "evidence_source": "native_pdf_positioned_word",
             "text": "Neighbor", "coordinates": [600, 380, 75, 18]},
            {"evidence_id": "p001-native-0002", "evidence_source": "native_pdf_positioned_word",
             "text": "Sidebar", "coordinates": [765, 380, 75, 18]},
            {"evidence_id": "p001-native-0003", "evidence_source": "native_pdf_positioned_word",
             "text": "Above", "coordinates": [765, 115, 75, 18]},
        ])
        owners = _exclusive_region_lines(regions, lines)
        self.assertNotIn("p001-native-0001", [line["evidence_id"] for line in owners["visual"]])
        self.assertNotIn("p001-native-0003", [line["evidence_id"] for line in owners["visual"]])
        self.assertIn("p001-native-0002", [line["evidence_id"] for line in owners["visual"]])

    def test_native_word_outside_visual_lane_joins_unique_adjacent_text(self) -> None:
        regions = [
            Region("visual", 1, "visual", [400, 100, 600, 800], 1, "test", 0.8,
                   metadata={"member_coordinates": [[650, 100, 350, 800]]}),
            Region("text", 1, "normal_text", [350, 300, 300, 150], 2, "test", 0.9),
        ]
        lines = [
            {"evidence_id": f"p001-ocr-{index:04d}", "text": f"Sidebar line {index}",
             "coordinates": [760, 200 + index * 30, 170, 20]}
            for index in range(1, 5)
        ]
        lines.append({"evidence_id": "p001-native-0001",
                      "evidence_source": "native_pdf_positioned_word", "text": "paragraph",
                      "coordinates": [651, 380, 75, 18]})
        owners = _exclusive_region_lines(regions, lines)
        self.assertEqual([line["evidence_id"] for line in owners["text"]], ["p001-native-0001"])

    def test_ocr_line_crossing_visual_margin_does_not_join_visual(self) -> None:
        regions = [Region(
            "visual", 1, "visual", [400, 100, 600, 800], 1, "test", 0.8,
            metadata={"member_coordinates": [[650, 100, 350, 800]]},
        )]
        lines = [
            {"evidence_id": f"p001-ocr-{index:04d}", "text": f"Inside line {index}",
             "coordinates": [760, 200 + index * 30, 170, 20]}
            for index in range(1, 5)
        ]
        lines.append({"evidence_id": "p001-ocr-0100", "text": "Adjacent footer text",
                      "coordinates": [540, 870, 285, 20]})
        owners = _exclusive_region_lines(regions, lines)
        self.assertNotIn("p001-ocr-0100", [line["evidence_id"] for line in owners["visual"]])

    def test_text_at_physical_image_edge_stays_with_visual_region(self) -> None:
        regions = [Region(
            "visual", 1, "visual", [0, 350, 1000, 400], 1, "test", 0.8,
            metadata={"member_coordinates": [[0, 350, 1000, 400]]},
        )]
        lines = [
            {"evidence_id": f"p001-ocr-{index:04d}", "text": f"Body line {index}",
             "coordinates": [120, 390 + index * 25, 300, 20]}
            for index in range(4)
        ]
        lines.append({"evidence_id": "p001-ocr-0100", "text": "Start of paragraph",
                      "coordinates": [120, 335, 300, 30]})
        owners = _exclusive_region_lines(regions, lines)
        self.assertIn("p001-ocr-0100", [line["evidence_id"] for line in owners["visual"]])

    def test_text_region_beats_image_at_shared_edge(self) -> None:
        regions = [
            Region("visual", 1, "visual", [0, 350, 1000, 400], 1, "test", 0.8,
                   metadata={"member_coordinates": [[0, 350, 1000, 400]]}),
            Region("text", 1, "normal_text", [500, 735, 390, 40], 2, "test", 0.9),
        ]
        lines = [
            {"evidence_id": f"p001-ocr-{index:04d}", "text": f"Body line {index}",
             "coordinates": [120, 390 + index * 25, 300, 20]}
            for index in range(4)
        ]
        edge = {"evidence_id": "p001-ocr-0100", "text": "Next section title",
                "coordinates": [525, 735, 350, 30]}
        owners = _exclusive_region_lines(regions, [*lines, edge])
        self.assertEqual([line["evidence_id"] for line in owners["text"]], [edge["evidence_id"]])

    def test_chart_labels_can_extend_beyond_image_member(self) -> None:
        regions = [Region(
            "chart", 1, "visual", [100, 100, 700, 600], 1, "test", 0.8,
            metadata={"chart_type_hint": "bar", "member_coordinates": [[200, 200, 500, 400]]},
        )]
        label = {"evidence_id": "p001-ocr-0001", "text": "Axis label",
                 "coordinates": [110, 300, 90, 20]}
        owners = _exclusive_region_lines(regions, [label])
        self.assertEqual([line["evidence_id"] for line in owners["chart"]], ["p001-ocr-0001"])

    def test_visual_search_crop_does_not_steal_text_outside_ownership_box(self) -> None:
        regions = [
            Region(
                "visual", 1, "visual", [0, 100, 1000, 700], 1, "test", 0.9,
                metadata={"ownership_coordinates": [0, 250, 1000, 550]},
            ),
            Region("text", 1, "normal_text", [50, 150, 900, 80], 2, "test", 0.98),
        ]
        lines = [{"evidence_id": "e1", "coordinates": [100, 170, 500, 30], "text": "independent prose"}]

        owners = _exclusive_region_lines(regions, lines)

        self.assertEqual(owners["visual"], [])
        self.assertEqual([item["evidence_id"] for item in owners["text"]], ["e1"])

    def test_broad_table_box_does_not_steal_chart_label_without_cell_support(self) -> None:
        line = {"evidence_id": "chart-label", "coordinates": [190, 310, 100, 20], "text": "$351 | 0 yrs"}
        regions = [
            Region("page-grid", 1, "table", [0, 0, 1000, 1000], 1, "pdf table finder", 0.92),
            Region("chart", 1, "visual", [80, 180, 540, 700], 2, "PDF image", 0.72),
        ]
        decisions = []
        owners = _exclusive_region_lines(regions, [line], decisions)
        self.assertEqual([item["evidence_id"] for item in owners["chart"]], ["chart-label"])
        self.assertEqual(owners["page-grid"], [])
        self.assertEqual(decisions[0]["owner_region_id"], "chart")
        regions[0].metadata["cell_coordinates"] = [[[180, 300, 200, 50]]]
        owners = _exclusive_region_lines(regions, [line])
        self.assertEqual([item["evidence_id"] for item in owners["page-grid"]], ["chart-label"])

    def test_two_column_text_is_not_interleaved(self) -> None:
        words = []
        for row in range(12):
            top = 20 + row * 12
            words.extend([
                {"text": f"L{row}", "x0": 20, "x1": 50, "top": top, "bottom": top + 9, "size": 10},
                {"text": "left", "x0": 55, "x1": 85, "top": top, "bottom": top + 9, "size": 10},
                {"text": f"R{row}", "x0": 420, "x1": 450, "top": top, "bottom": top + 9, "size": 10},
                {"text": "right", "x0": 455, "x1": 490, "top": top, "bottom": top + 9, "size": 10},
            ])
        paragraphs = _group_words(words)
        text = "\n".join(item["text"] for item in paragraphs)
        self.assertLess(text.index("L11"), text.index("R0"))

    def test_wide_paragraph_crossing_page_center_stays_one_lane(self) -> None:
        words = []
        for row in range(6):
            top = 20 + row * 12
            words.extend([
                {"text": f"L{row}", "x0": 300, "x1": 350, "top": top, "bottom": top + 9, "size": 10},
                {"text": f"R{row}", "x0": 356, "x1": 410, "top": top, "bottom": top + 9, "size": 10},
            ])
        # Add enough lane words for the two-column detector to engage.
        words *= 4
        paragraphs = _group_words(words)
        self.assertTrue(any("L0 R0" in paragraph["text"] for paragraph in paragraphs))

    def test_aligned_columns_with_narrow_persistent_gutter_stay_separate(self) -> None:
        words = []
        for row in range(12):
            top = 20 + row * 12
            words.extend([
                {"text": f"A{row}", "x0": 40, "x1": 150, "top": top, "bottom": top + 9, "size": 10},
                {"text": "alpha", "x0": 155, "x1": 350, "top": top, "bottom": top + 9, "size": 10},
                {"text": f"B{row}", "x0": 370, "x1": 480, "top": top, "bottom": top + 9, "size": 10},
                {"text": "beta", "x0": 485, "x1": 680, "top": top, "bottom": top + 9, "size": 10},
            ])
        paragraphs = _group_words(words)
        text = "\n".join(item["text"] for item in paragraphs)
        self.assertLess(text.index("A11"), text.index("B0"))
        self.assertEqual({item.get("reading_lane") for item in paragraphs}, {"left", "right"})
        self.assertTrue(all(item.get("column_gutter_points") == [350.0, 370.0] for item in paragraphs))

    def test_column_layout_keeps_page_footer_after_both_reading_lanes(self) -> None:
        def block(block_id: str, kind: str, role: str, coordinates: list[float]) -> SourceBlock:
            return SourceBlock(
                document_id="generic-document", type=kind, page=1, block_id=block_id,
                content={"text": block_id}, coordinates=coordinates,
                extraction_method=["synthetic"], confidence=0.99,
                validation_status="passed", semantic_role=role,
            )

        blocks = [
            block("right-2", "text", "body_text", [560, 390, 360, 120]),
            block("footer", "group", "page_footer", [40, 930, 920, 45]),
            block("left-2", "text", "body_text", [60, 410, 360, 120]),
            block("title", "heading", "page_title", [60, 60, 880, 70]),
            block("right-1", "text", "body_text", [560, 220, 360, 120]),
            block("left-1", "text", "body_text", [60, 210, 360, 120]),
        ]

        _arrange_page_blocks(blocks)

        self.assertEqual(
            [item.block_id for item in blocks],
            ["title", "left-1", "left-2", "right-1", "right-2", "footer"],
        )

    def test_aligned_image_cards_receive_photograph_hint(self) -> None:
        regions = [
            Region(f"r{index}", 1, "visual", [30, 100 + index * 250, 170, 210], index, "image", 0.58,
                   metadata={"image_indices": [index]})
            for index in range(3)
        ]
        _mark_repeated_portrait_visuals(regions)
        self.assertTrue(all(region.metadata.get("visual_hint") == "photograph" for region in regions))

    def test_vector_bar_bindings_use_category_value_and_mark(self) -> None:
        lines = [
            {"evidence_id": "v1", "text": "11.3%", "coordinates": [105, 300, 20, 50]},
            {"evidence_id": "v2", "text": "12.9%", "coordinates": [205, 280, 20, 50]},
            {"evidence_id": "v3", "text": "16.7%", "coordinates": [305, 230, 20, 50]},
            {"evidence_id": "c1", "text": "2014", "coordinates": [100, 510, 40, 20]},
            {"evidence_id": "c2", "text": "2015", "coordinates": [200, 510, 40, 20]},
            {"evidence_id": "c3", "text": "2016", "coordinates": [300, 510, 40, 20]},
        ]
        marks = [[100, 200, 30, 300], [200, 180, 30, 320], [300, 130, 30, 370]]
        bindings = _bar_bindings(lines, "Annual Returns", marks)
        self.assertEqual([(item["label"], item["value"]) for item in bindings], [
            ("2014", "11.3%"), ("2015", "12.9%"), ("2016", "16.7%"),
        ])
        self.assertEqual(len({tuple(item["visual_mark_coordinates"]) for item in bindings}), 3)
        self.assertTrue(all(item["grounding_method"].startswith("x-aligned") for item in bindings))

    def test_rotated_native_percentage_is_reconstructed_generically(self) -> None:
        tokens = [
            {"text": "81", "coordinates": [100, 160, 20, 25]},
            {"text": ".", "coordinates": [100, 150, 20, 8]},
            {"text": "5", "coordinates": [100, 135, 20, 13]},
            {"text": "%", "coordinates": [100, 115, 20, 18]},
        ]
        values = _reconstruct_vertical_values(tokens)
        self.assertEqual([item["text"] for item in values], ["18.5%"])

    def test_vertical_native_unit_is_reconstructed_generically(self) -> None:
        tokens = [
            {"text": text, "coordinates": [700, 100 + index * 12, 20, 12]}
            for index, text in enumerate(["M", "I", "LL", "I", "O", "N"])
        ]
        self.assertEqual([item["text"] for item in _reconstruct_vertical_words(tokens)], ["MILLION"])

    def test_kpi_cards_reconstruct_amount_count_label_and_date(self) -> None:
        def line(evidence_id: str, text: str, coordinates: list[float]) -> dict[str, object]:
            return {"evidence_id": evidence_id, "text": text, "coordinates": coordinates, "confidence": 0.99}

        lines = [
            line("p001-ocr-0001", "As of 12/31/23", [520, 100, 180, 25]),
            line("p001-native-0001", "$", [480, 220, 20, 50]),
            line("p001-native-0002", "277", [510, 210, 150, 90]),
            line("p001-ocr-0002", "MILLION", [665, 215, 35, 90]),
            line("p001-native-0003", "168", [790, 230, 60, 45]),
            line("p001-ocr-0003", "Investments", [780, 285, 130, 25]),
            line("p001-ocr-0004", "Investments Funded to", [510, 325, 230, 25]),
            line("p001-ocr-0005", "Date", [510, 355, 60, 25]),
            line("p001-native-0004", "$", [500, 510, 20, 50]),
            line("p001-native-0005", "146", [530, 500, 150, 90]),
            line("p001-ocr-0006", "MILLION", [685, 505, 35, 90]),
            line("p001-native-0006", "61", [800, 520, 45, 45]),
            line("p001-native-0007", "LL", [785, 560, 20, 30]),
            line("p001-ocr-0007", "Investments", [790, 575, 130, 25]),
            line("p001-ocr-0008", "Current Portfolio at cost", [530, 615, 245, 25]),
            line("p001-ocr-0009", "basis", [530, 645, 60, 25]),
        ]
        bindings, as_of, completed = _kpi_bindings(lines, [[500, 180, 360, 220], [520, 470, 340, 220]])
        self.assertEqual(as_of, "2023-12-31")
        self.assertEqual(completed, 2)
        actual = {(item["series"], item["label"]): item["normalized_value"] for item in bindings}
        self.assertEqual(actual[("Investments Funded to Date", "amount")], 277_000_000)
        self.assertEqual(actual[("Investments Funded to Date", "Investments")], 168)
        self.assertEqual(actual[("Current Portfolio at cost basis", "amount")], 146_000_000)
        self.assertEqual(actual[("Current Portfolio at cost basis", "Investments")], 61)

    def test_kpi_count_label_is_not_investment_specific(self) -> None:
        def line(evidence_id: str, text: str, coordinates: list[float]) -> dict[str, object]:
            return {"evidence_id": evidence_id, "text": text, "coordinates": coordinates, "confidence": 0.99}

        lines = [
            line("p001-ocr-0001", "As of 06/30/26", [520, 100, 180, 25]),
            line("p001-native-0001", "$", [480, 220, 20, 50]),
            line("p001-native-0002", "42", [510, 210, 120, 90]),
            line("p001-ocr-0002", "MILLION", [650, 215, 35, 90]),
            line("p001-native-0003", "73", [790, 230, 60, 45]),
            line("p001-ocr-0003", "Properties", [780, 285, 130, 25]),
            line("p001-ocr-0004", "Assets Under Management", [510, 325, 250, 25]),
        ]
        bindings, as_of, completed = _kpi_bindings(lines, [[500, 180, 360, 220]])
        self.assertEqual(as_of, "2026-06-30")
        self.assertEqual(completed, 1)
        self.assertEqual(
            {(item["series"], item["label"]): item["normalized_value"] for item in bindings},
            {
                ("Assets Under Management", "amount"): 42_000_000,
                ("Assets Under Management", "Properties"): 73,
            },
        )

    def test_generic_subtotals_reconcile(self) -> None:
        rows = [
            ["Holdings", "Cost", "Share", "Exposure"],
            ["Total Region North", "$60,000", "40.0%", "$45,000"],
            ["Total Region South", "$90,000", "60.0%", "$55,000"],
            ["Grand Total Holdings", "$150,000", "100.0%", "$100,000"],
        ]
        result = _table_total_reconciliation(rows)
        self.assertIsNotNone(result)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 3)
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_generic_subtotal_mismatch_is_detected(self) -> None:
        rows = [
            ["Holdings", "Cost"],
            ["Total Segment One", "40"],
            ["Total Segment Two", "50"],
            ["Total Holdings", "100"],
        ]
        result = _table_total_reconciliation(rows)
        self.assertIsNotNone(result)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()

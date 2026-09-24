# Region fidelity audit — 2026-09-24

## Acceptance criterion

A source block should contain the exact printed text, values, and visual evidence in its own physical region, without borrowing content from a neighbor. Semantic interpretation is a later stage. JSON schema validity, OCR evidence being somewhere on the page, and a low count of review warnings do not prove this criterion.

## Method

- Run six complete PDFs (42 pages) through the independent vision pipeline. The vision plans come from hash-verified caches for the same rendered pages; RapidOCR, PaddleOCR on GPU, extraction, reconciliation, and validation run again.
- Compare page images with source blocks, including table cells, prose, photos, maps, and nearby sections. All 15 pages in the three added OMs were visually screened. Selected failure-prone pages in the earlier three OMs were checked side by side.
- Treat a wrong character, omitted line, merged neighboring region, or misplaced value as a failure for exact region fidelity. Do not infer a corpus accuracy percentage from these checks; they are not a complete hand-annotated reference for all 42 pages.

## General pipeline changes

- Split spatially separated residual text around Paddle tables and image-backed text into independent source blocks. Do not wrap unrelated regions in a page-wide group.
- Distinguish dense image-backed prose from charts and maps when the necessary plot evidence is absent.
- Recover sparse-text photographic regions from independent vision proposals only after deterministic extraction, with image texture and OCR sparsity checks. Keep them marked for review.
- Reconstruct letter-spaced native PDF banners as headings when positioned glyph gaps support the words.
- Repair merged table section/row labels, independently observed currency symbols, and malformed scalar cells only when positioned OCR evidence matches the same row or cell.
- Screen Paddle table HTML against independent numeric OCR inside that table's own crop. A grid transcript whose values instead appear elsewhere on the page is rejected, and its source image/OCR evidence remains available for review.

These rules use layout and evidence, not OM names or page numbers. The production code has no references to the six audited OMs.

## Results

| Complete PDF | Pages | Blocks / review | Schema/integrity errors | Region evidence from page comparison |
| --- | ---: | ---: | ---: | --- |
| Ameris Center | 6 | 23 / 8 | 0 / 0 | Page 4 now separates the financial table, two photos, headings, and body sections. Pages 2, 3, and 5 still collapse multiple physical regions into a broad visual block. |
| Haven Senior Living | 7 | 70 / 30 | 0 / 0 | Page 7 has three separate sensitivity tables and neighboring text panels. A risk line on page 6 is visibly corrupted by OCR; a page 7 table header also differs from print. |
| Marble Hill | 2 | 30 / 13 | 0 / 0 | Page 2 now separates Sources and Uses, repairs a merged row label and malformed percentage, and treats the letter-spaced banner as a heading. Page 1 still has an unclassified visual and fragmented or garbled prose. |
| Concourse Oxford | 20 | 141 / 75 | 0 / 0 | Page 4's three sponsor cards are still owned by one map block. Page 9's large building photo is treated as tables. Page 11's horizontal chart lacks verified bar values. |
| Winter Haven Industrial | 6 | 64 / 23 | 0 / 0 | Page 2 initially had two Paddle tables populated with values from different regions. The new gate rejects both transcripts. OCR geometry reconstructs those two grids, but it merges rows in Rent Credit; all five table blocks remain review-required. |
| Torquoise Bay | 1 | 16 / 11 | 0 / 0 | Two investment tables and several text/sidebar sections are separate; image and sidebar ownership remain incomplete. |

The key result is that the pipeline is **not yet lossless or fully region accurate** across these OMs. Structural validation can pass while a page remains wrong by the requested physical-region standard. Review warnings must remain on unresolved areas.

Across these runs there are **42 pages, 344 source blocks, and 160 review blocks**. The review count is a diagnostic, not an accuracy score. The two rejected Paddle transcripts were the only source-page changes made by the numeric gate when compared byte for byte with the preceding same-code runs; Winter Haven page 2 changed and the other 41 pages did not.

### Side-by-side checks

| Page region | Printed page | Final source block | Assessment |
| --- | --- | --- | --- |
| Ameris p4 photo panel | Aerial photograph and interior photograph are separate | Two photograph blocks with their own crops | Boundary improved; both remain review-required. |
| Ameris p4 headings | `Capital Improvements and Holding Strategy`; `Conclusion` | `Capltal Improvements and HoldIng Strategy`; `Concluslon` | Separate owners, but OCR text is not exact. |
| Haven p6 risk panel | A complete `Thin-Market Liquidity` mitigation sentence | The corresponding sentence begins `TMhin s s n M r-s...` | OCR corruption; exact content fails. |
| Haven p7 sensitivity grid | Three distinct small tables | Three table blocks; one header reads `96%(Base~）` | Geometry improved; header transcription fails. |
| Marble Hill p2 Sources/Uses | `CCC Equity` section, `CCC Pref` row with `$7,250,000`; Uses `62.5%` | These appear in the correct separate tables after repair | These sampled cells pass; this is not a full table annotation. |
| Concourse p4 sponsor cards | Three separate person/company cards | One large `map` block covering the cards | Region ownership fails. |
| Winter Haven p2 Rent Credit | Distinct Contractual Rent and Underwritten Rent rows | OCR fallback joins values from both in one row | Cross-region values are removed, but row fidelity still fails. |

The source PDF page images and block JSON in the run directories are the evidence for these checks. The examples are deliberately a mix of successes and counterexamples; a passing sample does not certify its entire page.

## Reproduction

The complete run directories are under `runs/` and are ignored by Git:

- `ameris-region-audit-screened-full-20260924`
- `haven-senior-region-audit-screened-full-20260924`
- `marble-hill-region-audit-screened-full-20260924`
- `concourse-region-audit-screened-full-20260924`
- `winter-haven-region-audit-screened-full-20260924`
- `torquoise-region-audit-screened-full-20260924`

Each directory contains rendered pages, vision receipts, OCR/Paddle outputs, source blocks, and `validation.json`. The final full automated suite passed **166 tests**. This is a functional and targeted physical-region audit, not a percentage accuracy benchmark.

## Next technical priority

Add a page-region decomposition stage for mixed visual layouts. It should propose candidate photos, maps, text panels, tenant cards, and tables, then require independent OCR/pixel support before assigning each region an owner. After that, add an OCR disagreement lane for low-confidence prose and table glyphs, retaining competing transcripts and the crop until the printed value can be resolved. These are the remaining blockers to a defensible exact-content claim.

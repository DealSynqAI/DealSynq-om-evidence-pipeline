# Region decomposition and OCR disagreement audit — 2026-09-24

## Scope and decision

This change adds an evidence-gated region decomposition pass and an OCR disagreement lane to the independent OM evidence pipeline. It improves physical ownership on several mixed-layout pages. It **does not establish a lossless or exact-content guarantee**. A block can still omit print or contain a wrong glyph even when the collection validates.

The production rules use no OM name, page number, sponsor name, or expected answer. Vision supplies candidate boundaries only. PDF image geometry, rendered pixels, RapidOCR, and PaddleOCR provide independent checks. Model text and model values do not enter source blocks through this stage.

## What changed

- Propose photographs, maps, text panels, profile/tenant cards, and tables from PDF image members, pixel components, OCR clusters, and model boundaries. Each proposal records its support and acceptance reason in `reconciliation/page-NNN.json`.
- Reject photo claims for flat graphics using coarse color and foreground detail. A map proposal also prevents the same region from being silently labeled a photo. Colored card panels use their pixel boundary; overhanging logo edges extend a card crop only when the printed edge is visible.
- Replace a broad visual owner only if at least two independently supported regions account for at least 85% of the RapidOCR lines and include visual evidence or multiple profile cards. Otherwise retain the broad block for review. A photo can be added from a supported PDF image member when no broad owner needs replacement. Empty PDF-grid bands crossing such a photo are removed only when they have no printed row, cell, or OCR content.
- Run PaddleOCR on likely table pages and pages with low-confidence RapidOCR prose. Compare positioned OCR lines and table row, column, and cell glyphs. Store competing transcripts, a crop, and a crop SHA-256 on the owning USB as `ocr_disagreements`; mark that block `needs_review`. The lane does not choose a winner.
- Attach an OCR variant inside one uniquely bounded decomposed card to that card as raw evidence, rather than creating a separate neighboring text block. The validator checks referenced crops and forbids a passed block with unresolved disagreements.

## Complete-PDF check

Six complete PDFs were run with cached, hash-verified full-page vision plans, fresh RapidOCR and GPU PaddleOCR, fresh extraction/reconciliation, and validation. The 42 deterministic pre-fusion page snapshots are byte-for-byte identical to the prior audit. All 42 final pages validate with **zero schema and integrity errors**.

| OM | Pages | Blocks before → after | Review blocks before → after | Unresolved OCR records |
| --- | ---: | ---: | ---: | ---: |
| Ameris Center | 6 | 23 → 32 | 8 → 22 | 53 |
| Haven Senior Living | 7 | 70 → 70 | 30 → 36 | 24 |
| Marble Hill | 2 | 30 → 30 | 13 → 18 | 32 |
| Concourse Oxford | 20 | 141 → 153 | 75 → 89 | 63 |
| Winter Haven Industrial | 6 | 64 → 64 | 23 → 27 | 54 |
| Torquoise Bay | 1 | 16 → 17 | 11 → 13 | 5 |
| **Total** | **42** | **344 → 366** | **160 → 205** | **231** |

The extra review flags expose disagreements that previously went unmarked; they are not an accuracy score. These runs do not have a complete hand-annotated reference for every printed region or glyph, so no corpus accuracy percentage is justified.

## Physical page checks

- **Ameris p5:** one page-wide `unclassified_visual` owner became two profile cards, a narrative panel, and four distinct office photos. The card crops extend to the colored panel and raised logo. The printed `SCRIPPS` heading stays with its own card. A damaged RapidOCR `O R` line is paired with PaddleOCR `% OF RSF` and a crop; the printed reading remains unresolved in the USB.
- **Concourse p4:** one incorrect broad `map` owner became three separate developer/profile cards. OCR variants inside the cards stay as raw card evidence; title and footer stay separate.
- **Concourse p9:** a large building photograph has its own crop. Two empty side-band regions previously called tables were removed because neither contained OCR, a row label, nor a cell value.
- **Concourse p1 and p20:** the US silhouette and decorative script are not emitted as photographs. Their unverified visual interpretation remains conservative.
- **Haven p6 and p7:** the corrupt `Thin-Market Liquidity` prose and the `96%(Base~）` sensitivity header remain visible as unresolved OCR readings with crops. These have not been silently corrected.
- **Winter Haven p2:** disputed currency signs, a table row label, and a `2.1x` glyph are in the disagreement lane. Its previously noted Rent Credit row merging is still not fully resolved.

## Remaining limits

- Ameris p3 still has a broad map owner. The candidate regions did not supply enough independent visual evidence to replace it safely.
- Marble Hill p1 and parts of Torquoise Bay still contain unresolved visual ownership or fragmented prose.
- Some image tables and chart marks remain structurally unverified. Two OCR engines can agree on the same wrong glyph, so absence of a disagreement is not proof of exact transcription.
- A review status is an honest uncertainty flag, not a human resolution. Exact content should be claimed only after the outstanding regions and OCR conflicts are checked against the preserved PDF/crops.

## Reproduction and evidence

The full local run directories are ignored by Git and remain under `runs/`:

- `ameris-region-decomposition-verified-full-20260924`
- `haven-senior-region-decomposition-final-full-20260924`
- `marble-hill-region-decomposition-final-full-20260924`
- `concourse-region-decomposition-verified-full-20260924`
- `winter-haven-region-decomposition-final-full-20260924`
- `torquoise-region-decomposition-final-full-20260924`

Each run contains the source PDF, rendered pages, `deterministic/` snapshots, `reconciliation/` proposal decisions, `diagnostics/ocr-disagreements/` receipts, `disagreement-crops/`, final USBs, and `validation.json`. The automated suite passed **173 tests**.

# Review policy recheck — 2026-09-24

The region decomposition upgrade increased review blocks from 160 to 205 on six complete OMs. That failed the intended review reduction goal. The first implementation put every recovered text/card region and photo into review even when the independent evidence checks passed. It also generated false OCR disagreements by sorting word boxes into fixed vertical bands, which could move a word to the end of its printed line.

The review rule now passes a recovered text region only when its OCR layout and transcript pass the ordinary block checks. A photo can pass when its PDF image member gives the physical boundary, or when a model boundary closely matches an independently detected pixel component. Coarse pixel-only photo extents, table cell ownership, maps, low-confidence OCR, omitted structured text, and unresolved OCR readings remain reviewable. Paddle word boxes on the same printed line are ordered left to right using their vertical centers and heights.

## Complete-PDF rerun

The same six complete PDFs were rerun with hash-verified cached full-page vision plans, fresh RapidOCR, GPU PaddleOCR, extraction, reconciliation, and validation. Every new run has zero schema and integrity errors. All 175 tests pass.
The original pipeline emitted 344 blocks; both upgrade versions emit 366.

| OM | Pages | Upgrade blocks | Original review | First upgrade review | Corrected review | Corrected OCR disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ameris Center | 6 | 32 | 8 | 22 | 22 | 42 |
| Haven Senior Living | 7 | 70 | 30 | 36 | 36 | 24 |
| Marble Hill | 2 | 30 | 13 | 18 | 18 | 32 |
| Concourse Oxford | 20 | 153 | 75 | 89 | 76 | 62 |
| Winter Haven Industrial | 6 | 64 | 23 | 27 | 27 | 55 |
| Torquoise Bay | 1 | 17 | 11 | 13 | 12 | 5 |
| **Total** | **42** | **366** | **160** | **205** | **191** | **220** |

The corrected policy removes 14 review flags from the first upgrade, primarily supported photographs. Corrected OCR reading order removes 11 false conflict records overall (231 to 220). The remaining 191 review blocks are **31 above the original baseline**; this change does not meet the requested review reduction goal.

All 366 block IDs have the same type, coordinates, and `content.text` in the first upgrade and corrected runs. This correction changes review decisions and OCR conflict diagnostics; it does **not** improve printed text extraction or prove better region accuracy. In particular, Ameris page 5 still has substantive OCR disagreements such as `O R` versus `% OF RSF` and `ÓH` versus `OH`. Those require a reliable transcript resolution method, not a status downgrade.

## Reproduction

The corrected full runs are under `runs/` with these names:

- `ameris-review-policy-full-20260924-v3`
- `haven-senior-review-policy-full-20260924-v3`
- `marble-hill-review-policy-full-20260924-v3`
- `concourse-review-policy-full-20260924-v3`
- `winter-haven-review-policy-full-20260924-v3`
- `torquoise-review-policy-full-20260924-v3`

The retained vision plans were accepted only after model, page-image, prompt, and schema hash checks. Cached plans avoid rerunning Qwen; these are not fresh model latency measurements. The earlier counts remain in `REGION_DECOMPOSITION_AUDIT_2026-09-24.md`.

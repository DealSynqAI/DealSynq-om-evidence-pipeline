# DealSynq OM evidence pipeline

Independent full-page vision and OCR evidence reconciliation for offering memoranda. The output is a page-organized Unified Source Block (USB) collection with coordinates, evidence IDs, raw observations, and review status. It is a source-capture stage, not a guarantee that every printed value has the correct owner. The source PDFs, generated page images, OCR logs, and run outputs are deliberately excluded from this repository.

This is a separate DealSynq pipeline. Its full-page Qwen3-VL call sees only the rendered page image and a generic schema prompt. Native PDF inspection, RapidOCR, and OpenCV run without the model's proposal or hints. Their source blocks are saved in `deterministic/page-NNN.json` before any fusion. Model proposals, raw responses, attempts, failures, and timing remain in `vision-plans/`. OCR lines and visual geometry diagnostics remain in their own work and diagnostics folders.

After both branches finish, `reconciliation/page-NNN.json` compares proposed structured values with printed OCR/native evidence and deterministic source blocks. A chart label can be corrected only when the model's value and label have unique printed evidence, their positions support the pair, and the target observation is unambiguous. A percentage printed directly below a confirmed amount can inherit that label when its own OCR line and target observation are unique. A vision-proposed map can replace an unresolved deterministic route only after US-state reference registration succeeds with strong silhouette overlap and at least five geometry-backed bindings. A missed pie chart can be selected only when its PDF slice geometry verifies and the printed observations reconcile to 100%. The numeric value is never rewritten from model text alone. Corrected charts stay `needs_review` when slice or mark ownership is unverified. For native PDF tables, positioned cell ownership wins over a conflicting vision proposal. Unresolved or missing model claims are recorded in the ledger; they are not turned into verified facts.

A vision-proposed table can trigger a second OCR-based route when the independent route missed that table. The new route is accepted only when it creates at least three row-owned numeric cells from OCR and does not reduce nearby numeric-text coverage or introduce validation errors. The model's proposed cell values are never copied into the USB.

When the page inspector wraps several small tables in one empty region, Paddle's separate, nonoverlapping numeric grids become separate table regions. A broad Paddle detection covering the same grids is treated as a duplicate. Native table rows are never replaced by this split.

The USB is the **source capture** boundary before embeddings, semantic input preparation, context building, or DeepSeek. Every nonblank OCR line is retained in a source block with its page image, coordinates, confidence, and evidence ID. Lines already present in structured content keep that owner. An owned line omitted by the parser is attached to its block as `raw_evidence_lines`; an unowned line becomes an `unresolved_source_text` block. Neither path invents a table cell or changes a printed value. The reconciliation receipt reports how many lines were structured, attached as raw evidence, or given fallback blocks. Structured completeness and raw capture are reported separately, so retaining a line cannot silently claim that its row or chart mark is correct.

The page evidence ledger also records native PDF words and Paddle layout text. Each observation either points to matching RapidOCR evidence or remains visible as raw source evidence in a USB block. A broad layout-only region is split into a key/value table and adjacent prose only when OCR shows repeated aligned label/value pairs and a distinct, denser prose lane. Such table rows carry separate label and value evidence IDs and coordinates. A second pass grounds other table cells only when exact OCR value text is aligned with an exact OCR row label, and, for wider tables, a column header. Each cell reports `grounding_status`; unsupported row/value associations stay explicit candidates and keep the table in review.

The CLI requires a vision endpoint, RapidOCR, PaddleOCR PP-StructureV3, and Poppler's `pdftoppm`. Install the base package and OCR dependencies in the main Python environment with `python -m pip install -e ".[ocr]"`. Install Poppler separately and put `pdftoppm` on `PATH`, or pass its path through `--pdftoppm`. The default vision endpoint is a local Ollama OpenAI-compatible endpoint and the default model name is `dealsynq-qwen3-vl:4b-instruct-16k`; configure `--qwen-endpoint` and `--qwen-model` for another compatible vision deployment. API tokens, when needed, are read from `QWEN_API_KEY` in the environment.

It attempts full-page Qwen vision on every selected page, retaining a failure receipt if a call cannot produce a valid proposal. It runs PP-StructureV3 on every likely table page, including pages where the native PDF parser already found tables. Native table cells remain authoritative on disagreement; Paddle-detected image tables are retained for review. Every run writes a new immutable directory and performs independent schema/integrity validation.

On Windows, use Python 3.12 for PaddleOCR in a separate `.venv-paddle` environment. The RapidOCR runtime can use another supported Python environment. Install PaddlePaddle's CPU wheel and PaddleOCR's document-parser extra per the official Paddle documentation. The worker automatically discovers `.venv-paddle\\Scripts\\python.exe`, or accepts `--paddle-python`. CPU inference avoids contending with the Qwen GPU process. First use downloads PP-StructureV3 model weights into PaddleX's user cache.

```powershell
py -3.12 -m venv .venv-paddle
.venv-paddle\Scripts\python.exe -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
.venv-paddle\Scripts\python.exe -m pip install "paddleocr[doc-parser]>=3.3,<3.6"
```

For GPU table analysis on a compatible NVIDIA GPU, use a separate Python 3.12 environment. The tested Windows CUDA 12.6 runtime is PaddlePaddle GPU 3.2.2. Select `gpu:0` explicitly; the default remains `cpu`. With `--paddle-device gpu:0`, the worker discovers `.venv-paddle-gpu` automatically if `--paddle-python` is omitted.

```powershell
py -3.12 -m venv .venv-paddle-gpu
.venv-paddle-gpu\Scripts\python.exe -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
.venv-paddle-gpu\Scripts\python.exe -m pip install "paddleocr[doc-parser]==3.5.0"
```

```powershell
python -m ocr_pipeline input.pdf --output runs/new-run --paddle-device gpu:0
```

`--vision-plan-cache path/to/vision-plans` may reuse image-only proposals only when the model, rendered image hash, prompt hash, and proposal schema match. This reuses model evidence while rerunning OCR, OpenCV, extraction, and reconciliation. A cached run is not a new model latency measurement.

`validation.json` establishes output shape and evidence integrity. It does not certify that every printed chart or table value has the correct owner. Use the reconciliation ledger and source page to review disputed values.

Each reconciliation page records nearby high-confidence numeric OCR lines that did not survive into source-block content, plus any pre-Paddle route comparison. Image-only tables, chart mark ownership, and reading order can still require review; schema validation alone does not establish complete or accurate extraction.

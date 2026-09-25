"""Offline replay and text-fidelity regression gate for the source-block pipeline.

The gate replays the in-process pipeline logic against stored rendering, OCR,
Paddle, and vision-plan outputs, then scores each result against the PDF's own
text layer. See ``python -m regression.gate --help``.
"""

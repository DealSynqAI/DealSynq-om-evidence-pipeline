from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from typing import Any
from PIL import Image


def normalize_vision_payload(payload: Any) -> dict[str, Any]:
    """Normalize a model response to the small routing contract consumed by the pipeline."""
    if not isinstance(payload, dict):
        raise ValueError("vision model response must be a JSON object")
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    bindings = payload.get("bindings", [])
    if not isinstance(bindings, list):
        bindings = []
    bindings = [binding for binding in bindings if isinstance(binding, dict)]
    chart_type = payload.get("chart_type")
    normalized = {
        "type": str(payload.get("type") or "").strip().lower(),
        "confidence": confidence,
        "chart_type": None if chart_type is None else str(chart_type).strip().lower(),
        "bindings": bindings,
    }
    table_review = payload.get("table_review")
    if isinstance(table_review, dict):
        normalized["table_review"] = table_review
    panel_review = payload.get("panel_review")
    if isinstance(panel_review, dict):
        normalized["panel_review"] = panel_review
    return normalized


def discover_rapidocr_python() -> Path | None:
    candidates = [Path(sys.executable)]
    configured = os.environ.get("DEALSYNQ_RAPIDOCR_PYTHON")
    if configured:
        candidates.insert(0, Path(configured))
    for candidate in candidates:
        if not candidate.exists():
            continue
        result = subprocess.run(
            [str(candidate), "-c", "import rapidocr,cv2"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return candidate.resolve()
    return None


def discover_model_root() -> Path | None:
    configured = os.environ.get("DEALSYNQ_RAPIDOCR_MODELS")
    if not configured:
        return None
    path = Path(configured)
    return path.resolve() if path.is_dir() else None


def run_rapidocr_worker(
    jobs: list[dict[str, Any]], work_dir: Path, python_executable: Path | None = None,
    model_root: Path | None = None,
) -> dict[str, Any]:
    python_executable = python_executable or discover_rapidocr_python()
    if python_executable is None:
        raise RuntimeError(
            "RapidOCR is unavailable. Install the 'ocr' extra or set DEALSYNQ_RAPIDOCR_PYTHON."
        )
    request_path = work_dir / "rapidocr-jobs.json"
    result_path = work_dir / "rapidocr-results.json"
    request_path.write_text(json.dumps({"jobs": jobs}, indent=2) + "\n", encoding="utf-8")
    command = [
        str(python_executable), str(Path(__file__).with_name("rapidocr_worker.py")),
        "--jobs", str(request_path), "--output", str(result_path),
    ]
    model_root = model_root or discover_model_root()
    if model_root:
        command.extend(["--model-root", str(model_root)])
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (work_dir / "rapidocr.log").write_text(
        completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""), encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"RapidOCR worker failed with exit code {completed.returncode}; see {work_dir / 'rapidocr.log'}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def discover_paddle_python(device: str = "cpu") -> Path | None:
    configured = os.environ.get("DEALSYNQ_PADDLE_PYTHON")
    root = Path(__file__).resolve().parents[1]
    bundled_cpu = root / ".venv-paddle" / "Scripts" / "python.exe"
    bundled_gpu = root / ".venv-paddle-gpu" / "Scripts" / "python.exe"
    bundled = [bundled_gpu, bundled_cpu] if device.startswith("gpu") else [bundled_cpu, bundled_gpu]
    for candidate in [Path(configured) if configured else None, *bundled, Path(sys.executable)]:
        if candidate is None or not candidate.is_file():
            continue
        probe = (
            "import paddle,paddleocr; from paddleocr import PPStructureV3; "
            "assert paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0"
            if device.startswith("gpu") else
            "import paddle,paddleocr; from paddleocr import PPStructureV3"
        )
        result = subprocess.run(
            [str(candidate), "-c", probe],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return candidate.resolve()
    return None


def run_paddle_table_worker(
    rendered: dict[int, Path], output: Path, python_executable: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    python_executable = python_executable or discover_paddle_python(device)
    if python_executable is None:
        raise RuntimeError("PaddleOCR PP-StructureV3 is unavailable; install the doc-parser extra in a Python 3.9–3.13 runtime or pass --paddle-python")
    output.mkdir(parents=True, exist_ok=True)
    request = output / "jobs.json"
    summary = output / "summary.json"
    request.write_text(json.dumps([
        {"page": page, "image": str(image)} for page, image in sorted(rendered.items())
    ], indent=2) + "\n", encoding="utf-8")
    command = [str(python_executable), str(Path(__file__).with_name("paddle_table_worker.py")),
               str(request), str(output), str(summary), "--device", device]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (output / "worker.log").write_text(
        completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""), encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"PaddleOCR table worker failed with exit code {completed.returncode}; see {output / 'worker.log'}")
    result = json.loads(summary.read_text(encoding="utf-8"))
    result["python"] = str(python_executable)
    return result


class QwenVisionClient:
    """Small OpenAI-compatible adapter used only for classification and visual grounding."""

    def __init__(self, endpoint: str, model: str, api_key_env: str = "QWEN_API_KEY", timeout: int = 120):
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout

    def analyze(self, image_path: Path, prompt: str, max_bindings: int = 60) -> dict[str, Any]:
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        with Image.open(image_path) as source:
            source.load()
            image = source.convert("RGB")
            image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        response_schema = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "normal_text", "table", "chart", "map", "kpi_panel", "photograph",
                        "brand_mark", "unclassified_visual",
                    ],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "chart_type": {"type": ["string", "null"]},
                "bindings": {
                    "type": "array",
                    # Prevent a vision model from looping until the response is
                    # truncated while still covering a dense US map.
                    "maxItems": max(0, int(max_bindings)),
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "label_evidence_id": {"type": ["string", "null"]},
                            "value_evidence_id": {"type": "string"},
                        },
                        "required": ["label", "label_evidence_id", "value_evidence_id"],
                        "additionalProperties": False,
                    },
                },
                "table_review": {
                    "type": "object",
                    "properties": {
                        "data_row_count": {"type": "integer", "minimum": 0},
                        "column_count": {"type": "integer", "minimum": 0},
                        "headers": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
                        "structure_matches": {"type": "boolean"},
                    },
                    "required": ["data_row_count", "column_count", "headers", "structure_matches"],
                    "additionalProperties": False,
                },
                "panel_review": {
                    "type": "object",
                    "properties": {
                        "lane_count": {"type": "integer", "minimum": 0},
                        "leaf_titles": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                        "claim_counts": {"type": "array", "items": {"type": "integer", "minimum": 0}, "maxItems": 12},
                        "structure_matches": {"type": "boolean"},
                    },
                    "required": ["lane_count", "leaf_titles", "claim_counts", "structure_matches"],
                    "additionalProperties": False,
                },
            },
            "required": ["type", "confidence", "chart_type", "bindings"],
            "additionalProperties": False,
        }
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 4096,
            # These are understood by common Qwen and Ollama-compatible
            # servers. Servers that ignore them still receive a bounded call.
            "enable_thinking": False,
            "think": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "document_region_result", "strict": True, "schema": response_schema},
            },
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }],
        }
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        content = str(content).strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            if content.lstrip().startswith("json"):
                content = content.lstrip()[4:].lstrip()
        return normalize_vision_payload(json.loads(content))

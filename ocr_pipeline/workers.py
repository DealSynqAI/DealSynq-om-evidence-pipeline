from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


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



"""Experiment records produced by the offline training and evaluation scripts.

The assignment requires the demonstration to show experiment-tracking records.
The training runs logged to Weights & Biases; this endpoint surfaces the
resulting metric files so the browser can display them without a W&B account or
network access. Every number here comes from ``results/`` -- nothing is
recomputed, so the interface can never disagree with the technical report.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import ASSETS_DIR

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

EXPERIMENTS_DIR = ASSETS_DIR / "experiments"

#: Figures copied from ``results/`` so they render inside the container.
FIGURES: dict[str, dict[str, str]] = {
    "task1_representative": {
        "file": "task1/grid_representative.png",
        "title": "Task 1 - representative restorations",
        "caption": "Clean target, corrupted input, reconstruction and absolute error map.",
    },
    "task1_failures": {
        "file": "task1/grid_failures.png",
        "title": "Task 1 - failure cases",
        "caption": "Cases where the universal autoencoder under-restores.",
    },
    "task2_confusion_matrix": {
        "file": "task2/confusion_matrix.png",
        "title": "Task 2 - normalised confusion matrix",
        "caption": "Four-class corruption classifier on the deterministic test manifest.",
    },
    "task2_oracle": {
        "file": "task2/grid_oracle.png",
        "title": "Task 2 - oracle routing",
        "caption": "Specialists selected by the known corruption label.",
    },
    "task2_predicted": {
        "file": "task2/grid_predicted.png",
        "title": "Task 2 - predicted routing",
        "caption": "The complete operational system, including classifier errors.",
    },
    "task2_worst_cases": {
        "file": "task2/grid_worst_cases.png",
        "title": "Task 2 - worst misrouted cases",
        "caption": "Inputs where a classifier error causes a restoration failure.",
    },
}


def _load(*parts: str) -> dict[str, Any]:
    path = EXPERIMENTS_DIR.joinpath(*parts)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{'/'.join(parts)} is not bundled")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def task1_summary() -> dict[str, Any]:
    """Per-corruption and per-severity restoration quality for Task 1."""
    metrics = _load("task1", "test_metrics.json")
    groups = metrics.get("by_group", {})
    by_corruption: dict[str, dict[str, Any]] = {}
    by_severity: dict[str, dict[str, list[float]]] = {}
    for key, entry in groups.items():
        corruption, _, severity = key.partition("/")
        by_corruption.setdefault(
            corruption, {"corruption": corruption, "l1": [], "ssim": [], "rank": [], "n": 0}
        )
        for metric in ("l1", "ssim", "rank"):
            by_corruption[corruption][metric].append(entry[metric]["mean"])
            by_corruption[corruption]["n"] += entry[metric]["n"]
        by_severity.setdefault(severity, {"severity": severity, "l1": [], "ssim": [], "n": 0})
        by_severity[severity]["l1"].append(entry["l1"]["mean"])
        by_severity[severity]["ssim"].append(entry["ssim"]["mean"])
        by_severity[severity]["n"] += entry["l1"]["n"]

    for entry in by_corruption.values():
        for metric in ("l1", "ssim", "rank"):
            values = entry[metric]
            entry[metric] = round(sum(values) / len(values), 6)
    for entry in by_severity.values():
        for metric in ("l1", "ssim"):
            values = entry[metric]
            entry[metric] = round(sum(values) / len(values), 6)

    return {
        "overall": metrics.get("overall", {}),
        "by_corruption": sorted(by_corruption.values(), key=lambda item: item["corruption"]),
        "by_severity": sorted(by_severity.values(), key=lambda item: item["severity"]),
        "onnx_verification": _load("task1", "onnx_verification.json"),
    }


@lru_cache(maxsize=1)
def task2_summary() -> dict[str, Any]:
    """Classifier metrics plus oracle/predicted restoration quality."""
    evaluation = _load("task2", "evaluation_results.json")
    best = _load("task2", "best_params.json")
    predicted = evaluation.get("predicted", {})
    oracle = evaluation.get("oracle", {})
    return {
        "classifier": predicted.get("classifier", {}),
        "predicted": {
            "reconstruction": predicted.get("reconstruction", {}),
            "routing_failures": evaluation.get("routing_failures", {}),
        },
        "oracle": {"reconstruction": oracle.get("reconstruction", {})},
        "optuna": best,
        "onnx_parity": _load("task2", "onnx_parity.json"),
    }


@lru_cache(maxsize=1)
def task4_summary() -> dict[str, Any]:
    """Per-style sketch generation quality."""
    return {"metrics": _load("task4", "metrics.json")}


@router.get("/summary")
def summary() -> dict[str, Any]:
    """Everything the Experiments workspace renders."""
    return {
        "task1": task1_summary(),
        "task2": task2_summary(),
        "task4": task4_summary(),
        "figures": [
            {"id": key, **meta, "url": f"/api/experiments/figures/{key}"}
            for key, meta in FIGURES.items()
        ],
        "notes": {
            "tracking": "Training runs were logged to Weights & Biases; these files are the "
            "exported metric records from results/.",
            "source": "Every value is read verbatim from the repository's results folder.",
        },
    }


@router.get("/figures/{figure_id}")
def figure(figure_id: str) -> Any:
    """Serve one bundled evaluation figure."""
    from fastapi.responses import FileResponse

    meta = FIGURES.get(figure_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown figure {figure_id!r}")
    path = EXPERIMENTS_DIR / meta["file"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{meta['file']} is not bundled")
    return FileResponse(path, media_type="image/png")

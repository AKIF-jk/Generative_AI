"""Liveness, model inventory and static workspace metadata."""

from __future__ import annotations

import os
import platform
import time
from typing import Any

from fastapi import APIRouter

from .. import __version__
from ..config import (
    ASSETS_DIR,
    IMAGE_SIZE,
    MODEL_ROOT,
    TASK_IDS,
    WORKSPACES,
)
from ..corruptions import catalog as corruption_catalog
from ..registry import registry
from ..services.samples import list_samples

router = APIRouter(prefix="/api", tags=["system"])

_STARTED_AT = time.time()


@router.get("/health")
def health() -> dict[str, Any]:
    """Report API liveness plus which ONNX artefacts are currently usable."""
    models = registry.status()
    ready = [model for model in models if model["available"]]
    tasks_ready = sorted({model["task"] for model in ready})
    return {
        "status": "ok" if len(ready) == len(models) else "degraded",
        "version": __version__,
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "image_size": IMAGE_SIZE,
        "model_root": str(MODEL_ROOT),
        "assets_root": str(ASSETS_DIR),
        "tasks": [
            {
                "task": task,
                **WORKSPACES[task],
                "ready": task in tasks_ready,
                "models": [model["key"] for model in models if model["task"] == task],
            }
            for task in TASK_IDS
        ],
        "models": models,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
    }


@router.get("/catalog")
def catalog() -> dict[str, Any]:
    """Everything the frontend needs to render its controls."""
    return {
        "workspaces": [{"task": task, **meta} for task, meta in WORKSPACES.items()],
        "corruptions": corruption_catalog(),
        "samples": list_samples(),
        "image_size": IMAGE_SIZE,
    }

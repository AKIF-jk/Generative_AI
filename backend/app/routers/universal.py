"""Task 1 - Universal multi-corruption restoration."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import MODEL_SPECS_BY_KEY  # noqa: F401  (kept for spec lookups)
from ..imaging import to_nchw
from ..registry import ModelUnavailableError, registry
from ..services.pipeline import RequestError, build_response, prepare_input

router = APIRouter(prefix="/api/universal", tags=["task1"])

MODEL_KEY = "universal"


@router.post("/restore")
async def restore(
    image: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
    corruption: str | None = Form(default=None),
    severity: str | None = Form(default=None),
    seed: int | None = Form(default=None),
) -> dict[str, Any]:
    """Restore one image with the universal autoencoder.

    Accepts an uploaded file or a bundled clean ``sample_id``. When a
    ``corruption`` is supplied it is applied first, which both demonstrates the
    training corruption pipeline and provides a clean reference so L1, PSNR and
    SSIM are measured against the true target.
    """
    total_start = time.perf_counter()

    try:
        prepared = prepare_input(
            upload=await image.read() if image is not None else None,
            sample_id=sample_id,
            corruption=corruption,
            severity=severity,
            seed=seed,
        )
    except RequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    preprocess_ms = (time.perf_counter() - total_start) * 1000.0

    tensor = to_nchw(prepared.network_input)
    inference_start = time.perf_counter()
    try:
        outputs = registry.run(MODEL_KEY, {"input": tensor})
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "model": exc.path,
                "hint": "Place the exported ONNX weights in the models directory, then retry.",
            },
        ) from exc
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    postprocess_start = time.perf_counter()
    restored = np.asarray(outputs[0], dtype=np.float32)

    payload = build_response(
        prepared,
        restored,
        inference_ms=inference_ms,
        preprocess_ms=preprocess_ms,
        postprocess_ms=(time.perf_counter() - postprocess_start) * 1000.0,
        total_ms=(time.perf_counter() - total_start) * 1000.0,
        extra={
            "task": "task1",
            "workspace": "Universal Restoration",
            "model": model_metadata(),
        },
    )
    return payload


def model_metadata() -> dict[str, Any]:
    """Describe the loaded Task 1 artefact for the interface."""
    return {
        **registry.info(MODEL_KEY),
        "opset": 17,
        "architecture": "5-stage convolutional encoder, flattened vector bottleneck, "
        "convolutional decoder with two gated skips",
        "bottleneck": "vector (no spatial skip at 128x128 or 64x64)",
    }

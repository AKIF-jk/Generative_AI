"""Task 4 - Style-conditioned face-to-sketch generation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import MAX_UPLOAD_BYTES, STYLE_NAMES
from ..imaging import (
    InvalidImageError,
    decode_upload,
    encode_data_url,
    from_unit_range,
    resize_to_model_input,
    to_unit_range,
)
from ..registry import ModelUnavailableError, registry

router = APIRouter(prefix="/api/face-to-sketch", tags=["task4"])

MODEL_KEY = "face_to_sketch"

VALID_STYLES = tuple(sorted(STYLE_NAMES))


@router.post("/generate")
async def generate(
    photo: UploadFile = File(...),
    style: int = Form(...),
) -> dict[str, Any]:
    """Generate the sketch that matches ``photo`` under the selected style.

    The style condition is a learned categorical embedding inside the
    generator, so ``style`` selects one of the three FS2K sketch categories
    rather than acting as a free-text prompt.
    """
    if style not in VALID_STYLES:
        raise HTTPException(
            status_code=400, detail=f"style must be one of {', '.join(map(str, VALID_STYLES))}"
        )

    payload = await photo.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"the uploaded file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    try:
        image = decode_upload(payload)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    total_start = time.perf_counter()
    resized = resize_to_model_input(image)
    preprocess_ms = (time.perf_counter() - total_start) * 1000.0

    # Generator convention: NCHW float32 in [-1, 1] plus an int64 style index.
    photo_tensor = np.ascontiguousarray(
        np.transpose(to_unit_range(resized), (2, 0, 1))[None, ...], dtype=np.float32
    )
    style_tensor = np.array([style - 1], dtype=np.int64)

    inference_start = time.perf_counter()
    try:
        outputs = registry.run(MODEL_KEY, {"photo": photo_tensor, "style_id": style_tensor})
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "model": exc.path,
                "hint": "Place generator.onnx in the Task 4 models directory, then retry.",
            },
        ) from exc
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    postprocess_start = time.perf_counter()
    sketch = from_unit_range(np.asarray(outputs[0], dtype=np.float32))
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    return {
        "task": "task4",
        "workspace": "Face-to-Sketch Generator",
        "style": {"id": style, "name": STYLE_NAMES[style], "embedding_index": style - 1},
        "images": {
            "photo": encode_data_url(resized),
            "sketch": encode_data_url(sketch),
        },
        "timing_ms": {
            "preprocess": round(preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "postprocess": round(postprocess_ms, 2),
            "total": round(total_ms, 2),
        },
        "model": {
            **registry.info(MODEL_KEY),
            "opset": 14,
            "architecture": "U-Net generator, 7 down / 6 up stages, style embedding "
            "concatenated as extra input channels",
            "output_range": "[-1, 1] (tanh)",
        },
    }


@router.get("/styles")
def styles() -> dict[str, Any]:
    """List the three FS2K sketch-style conditions."""
    return {
        "styles": [{"id": style, "name": STYLE_NAMES[style]} for style in VALID_STYLES],
        "note": "The style condition is a learned categorical embedding consumed by "
        "the generator's first convolution.",
    }

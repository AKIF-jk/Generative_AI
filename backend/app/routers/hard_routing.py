"""Task 2 - Corruption classification and hard-routed specialist restoration."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import (
    CORRUPTION_LABELS,
    LABEL_TO_CORRUPTION,
    SPECIALIST_EXPERTS,
)
from ..imaging import from_float01, to_float01, to_nchw
from ..registry import ModelUnavailableError, registry
from ..services.pipeline import (
    RequestError,
    build_response,
    prepare_input,
    validate_true_label,
)

router = APIRouter(prefix="/api/hard-routing", tags=["task2"])

CLASSIFIER_KEY = "classifier"

#: Maps a corruption label to the ONNX artefact of the expert responsible for it.
EXPERT_MODELS: dict[int, str] = {
    CORRUPTION_LABELS["salt_pepper"]: "specialist_salt_pepper",
    CORRUPTION_LABELS["gaussian_blur"]: "specialist_gaussian_blur",
    CORRUPTION_LABELS["occlusion"]: "specialist_occlusion",
}

EXPERT_TITLES: dict[int, str] = {
    CORRUPTION_LABELS["clean"]: "Identity bypass",
    CORRUPTION_LABELS["salt_pepper"]: "Salt-and-pepper specialist",
    CORRUPTION_LABELS["gaussian_blur"]: "Gaussian blur specialist",
    CORRUPTION_LABELS["occlusion"]: "Occlusion specialist",
}


@router.post("/restore")
async def restore(
    image: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
    corruption: str | None = Form(default=None),
    severity: str | None = Form(default=None),
    seed: int | None = Form(default=None),
    routing: Literal["predicted", "oracle"] = Form(default="predicted"),
    true_label: str | None = Form(default=None),
) -> dict[str, Any]:
    """Classify the input, route it, and restore it with the selected expert.

    ``routing=predicted`` is the operational system. ``routing=oracle`` uses the
    known corruption label so the specialist's own ability can be measured
    without classifier errors interfering.
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
        oracle_label = validate_true_label(true_label)
    except RequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if routing == "oracle":
        if oracle_label is None and prepared.settings is not None:
            # A corruption applied through the interface supplies its own label.
            oracle_label = CORRUPTION_LABELS[prepared.settings.corruption]
        if oracle_label is None:
            raise HTTPException(
                status_code=400,
                detail="oracle routing needs true_label, or a corruption applied to a clean sample",
            )

    preprocess_ms = (time.perf_counter() - total_start) * 1000.0
    tensor = to_nchw(prepared.network_input)

    # -- stage 1: classify ---------------------------------------------------
    classifier_start = time.perf_counter()
    try:
        logits = np.asarray(registry.run(CLASSIFIER_KEY, {"input": tensor})[0], dtype=np.float32)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=_unavailable(exc)) from exc
    classifier_ms = (time.perf_counter() - classifier_start) * 1000.0

    probabilities = _softmax(logits)[0]
    predicted_index = int(np.argmax(probabilities))

    # -- stage 2: route ------------------------------------------------------
    if routing == "oracle":
        routed_index = int(oracle_label)
    else:
        routed_index = predicted_index

    corruption_name = LABEL_TO_CORRUPTION[routed_index]
    applied_corruption = prepared.settings.corruption if prepared.settings else None
    agreement = {
        "routing": routing,
        "agrees_with_prediction": routed_index == predicted_index,
        "applied_corruption": applied_corruption,
        "applied_label": CORRUPTION_LABELS[applied_corruption] if applied_corruption else None,
        "oracle_label": LABEL_TO_CORRUPTION[oracle_label] if oracle_label is not None else None,
    }
    if applied_corruption is not None and routed_index != CORRUPTION_LABELS[applied_corruption]:
        agreement["misrouted"] = True
        agreement["note"] = (
            f"the {routing} router selected '{corruption_name}' but the applied "
            f"corruption was '{applied_corruption}'"
        )
    else:
        agreement["misrouted"] = False

    # -- stage 3: restore ---------------------------------------------------
    expert_start = time.perf_counter()
    expert_ms = 0.0
    if routed_index == CORRUPTION_LABELS["clean"]:
        # Identity bypass: a clean input must not be pushed through an expert.
        restored = tensor
        used_model: str | None = None
    else:
        model_key = EXPERT_MODELS[routed_index]
        try:
            restored = np.asarray(registry.run(model_key, {"input": tensor})[0], dtype=np.float32)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=_unavailable(exc)) from exc
        used_model = model_key
    expert_ms = (time.perf_counter() - expert_start) * 1000.0

    postprocess_start = time.perf_counter()
    payload = build_response(
        prepared,
        restored,
        inference_ms=classifier_ms + expert_ms,
        preprocess_ms=preprocess_ms,
        postprocess_ms=(time.perf_counter() - postprocess_start) * 1000.0,
        total_ms=(time.perf_counter() - total_start) * 1000.0,
        extra={
            "task": "task2",
            "workspace": "Hard-Routed Restoration",
            "routing": {
                "mode": routing,
                "classes": list(CORRUPTION_LABELS.keys()),
                "logits": [round(float(value), 6) for value in logits[0]],
                "probabilities": [round(float(value), 6) for value in probabilities],
                "predicted_index": predicted_index,
                "predicted_corruption": LABEL_TO_CORRUPTION[predicted_index],
                "routed_index": routed_index,
                "routed_corruption": corruption_name,
                "selected_expert": EXPERT_TITLES[routed_index],
                "selected_model": used_model,
                "identity_bypass": routed_index == CORRUPTION_LABELS["clean"],
                "confidence": round(float(probabilities[predicted_index]), 6),
                "margin": round(
                    float(
                        np.sort(probabilities)[-1] - np.sort(probabilities)[-2]
                    ),
                    6,
                ),
                **agreement,
            },
            "timing_ms": {
                "preprocess": round(preprocess_ms, 2),
                "classifier": round(classifier_ms, 2),
                "expert": round(expert_ms, 2),
                "total": round((time.perf_counter() - total_start) * 1000.0, 2),
            },
            "models": {
                "classifier": registry.info(CLASSIFIER_KEY),
                "selected_expert": registry.info(used_model) if used_model else None,
            },
        },
    )
    return payload


@router.get("/experts")
def experts() -> dict[str, Any]:
    """Describe the routing table and the availability of every expert."""
    return {
        "classes": [
            {
                "index": index,
                "name": name,
                "title": EXPERT_TITLES[index],
                "model": EXPERT_MODELS.get(index),
                "is_identity": index == CORRUPTION_LABELS["clean"],
            }
            for name, index in CORRUPTION_LABELS.items()
        ],
        "models": {
            key: registry.info(key)
            for key in (CLASSIFIER_KEY, *EXPERT_MODELS.values())
        },
    }


def _unavailable(exc: ModelUnavailableError) -> dict[str, str]:
    return {
        "message": str(exc),
        "model": exc.path,
        "hint": "Place the exported ONNX weights in the models directory, then retry.",
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

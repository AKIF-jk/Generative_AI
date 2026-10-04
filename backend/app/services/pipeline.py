"""Shared request handling for the two restoration workspaces.

Tasks 1 and 2 accept the same kinds of input -- an uploaded image, a bundled
clean sample, and optionally a corruption to apply first -- so the resolution,
validation and metric-gathering logic lives here instead of being duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from .. import corruptions as corruption_ops
from ..config import IMAGE_SIZE, MAX_UPLOAD_BYTES
from ..corruptions import CorruptionSettings
from ..imaging import (
    InvalidImageError,
    absolute_error_heatmap,
    absolute_error_map,
    decode_upload,
    encode_data_url,
    encode_png,
    from_float01,
    mean_absolute_error,
    psnr,
    resize_to_model_input,
    ssim,
    to_data_url,
    to_float01,
    to_uint8,
)
from .samples import SAMPLES, load_sample


class RequestError(ValueError):
    """A client-side problem; the routers translate this into HTTP 400."""


@dataclass
class PreparedInput:
    """The tensors and metadata every restoration request starts from."""

    clean: Image.Image
    """The unaltered 128x128 image; the reconstruction target when known."""

    network_input: Image.Image
    """What the model actually receives (possibly corrupted)."""

    settings: CorruptionSettings | None
    source: str
    filename: str | None = None
    clean_float: np.ndarray = field(init=False)
    input_float: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.clean_float = to_float01(self.clean)
        self.input_float = to_float01(self.network_input)


def prepare_input(
    *,
    upload: bytes | None,
    sample_id: str | None,
    corruption: str | None,
    severity: str | None,
    seed: int | None,
) -> PreparedInput:
    """Resolve the upload/sample choice and optionally corrupt it."""
    if upload is not None:
        if len(upload) > MAX_UPLOAD_BYTES:
            raise RequestError(
                f"the uploaded file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
            )
        try:
            image = decode_upload(upload)
        except InvalidImageError as exc:
            raise RequestError(str(exc)) from exc
        source, filename = "upload", None
    elif sample_id:
        if sample_id not in SAMPLES:
            raise RequestError(f"unknown sample id {sample_id!r}")
        image = load_sample(sample_id)
        source, filename = "sample", SAMPLES[sample_id].filename
    else:
        raise RequestError("provide either an uploaded image or a sample id")

    clean = resize_to_model_input(image, IMAGE_SIZE)

    if not corruption or corruption == "clean":
        return PreparedInput(clean=clean, network_input=clean, settings=None, source=source,
                             filename=filename)

    array = to_uint8(clean)
    try:
        settings = corruption_ops.resolve_settings(
            corruption, severity, seed=seed, shape=array.shape[:2]
        )
        corrupted = corruption_ops.apply_settings(array, settings)
    except corruption_ops.UnknownCorruptionError as exc:
        raise RequestError(str(exc)) from exc

    return PreparedInput(
        clean=clean,
        network_input=Image.fromarray(corrupted, mode="RGB"),
        settings=settings,
        source=source,
        filename=filename,
    )


def build_response(
    prepared: PreparedInput,
    restored: np.ndarray,
    *,
    inference_ms: float,
    preprocess_ms: float,
    postprocess_ms: float,
    total_ms: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON payload shared by the Task 1 and Task 2 endpoints."""
    restored_image = from_float01(restored)
    target = prepared.clean_float
    prediction = to_float01(restored_image)

    payload: dict[str, Any] = {
        "source": {
            "kind": prepared.source,
            "filename": prepared.filename,
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "has_clean_reference": prepared.settings is not None,
        },
        "corruption": prepared.settings.describe() if prepared.settings else {
            "corruption": "clean",
            "severity": None,
            "seed": None,
            "parameters": {},
        },
        "images": {
            "clean": encode_data_url(prepared.clean),
            "input": encode_data_url(prepared.network_input),
            "output": encode_data_url(restored_image),
            "error_map": to_data_url(encode_png(_as_png_image(absolute_error_map(target, prediction)))),
            "error_heatmap": encode_data_url(_as_png_image(absolute_error_heatmap(target, prediction))),
        },
        "downloads": {
            "output": to_data_url(encode_png(restored_image)),
            "error_map": to_data_url(
                encode_png(_as_png_image(absolute_error_map(target, prediction)))
            ),
        },
        "metrics": {
            "reference": "clean_target" if prepared.settings is not None else "input_image",
            "l1": round(mean_absolute_error(target, prediction), 6),
            "psnr": round(psnr(target, prediction), 3),
            "ssim": round(ssim(target, prediction), 6),
            "input_l1": round(mean_absolute_error(prepared.input_float, prediction), 6),
            "input_ssim": round(ssim(prepared.input_float, prediction), 6),
        },
        "timing_ms": {
            "preprocess": round(preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "postprocess": round(postprocess_ms, 2),
            "total": round(total_ms, 2),
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _as_png_image(array: np.ndarray) -> Image.Image:
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L")
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def validate_true_label(value: str | None) -> int | None:
    """Accept a corruption name or a numeric label index for oracle routing."""
    from ..config import CORRUPTION_LABELS, LABEL_TO_CORRUPTION

    if value in (None, "", "auto"):
        return None
    text = str(value).strip()
    if text in CORRUPTION_LABELS:
        return CORRUPTION_LABELS[text]
    if text in LABEL_TO_CORRUPTION:
        return LABEL_TO_CORRUPTION[int(text)]
    raise RequestError(
        f"unknown corruption label {value!r}; expected one of "
        f"{', '.join(CORRUPTION_LABELS)} or an index 0-3"
    )

"""Image corruptions used by the restoration tasks.

All public functions use uint8 ``(H, W, 3)`` RGB NumPy arrays.  Keeping the
corruptions independent from the Dataset makes them straightforward to test
and lets the application use exactly the training/evaluation implementation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _validate_rgb(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a uint8 RGB array with shape (H, W, 3)")


def salt_and_pepper(
    image: np.ndarray, probability: float, rng: np.random.Generator
) -> np.ndarray:
    """Replace ``probability`` of pixels with an equal black/white split."""
    _validate_rgb(image)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    corrupted = image.copy()
    height, width = image.shape[:2]
    count = int(round(probability * height * width))
    if count == 0:
        return corrupted
    flat_indices = rng.choice(height * width, size=count, replace=False)
    black_count = count // 2
    flat = corrupted.reshape(-1, 3)
    flat[flat_indices[:black_count]] = 0
    flat[flat_indices[black_count:]] = 255
    return corrupted


def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply a channel-wise Gaussian blur with the requested finite kernel."""
    _validate_rgb(image)
    if kernel_size not in (3, 5, 7):
        raise ValueError("kernel_size must be one of 3, 5, or 7")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    # Separable convolution preserves the requested 3/5/7-pixel kernel and
    # avoids an OpenCV/scipy dependency.
    padded_x = np.pad(image.astype(np.float32), ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    horizontal = np.zeros_like(image, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        horizontal += weight * padded_x[:, offset : offset + image.shape[1], :]
    padded_y = np.pad(horizontal, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    blurred = np.zeros_like(horizontal, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        blurred += weight * padded_y[offset : offset + image.shape[0], :, :]
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def occlude(image: np.ndarray, rectangles: Sequence[dict[str, int]]) -> np.ndarray:
    """Overlay black half-open ``x, y, width, height`` rectangles."""
    _validate_rgb(image)
    height, width = image.shape[:2]
    corrupted = image.copy()
    for rectangle in rectangles:
        x, y = int(rectangle["x"]), int(rectangle["y"])
        rect_width, rect_height = int(rectangle["width"]), int(rectangle["height"])
        if rect_width <= 0 or rect_height <= 0:
            raise ValueError("occlusion rectangle dimensions must be positive")
        if x < 0 or y < 0 or x + rect_width > width or y + rect_height > height:
            raise ValueError("occlusion rectangle is outside the image")
        corrupted[y : y + rect_height, x : x + rect_width] = 0
    return corrupted


def apply_manifest_corruption(image: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    """Apply one deterministic manifest record to an RGB image."""
    corruption = record["corruption"]
    parameters = record.get("parameters", {})
    if corruption == "clean":
        return image.copy()
    if corruption == "salt_pepper":
        return salt_and_pepper(image, float(parameters["probability"]), np.random.default_rng(int(record["seed"])))
    if corruption == "gaussian_blur":
        return gaussian_blur(image, int(parameters["kernel_size"]), float(parameters["sigma"]))
    if corruption == "occlusion":
        return occlude(image, parameters["rectangles"])
    raise ValueError(f"Unknown corruption type: {corruption!r}")

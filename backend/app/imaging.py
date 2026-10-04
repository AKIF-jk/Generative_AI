"""Image preprocessing, postprocessing and reference-free evaluation helpers.

The application must reproduce the exact tensor conventions used during
training, otherwise the ONNX models receive out-of-distribution inputs:

* Tasks 1 and 2 -- ``float32`` NCHW tensors scaled to ``[0, 1]``; the models
  end in a sigmoid, so their outputs are already in ``[0, 1]``.
* Task 4 -- ``float32`` NCHW tensors scaled to ``[-1, 1]``; the generator ends
  in a tanh.

Resizing always uses bicubic interpolation to 128x128, matching
``scripts/prepare_pet_data.py`` and the Task 4 augmentation pipeline.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import IMAGE_SIZE

RESAMPLE = Image.Resampling.BICUBIC

#: Areas below this many pixels are treated as background by the heat-map.
_ERROR_MAP_FLOOR = 1e-8


class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot be decoded as an RGB image."""


@dataclass(frozen=True)
class EncodedImage:
    """A PNG payload plus the pixel array it was produced from."""

    png: bytes
    array: np.ndarray


def decode_upload(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an RGB :class:`PIL.Image.Image`."""
    if not data:
        raise InvalidImageError("the uploaded file is empty")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("the uploaded file is not a readable image") from exc
    return image.convert("RGB")


def resize_to_model_input(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize to the square resolution every model in this project expects."""
    return image.convert("RGB").resize((size, size), RESAMPLE)


def to_uint8(image: Image.Image) -> np.ndarray:
    """Return an ``(H, W, 3)`` uint8 array for an RGB image."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def to_float01(image: Image.Image) -> np.ndarray:
    """Return an ``(H, W, 3)`` float32 array scaled to ``[0, 1]``."""
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def to_unit_range(image: Image.Image) -> np.ndarray:
    """Return an ``(H, W, 3)`` float32 array scaled to ``[-1, 1]``."""
    return (np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5) - 1.0


def to_nchw(image: Image.Image, *, signed: bool = False) -> np.ndarray:
    """Return a ``(1, 3, H, W)`` float32 batch in ``[0, 1]`` or ``[-1, 1]``."""
    array = to_unit_range(image) if signed else to_float01(image)
    return np.ascontiguousarray(np.transpose(array, (2, 0, 1))[None, ...])


def from_float01(chw: np.ndarray) -> Image.Image:
    """Convert a ``(3, H, W)`` or ``(1, 3, H, W)`` tensor in ``[0, 1]`` to an image."""
    array = np.asarray(chw, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) tensor, received shape {array.shape}")
    hwc = np.clip(np.transpose(array, (1, 2, 0)), 0.0, 1.0)
    return Image.fromarray(np.rint(hwc * 255.0).astype(np.uint8), mode="RGB")


def from_unit_range(chw: np.ndarray) -> Image.Image:
    """Convert a ``(3, H, W)`` or ``(1, 3, H, W)`` tensor in ``[-1, 1]`` to an image."""
    array = np.asarray(chw, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) tensor, received shape {array.shape}")
    hwc = np.clip((np.transpose(array, (1, 2, 0)) + 1.0) * 0.5, 0.0, 1.0)
    return Image.fromarray(np.rint(hwc * 255.0).astype(np.uint8), mode="RGB")


def encode_png(image: Image.Image) -> bytes:
    """Serialise an image to PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def encode_array_png(array: np.ndarray) -> bytes:
    """Serialise a uint8 ``(H, W, 3)`` or ``(H, W)`` array to PNG bytes."""
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 2:
        return encode_png(Image.fromarray(array.astype(np.uint8), mode="L"))
    return encode_png(Image.fromarray(array.astype(np.uint8), mode="RGB"))


def to_data_url(png: bytes) -> str:
    """Return a ``data:`` URL so the SPA can render an image without a round trip."""
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def encode_data_url(image: Image.Image) -> str:
    return to_data_url(encode_png(image))


# --------------------------------------------------------------------------
# Error maps and metrics
# --------------------------------------------------------------------------


def absolute_error_map(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return the per-pixel mean absolute error as a uint8 grayscale map.

    ``target`` and ``prediction`` are ``(H, W, 3)`` float arrays in ``[0, 1]``.
    The output is linearly rescaled so the worst pixel in the pair maps to 255,
    which makes faint residuals visible in the browser.
    """
    error = np.abs(np.asarray(target, np.float32) - np.asarray(prediction, np.float32))
    error = error.mean(axis=2)
    peak = float(error.max())
    if peak <= _ERROR_MAP_FLOOR:
        return np.zeros(error.shape, dtype=np.uint8)
    return np.rint(error / peak * 255.0).astype(np.uint8)


def absolute_error_heatmap(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return the absolute error as an inferno-style RGB heat map."""
    error = np.abs(np.asarray(target, np.float32) - np.asarray(prediction, np.float32))
    error = error.mean(axis=2)
    peak = float(error.max())
    normalised = error / peak if peak > _ERROR_MAP_FLOOR else error
    return np.rint(_inferno(normalised) * 255.0).astype(np.uint8)


def _inferno(t: np.ndarray) -> np.ndarray:
    """Piecewise-linear approximation of matplotlib's ``inferno`` colormap.

    Interpolating a small table of control points keeps matplotlib out of the
    runtime image; the table is sampled from ``inferno`` so the resulting
    colours match the error maps in the technical report.
    """
    t = np.clip(t, 0.0, 1.0)
    stops = np.array(
        [
            0.000, 0.114, 0.235, 0.352, 0.474, 0.588, 0.697, 0.797, 0.887, 0.966, 1.000,
        ]
    )
    red = np.array(
        [0.001, 0.045, 0.145, 0.255, 0.372, 0.505, 0.658, 0.806, 0.921, 0.987, 0.988]
    )
    green = np.array(
        [0.000, 0.031, 0.145, 0.282, 0.413, 0.548, 0.665, 0.758, 0.865, 0.944, 0.999]
    )
    blue = np.array(
        [0.014, 0.168, 0.298, 0.341, 0.353, 0.442, 0.518, 0.588, 0.702, 0.851, 0.645]
    )
    out = np.empty(t.shape + (3,), dtype=np.float64)
    for channel, table in enumerate((red, green, blue)):
        flat = t.reshape(-1)
        values = np.interp(flat, stops, table).astype(np.float64)
        out[..., channel] = values.reshape(t.shape)
    return out


def mean_absolute_error(target: np.ndarray, prediction: np.ndarray) -> float:
    """Mean absolute error over all pixels and channels, in ``[0, 1]`` units."""
    return float(
        np.abs(np.asarray(target, np.float32) - np.asarray(prediction, np.float32)).mean()
    )


def psnr(target: np.ndarray, prediction: np.ndarray, *, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in decibels."""
    mse = float(np.mean((np.asarray(target, np.float32) - np.asarray(prediction, np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range**2) / mse))


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coordinates = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def ssim(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Structural similarity index using the standard 11x11 Gaussian window.

    Mirrors the ``skimage``/Wang et al. formulation used during training so the
    numbers shown in the browser are comparable with the report tables.
    """
    x = np.asarray(target, np.float64)
    y = np.asarray(prediction, np.float64)
    if x.shape != y.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    if x.ndim == 2:
        x = x[:, :, None]
        y = y[:, :, None]
    if x.shape[2] != 3:
        # Luminance conversion for non-RGB inputs.
        x = np.dot(x[..., :3], [0.2125, 0.7154, 0.0721])[..., None]
        y = np.dot(y[..., :3], [0.2125, 0.7154, 0.0721])[..., None]

    kernel = _gaussian_window(window_size, sigma)
    window = np.outer(kernel, kernel)

    def _filter(image: np.ndarray) -> np.ndarray:
        radius = window_size // 2

        def smooth(channel: np.ndarray) -> np.ndarray:
            """Apply the 2-D Gaussian window as two 1-D convolutions."""
            padded_rows = np.pad(channel, ((radius, radius), (0, 0)), mode="reflect")
            rows = np.apply_along_axis(
                lambda row: np.convolve(row, kernel, mode="valid"), 0, padded_rows
            )
            padded_columns = np.pad(rows, ((0, 0), (radius, radius)), mode="reflect")
            return np.apply_along_axis(
                lambda column: np.convolve(column, kernel, mode="valid"),
                1,
                padded_columns,
            )

        channels = [smooth(image[..., c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x, mu_y = _filter(x), _filter(y)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_xx = _filter(x * x) - mu_xx
    sigma_yy = _filter(y * y) - mu_yy
    sigma_xy = _filter(x * y) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    return float(np.mean(numerator / denominator))

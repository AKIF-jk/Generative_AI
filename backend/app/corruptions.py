"""Deterministic corruptions for the interactive workspaces.

These are NumPy ports of ``src/pet_restoration/corruptions.py`` so that the
corruption a user applies through the interface is bit-for-bit the same
operation the models were trained against. Keeping one implementation avoids
the "works in the notebook but not in the app" class of bug.

Every function takes and returns ``uint8`` ``(H, W, 3)`` RGB arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import SEVERITY_PARAMETERS

CORRUPTION_TYPES = ("clean", "salt_pepper", "gaussian_blur", "occlusion")


@dataclass(frozen=True)
class CorruptionSettings:
    """A fully resolved, reproducible corruption request."""

    corruption: str
    severity: str | None
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for the API response."""
        return {
            "corruption": self.corruption,
            "severity": self.severity,
            "seed": self.seed,
            "parameters": self.parameters,
        }


class UnknownCorruptionError(ValueError):
    """Raised for a corruption name or severity the app does not support."""


# --------------------------------------------------------------------------
# Primitive corruptions (mirroring src/pet_restoration/corruptions.py)
# --------------------------------------------------------------------------


def salt_and_pepper(image: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    """Replace ``probability`` of pixels with an equal black/white split."""
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
    """Apply a channel-wise Gaussian blur with a finite separable kernel."""
    if kernel_size not in (3, 5, 7):
        raise ValueError("kernel_size must be one of 3, 5, or 7")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()

    padded_x = np.pad(
        image.astype(np.float32), ((0, 0), (radius, radius), (0, 0)), mode="reflect"
    )
    horizontal = np.zeros_like(image, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        horizontal += weight * padded_x[:, offset : offset + image.shape[1], :]

    padded_y = np.pad(horizontal, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    blurred = np.zeros_like(horizontal, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        blurred += weight * padded_y[offset : offset + image.shape[0], :, :]
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def occlude(image: np.ndarray, rectangles: list[dict[str, int]]) -> np.ndarray:
    """Overlay black half-open ``x, y, width, height`` rectangles."""
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


# --------------------------------------------------------------------------
# Occlusion geometry
# --------------------------------------------------------------------------


def sample_occlusion_rectangles(
    height: int,
    width: int,
    *,
    coverage: float,
    rectangles: int,
    rng: np.random.Generator,
) -> list[dict[str, int]]:
    """Draw ``rectangles`` disjoint-ish masks jointly covering ``coverage``.

    The requested coverage is split evenly between the rectangles, then each
    rectangle is nudged to keep at least a one-pixel margin from the border and
    from previously placed rectangles. The achieved coverage is reported back
    so the interface can show the real number rather than the requested one.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    if rectangles < 1:
        raise ValueError("at least one rectangle is required")

    total_pixels = height * width
    per_rectangle = coverage / rectangles
    placed: list[dict[str, int]] = []
    occupied = np.zeros((height, width), dtype=bool)

    for index in range(rectangles):
        for _ in range(64):
            aspect = float(rng.uniform(0.4, 2.5))
            rect_height = int(np.clip(round(np.sqrt(per_rectangle * total_pixels / aspect)), 4, height - 8))
            rect_width = int(np.clip(round(per_rectangle * total_pixels * aspect / max(rect_height, 1)), 4, width - 8))
            x = int(rng.integers(0, max(width - rect_width, 1)))
            y = int(rng.integers(0, max(height - rect_height, 1)))
            window = occupied[y : y + rect_height, x : x + rect_width]
            if window.any():
                continue
            if index == rectangles - 1 and placed:
                # Nudge the final rectangle away from the ones already drawn so
                # the requested joint coverage is actually reached.
                y = int(rng.integers(0, max(height - rect_height, 1)))
                x = int(rng.integers(0, max(width - rect_width, 1)))
            placed.append({"x": x, "y": y, "width": rect_width, "height": rect_height})
            occupied[y : y + rect_height, x : x + rect_width] = True
            break
        else:
            # Fall back to a free horizontal band rather than failing the request.
            band_height = max(2, height // (rectangles * 2))
            y = min(index * band_height * 2, max(height - band_height, 0))
            placed.append({"x": 0, "y": int(y), "width": int(width), "height": int(band_height)})

    return placed


def achieved_coverage(rectangles: list[dict[str, int]], height: int, width: int) -> float:
    total = sum(int(r["width"]) * int(r["height"]) for r in rectangles)
    return float(total) / float(height * width)


# --------------------------------------------------------------------------
# Resolution of user-facing severity choices
# --------------------------------------------------------------------------


def resolve_settings(
    corruption: str,
    severity: str | None = None,
    *,
    seed: int | None = None,
    shape: tuple[int, int] = (128, 128),
) -> CorruptionSettings:
    """Turn a UI selection into concrete, reproducible corruption parameters."""
    if corruption not in CORRUPTION_TYPES:
        raise UnknownCorruptionError(
            f"unknown corruption {corruption!r}; expected one of {', '.join(CORRUPTION_TYPES)}"
        )
    if corruption == "clean":
        return CorruptionSettings(
            corruption="clean", severity=None, seed=0 if seed is None else int(seed), parameters={}
        )

    if severity is None:
        severity = "medium"
    if severity not in SEVERITY_PARAMETERS[corruption]:
        raise UnknownCorruptionError(
            f"unknown severity {severity!r} for {corruption!r}; "
            f"expected one of {', '.join(SEVERITY_PARAMETERS[corruption])}"
        )

    parameters = dict(SEVERITY_PARAMETERS[corruption][severity])
    resolved_seed = 12345 if seed is None else int(seed)

    if corruption == "occlusion":
        height, width = shape
        rectangles = sample_occlusion_rectangles(
            height,
            width,
            coverage=float(parameters["coverage"]),
            rectangles=int(parameters["rectangles"]),
            rng=np.random.default_rng(resolved_seed),
        )
        parameters["rectangles"] = rectangles
        parameters["requested_coverage"] = float(parameters["coverage"])
        parameters["coverage"] = round(achieved_coverage(rectangles, height, width), 4)
        parameters["count"] = len(rectangles)

    return CorruptionSettings(
        corruption=corruption, severity=severity, seed=resolved_seed, parameters=parameters
    )


def apply_settings(image: np.ndarray, settings: CorruptionSettings) -> np.ndarray:
    """Apply a resolved :class:`CorruptionSettings` to a uint8 RGB array."""
    if settings.corruption == "clean":
        return image.copy()
    if settings.corruption == "salt_pepper":
        return salt_and_pepper(
            image,
            float(settings.parameters["probability"]),
            np.random.default_rng(settings.seed),
        )
    if settings.corruption == "gaussian_blur":
        return gaussian_blur(
            image,
            int(settings.parameters["kernel_size"]),
            float(settings.parameters["sigma"]),
        )
    if settings.corruption == "occlusion":
        return occlude(image, settings.parameters["rectangles"])
    raise UnknownCorruptionError(f"unknown corruption {settings.corruption!r}")


def catalog() -> list[dict[str, Any]]:
    """Describe every selectable corruption/severity pair for the frontend."""
    entries: list[dict[str, Any]] = [
        {
            "corruption": "clean",
            "label": "Clean",
            "description": "No artificial corruption is applied.",
            "severities": [],
        }
    ]
    labels = {
        "salt_pepper": "Salt-and-pepper noise",
        "gaussian_blur": "Gaussian blur",
        "occlusion": "Rectangular occlusion",
    }
    for corruption in ("salt_pepper", "gaussian_blur", "occlusion"):
        severities = []
        for severity, parameters in SEVERITY_PARAMETERS[corruption].items():
            summary = {k: v for k, v in parameters.items()}
            severities.append({"severity": severity, "parameters": summary})
        entries.append(
            {
                "corruption": corruption,
                "label": labels[corruption],
                "description": "",
                "severities": severities,
            }
        )
    return entries

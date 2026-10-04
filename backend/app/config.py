"""Runtime configuration for the Generative AI restoration application.

Every path can be overridden with an environment variable so the same image can
be pointed at the files in ``results/`` (development), at ``/models`` (Docker
Compose) or at a download location chosen by the evaluator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------

#: Root that holds one sub-directory per task. Inside Docker Compose this is
#: ``/models`` with ``task1``/``task2``/``task4`` mounted read-only from the
#: repository's ``results`` folder.
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))

#: Checkpoints/manifests are only needed for the offline experiments view.
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "/results"))

#: Bundled sample images and pre-computed experiment summaries.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

IMAGE_SIZE = 128
"""All models in this assignment operate on 128x128 RGB tensors."""

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
"""Hard ceiling on a single uploaded file (20 MB by default)."""

ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "0")) or None
"""Intra-op thread count; ``None`` lets onnxruntime decide."""


# --------------------------------------------------------------------------
# Task identifiers and corruption vocabulary
# --------------------------------------------------------------------------

TASK_IDS = ("task1", "task2", "task4")

WORKSPACES = {
    "task1": {
        "title": "Universal Restoration",
        "task": "Task 1",
        "summary": "Single multi-corruption denoising autoencoder with a vector bottleneck.",
    },
    "task2": {
        "title": "Hard-Routed Restoration",
        "task": "Task 2",
        "summary": "Corruption classifier routes to one of three specialist autoencoders.",
    },
    "task4": {
        "title": "Face-to-Sketch Generator",
        "task": "Task 4",
        "summary": "Style-conditioned U-Net generator trained with a PatchGAN discriminator.",
    },
}

#: Label order shared by the Task 1/2 models. Index 0 is the identity/clean path.
CORRUPTION_LABELS: dict[str, int] = {
    "clean": 0,
    "salt_pepper": 1,
    "gaussian_blur": 2,
    "occlusion": 3,
}

LABEL_TO_CORRUPTION: dict[int, str] = {v: k for k, v in CORRUPTION_LABELS.items()}

SPECIALIST_EXPERTS: dict[int, str] = {
    CORRUPTION_LABELS["salt_pepper"]: "salt_pepper",
    CORRUPTION_LABELS["gaussian_blur"]: "gaussian_blur",
    CORRUPTION_LABELS["occlusion"]: "occlusion",
}

SEVERITIES = ("low", "medium", "high")

#: Fixed evaluation severities mandated by the assignment.
SEVERITY_PARAMETERS: dict[str, dict[str, dict]] = {
    "salt_pepper": {
        "low": {"probability": 0.03},
        "medium": {"probability": 0.08},
        "high": {"probability": 0.15},
    },
    "gaussian_blur": {
        "low": {"kernel_size": 3, "sigma": 0.7},
        "medium": {"kernel_size": 5, "sigma": 1.5},
        "high": {"kernel_size": 7, "sigma": 2.5},
    },
    "occlusion": {
        "low": {"coverage": 0.10, "rectangles": 1},
        "medium": {"coverage": 0.20, "rectangles": 2},
        "high": {"coverage": 0.35, "rectangles": 3},
    },
}

STYLE_NAMES = {1: "Style 1", 2: "Style 2", 3: "Style 3"}


@dataclass(frozen=True)
class ModelSpec:
    """A single ONNX artefact required for inference."""

    key: str
    task: str
    path: Path
    role: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    description: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="universal",
        task="task1",
        path=MODEL_ROOT / "task1" / "model.onnx",
        role="universal_autoencoder",
        inputs=("input",),
        outputs=("output",),
        description="Task 1 universal multi-corruption restoration autoencoder",
    ),
    ModelSpec(
        key="classifier",
        task="task2",
        path=MODEL_ROOT / "task2" / "classifier.onnx",
        role="corruption_classifier",
        inputs=("input",),
        outputs=("output",),
        description="Task 2 four-class corruption classifier",
    ),
    ModelSpec(
        key="specialist_salt_pepper",
        task="task2",
        path=MODEL_ROOT / "task2" / "specialist_salt_pepper.onnx",
        role="specialist",
        inputs=("input",),
        outputs=("output",),
        description="Task 2 salt-and-pepper specialist",
    ),
    ModelSpec(
        key="specialist_gaussian_blur",
        task="task2",
        path=MODEL_ROOT / "task2" / "specialist_gaussian_blur.onnx",
        role="specialist",
        inputs=("input",),
        outputs=("output",),
        description="Task 2 Gaussian blur specialist",
    ),
    ModelSpec(
        key="specialist_occlusion",
        task="task2",
        path=MODEL_ROOT / "task2" / "specialist_occlusion.onnx",
        role="specialist",
        inputs=("input",),
        outputs=("output",),
        description="Task 2 rectangular occlusion specialist",
    ),
    ModelSpec(
        key="face_to_sketch",
        task="task4",
        path=MODEL_ROOT / "task4" / "generator.onnx",
        role="generator",
        inputs=("photo", "style_id"),
        outputs=("sketch",),
        description="Task 4 style-conditioned U-Net sketch generator",
    ),
)

MODEL_SPECS_BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in MODEL_SPECS}

MODELS_BY_TASK: dict[str, tuple[ModelSpec, ...]] = {
    task: tuple(spec for spec in MODEL_SPECS if spec.task == task) for task in TASK_IDS
}


def relative_model_path(path: Path) -> str:
    """Render a model path relative to :data:`MODEL_ROOT` for API responses."""
    try:
        return str(path.relative_to(MODEL_ROOT))
    except ValueError:
        return str(path)

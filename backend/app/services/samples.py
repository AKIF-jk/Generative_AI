"""Access to the bundled clean sample images.

The Oxford-IIIT Pet development split is far too large to ship, so a small,
representative strip of clean 128x128 images is bundled with the backend. It
lets an evaluator reproduce the corruption pipeline without uploading anything,
which is what the demonstration video needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from ..config import ASSETS_DIR

SAMPLES_DIR = ASSETS_DIR / "samples" / "pet"

#: Maps a stable sample id to its bundled file.
_SAMPLE_FILES: dict[str, str] = {}


@dataclass(frozen=True)
class Sample:
    """A bundled clean image offered by the interface."""

    id: str
    filename: str
    label: str

    def describe(self) -> dict[str, str]:
        return {"id": self.id, "filename": self.filename, "label": self.label}


def _discover() -> dict[str, Sample]:
    if not SAMPLES_DIR.is_dir():
        return {}
    samples: dict[str, Sample] = {}
    for path in sorted(SAMPLES_DIR.glob("*.jpg")):
        sample_id = path.stem
        label = sample_id.rsplit("_", 1)[0].replace("_", " ").title()
        samples[sample_id] = Sample(id=sample_id, filename=path.name, label=label)
    return samples


SAMPLES: dict[str, Sample] = _discover()


@lru_cache(maxsize=32)
def load_sample(sample_id: str) -> Image.Image:
    """Return the clean RGB image for ``sample_id``."""
    sample = SAMPLES.get(sample_id)
    if sample is None:
        raise KeyError(sample_id)
    return Image.open(SAMPLES_DIR / sample.filename).convert("RGB")


def list_samples() -> list[dict[str, str]]:
    return [sample.describe() for sample in SAMPLES.values()]

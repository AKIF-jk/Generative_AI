#!/usr/bin/env python3
"""Save a visual sanity-check grid for the four corruption conditions."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pet_restoration.corruptions import gaussian_blur, occlude, salt_and_pepper
from pet_restoration.manifests import official_image_paths, sample_occlusion_rectangles


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/corruption_sanity_check.png"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    image_paths = [arguments.dataset_root / path for path in official_image_paths(arguments.dataset_root, "trainval.txt")[: arguments.samples]]
    if not image_paths:
        raise FileNotFoundError("No images found; run scripts/prepare_pet_data.py --download first")
    figure, axes = plt.subplots(len(image_paths), 5, figsize=(15, 3 * len(image_paths)))
    axes = np.atleast_2d(axes)
    rng = np.random.default_rng(arguments.seed)
    for row, image_path in enumerate(image_paths):
        clean = _load_rgb(image_path)
        rectangles, fraction = sample_occlusion_rectangles(random.Random(arguments.seed + row), 128, 2, 0.20)
        examples = [
            ("clean", clean),
            ("salt-pepper p=0.08", salt_and_pepper(clean, 0.08, rng)),
            ("blur k=5, sigma=1.5", gaussian_blur(clean, 5, 1.5)),
            (f"occlusion {fraction:.1%}", occlude(clean, rectangles)),
            ("target", clean),
        ]
        for column, (title, image) in enumerate(examples):
            axes[row, column].imshow(image)
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=150)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()

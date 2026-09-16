"""PyTorch dataset for dynamic training and manifest-driven evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from .corruptions import apply_manifest_corruption, gaussian_blur, occlude, salt_and_pepper
from .manifests import sample_occlusion_rectangles


CORRUPTION_LABELS = {"clean": 0, "salt_pepper": 1, "gaussian_blur": 2, "occlusion": 3}


class PetRestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Return ``(corrupted, clean, corruption_label)`` images resized to 128².

    Training dynamically selects clean, salt-and-pepper, blur, and occlusion
    with probability 0.25 each. Validation and test require a manifest and are
    consequently repeatable across all Tasks 1--3.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        mode: Literal["train", "validation", "test"],
        *,
        split_csv: str | Path | None = None,
        manifest_path: str | Path | None = None,
        training_seed: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.mode = mode
        self.training_seed = training_seed
        self._rng: np.random.Generator | None = None
        self._rng_worker_id: int | None = None
        if mode == "train":
            if split_csv is None:
                raise ValueError("split_csv is required in training mode")
            self.records = self._read_training_records(Path(split_csv))
        elif mode in ("validation", "test"):
            if manifest_path is None:
                raise ValueError("manifest_path is required in validation/test mode")
            self.records = self._read_manifest(Path(manifest_path))
        else:
            raise ValueError("mode must be 'train', 'validation', or 'test'")
        if not self.records:
            raise ValueError("Dataset has no records")

    @staticmethod
    def _read_training_records(split_csv: Path) -> list[dict[str, Any]]:
        with split_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or not {"image_path", "split"}.issubset(rows[0]):
            raise ValueError("split CSV must contain image_path and split columns")
        return [{"image_path": row["image_path"]} for row in rows if row["split"] == "train"]

    @staticmethod
    def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        records = manifest.get("records") if isinstance(manifest, dict) else manifest
        if not isinstance(records, list):
            raise ValueError("manifest must be a list or an object with a records list")
        required = {"image_path", "corruption", "severity", "seed", "parameters"}
        for record in records:
            missing = required.difference(record)
            if missing:
                raise ValueError(f"Manifest record is missing fields: {sorted(missing)}")
            if record["corruption"] not in CORRUPTION_LABELS:
                raise ValueError(f"Unknown manifest corruption: {record['corruption']!r}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _get_training_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if self._rng is None or self._rng_worker_id != worker_id:
            base_seed = self.training_seed if self.training_seed is not None else torch.initial_seed()
            self._rng = np.random.default_rng(int(base_seed) + worker_id + 1)
            self._rng_worker_id = worker_id
        return self._rng

    @staticmethod
    def _load_rgb(image_path: Path) -> np.ndarray:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed by dataset is missing: {image_path}")
        with Image.open(image_path) as image:
            resized = image.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
            return np.asarray(resized, dtype=np.uint8).copy()

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)

    def _sample_training_corruption(self, clean: np.ndarray) -> tuple[np.ndarray, int]:
        rng = self._get_training_rng()
        label = int(rng.integers(0, 4))
        if label == CORRUPTION_LABELS["clean"]:
            return clean.copy(), label
        if label == CORRUPTION_LABELS["salt_pepper"]:
            return salt_and_pepper(clean, float(rng.uniform(0.02, 0.15)), rng), label
        if label == CORRUPTION_LABELS["gaussian_blur"]:
            return gaussian_blur(clean, int(rng.choice((3, 5, 7))), float(rng.uniform(0.5, 2.5))), label
        import random

        rectangles, _ = sample_occlusion_rectangles(
            random.Random(int(rng.integers(0, 2**32))),
            128,
            int(rng.integers(1, 4)),
            # Keep an 0.8%-tolerance buffer so the realised integer rectangle
            # area always remains inside the required 10--35% training range.
            float(rng.uniform(0.108, 0.342)),
        )
        return occlude(clean, rectangles), label

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        clean = self._load_rgb(self.dataset_root / record["image_path"])
        if self.mode == "train":
            corrupted, label = self._sample_training_corruption(clean)
        else:
            corrupted = apply_manifest_corruption(clean, record)
            label = CORRUPTION_LABELS[record["corruption"]]
        return self._to_tensor(corrupted), self._to_tensor(clean), label

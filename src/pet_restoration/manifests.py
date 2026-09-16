"""Reproducible split and deterministic-manifest construction."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable


TEST_SEVERITIES: dict[str, tuple[dict[str, Any], ...]] = {
    "salt_pepper": (
        {"severity": "low", "probability": 0.03},
        {"severity": "medium", "probability": 0.08},
        {"severity": "high", "probability": 0.15},
    ),
    "gaussian_blur": (
        {"severity": "low", "kernel_size": 3, "sigma": 0.7},
        {"severity": "medium", "kernel_size": 5, "sigma": 1.5},
        {"severity": "high", "kernel_size": 7, "sigma": 2.5},
    ),
    "occlusion": (
        {"severity": "low", "rectangles": 1, "target_area_fraction": 0.10},
        {"severity": "medium", "rectangles": 2, "target_area_fraction": 0.20},
        {"severity": "high", "rectangles": 3, "target_area_fraction": 0.35},
    ),
}


def official_image_paths(dataset_root: Path, annotation_name: str) -> list[str]:
    """Return official image paths from an Oxford-IIIT annotation list only."""
    annotation_path = dataset_root / "annotations" / annotation_name
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing Oxford-IIIT annotation file: {annotation_path}")
    paths: list[str] = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        image_id = line.split()[0]
        relative_path = f"images/{image_id}.jpg"
        if not (dataset_root / relative_path).is_file():
            raise FileNotFoundError(f"Annotation references a missing image: {relative_path}")
        paths.append(relative_path)
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate image IDs in {annotation_path}")
    return paths


def make_development_split(paths: Iterable[str], seed: int = 42) -> tuple[list[str], list[str]]:
    """Create the assignment's 80/20 development split with a local RNG."""
    shuffled = sorted(paths)
    random.Random(seed).shuffle(shuffled)
    train_count = int(0.8 * len(shuffled))
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def write_split_csv(train_paths: list[str], validation_paths: list[str], output_path: Path, seed: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "split", "split_seed"))
        writer.writeheader()
        for split, paths in (("train", train_paths), ("validation", validation_paths)):
            for image_path in paths:
                writer.writerow({"image_path": image_path, "split": split, "split_seed": seed})


def _non_overlapping(rectangles: list[dict[str, int]], candidate: dict[str, int]) -> bool:
    left, top = candidate["x"], candidate["y"]
    right, bottom = left + candidate["width"], top + candidate["height"]
    for rectangle in rectangles:
        other_left, other_top = rectangle["x"], rectangle["y"]
        other_right = other_left + rectangle["width"]
        other_bottom = other_top + rectangle["height"]
        if max(left, other_left) < min(right, other_right) and max(top, other_top) < min(bottom, other_bottom):
            return False
    return True


def sample_occlusion_rectangles(
    rng: random.Random, image_size: int, rectangle_count: int, target_area_fraction: float, tolerance: float = 0.008
) -> tuple[list[dict[str, int]], float]:
    """Sample non-overlapping rectangles whose union closely matches a target."""
    if not 1 <= rectangle_count <= 3:
        raise ValueError("rectangle_count must be between 1 and 3")
    if not 0 < target_area_fraction < 1:
        raise ValueError("target_area_fraction must be in (0, 1)")
    image_area = image_size * image_size
    for _ in range(500):
        shares = [rng.uniform(0.7, 1.3) for _ in range(rectangle_count)]
        share_total = sum(shares)
        rectangles: list[dict[str, int]] = []
        for share in shares:
            desired_area = target_area_fraction * image_area * share / share_total
            aspect_ratio = rng.uniform(0.5, 2.0)
            rect_width = max(1, min(image_size, round((desired_area * aspect_ratio) ** 0.5)))
            rect_height = max(1, min(image_size, round(desired_area / rect_width)))
            for _ in range(300):
                candidate = {"x": rng.randint(0, image_size - rect_width), "y": rng.randint(0, image_size - rect_height), "width": rect_width, "height": rect_height}
                if _non_overlapping(rectangles, candidate):
                    rectangles.append(candidate)
                    break
            else:
                break
        if len(rectangles) != rectangle_count:
            continue
        actual_fraction = sum(item["width"] * item["height"] for item in rectangles) / image_area
        if abs(actual_fraction - target_area_fraction) <= tolerance:
            return rectangles, actual_fraction
    raise RuntimeError("Could not sample non-overlapping occlusion masks near requested area")


def _record(image_path: str, corruption: str, severity: str, seed: int, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"image_path": image_path, "corruption": corruption, "severity": severity, "seed": seed, "parameters": parameters}


def build_fixed_manifest(image_paths: Iterable[str], seed: int = 42, image_size: int = 128) -> dict[str, Any]:
    """Build clean plus all three fixed severities for every corruption/image."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for image_path in sorted(image_paths):
        records.append(_record(image_path, "clean", "clean", rng.randrange(2**32), {}))
        for corruption, settings in TEST_SEVERITIES.items():
            for setting in settings:
                record_seed = rng.randrange(2**32)
                parameters = dict(setting)
                severity = str(parameters.pop("severity"))
                if corruption == "occlusion":
                    rectangle_count = int(parameters.pop("rectangles"))
                    target_fraction = float(parameters["target_area_fraction"])
                    rectangles, actual_fraction = sample_occlusion_rectangles(random.Random(record_seed), image_size, rectangle_count, target_fraction)
                    parameters.update({"rectangle_count": rectangle_count, "rectangles": rectangles, "actual_area_fraction": actual_fraction})
                records.append(_record(image_path, corruption, severity, record_seed, parameters))
    return {"schema_version": 1, "seed": seed, "image_size": image_size, "records": records}


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, separators=(",", ":"))

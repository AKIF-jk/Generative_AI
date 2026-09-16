#!/usr/bin/env python3
"""Download (optional) and prepare shared Oxford-IIIT Pet data artefacts."""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

#sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pet_restoration.manifests import build_fixed_manifest, make_development_split, official_image_paths, write_manifest, write_split_csv


DATASET_URLS = {
    "images.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    archive.extractall(destination, filter="data")


def download_dataset(dataset_root: Path) -> None:
    if (dataset_root / "images").is_dir() and (dataset_root / "annotations").is_dir():
        return
    dataset_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="oxford_pets_") as temporary:
        for filename, url in DATASET_URLS.items():
            archive_path = Path(temporary) / filename
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive_path)
            with tarfile.open(archive_path, "r:gz") as archive:
                _safe_extract(archive, dataset_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--download", action="store_true", help="download missing official archives")
    parser.add_argument("--drive", action="store_true", help="use dataset from Google Drive at /content/drive/MyDrive/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="replace existing generated artefacts")
    arguments = parser.parse_args()
    if arguments.drive:
        dataset_root = Path("/content/drive/MyDrive/data")
    elif arguments.dataset_root:
        dataset_root = arguments.dataset_root
    else:
        dataset_root = Path("data")
    if arguments.download:
        download_dataset(dataset_root)
    development_paths = official_image_paths(dataset_root, "trainval.txt")
    test_paths = official_image_paths(dataset_root, "test.txt")
    train_paths, validation_paths = make_development_split(development_paths, arguments.seed)
    manifests = dataset_root / "manifests"
    output_paths = [manifests / "development_split.csv", manifests / "validation_manifest.json", manifests / "test_manifest.json"]
    existing = [path for path in output_paths if path.exists()]
    if existing and not arguments.force:
        raise FileExistsError(f"Generated artefacts already exist ({', '.join(map(str, existing))}). Use --force only to regenerate them.")
    if arguments.force:
        for path in existing:
            path.unlink()
    write_split_csv(train_paths, validation_paths, output_paths[0], arguments.seed)
    write_manifest(build_fixed_manifest(validation_paths, arguments.seed), output_paths[1])
    write_manifest(build_fixed_manifest(test_paths, arguments.seed), output_paths[2])
    print(f"Development split: {len(train_paths)} train / {len(validation_paths)} validation")
    print(f"Validation manifest: {len(validation_paths) * 10} records")
    print(f"Test manifest: {len(test_paths) * 10} records")


if __name__ == "__main__":
    main()

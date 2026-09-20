from __future__ import annotations

import random
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

from pet_restoration.corruptions import apply_manifest_corruption, gaussian_blur, occlude, salt_and_pepper
from pet_restoration.manifests import build_fixed_manifest, sample_occlusion_rectangles


class CorruptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((128, 128, 3), 127, dtype=np.uint8)

    def test_salt_and_pepper_uses_equal_black_white_split(self) -> None:
        result = salt_and_pepper(self.image, 0.10, np.random.default_rng(4))
        black = np.all(result == 0, axis=2).sum()
        white = np.all(result == 255, axis=2).sum()
        self.assertEqual(black + white, round(0.10 * 128 * 128))
        self.assertLessEqual(abs(int(black) - int(white)), 1)

    def test_blur_preserves_shape_and_changes_a_sharp_image(self) -> None:
        image = self.image.copy()
        image[:, 64:] = 255
        result = gaussian_blur(image, 5, 1.5)
        self.assertEqual(result.shape, image.shape)
        self.assertFalse(np.array_equal(result, image))

    def test_occlusion_sampler_hits_requested_training_range(self) -> None:
        rectangles, fraction = sample_occlusion_rectangles(random.Random(8), 128, 3, 0.25)
        result = occlude(self.image, rectangles)
        self.assertEqual(len(rectangles), 3)
        self.assertGreaterEqual(fraction, 0.10)
        self.assertLessEqual(fraction, 0.35)
        self.assertEqual(np.all(result == 0, axis=2).sum(), round(fraction * 128 * 128))

    def test_manifest_replay_is_bitwise_deterministic(self) -> None:
        record = {"corruption": "salt_pepper", "seed": 99, "parameters": {"probability": 0.08}}
        first = apply_manifest_corruption(self.image, record)
        second = apply_manifest_corruption(self.image, record)
        self.assertTrue(np.array_equal(first, second))

    def test_fixed_manifest_has_all_test_conditions(self) -> None:
        manifest = build_fixed_manifest(["images/example.jpg"], seed=42)
        self.assertEqual(len(manifest["records"]), 10)
        by_type = [record["corruption"] for record in manifest["records"]]
        self.assertEqual(by_type.count("clean"), 1)
        self.assertEqual(by_type.count("salt_pepper"), 3)
        self.assertEqual(by_type.count("gaussian_blur"), 3)
        self.assertEqual(by_type.count("occlusion"), 3)

    def test_dataset_replays_a_manifest_record_when_torch_is_available(self) -> None:
        try:
            import torch
            from pet_restoration import PetRestorationDataset
        except ImportError:
            self.skipTest("PyTorch is not installed in this lightweight test environment")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            Image.fromarray(self.image).save(root / "images" / "sample.jpg")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"records": [{
                "image_path": "images/sample.jpg", "corruption": "gaussian_blur",
                "severity": "medium", "seed": 1,
                "parameters": {"kernel_size": 5, "sigma": 1.5},
            }]}), encoding="utf-8")
            dataset = PetRestorationDataset(root, "validation", manifest_path=manifest_path)
            first = dataset[0]
            second = dataset[0]
            self.assertEqual(tuple(first[0].shape), (3, 128, 128))
            self.assertEqual(first[2], 2)
            self.assertTrue(torch.equal(first[0], second[0]))


if __name__ == "__main__":
    unittest.main()

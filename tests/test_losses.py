"""Unit tests for the loss functions.

The SSIM correctness test compares our differentiable implementation against
scikit-image's ``structural_similarity`` on fixed uint8 image pairs.  The
asserted tolerance (< 1e-3) is the evidence logged in the report under
"loss-correctness verification".
"""

from __future__ import annotations

import importlib.util
import unittest


def _has_skimage() -> bool:
    return importlib.util.find_spec("skimage") is not None


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_has_torch(), "PyTorch is not installed")
class DifferentiableSSIMTests(unittest.TestCase):
    """Verify differentiable_ssim matches scikit-image's reference SSIM."""

    @staticmethod
    def _reference_ssim(img1, img2) -> float:
        """Compute SSIM via scikit-image (ground truth)."""
        from skimage.metrics import structural_similarity

        return structural_similarity(
            img1,
            img2,
            multichannel=True,
            channel_axis=-1,
            data_range=255,
        )

    @unittest.skipUnless(_has_skimage(), "scikit-image is not installed")
    def test_ssim_matches_skimage_on_fixed_uint8_pair(self) -> None:
        """Core correctness assertion for the report."""
        import numpy as np
        import torch

        rng = np.random.RandomState(42)
        a = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        b = a.copy()
        # Corrupt ~5 % of pixels to produce a non-trivial SSIM.
        mask = rng.rand(128, 128) < 0.05
        b[mask] = rng.randint(0, 256, (mask.sum(), 3), dtype=np.uint8)

        ref = self._reference_ssim(a, b)

        from pet_restoration.losses import differentiable_ssim

        pred = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tgt = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ours = differentiable_ssim(pred, tgt).item()

        self.assertAlmostEqual(ours, ref, delta=1e-3)

    @unittest.skipUnless(_has_skimage(), "scikit-image is not installed")
    def test_ssim_identical_images_gives_one(self) -> None:
        import numpy as np
        import torch

        a = np.full((128, 128, 3), 128, dtype=np.uint8)
        ref = self._reference_ssim(a, a)

        from pet_restoration.losses import differentiable_ssim

        t = torch.full((1, 3, 128, 128), 128.0 / 255.0)
        ours = differentiable_ssim(t, t).item()

        self.assertAlmostEqual(ours, ref, delta=1e-6)
        self.assertAlmostEqual(ours, 1.0, delta=1e-6)

    @unittest.skipUnless(_has_skimage(), "scikit-image is not installed")
    def test_ssim_random_pair_correlation(self) -> None:
        """SSIM of two random images should be near 0 and close to skimage."""
        import numpy as np
        import torch

        rng = np.random.RandomState(7)
        a = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        b = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)

        ref = self._reference_ssim(a, b)

        from pet_restoration.losses import differentiable_ssim

        pred = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tgt = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ours = differentiable_ssim(pred, tgt).item()

        self.assertAlmostEqual(ours, ref, delta=1e-3)


@unittest.skipUnless(_has_torch(), "PyTorch is not installed")
class CombinedLossTests(unittest.TestCase):
    """Smoke tests for combined_loss, l1_loss, ssim_loss."""

    def test_combined_loss_components(self) -> None:
        import torch

        from pet_restoration.losses import combined_loss, l1_loss, ssim_loss

        pred = torch.rand(2, 3, 128, 128)
        tgt = torch.rand(2, 3, 128, 128)

        alpha = 0.8
        combined = combined_loss(pred, tgt, alpha=alpha)
        l1 = l1_loss(pred, tgt)
        ssim_l = ssim_loss(pred, tgt)

        expected = alpha * l1 + (1.0 - alpha) * ssim_l
        self.assertAlmostEqual(combined.item(), expected.item(), places=7)

    def test_alpha_boundary_values(self) -> None:
        import torch

        from pet_restoration.losses import combined_loss, l1_loss, ssim_loss

        pred = torch.rand(2, 3, 128, 128)
        tgt = torch.rand(2, 3, 128, 128)

        self.assertAlmostEqual(
            combined_loss(pred, tgt, alpha=0.0).item(),
            ssim_loss(pred, tgt).item(),
            places=7,
        )
        self.assertAlmostEqual(
            combined_loss(pred, tgt, alpha=1.0).item(),
            l1_loss(pred, tgt).item(),
            places=7,
        )

    def test_identical_images_gives_zero_loss(self) -> None:
        import torch

        from pet_restoration.losses import combined_loss

        x = torch.rand(2, 3, 128, 128)
        self.assertAlmostEqual(combined_loss(x, x, alpha=0.8).item(), 0.0, places=6)


@unittest.skipUnless(_has_torch(), "PyTorch is not installed")
class AlphaSweepTests(unittest.TestCase):
    """Tests for sweep_alpha_per_corruption."""

    def test_sweep_returns_best_alpha_per_corruption(self) -> None:
        import torch

        from pet_restoration.losses import sweep_alpha_per_corruption

        B = 20
        pred = torch.rand(B, 3, 128, 128)
        tgt = torch.rand(B, 3, 128, 128)
        # 0=clean, 1=salt_pepper, 2=gaussian_blur, 3=occlusion
        labels = torch.arange(B) % 4

        results = sweep_alpha_per_corruption(pred, tgt, labels, alphas=[0.0, 0.5, 0.7, 1.0])
        self.assertIn("clean", results)
        self.assertIn("salt_pepper", results)
        for name, (best_a, best_loss) in results.items():
            self.assertIn(best_a, [0.0, 0.5, 0.7, 1.0])
            self.assertGreaterEqual(best_loss, 0.0)

    def test_sweep_includes_boundaries_by_default(self) -> None:
        import torch

        from pet_restoration.losses import sweep_alpha_per_corruption

        B = 8
        pred = torch.rand(B, 3, 128, 128)
        tgt = torch.rand(B, 3, 128, 128)
        labels = torch.arange(B) % 4

        results = sweep_alpha_per_corruption(pred, tgt, labels)
        # Default sweep should include both boundaries.
        seen_alphas = {a for a, _ in results.values()}
        self.assertIn(0.0, seen_alphas)
        self.assertIn(1.0, seen_alphas)


if __name__ == "__main__":
    unittest.main()

"""Minimal tests for src/task1/train.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import torch

# Ensure src/ is on the path for imports
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from task1.train import TrainConfig, ranking_score, train, validate
from pet_restoration.dataset import CORRUPTION_LABELS
from pet_restoration.models import UniversalRestorationAutoencoder


# ---------------------------------------------------------------------------
# Helpers: synthetic data for tests that do not need real images
# ---------------------------------------------------------------------------

class _SyntheticDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Deterministic synthetic dataset for unit tests."""

    def __init__(
        self,
        n: int,
        corruption_labels: list[int] | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.n = n
        rng = torch.Generator().manual_seed(seed)
        self.cleans = torch.rand(n, 3, 128, 128, generator=rng)
        self.corrupteds = torch.rand(n, 3, 128, 128, generator=rng)
        self.labels = corruption_labels if corruption_labels is not None else [0] * n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.corrupteds[idx], self.cleans[idx], self.labels[idx]


def _make_cfg(tmp: Path, **overrides: object) -> TrainConfig:
    """Build a minimal TrainConfig pointing at *tmp*."""
    kw = dict(
        dataset_root=tmp,
        split_csv=tmp / "dummy.csv",
        val_manifest=tmp / "dummy.json",
        output_dir=tmp / "out",
        epochs=2,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    kw.update(overrides)
    return TrainConfig(**kw)


# ===================================================================
# Tests
# ===================================================================


class TestRankingScoreIsAlphaFree(unittest.TestCase):
    """ranking_score must not depend on cfg.alpha."""

    def test_validate_same_ranking_under_different_alpha(self) -> None:
        ds = _SyntheticDataset(6, corruption_labels=[0, 1, 2, 3, 0, 1], seed=42)
        loader = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False)

        model = UniversalRestorationAutoencoder(base_channels=8, latent_dim=32)
        model.eval()

        with tempfile.TemporaryDirectory() as tmp:
            cfg_a = _make_cfg(Path(tmp), alpha=0.2, epochs=1)
            cfg_b = _make_cfg(Path(tmp), alpha=0.9, epochs=1)

            result_a = validate(model, loader, cfg_a)
            result_b = validate(model, loader, cfg_b)

        self.assertAlmostEqual(
            result_a["ranking_score"],
            result_b["ranking_score"],
            places=6,
            msg="ranking_score differs when only cfg.alpha changes — the metric "
            "is not alpha-independent as required",
        )

    def test_ranking_score_formula(self) -> None:
        """Verify the fixed 50/50 weighting: 0.5*L1 + 0.5*(1-SSIM)."""
        self.assertAlmostEqual(ranking_score(0.1, 0.9), 0.5 * 0.1 + 0.5 * 0.1, places=7)
        self.assertAlmostEqual(ranking_score(0.0, 1.0), 0.0, places=7)
        self.assertAlmostEqual(ranking_score(1.0, -1.0), 0.5 * 1.0 + 0.5 * 2.0, places=7)


class TestValidateIgnoresAbsentCorruptionTypes(unittest.TestCase):
    """Per-corruption aggregates must only include batches that contain them."""

    def test_absent_corruption_not_polluted_by_zeros(self) -> None:
        # Batch 0: clean=2, salt_pepper=1, no occlusion
        # Batch 1: occlusion=2, gaussian_blur=1, no clean
        # We verify that clean_l1 is averaged only from batches with clean,
        # and occlusion_l1 is averaged only from batches with occlusion.
        labels_batch_0 = [0, 0, 1]  # clean, clean, salt_pepper
        labels_batch_1 = [3, 3, 2]  # occlusion, occlusion, gaussian_blur
        all_labels = labels_batch_0 + labels_batch_1

        ds = _SyntheticDataset(6, corruption_labels=all_labels, seed=7)
        loader = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False)

        model = UniversalRestorationAutoencoder(base_channels=8, latent_dim=32)
        model.eval()

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(Path(tmp), alpha=0.5, epochs=1)
            result = validate(model, loader, cfg)

        # clean_l1 should come only from batch 0 (2 clean samples).
        # If batch 1's 0.0 placeholder were included, the average would differ.
        self.assertNotEqual(result["clean_l1"], 0.0, "clean_l1 is zero — possibly polluted by absent-class zeros")
        self.assertNotEqual(result["occlusion_l1"], 0.0, "occlusion_l1 is zero — possibly polluted by absent-class zeros")

        # Verify manually: recompute batch 0 clean metrics
        from pet_restoration.losses import l1_loss as _l1
        with torch.no_grad():
            b0 = ds[0:3]
            pred_b0 = model(b0[0])
            clean_mask = torch.tensor([True, True, False])
            batch0_clean_l1 = _l1(pred_b0[clean_mask], b0[1][clean_mask]).item()

            b1 = ds[3:6]
            pred_b1 = model(b1[0])
            occ_mask = torch.tensor([True, True, False])
            batch1_occ_l1 = _l1(pred_b1[occ_mask], b1[1][occ_mask]).item()

        # clean_l1 should equal batch0_clean_l1 (only one batch with clean)
        self.assertAlmostEqual(result["clean_l1"], batch0_clean_l1, places=5)
        # occlusion_l1 should equal batch1_occ_l1 (only one batch with occlusion)
        self.assertAlmostEqual(result["occlusion_l1"], batch1_occ_l1, places=5)


class TestTrainSmoke(unittest.TestCase):
    """Integration smoke: tiny dataset, 2 epochs, CPU, all skip counts."""

    def _run_smoke(self, skip_count: int) -> dict:
        n = 4
        labels = [0, 1, 2, 3]

        train_ds = _SyntheticDataset(n, corruption_labels=labels, seed=10)
        val_ds = _SyntheticDataset(n, corruption_labels=labels, seed=20)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            with patch("task1.train.PetRestorationDataset") as MockDS:
                MockDS.side_effect = [train_ds, val_ds]
                cfg = _make_cfg(
                    tmp,
                    skip_count=skip_count,
                    epochs=2,
                    batch_size=2,
                    early_stop_patience=None,
                )
                result = train(cfg)

            # Check checkpoint was written
            ckpt_path = tmp / "out" / "best.pt"
            self.assertTrue(ckpt_path.exists(), f"Checkpoint not written at {ckpt_path}")

        return result

    def test_skip_count_0(self) -> None:
        result = self._run_smoke(skip_count=0)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(torch.isfinite(torch.tensor(result["best_ranking_score"])))

    def test_skip_count_1(self) -> None:
        result = self._run_smoke(skip_count=1)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(torch.isfinite(torch.tensor(result["best_ranking_score"])))

    def test_skip_count_2(self) -> None:
        result = self._run_smoke(skip_count=2)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(torch.isfinite(torch.tensor(result["best_ranking_score"])))

    def test_returns_expected_keys(self) -> None:
        result = self._run_smoke(skip_count=0)
        for key in (
            "best_ranking_score",
            "best_epoch",
            "epochs_run",
            "status",
            "final_val_metrics",
            "best_checkpoint_path",
        ):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_checkpoint_contains_required_keys(self) -> None:
        labels = [0, 1, 2, 3]
        train_ds = _SyntheticDataset(4, corruption_labels=labels, seed=10)
        val_ds = _SyntheticDataset(4, corruption_labels=labels, seed=20)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with patch("task1.train.PetRestorationDataset") as MockDS:
                MockDS.side_effect = [train_ds, val_ds]
                cfg = _make_cfg(tmp, skip_count=0, epochs=1, batch_size=2, early_stop_patience=None)
                train(cfg)

            ckpt = torch.load(tmp / "out" / "best.pt", weights_only=False)
            for key in ("model", "optimizer", "epoch", "cfg", "ranking_score", "val_metrics"):
                self.assertIn(key, ckpt, f"Checkpoint missing key: {key}")


if __name__ == "__main__":
    unittest.main()

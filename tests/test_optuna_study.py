"""Unit tests for src/task1/optuna_study.py.

Every test mocks ``task1.optuna_study.train`` (no dataset, no GPU, no real
training) and mocks ``task1.optuna_study.wandb`` (no W&B account, no network).
The Optuna studies used here are tiny in-memory studies only.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import optuna
import torch
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState

# Ensure src/ is on the path for imports
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from task1 import optuna_study


REQUIRED_BEST_CONFIG_KEYS = frozenset(
    {
        "skip_count",
        "study_name",
        "n_trials_requested",
        "n_trials_completed",
        "n_trials_pruned",
        "best_value",
        "best_trial_number",
        "best_params",
        "wandb_group",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_args(root: Path, **overrides: Any) -> Namespace:
    """Build a fully-resolved CLI namespace without touching the filesystem."""
    args = optuna_study.parse_args(
        [
            "--skip-count",
            "0",
            "--dataset-root",
            "data/pet_images",
            "--split-csv",
            "data/splits/train.csv",
            "--val-manifest",
            "data/manifests/val.json",
            "--output-root",
            str(root),
            "--n-trials",
            "3",
            "--max-epochs-per-trial",
            "1",
            "--num-workers",
            "0",
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def canned_summary(score: float) -> dict[str, Any]:
    """A train() return value that never touches a real model."""
    return {
        "best_ranking_score": score,
        "best_epoch": 3,
        "epochs_run": 5,
        "status": "ok",
        "final_val_metrics": {"ranking_score": score + 0.01},
        "best_checkpoint_path": "",
    }


def make_in_memory_study(direction: str = "minimize") -> optuna.Study:
    """A fresh in-memory study — no SQLite, nothing persisted."""
    return optuna.create_study(
        direction=direction, storage=optuna.storages.InMemoryStorage()
    )


def synthetic_distributions() -> dict[str, BaseDistribution]:
    """Distributions matching optuna_study's search space, for synthetic trials."""
    return {
        "base_channels": CategoricalDistribution(optuna_study.BASE_CHANNEL_CHOICES),
        "latent_dim": CategoricalDistribution(optuna_study.LATENT_DIM_CHOICES),
        "dropout": FloatDistribution(
            optuna_study.DROPOUT_LOW, optuna_study.DROPOUT_HIGH
        ),
        "lr": FloatDistribution(optuna_study.LR_LOW, optuna_study.LR_HIGH, log=True),
        "weight_decay": FloatDistribution(
            optuna_study.WEIGHT_DECAY_LOW,
            optuna_study.WEIGHT_DECAY_HIGH,
            log=True,
        ),
        "batch_size": CategoricalDistribution(optuna_study.BATCH_SIZE_CHOICES),
        "alpha": FloatDistribution(optuna_study.ALPHA_LOW, optuna_study.ALPHA_HIGH),
    }


def synthetic_params(base_channels: int, latent_dim: int) -> dict[str, Any]:
    """A valid parameter set inside synthetic_distributions()."""
    return {
        "base_channels": base_channels,
        "latent_dim": latent_dim,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "alpha": 0.8,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class OptunaStudyTest(unittest.TestCase):
    """objective / pruning / OOM / best_config.json behaviour."""

    @patch("task1.optuna_study.wandb")
    def test_objective_returns_ranking_score(self, wandb_mock: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as root:
            args = make_args(Path(root))
            objective = optuna_study.make_objective(args)
            study = make_in_memory_study()

            with patch.object(
                optuna_study, "train", return_value=canned_summary(0.42)
            ) as train_mock:
                study.optimize(objective, n_trials=1)

            self.assertAlmostEqual(study.best_value, 0.42)

            # Callbacks must arrive as TrainConfig fields, not train() kwargs.
            cfg = train_mock.call_args.args[0]
            self.assertIsInstance(cfg, optuna_study.TrainConfig)
            self.assertEqual(cfg.skip_count, 0)
            self.assertEqual(cfg.epochs, 1)
            self.assertEqual(cfg.early_stop_patience, optuna_study.EARLY_STOP_PATIENCE)
            self.assertIsNotNone(cfg.log_epoch)
            self.assertIsNotNone(cfg.report_to)

            # W&B: one re-initialised run per trial, always finished.
            self.assertTrue(wandb_mock.init.call_args.kwargs["reinit"])
            wandb_mock.finish.assert_called_once()

            # Scalar epoch logging shape (including per-corruption metrics).
            cfg.log_epoch(
                1,
                {"train_l1": 0.1, "train_ssim": 0.9, "train_combined_loss": 0.2},
                {
                    "val_l1": 0.2,
                    "val_ssim": 0.8,
                    "ranking_score": 0.3,
                    "clean_l1": 0.1,
                    "clean_ssim_loss": 0.2,
                    "salt_pepper_l1": 0.15,
                    "salt_pepper_ssim_loss": 0.25,
                },
            )
            logged = wandb_mock.log.call_args.args[0]
            self.assertEqual(logged["val/l1"], 0.2)
            self.assertEqual(logged["val/ranking_score"], 0.3)
            self.assertEqual(logged["val/clean_l1"], 0.1)
            self.assertEqual(logged["val/salt_pepper_ssim_loss"], 0.25)
            self.assertNotIn("val/val_l1", logged)

    @patch("task1.optuna_study.wandb")
    def test_pruned_trial_is_recorded(self, wandb_mock: MagicMock) -> None:
        def fake_train(cfg: optuna_study.TrainConfig) -> dict[str, Any]:
            report_to = cfg.report_to
            assert report_to is not None
            for epoch, score in enumerate([0.1, 0.2, 0.3], start=1):
                report_to(epoch, score)
            raise optuna.TrialPruned()

        with tempfile.TemporaryDirectory() as root:
            args = make_args(Path(root))
            objective = optuna_study.make_objective(args)
            study = make_in_memory_study()

            with patch.object(optuna_study, "train", side_effect=fake_train):
                study.optimize(objective, n_trials=1)

            trial = study.trials[0]
            self.assertEqual(trial.state, TrialState.PRUNED)
            self.assertEqual(sorted(trial.intermediate_values), [1, 2, 3])
            for epoch, score in zip([1, 2, 3], [0.1, 0.2, 0.3]):
                self.assertAlmostEqual(trial.intermediate_values[epoch], score)

            wandb_mock.finish.assert_called_once()
            wandb_mock.log.assert_called_with({"pruned": True})

    @patch("task1.optuna_study.wandb")
    def test_oom_returns_infinity(self, wandb_mock: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as root:
            args = make_args(Path(root))
            objective = optuna_study.make_objective(args)
            study = make_in_memory_study()

            with patch.object(
                optuna_study,
                "train",
                side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory"),
            ):
                trial = study.ask()
                value = objective(trial)
                study.tell(trial, value)

            # The exception must not escape the objective.
            self.assertEqual(value, float("inf"))
            wandb_mock.finish.assert_called_once()

    def test_best_config_json_shape(self) -> None:
        study = make_in_memory_study()
        distributions = synthetic_distributions()
        study.add_trial(
            optuna.trial.create_trial(
                state=TrialState.COMPLETE,
                value=0.3,
                params=synthetic_params(32, 128),
                distributions=distributions,
            )
        )
        study.add_trial(
            optuna.trial.create_trial(
                state=TrialState.COMPLETE,
                value=0.2,
                params=synthetic_params(64, 256),
                distributions=distributions,
            )
        )
        study.add_trial(
            optuna.trial.create_trial(
                state=TrialState.PRUNED,
                params=synthetic_params(96, 512),
                distributions=distributions,
                intermediate_values={1: 0.5},
            )
        )

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            args = make_args(root_path, n_trials=3)
            path = optuna_study.write_best_config(study, args)

            self.assertEqual(path, root_path / "skip0" / "best_config.json")
            self.assertTrue(path.is_file())
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(REQUIRED_BEST_CONFIG_KEYS.issubset(data.keys()))
        self.assertEqual(data["skip_count"], 0)
        self.assertEqual(data["study_name"], "task1_skip0")
        self.assertEqual(data["wandb_group"], "task1-skip0")
        self.assertEqual(data["n_trials_requested"], 3)
        self.assertEqual(data["n_trials_completed"], 2)
        self.assertEqual(data["n_trials_pruned"], 1)
        self.assertEqual(data["best_trial_number"], 1)
        self.assertAlmostEqual(data["best_value"], 0.2)
        self.assertEqual(data["best_params"]["base_channels"], 64)
        self.assertEqual(data["best_params"]["latent_dim"], 256)


if __name__ == "__main__":
    unittest.main()

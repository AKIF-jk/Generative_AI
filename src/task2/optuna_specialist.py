"""Optuna hyperparameter search for Task 2 specialist autoencoders.

A shared search finds the best architecture that works across all three
corruption types.  The three specialists are then independently trained
with those parameters.

Tunes: lr, latent_dim, base_channels, batch_size, alpha, weight_decay.

Usage
-----
    python -m task2.optuna_specialist \
        --dataset-root /path/to/data \
        --split-csv /path/to/development_split.csv \
        --val-manifest /path/to/validation_manifest.json \
        --output-dir output/task2/optuna_specialist \
        --n-trials 30 \
        --wandb-project GenAI-Assignment1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import wandb

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from task2.train_specialist import SpecialistConfig, train_specialist

CORRUPTION_TYPES = ["salt_pepper", "gaussian_blur", "occlusion"]


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Average ranking score across all three specialist types (lower = better)."""
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 512])
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 48])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    alpha = trial.suggest_float("alpha", 0.5, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    skip_count = trial.suggest_categorical("skip_count", [0, 1])

    total_rank = 0.0
    for i, corruption in enumerate(CORRUPTION_TYPES):
        cfg = SpecialistConfig(
            corruption_type=corruption,
            dataset_root=args.dataset_root,
            split_csv=args.split_csv,
            val_manifest=args.val_manifest,
            output_dir=args.output_dir / f"trial_{trial.number}" / corruption,
            base_channels=base_channels,
            latent_dim=latent_dim,
            dropout=0.0,
            skip_count=skip_count,
            alpha=alpha,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=args.epochs_per_trial,
            num_workers=args.num_workers,
            seed=42,
            early_stop_patience=5,
            min_epochs_before_stopping=10,
            report_to=lambda epoch, value, _i=i: trial.report(
                value, epoch * len(CORRUPTION_TYPES) + _i
            ),
        )

        try:
            summary = train_specialist(cfg)
        except optuna.exceptions.TrialPruned:
            raise

        rs = summary["best_ranking_score"]
        total_rank += rs

        if args.wandb_project:
            wandb.log({
                f"optuna/trial_{trial.number}/{corruption}_rank": rs,
            })

    mean_rank = total_rank / len(CORRUPTION_TYPES)

    if args.wandb_project:
        wandb.log({
            "optuna/trial": trial.number,
            "optuna/mean_rank": mean_rank,
            "optuna/lr": lr,
            "optuna/latent_dim": latent_dim,
            "optuna/base_channels": base_channels,
            "optuna/batch_size": batch_size,
            "optuna/alpha": alpha,
            "optuna/weight_decay": weight_decay,
            "optuna/skip_count": skip_count,
        })

    return mean_rank


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna study for Task 2 specialist autoencoders")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("output/task2/optuna_specialist"))
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs-per-trial", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--wandb-project", default="GenAI-Assignment1")
    p.add_argument("--study-name", default="task2_specialist")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.study_name, reinit=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    best = study.best_trial
    print(f"\nBest trial: {best.number}")
    print(f"  mean_rank = {best.value:.4f}")
    print(f"  params    = {best.params}")

    results = {
        "best_trial": best.number,
        "best_mean_ranking_score": best.value,
        "best_params": best.params,
    }
    (args.output_dir / "best_params.json").write_text(json.dumps(results, indent=2))

    if args.wandb_project:
        wandb.log({"optuna/best_mean_rank": best.value,
                   **{f"optuna/best_{k}": v for k, v in best.params.items()}})
        wandb.finish()


if __name__ == "__main__":
    main()

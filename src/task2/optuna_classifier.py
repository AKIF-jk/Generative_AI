"""Optuna hyperparameter search for the Task 2 corruption classifier.

Tunes: learning rate, batch size, base_channels, dropout, weight_decay.

Usage
-----
    python -m task2.optuna_classifier \
        --dataset-root /path/to/data \
        --split-csv /path/to/development_split.csv \
        --val-manifest /path/to/validation_manifest.json \
        --output-dir output/task2/optuna_classifier \
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

from task2.train_classifier import ClassifierConfig, train_classifier


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 48, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    cfg = ClassifierConfig(
        dataset_root=args.dataset_root,
        split_csv=args.split_csv,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir / f"trial_{trial.number}",
        base_channels=base_channels,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=args.epochs_per_trial,
        num_workers=args.num_workers,
        seed=42,
        early_stop_patience=5,
        min_epochs_before_stopping=8,
        report_to=lambda epoch, value: trial.report(value, epoch),
    )

    try:
        summary = train_classifier(cfg)
    except optuna.exceptions.TrialPruned:
        raise

    val_acc = summary["best_val_acc"]

    # Log to W&B
    if args.wandb_project:
        wandb.log({
            "optuna/trial": trial.number,
            "optuna/val_acc": val_acc,
            "optuna/lr": lr,
            "optuna/batch_size": batch_size,
            "optuna/base_channels": base_channels,
            "optuna/dropout": dropout,
            "optuna/weight_decay": weight_decay,
        })

    # Optuna minimises → return negative accuracy
    return -val_acc


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna study for Task 2 classifier")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("output/task2/optuna_classifier"))
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs-per-trial", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--wandb-project", default="GenAI-Assignment1")
    p.add_argument("--study-name", default="task2_classifier")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.study_name, reinit=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    best = study.best_trial
    print(f"\nBest trial: {best.number}")
    print(f"  val_acc = {-best.value:.4f}")
    print(f"  params  = {best.params}")

    results = {
        "best_trial": best.number,
        "best_val_acc": -best.value,
        "best_params": best.params,
    }
    (args.output_dir / "best_params.json").write_text(json.dumps(results, indent=2))

    if args.wandb_project:
        wandb.log({"optuna/best_val_acc": -best.value, **{f"optuna/best_{k}": v for k, v in best.params.items()}})
        wandb.finish()


if __name__ == "__main__":
    main()

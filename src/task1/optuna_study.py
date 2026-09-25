"""Optuna study driver for Task 1: image restoration.

Study structure — three studies, one per skip count
---------------------------------------------------
``skip_count ∈ {0, 1, 2}`` is a mandatory ablation axis for this project.  It
changes the *shape* of the autoencoder (each skip adds a gated connection
between encoder and decoder), so it is deliberately **not** part of the Optuna
search space: it is a categorical experiment, not a hyperparameter.  Every run
of this module drives exactly one of the three studies, and each study keeps
its own:

* SQLite storage (``--storage``), so a Colab disconnect and re-run resumes the
  same study via ``load_if_exists=True`` (``None`` falls back to in-memory,
  useful for smoke tests);
* W&B group (``--wandb-group``);
* best-trial record written to ``<output-root>/skip<N>/best_config.json``,
  which Phase 6 (final training) reads back to rebuild the ``TrainConfig``.

Objective — the α-free ranking metric
--------------------------------------
Each trial trains with its own ``alpha`` inside the combined loss
(``alpha·L1 + (1−alpha)·SSIM``), so raw training losses are *not* comparable
across trials.  The study therefore minimises ``ranking_score`` as computed by
``train.ranking_score``:

    ranking_score = 0.5·L1 + 0.5·(1 − SSIM)

Fixed 50/50 weights, no access to ``alpha`` — every trial is ranked on one
scales regardless of how it weighted its own loss.  Lower is better, hence
``direction="minimize"``.  The objective always returns
``summary["best_ranking_score"]`` (the best epoch), never the last epoch's
score.

Reproducibility
---------------
``TPESampler(seed=args.seed, n_startup_trials=10)`` makes the TPE proposal
sequence a pure function of ``--seed``, and every trial is trained with the
same ``--seed``; re-running the same command therefore reproduces the same
study trajectory (given the same pre-existing trial history in storage).

Other notes
-----------
* Scalars only: reconstructed images are never logged here.  Image logging is
  Phase 6/7; per-epoch images across 35 trials × 3 skip counts would exhaust
  the W&B free-tier storage quota.
* ``MedianPruner`` is the primary compute saver (``trial.report`` every epoch);
  ``early_stop_patience=4`` inside ``train()`` is the secondary safety net for
  trials the pruner has not warmed up to yet.
* ``gc_after_trial=True`` plus a ``torch.cuda.empty_cache()`` in the objective's
  ``finally`` release GPU memory between trials on a Colab T4.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import optuna
import torch
import wandb
from optuna.trial import TrialState

# Ensure the sibling ``task1`` package is importable regardless of how this
# script is invoked (``python -m task1.optuna_study`` or direct execution).
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from task1.train import TrainConfig, train


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# Capped at 128: models.py builds the encoder channel list as
# [b, min(2b, 512), min(4b, 512), min(8b, 512), 512].  Anything above 128 makes
# the very first doubling hit the 512 clamp, collapsing the doubling pattern
# (e.g. 256 -> 512 -> 512 -> 512 -> 512), so larger widths buy nothing but cost.
BASE_CHANNEL_CHOICES: list[int] = [32, 48, 64, 96, 128]
LATENT_DIM_CHOICES: list[int] = [128, 192, 256, 384, 512, 768, 1024]
BATCH_SIZE_CHOICES: list[int] = [16, 32, 64]

DROPOUT_LOW: float = 0.0
DROPOUT_HIGH: float = 0.3
LR_LOW: float = 1e-4
LR_HIGH: float = 3e-3
WEIGHT_DECAY_LOW: float = 1e-6
WEIGHT_DECAY_HIGH: float = 1e-3
# Biased toward L1 (α ≥ 0.5) because pixel fidelity is the primary quality
# signal for restoration, while the upper bound of 0.95 still lets SSIM exert
# meaningful influence on the training loss.
ALPHA_LOW: float = 0.5
ALPHA_HIGH: float = 0.95

# In-trial early stop inside train(); independent of — and secondary to — the
# Optuna pruner, catching diverging trials before the pruner has warmed up.
EARLY_STOP_PATIENCE: int = 4


# ---------------------------------------------------------------------------
# Per-trial plumbing
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Draw one hyperparameter configuration from the search space."""
    return {
        "base_channels": trial.suggest_categorical(
            "base_channels", BASE_CHANNEL_CHOICES
        ),
        "latent_dim": trial.suggest_categorical("latent_dim", LATENT_DIM_CHOICES),
        "dropout": trial.suggest_float("dropout", DROPOUT_LOW, DROPOUT_HIGH),
        "lr": trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", WEIGHT_DECAY_LOW, WEIGHT_DECAY_HIGH, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES),
        "alpha": trial.suggest_float("alpha", ALPHA_LOW, ALPHA_HIGH),
    }


def build_config(
    trial: optuna.Trial, params: dict[str, Any], args: argparse.Namespace
) -> TrainConfig:
    """Translate one trial's suggestions plus fixed CLI values into a TrainConfig.

    ``skip_count``, ``epochs``, ``early_stop_patience``, ``num_workers``,
    ``seed`` and ``device`` are fixed by the CLI — never suggested.
    """
    return TrainConfig(
        dataset_root=args.dataset_root,
        split_csv=args.split_csv,
        val_manifest=args.val_manifest,
        output_dir=args.output_root / f"skip{args.skip_count}" / f"trial_{trial.number}",
        base_channels=int(params["base_channels"]),
        latent_dim=int(params["latent_dim"]),
        dropout=float(params["dropout"]),
        skip_count=args.skip_count,
        alpha=float(params["alpha"]),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        batch_size=int(params["batch_size"]),
        epochs=args.max_epochs_per_trial,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )


def make_objective(args: argparse.Namespace) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective bound to a fully-resolved CLI namespace."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        cfg = build_config(trial, params, args)

        # reinit=True is required: wandb.init runs once per trial in this
        # process, and without it every trial would overwrite the same run.
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"skip{args.skip_count}-trial{trial.number}",
            config={**params, "skip_count": args.skip_count, "seed": args.seed},
            reinit=True,
        )

        def log_epoch(
            epoch: int,
            train_metrics: dict[str, float],
            val_metrics: dict[str, float],
        ) -> None:
            """Scalar-only per-epoch log (no images — see module docstring)."""
            wandb.log(
                {
                    "epoch": epoch,
                    "train/l1": train_metrics["train_l1"],
                    "train/ssim": train_metrics["train_ssim"],
                    "train/combined_loss": train_metrics["train_combined_loss"],
                    "val/l1": val_metrics["val_l1"],
                    "val/ssim": val_metrics["val_ssim"],
                    "val/ranking_score": val_metrics["ranking_score"],
                    # Per-corruption L1 and (1-SSIM); those keys are already in
                    # val_metrics.  "val_l1" matches the "_l1" suffix but is
                    # excluded here because it is logged as "val/l1" above.
                    **{
                        f"val/{k}": v
                        for k, v in val_metrics.items()
                        if (k.endswith("_l1") or k.endswith("_ssim_loss"))
                        and k not in {"val_l1"}
                    },
                }
            )

        def report_to(epoch: int, score: float) -> None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Callbacks are TrainConfig fields; train() takes no callback kwargs.
        cfg.log_epoch = log_epoch
        cfg.report_to = report_to

        try:
            summary = train(cfg)
            wandb.log(
                {
                    "best_ranking_score": summary["best_ranking_score"],
                    "best_epoch": summary["best_epoch"],
                    "status": summary["status"],
                }
            )
            return float(summary["best_ranking_score"])
        except optuna.TrialPruned:
            wandb.log({"pruned": True})
            raise
        except torch.cuda.OutOfMemoryError:
            wandb.log({"oom": True})
            # Optuna only rejects NaN values, so it records inf as a COMPLETE
            # trial whose value is the worst possible; the study keeps running
            # instead of being killed by the OOM.
            return float("inf")
        finally:
            wandb.finish()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


# ---------------------------------------------------------------------------
# Best-configuration persistence
# ---------------------------------------------------------------------------

def build_best_config(
    study: optuna.Study, args: argparse.Namespace
) -> dict[str, Any]:
    """Assemble the best_config.json payload for a finished study."""
    counts = Counter(trial.state for trial in study.trials)
    if counts[TrialState.COMPLETE] == 0:
        raise ValueError(
            f"Study '{study.study_name}' has no COMPLETE trial; "
            "cannot record a best configuration."
        )
    best = study.best_trial
    return {
        "skip_count": args.skip_count,
        "study_name": args.study_name,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": counts[TrialState.COMPLETE],
        "n_trials_pruned": counts[TrialState.PRUNED],
        "best_value": best.value,
        "best_trial_number": best.number,
        "best_params": dict(best.params),
        "wandb_group": args.wandb_group,
    }


def write_best_config(study: optuna.Study, args: argparse.Namespace) -> Path:
    """Write ``best_config.json`` under ``<output-root>/skip<N>/`` and return its path."""
    out_dir = args.output_root / f"skip{args.skip_count}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_config.json"
    payload = build_best_config(study, args)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and fully resolve the command line (defaults that depend on
    ``--skip-count`` and the device auto-detection are filled in here)."""
    parser = argparse.ArgumentParser(
        prog="python -m task1.optuna_study",
        description=(
            "Run one Optuna study (skip_count 0, 1 or 2) for Task 1 image "
            "restoration, logging scalars to W&B and persisting the study so "
            "it survives Colab disconnects."
        ),
    )
    parser.add_argument("--skip-count", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=35)
    parser.add_argument("--max-epochs-per-trial", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="default: task1_skip<skip-count>",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:////path/to/study.db; "
        "default: in-memory (smoke tests only, not resumable)",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/task1"))
    parser.add_argument("--wandb-project", type=str, default="GenAI-Assignment1")
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="default: task1-skip<skip-count>",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="default: auto-detect (cuda if available, else cpu)",
    )

    args = parser.parse_args(argv)
    if args.study_name is None:
        args.study_name = f"task1_skip{args.skip_count}"
    if args.wandb_group is None:
        args.wandb_group = f"task1-skip{args.skip_count}"
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _ensure_storage_parent(storage: str) -> None:
    """Create the SQLite database's parent directory when the URL implies one.

    A missing ``optuna/`` directory on Drive is a common first-run failure on
    Colab; relative ``sqlite:///relative.db`` URLs are left untouched.
    """
    if not storage.startswith("sqlite:///"):
        return
    raw_path = storage[len("sqlite:///") :]
    if not raw_path:
        return
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Create (or resume) the study, optimize it, and report the best trial."""
    args = parse_args(argv)

    # Seeded TPE for reproducibility; median pruning from epoch 5 onward as the
    # primary compute saver (see module docstring).
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1,
    )

    if args.storage is not None:
        _ensure_storage_parent(args.storage)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,  # None -> in-memory
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    storage_desc = args.storage if args.storage is not None else "in-memory"
    n_existing = len(study.trials)
    if n_existing > 0:
        print(
            f"Resuming study '{study.study_name}' in {storage_desc}: "
            f"{n_existing} trial(s) already recorded, running "
            f"{args.n_trials} new trial(s)."
        )
    else:
        print(
            f"Created study '{study.study_name}' in {storage_desc}: "
            f"running {args.n_trials} trial(s)."
        )

    study.optimize(
        make_objective(args),
        n_trials=args.n_trials,
        gc_after_trial=True,
        # Record unexpected objective errors as FAILED trials and move on
        # (retrying would mask systematic bugs).  KeyboardInterrupt is not an
        # Exception, so Ctrl-C still stops the study.
        catch=(Exception,),
    )

    # --- Final summary (the only place this module prints after startup) ---
    counts = Counter(trial.state for trial in study.trials)
    print("\n=== Study summary ===")
    print(f"study_name : {study.study_name}")
    print(f"storage    : {storage_desc}")
    print(f"skip_count : {args.skip_count}")
    print(f"trials     : {len(study.trials)} total")
    for state in (
        TrialState.COMPLETE,
        TrialState.PRUNED,
        TrialState.FAIL,
        TrialState.RUNNING,
        TrialState.WAITING,
    ):
        print(f"  {state.name:<9}: {counts[state]}")

    result: dict[str, Any] = {
        "study_name": study.study_name,
        "skip_count": args.skip_count,
        "trial_counts": {state.name: counts[state] for state in TrialState},
    }

    if counts[TrialState.COMPLETE] == 0:
        print("No COMPLETE trial — best_config.json not written.")
        return result

    best = study.best_trial
    print(f"best_trial : #{best.number}")
    print(f"best_value : {best.value}  (ranking score, lower is better)")
    print("best_params:")
    print(json.dumps(best.params, indent=2, sort_keys=True))

    path = write_best_config(study, args)
    print(f"best_config: {path}")
    print(path.read_text(encoding="utf-8"))

    result.update(
        best_value=best.value,
        best_trial_number=best.number,
        best_params=dict(best.params),
        best_config_path=str(path),
    )
    return result


if __name__ == "__main__":
    main()

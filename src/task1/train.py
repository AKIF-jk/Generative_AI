"""Training and validation module for Task 1: image restoration.

This module implements a training loop and validation procedure designed to be
called both by a standalone smoke test and by an Optuna study via the
``report_to`` callback.

**Key design: α vs. ranking metric**

The training loss (``combined_loss``) uses the trial's own α to weight pixel
(L1) vs. structural (SSIM) reconstruction during gradient updates.  This means
each Optuna trial trains with a potentially different α, shaping its own
optimization landscape.

The *ranking metric* that drives checkpointing, early stopping, and final trial
comparison is deliberately α-independent:

    ranking_score = 0.5 · L1 + 0.5 · (1 − SSIM)

This fixed 50/50 weighting ensures that trials trained with different α values
are directly comparable on a single scale.  The ranking metric never calls
``combined_loss`` and never receives the trial's α.  It is the single arbiter
of checkpoint quality and early-stopping decisions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Ensure the sibling ``pet_restoration`` package is importable regardless of
# how this script is invoked.
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pet_restoration.dataset import CORRUPTION_LABELS, PetRestorationDataset
from pet_restoration.losses import (
    combined_loss,
    differentiable_ssim,
    l1_loss,
    log_alpha_diagnostics,
)
from pet_restoration.models import UniversalRestorationAutoencoder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All tuneable and fixed parameters for one training run."""

    # Paths
    dataset_root: Path = field(default=Path("data/pet_images"))
    split_csv: Path = field(default=Path("data/splits/train.csv"))
    val_manifest: Path = field(default=Path("data/manifests/val.json"))
    output_dir: Path = field(default=Path("output/task1"))

    # Model
    base_channels: int = 32
    latent_dim: int = 256
    dropout: float = 0.0
    skip_count: int = 0

    # Loss
    alpha: float = 0.8

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Loop
    batch_size: int = 32
    epochs: int = 15
    num_workers: int = 4
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Early stop (in-trial)
    early_stop_patience: int | None = 4

    # Pruner callback
    report_to: Callable[[int, float], None] | None = None

    # Per-epoch logging callback (W&B, MLflow, etc.). Optional.
    log_epoch: Callable[[int, dict[str, float], dict[str, float]], None] | None = None

    # Checkpoint
    checkpoint_metric_name: str = "ranking_score"


# ---------------------------------------------------------------------------
# α-free ranking metric
# ---------------------------------------------------------------------------

def ranking_score(mean_l1: float, mean_ssim: float) -> float:
    """Fixed-weight validation score used to rank trials and checkpoints.

    Deliberately independent of the tuned α so that trials with different α
    values are directly comparable.  ``0.5·L1 + 0.5·(1−SSIM)``.

    Lower is better.
    """
    return 0.5 * mean_l1 + 0.5 * (1.0 - mean_ssim)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Run one training epoch; return aggregate metrics.

    The training loss uses ``cfg.alpha`` via ``combined_loss``.  Reporting
    metrics (L1, SSIM) are computed separately for logging only.
    """
    model.train()
    device = cfg.device

    total_l1 = 0.0
    total_ssim = 0.0
    total_combined = 0.0
    total_samples = 0

    for corrupted, clean, _labels in loader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)

        pred = model(corrupted)
        loss = combined_loss(pred, clean, alpha=cfg.alpha)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = corrupted.shape[0]
        with torch.no_grad():
            total_l1 += l1_loss(pred, clean).item() * batch_size
            total_ssim += differentiable_ssim(pred, clean).item() * batch_size
            total_combined += loss.item() * batch_size
        total_samples += batch_size

    n = max(total_samples, 1)
    return {
        "train_l1": total_l1 / n,
        "train_ssim": total_ssim / n,
        "train_combined_loss": total_combined / n,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Evaluate on the validation set; return flat metric dict.

    The ranking metric uses the fixed 50/50 weighting, **not** ``cfg.alpha``.
    Per-corruption aggregates are computed only over batches that actually
    contain samples of that corruption type (0.0 placeholders from
    ``log_alpha_diagnostics`` are discarded).
    """
    model.eval()
    device = cfg.device

    total_l1 = 0.0
    total_ssim = 0.0
    total_samples = 0

    per_corruption_l1: dict[str, list[float]] = defaultdict(list)
    per_corruption_ssim_loss: dict[str, list[float]] = defaultdict(list)
    per_corruption_count: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for corrupted, clean, labels in loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)

            pred = model(corrupted)

            batch_l1 = l1_loss(pred, clean).item()
            batch_ssim = differentiable_ssim(pred, clean).item()
            batch_size = corrupted.shape[0]

            total_l1 += batch_l1 * batch_size
            total_ssim += batch_ssim * batch_size
            total_samples += batch_size

            # Per-corruption diagnostics (alpha only echoes into the dict;
            # the L1/SSIM values are computed independently of it).
            diag = log_alpha_diagnostics(pred, clean, labels, alpha=cfg.alpha)

            for name in CORRUPTION_LABELS:
                idx = CORRUPTION_LABELS[name]
                count = (labels == idx).sum().item()
                if count > 0:
                    # Weight by count so aggregation is a true sample-weighted
                    # mean, not an unweighted mean of per-batch means.
                    per_corruption_l1[name].append(diag[f"{name}_l1"] * count)
                    per_corruption_ssim_loss[name].append(diag[f"{name}_ssim_loss"] * count)
                    per_corruption_count[name] += count

    n = max(total_samples, 1)
    mean_l1 = total_l1 / n
    mean_ssim = total_ssim / n

    result: dict[str, float] = {
        "val_l1": mean_l1,
        "val_ssim": mean_ssim,
        "ranking_score": ranking_score(mean_l1, mean_ssim),
    }

    for name in CORRUPTION_LABELS:
        total_c = per_corruption_count[name]
        if total_c > 0:
            result[f"{name}_l1"] = sum(per_corruption_l1[name]) / total_c
            result[f"{name}_ssim_loss"] = sum(per_corruption_ssim_loss[name]) / total_c
        else:
            result[f"{name}_l1"] = 0.0
            result[f"{name}_ssim_loss"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> dict[str, Any]:
    """Full training run.  Callable many times in one process (Optuna-safe)."""

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- Datasets / loaders ---
    train_ds = PetRestorationDataset(
        cfg.dataset_root,
        "train",
        split_csv=cfg.split_csv,
        training_seed=cfg.seed,
    )
    val_ds = PetRestorationDataset(
        cfg.dataset_root,
        "validation",
        manifest_path=cfg.val_manifest,
    )
    assert len(train_ds) > 0, "Training dataset is empty"
    assert len(val_ds) > 0, "Validation dataset is empty"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    # --- Model / optimizer ---
    model = UniversalRestorationAutoencoder(
        base_channels=cfg.base_channels,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        skip_count=cfg.skip_count,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # --- Book-keeping ---
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_ranking = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    epochs_without_improve = 0
    epochs_run = 0
    status = "ok"

    for epoch in range(1, cfg.epochs + 1):
        epochs_run = epoch
        t0 = time.monotonic()

        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg)
        val_metrics = validate(model, val_loader, cfg)

        elapsed = time.monotonic() - t0
        rs = val_metrics["ranking_score"]

        # --- Gate logging ---
        gate_str = ""
        if cfg.skip_count > 0:
            gates = model.skip_gate_values()
            g16 = gates.get("16x16")
            g32 = gates.get("32x32")
            g16_mean = float(g16.mean()) if g16 is not None else None
            g32_mean = float(g32.mean()) if g32 is not None else None
            if g16_mean is not None:
                gate_str += f"  gate_16x16={g16_mean:.4f}"
            if g32_mean is not None:
                gate_str += f"  gate_32x32={g32_mean:.4f}"

        print(
            f"epoch {epoch:3d}  "
            f"train_l1={train_metrics['train_l1']:.4f}  "
            f"train_ssim={train_metrics['train_ssim']:.4f}  "
            f"train_comb={train_metrics['train_combined_loss']:.4f}  "
            f"val_l1={val_metrics['val_l1']:.4f}  "
            f"val_ssim={val_metrics['val_ssim']:.4f}  "
            f"rank={rs:.4f}  "
            f"({elapsed:.1f}s)"
            f"{gate_str}",
            flush=True,
        )

        # --- Per-epoch logging (W&B, MLflow, etc.) ---
        if cfg.log_epoch is not None:
            cfg.log_epoch(epoch, train_metrics, val_metrics)

        # --- Report to pruner ---
        if cfg.report_to is not None:
            cfg.report_to(epoch, rs)

        # --- NaN / Inf check ---
        if not (np.isfinite(val_metrics["val_l1"]) and np.isfinite(val_metrics["val_ssim"])):
            print("  -> Diverged (non-finite validation metric)")
            status = "diverged"
            break

        # --- Checkpoint ---
        if rs < best_ranking:
            best_ranking = rs
            best_epoch = epoch
            epochs_without_improve = 0
            # Exclude report_to: it may be an unpicklable lambda/closure.
            cfg_dict = {k: v for k, v in asdict(cfg).items() if k not in {"report_to", "log_epoch"}}    
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "cfg": cfg_dict,
                "ranking_score": rs,
                "val_metrics": val_metrics,
            }
            ckpt_path = cfg.output_dir / "best.pt"
            torch.save(best_state, ckpt_path)
        else:
            epochs_without_improve += 1

        # --- Early stopping ---
        if (
            cfg.early_stop_patience is not None
            and epochs_without_improve >= cfg.early_stop_patience
        ):
            print(f"  -> Early stopping (no improvement for {cfg.early_stop_patience} epochs)")
            status = "early_stopped"
            break

    # --- Restore best weights ---
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        ckpt_path = cfg.output_dir / "best.pt"
    else:
        ckpt_path = ""

    summary = {
        "best_ranking_score": best_ranking if best_state is not None else float("inf"),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "status": status,
        "final_val_metrics": best_state["val_metrics"] if best_state is not None else {},
        "best_checkpoint_path": str(ckpt_path),
    }

    # --- Cleanup ---
    del model, optimizer
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Task 1 image-restoration model")
    p.add_argument("--dataset-root", type=Path, default=Path("data/pet_images"))
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/train.csv"))
    p.add_argument("--val-manifest", type=Path, default=Path("data/manifests/val.json"))
    p.add_argument("--output-dir", type=Path, default=Path("output/task1"))
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--skip-count", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--early-stop-patience", type=int, default=4)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        split_csv=args.split_csv,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        skip_count=args.skip_count,
        alpha=args.alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
    )
    summary = train(cfg)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()

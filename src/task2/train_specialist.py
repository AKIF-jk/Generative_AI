"""Specialist autoencoder training for Task 2.

Trains ONE specialist denoising autoencoder that sees only ONE corruption
type during training.  Running three times (with different corruption
filters) produces the three specialist models:
  - salt_pepper specialist
  - gaussian_blur specialist
  - occlusion specialist

The specialist re-uses the ``UniversalRestorationAutoencoder`` backbone
from Task 1 — parameters are trained independently for each specialist.

The training dataset is filtered to only yield the target corruption type.
A ``SpecialistDataset`` wrapper applies the filter at __getitem__ time
using rejection sampling (up to MAX_RESAMPLE attempts), so the DataLoader
sees only the requested type without permanently saving filtered copies.

Optuna tunes: lr, latent_dim, base_channels, batch_size, alpha (L1/SSIM
weighting), weight_decay.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pet_restoration.dataset import CORRUPTION_LABELS, PetRestorationDataset
from pet_restoration.losses import combined_loss, differentiable_ssim, l1_loss
from pet_restoration.models import UniversalRestorationAutoencoder


# ---------------------------------------------------------------------------
# Dataset filter for specialist training
# ---------------------------------------------------------------------------

class SpecialistDataset(Dataset):
    """Wraps PetRestorationDataset to only yield one corruption type.

    In training mode the dataset applies corruptions at runtime.  We
    filter by re-sampling until the randomly chosen corruption matches
    ``target_label``.  The wrapped dataset must be in 'train' mode.
    """

    MAX_RESAMPLE = 20  # give up after this many re-draws per __getitem__

    def __init__(
        self,
        base_dataset: PetRestorationDataset,
        target_label: int,
    ) -> None:
        if base_dataset.mode != "train":
            raise ValueError("SpecialistDataset only supports training mode datasets")
        self.base = base_dataset
        self.target_label = target_label
        # Expose the same length as the base — each call may re-sample.
        self._len = len(base_dataset)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int):
        # Keep drawing from the same image index until we get the right corruption.
        for _ in range(self.MAX_RESAMPLE):
            corrupted, clean, label = self.base[index]
            if label == self.target_label:
                return corrupted, clean, label
        # Fallback: scan forward until the label matches (rare)
        for offset in range(1, self._len):
            idx = (index + offset) % self._len
            corrupted, clean, label = self.base[idx]
            if label == self.target_label:
                return corrupted, clean, label
        # Should never reach here in practice
        raise RuntimeError(
            f"Could not sample corruption label {self.target_label} "
            f"after exhaustive search"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpecialistConfig:
    """Parameters for one specialist autoencoder training run."""

    # What this specialist restores
    corruption_type: str = "salt_pepper"   # one of the CORRUPTION_LABELS keys (not 'clean')

    # Paths
    dataset_root: Path = field(default=Path("data"))
    split_csv: Path = field(default=Path("data/manifests/development_split.csv"))
    val_manifest: Path = field(default=Path("data/manifests/validation_manifest.json"))
    output_dir: Path = field(default=Path("output/task2/specialists"))

    # Model (re-uses UniversalRestorationAutoencoder)
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
    epochs: int = 50
    num_workers: int = 2
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Early stop
    early_stop_patience: int | None = 8
    min_epochs_before_stopping: int = 15

    # Callbacks
    report_to: Callable[[int, float], None] | None = None
    log_epoch: Callable[[int, dict[str, float], dict[str, float]], None] | None = None


# ---------------------------------------------------------------------------
# Ranking metric (same as Task 1 for consistency)
# ---------------------------------------------------------------------------

def ranking_score(mean_l1: float, mean_ssim: float) -> float:
    """0.5·L1 + 0.5·(1−SSIM) — lower is better."""
    return 0.5 * mean_l1 + 0.5 * (1.0 - mean_ssim)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: SpecialistConfig,
) -> dict[str, float]:
    model.train()
    device = cfg.device
    total_l1 = total_ssim = total_comb = n = 0

    for corrupted, clean, _labels in loader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)

        pred = model(corrupted)
        loss = combined_loss(pred, clean, alpha=cfg.alpha)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = corrupted.shape[0]
        with torch.no_grad():
            total_l1 += l1_loss(pred, clean).item() * bs
            total_ssim += differentiable_ssim(pred, clean).item() * bs
            total_comb += loss.item() * bs
        n += bs

    n = max(n, 1)
    return {
        "train_l1": total_l1 / n,
        "train_ssim": total_ssim / n,
        "train_combined_loss": total_comb / n,
    }


def validate_specialist(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: SpecialistConfig,
    target_label: int,
) -> dict[str, float]:
    """Evaluate on validation set; compute metrics only for target corruption."""
    model.eval()
    device = cfg.device
    total_l1 = total_ssim = n = 0

    with torch.no_grad():
        for corrupted, clean, labels in loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            labels_t = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)

            # Compute metrics only on samples of the target corruption
            mask = labels_t == target_label
            if not mask.any():
                continue

            pred = model(corrupted)
            pred_masked = pred[mask]
            clean_masked = clean[mask]

            bs = mask.sum().item()
            total_l1 += l1_loss(pred_masked, clean_masked).item() * bs
            total_ssim += differentiable_ssim(pred_masked, clean_masked).item() * bs
            n += bs

    if n == 0:
        return {"val_l1": 0.0, "val_ssim": 0.0, "ranking_score": 1.0}

    mean_l1 = total_l1 / n
    mean_ssim = total_ssim / n
    return {
        "val_l1": mean_l1,
        "val_ssim": mean_ssim,
        "ranking_score": ranking_score(mean_l1, mean_ssim),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def train_specialist(cfg: SpecialistConfig) -> dict[str, Any]:
    """Train one specialist autoencoder.  Callable many times (Optuna-safe)."""

    if cfg.corruption_type not in CORRUPTION_LABELS or cfg.corruption_type == "clean":
        raise ValueError(
            f"corruption_type must be one of {list(CORRUPTION_LABELS)} (not 'clean'), "
            f"got {cfg.corruption_type!r}"
        )
    target_label = CORRUPTION_LABELS[cfg.corruption_type]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- Datasets ---
    base_train_ds = PetRestorationDataset(
        cfg.dataset_root,
        "train",
        split_csv=cfg.split_csv,
        training_seed=cfg.seed,
    )
    specialist_train_ds = SpecialistDataset(base_train_ds, target_label)

    val_ds = PetRestorationDataset(
        cfg.dataset_root,
        "validation",
        manifest_path=cfg.val_manifest,
    )

    train_loader = DataLoader(
        specialist_train_ds,
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

    # --- Model ---
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
    specialist_dir = cfg.output_dir / cfg.corruption_type
    specialist_dir.mkdir(parents=True, exist_ok=True)

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
        val_metrics = validate_specialist(model, val_loader, cfg, target_label)

        elapsed = time.monotonic() - t0
        rs = val_metrics["ranking_score"]

        print(
            f"[{cfg.corruption_type}] epoch {epoch:3d}  "
            f"train_l1={train_metrics['train_l1']:.4f}  "
            f"train_ssim={train_metrics['train_ssim']:.4f}  "
            f"val_l1={val_metrics['val_l1']:.4f}  "
            f"val_ssim={val_metrics['val_ssim']:.4f}  "
            f"rank={rs:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

        if cfg.log_epoch is not None:
            cfg.log_epoch(epoch, train_metrics, val_metrics)

        if cfg.report_to is not None:
            cfg.report_to(epoch, rs)

        if not (np.isfinite(val_metrics["val_l1"]) and np.isfinite(val_metrics["val_ssim"])):
            print("  -> Diverged")
            status = "diverged"
            break

        if rs < best_ranking:
            best_ranking = rs
            best_epoch = epoch
            epochs_without_improve = 0
            cfg_dict = {k: v for k, v in asdict(cfg).items()
                        if k not in {"report_to", "log_epoch"}}
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "cfg": cfg_dict,
                "ranking_score": rs,
                "val_metrics": val_metrics,
                "corruption_type": cfg.corruption_type,
            }
            ckpt_name = f"best_{cfg.corruption_type}.pt"
            torch.save(best_state, specialist_dir / ckpt_name)
        else:
            epochs_without_improve += 1

        if (
            cfg.early_stop_patience is not None
            and epoch >= cfg.min_epochs_before_stopping
            and epochs_without_improve >= cfg.early_stop_patience
        ):
            print(f"  -> Early stopping")
            status = "early_stopped"
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    summary = {
        "corruption_type": cfg.corruption_type,
        "best_ranking_score": best_ranking if best_state else float("inf"),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "status": status,
        "final_val_metrics": best_state["val_metrics"] if best_state else {},
        "checkpoint_path": str(specialist_dir / f"best_{cfg.corruption_type}.pt"),
    }

    del model, optimizer
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train Task 2 specialist autoencoder")
    p.add_argument("--corruption-type", choices=["salt_pepper", "gaussian_blur", "occlusion"],
                   required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("output/task2/specialists"))
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--skip-count", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cfg = SpecialistConfig(
        corruption_type=args.corruption_type,
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
    )
    summary = train_specialist(cfg)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

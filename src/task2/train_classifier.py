"""Training module for Task 2 corruption classifier.

Trains a CorruptionClassifier on the dynamic training pipeline (same
runtime corruption as Task 1).  The training batches are implicitly
balanced because the dataset samples each of the four corruptions with
equal probability (0.25 each) per image.

Key design choices
------------------
- Cross-entropy loss (standard multiclass classification)
- AdamW optimiser with configurable lr / weight_decay
- Per-epoch W&B logging via optional ``log_epoch`` callback
- Optuna-compatible: ``report_to`` callback for pruning
- Early stopping on validation accuracy (higher is better)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pet_restoration.dataset import CORRUPTION_LABELS, PetRestorationDataset
from task2.classifier import CorruptionClassifier


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """All tuneable and fixed parameters for one classifier training run."""

    # Paths
    dataset_root: Path = field(default=Path("data/pet_images"))
    split_csv: Path = field(default=Path("data/manifests/development_split.csv"))
    val_manifest: Path = field(default=Path("data/manifests/validation_manifest.json"))
    output_dir: Path = field(default=Path("output/task2/classifier"))

    # Model
    base_channels: int = 32
    dropout: float = 0.0

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Loop
    batch_size: int = 32
    epochs: int = 30
    num_workers: int = 2
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Early stopping (on val accuracy — higher is better)
    early_stop_patience: int | None = 5
    min_epochs_before_stopping: int = 10

    # Callbacks
    report_to: Callable[[int, float], None] | None = None
    log_epoch: Callable[[int, dict[str, float], dict[str, float]], None] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    model.train()
    total_loss = total_acc = total_n = 0

    for corrupted, _clean, labels in loader:
        corrupted = corrupted.to(device)
        labels = labels.to(device)

        logits = model(corrupted)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = corrupted.shape[0]
        total_loss += loss.item() * bs
        total_acc += _accuracy(logits.detach(), labels) * bs
        total_n += bs

    n = max(total_n, 1)
    return {"train_loss": total_loss / n, "train_acc": total_acc / n}


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    model.eval()
    total_loss = total_acc = total_n = 0
    # per-class TP / support for macro metrics
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for corrupted, _clean, labels in loader:
            corrupted = corrupted.to(device)
            labels = labels.to(device)
            logits = model(corrupted)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)

            bs = corrupted.shape[0]
            total_loss += loss.item() * bs
            total_acc += _accuracy(logits, labels) * bs
            total_n += bs

            for p, t in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                if p == t:
                    tp[t] += 1
                else:
                    fp[p] += 1
                    fn[t] += 1

    n = max(total_n, 1)
    mean_loss = total_loss / n
    mean_acc = total_acc / n

    # Macro precision / recall / F1
    num_classes = 4
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        denom_p = tp[c] + fp[c]
        denom_r = tp[c] + fn[c]
        p_val = tp[c] / denom_p if denom_p > 0 else 0.0
        r_val = tp[c] / denom_r if denom_r > 0 else 0.0
        f1 = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        precisions.append(p_val)
        recalls.append(r_val)
        f1s.append(f1)

    return {
        "val_loss": mean_loss,
        "val_acc": mean_acc,
        "val_macro_precision": float(np.mean(precisions)),
        "val_macro_recall": float(np.mean(recalls)),
        "val_macro_f1": float(np.mean(f1s)),
    }


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train_classifier(cfg: ClassifierConfig) -> dict[str, Any]:
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
    model = CorruptionClassifier(
        base_channels=cfg.base_channels,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # --- Book-keeping ---
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    epochs_without_improve = 0
    epochs_run = 0
    status = "ok"

    for epoch in range(1, cfg.epochs + 1):
        epochs_run = epoch
        t0 = time.monotonic()

        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg.device)
        val_metrics = validate(model, val_loader, cfg.device)

        elapsed = time.monotonic() - t0
        acc = val_metrics["val_acc"]

        print(
            f"epoch {epoch:3d}  "
            f"train_loss={train_metrics['train_loss']:.4f}  "
            f"train_acc={train_metrics['train_acc']:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_acc={acc:.4f}  "
            f"val_f1={val_metrics['val_macro_f1']:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        if cfg.log_epoch is not None:
            cfg.log_epoch(epoch, train_metrics, val_metrics)

        if cfg.report_to is not None:
            # report negative accuracy so pruner minimises
            cfg.report_to(epoch, -acc)

        if not np.isfinite(val_metrics["val_loss"]):
            print("  -> Diverged (non-finite val_loss)")
            status = "diverged"
            break

        # checkpoint (higher accuracy is better)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            epochs_without_improve = 0
            cfg_dict = {k: v for k, v in asdict(cfg).items()
                        if k not in {"report_to", "log_epoch"}}
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "cfg": cfg_dict,
                "val_acc": acc,
                "val_metrics": val_metrics,
            }
            torch.save(best_state, cfg.output_dir / "best_classifier.pt")
        else:
            epochs_without_improve += 1

        if (
            cfg.early_stop_patience is not None
            and epoch >= cfg.min_epochs_before_stopping
            and epochs_without_improve >= cfg.early_stop_patience
        ):
            print(f"  -> Early stopping (no improvement for {cfg.early_stop_patience} epochs)")
            status = "early_stopped"
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    summary = {
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "status": status,
        "final_val_metrics": best_state["val_metrics"] if best_state is not None else {},
        "best_checkpoint_path": str(cfg.output_dir / "best_classifier.pt"),
    }

    del model, optimizer
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train Task 2 corruption classifier")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("output/task2/classifier"))
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cfg = ClassifierConfig(
        dataset_root=args.dataset_root,
        split_csv=args.split_csv,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        base_channels=args.base_channels,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    summary = train_classifier(cfg)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

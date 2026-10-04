"""Test-set evaluation for Task 2: Hard-Routed Restoration.

Evaluates the full pipeline on the deterministic test manifest in two
modes:
  1. **Oracle routing** — uses the known ground-truth corruption label to
     select the specialist.  Isolates specialist quality from classifier
     accuracy.
  2. **Predicted routing** — uses the classifier prediction to select the
     specialist.  Reflects real operational performance.

Also computes:
  - Classifier overall accuracy + macro P/R/F1
  - Normalised 4-class confusion matrix
  - Per-corruption reconstruction metrics for both routing modes
  - Cases where classifier errors caused restoration failures

Outputs (in --output-dir):
  - evaluation_results.json
  - confusion_matrix.png
  - grid_oracle.png, grid_predicted.png
  - routing_failures.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pet_restoration.dataset import CORRUPTION_LABELS, PetRestorationDataset
from pet_restoration.losses import differentiable_ssim, l1_loss
from task2.router import (
    HardRouter,
    LABEL_TO_NAME,
    load_classifier,
    load_specialist,
    route_batch,
)
from task1.train import ranking_score


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_routing(
    router: HardRouter,
    loader: DataLoader,
    dataset: PetRestorationDataset,
    device: str,
    routing_mode: str,
) -> list[dict]:
    from pet_restoration.losses import _ssim_channel

    per_sample = []
    global_idx = 0

    with torch.no_grad():
        for corrupted, clean, labels in loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            labels_t = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)

            result = route_batch(router, corrupted, routing=routing_mode, true_labels=labels_t)
            restored = result["restored"]

            # L1 per sample
            l1_per = (restored - clean).abs().mean(dim=(1, 2, 3)).cpu().tolist()

            # SSIM per sample
            C = restored.shape[1]
            ssim_maps = [_ssim_channel(restored[:, c:c+1], clean[:, c:c+1]) for c in range(C)]
            ssim_per = torch.cat(ssim_maps, dim=1).mean(dim=(1, 2, 3)).cpu().tolist()

            pred_cls = result["predicted_class_idx"].tolist()

            for i in range(corrupted.shape[0]):
                record = dataset.records[global_idx]
                true_label = int(labels_t[i])
                predicted_label = int(pred_cls[i])
                per_sample.append({
                    "index": global_idx,
                    "image_path": record["image_path"],
                    "corruption": record["corruption"],
                    "severity": record["severity"],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "classifier_correct": true_label == predicted_label,
                    "routed_to": LABEL_TO_NAME[predicted_label if routing_mode == "predicted" else true_label],
                    "l1": l1_per[i],
                    "ssim": ssim_per[i],
                    "rank": ranking_score(l1_per[i], ssim_per[i]),
                })
                global_idx += 1

    return per_sample


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_per_sample(per_sample: list[dict]) -> dict:
    overall = {"l1": [], "ssim": [], "rank": []}
    by_corruption: dict[str, dict] = defaultdict(lambda: {"l1": [], "ssim": [], "rank": []})

    for s in per_sample:
        overall["l1"].append(s["l1"])
        overall["ssim"].append(s["ssim"])
        overall["rank"].append(s["rank"])
        by_corruption[s["corruption"]]["l1"].append(s["l1"])
        by_corruption[s["corruption"]]["ssim"].append(s["ssim"])
        by_corruption[s["corruption"]]["rank"].append(s["rank"])

    def _stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    return {
        "overall": {k: _stats(v) for k, v in overall.items()},
        "by_corruption": {
            name: {k: _stats(v) for k, v in vals.items()}
            for name, vals in sorted(by_corruption.items())
        },
    }


def classifier_metrics(per_sample: list[dict]) -> dict:
    """Overall accuracy + confusion matrix."""
    num_classes = len(CORRUPTION_LABELS)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    correct = 0

    for s in per_sample:
        t = s["true_label"]
        p = s["predicted_label"]
        confusion[t, p] += 1
        if t == p:
            correct += 1

    acc = correct / max(len(per_sample), 1)

    # Normalised confusion matrix (row-normalised)
    row_sums = confusion.sum(axis=1, keepdims=True).clip(min=1)
    norm_conf = confusion / row_sums

    # Per-class P / R / F1
    per_class = {}
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        p_val = tp / max(tp + fp, 1)
        r_val = tp / max(tp + fn, 1)
        f1 = 2 * p_val * r_val / max(p_val + r_val, 1e-9)
        per_class[LABEL_TO_NAME[c]] = {"precision": float(p_val), "recall": float(r_val), "f1": float(f1)}

    macro_p = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_r = float(np.mean([v["recall"] for v in per_class.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    return {
        "accuracy": float(acc),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "normalised_confusion_matrix": norm_conf.tolist(),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def render_confusion_matrix(cm_norm: list[list[float]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [LABEL_TO_NAME[i] for i in range(len(cm_norm))]
    cm = np.array(cm_norm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.6 else "black", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_routing_grid(
    indices: list[int],
    dataset: PetRestorationDataset,
    router: HardRouter,
    device: str,
    out_path: Path,
    routing_mode: str,
    suptitle: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        corrupted, clean, label_int = dataset[idx]
        label_t = torch.tensor([label_int])
        with torch.no_grad():
            result = route_batch(
                router,
                corrupted.unsqueeze(0).to(device),
                routing=routing_mode,
                true_labels=label_t.to(device),
            )
        recon = result["restored"][0].cpu()
        pred_name = result["predicted_class_name"][0]
        record = dataset.records[idx]

        c = clean.permute(1, 2, 0).numpy()
        x = corrupted.permute(1, 2, 0).numpy()
        r = recon.permute(1, 2, 0).numpy()
        err = np.abs(c - r).mean(axis=2)
        rs = ranking_score(l1_loss(recon.unsqueeze(0), clean.unsqueeze(0)).item(),
                           differentiable_ssim(recon.unsqueeze(0), clean.unsqueeze(0)).item())

        panels = [
            (c, "clean target", None),
            (x, f"input ({record['corruption']})", None),
            (r, f"restored (pred={pred_name})", None),
            (err, f"|error|  rank={rs:.3f}", "hot"),
        ]
        for col, (img, ttl, cmap) in enumerate(panels):
            axes[row, col].imshow(np.clip(img, 0, 1), cmap=cmap)
            axes[row, col].set_title(ttl, fontsize=8)
            axes[row, col].axis("off")

    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Task 2 hard-routed restoration")
    p.add_argument("--classifier-ckpt", type=Path, required=True)
    p.add_argument("--specialist-salt-ckpt", type=Path, required=True)
    p.add_argument("--specialist-blur-ckpt", type=Path, required=True)
    p.add_argument("--specialist-occlusion-ckpt", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load models ----
    print("Loading classifier...")
    classifier = load_classifier(args.classifier_ckpt, device)
    print("Loading specialists...")
    sp_salt = load_specialist(args.specialist_salt_ckpt, device)
    sp_blur = load_specialist(args.specialist_blur_ckpt, device)
    sp_occ  = load_specialist(args.specialist_occlusion_ckpt, device)

    router = HardRouter(classifier, sp_salt, sp_blur, sp_occ).to(device)
    router.eval()

    # ---- Dataset ----
    test_ds = PetRestorationDataset(
        args.dataset_root, "test", manifest_path=args.test_manifest
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Test set: {len(test_ds)} samples")

    results = {}

    for mode in ("oracle", "predicted"):
        print(f"\nEvaluating {mode} routing...")
        per_sample = evaluate_routing(router, loader, test_ds, device, mode)
        agg = aggregate_per_sample(per_sample)
        cls_metrics = classifier_metrics(per_sample)

        results[mode] = {
            "reconstruction": agg,
            "classifier": cls_metrics,
        }

        # Print summary
        o = agg["overall"]
        print(f"  [{mode}] L1={o['l1']['mean']:.4f}  SSIM={o['ssim']['mean']:.4f}  "
              f"rank={o['rank']['mean']:.4f}  acc={cls_metrics['accuracy']:.4f}  "
              f"macro_f1={cls_metrics['macro_f1']:.4f}")

        # Per-CSV
        with (args.output_dir / f"per_sample_{mode}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
            w.writeheader()
            w.writerows(per_sample)

        # Confusion matrix
        if mode == "predicted":
            render_confusion_matrix(
                cls_metrics["normalised_confusion_matrix"],
                args.output_dir / "confusion_matrix.png",
            )
            print("  Saved confusion_matrix.png")

        # Routing failures (classifier wrong AND rank > 0.3)
        failures = [
            s for s in per_sample
            if not s["classifier_correct"] and s["rank"] > 0.3
        ]
        failures.sort(key=lambda s: s["rank"], reverse=True)
        (args.output_dir / f"routing_failures_{mode}.json").write_text(
            json.dumps(failures[:20], indent=2)
        )

        # Visual grid
        idxs = [s["index"] for s in per_sample[::max(1, len(per_sample)//8)][:8]]
        render_routing_grid(
            idxs, test_ds, router, device,
            args.output_dir / f"grid_{mode}.png",
            routing_mode=mode,
            suptitle=f"Task 2 Hard-Routed Restoration — {mode.capitalize()} Routing",
        )
        print(f"  Saved grid_{mode}.png")

    # ---- Save all results ----
    (args.output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

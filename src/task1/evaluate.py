"""Test-manifest evaluation for Task 1.

Produces:
  - Overall + per (corruption, severity) metrics table (JSON + CSV)
  - Per-sample metrics CSV for deeper analysis
  - 12+ representative visual grids (clean | corrupted | reconstruction | |error|)
  - 4 failure cases, one per corruption type, with analysis
  - Copy-input baseline on the same test manifest

Usage:
    python -m task1.evaluate \
        --checkpoint /path/to/best.pt \
        --test-manifest /path/to/test_manifest.json \
        --output-dir /path/to/eval_out
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
from pet_restoration.models import UniversalRestorationAutoencoder
from task1.train import ranking_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = UniversalRestorationAutoencoder(
        base_channels=cfg["base_channels"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
        skip_count=cfg["skip_count"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, ckpt





def aggregate(per_sample: list[dict]) -> dict:
    """Aggregate per-sample metrics into overall + per (corruption, severity)."""
    overall = {"l1": [], "ssim": [], "rank": []}
    by_group: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"l1": [], "ssim": [], "rank": []}
    )

    for s in per_sample:
        overall["l1"].append(s["l1"])
        overall["ssim"].append(s["ssim"])
        overall["rank"].append(s["rank"])
        key = (s["corruption"], s["severity"])
        by_group[key]["l1"].append(s["l1"])
        by_group[key]["ssim"].append(s["ssim"])
        by_group[key]["rank"].append(s["rank"])

    def _stats(vals: list[float]) -> dict:
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }

    result = {
        "overall": {
            "l1": _stats(overall["l1"]),
            "ssim": _stats(overall["ssim"]),
            "rank": _stats(overall["rank"]),
        },
        "by_group": {
            f"{corruption}/{severity}": {
                "corruption": corruption,
                "severity": severity,
                "l1": _stats(v["l1"]),
                "ssim": _stats(v["ssim"]),
                "rank": _stats(v["rank"]),
            }
            for (corruption, severity), v in sorted(by_group.items())
        },
    }
    return result

def evaluate_per_sample(model, loader, dataset, device):
    """Run inference, return per-sample metrics (batched)."""
    from pet_restoration.losses import _ssim_channel

    per_sample = []
    global_idx = 0

    with torch.no_grad():
        for corrupted, clean, labels in loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            pred = model(corrupted)

            # L1 per sample — one op, batched
            l1_per = (pred - clean).abs().mean(dim=(1, 2, 3)).cpu().tolist()

            # SSIM per sample — one call per channel, batched over B
            C = pred.shape[1]
            ssim_maps = [_ssim_channel(pred[:, c:c+1], clean[:, c:c+1]) for c in range(C)]
            ssim_per = torch.stack(ssim_maps, dim=1).mean(dim=(1, 2, 3)).cpu().tolist()

            for i in range(corrupted.shape[0]):
                record = dataset.records[global_idx]
                per_sample.append({
                    "index": global_idx,
                    "image_path": record["image_path"],
                    "corruption": record["corruption"],
                    "severity": record["severity"],
                    "l1": l1_per[i],
                    "ssim": ssim_per[i],
                    "rank": ranking_score(l1_per[i], ssim_per[i]),
                })
                global_idx += 1

            if global_idx % 2000 < corrupted.shape[0]:
                print(f"  processed {global_idx}/{len(dataset)}", flush=True)

    return per_sample


def compute_copy_baseline(loader, dataset):
    """Ranking score for the trivial 'output = input' model (batched)."""
    from pet_restoration.losses import _ssim_channel

    total_l1 = total_ssim = n = 0
    per_group = defaultdict(lambda: {"l1": [], "ssim": []})
    global_idx = 0

    with torch.no_grad():
        for corrupted, clean, _ in loader:
            l1_per = (corrupted - clean).abs().mean(dim=(1, 2, 3)).cpu().tolist()
            C = corrupted.shape[1]
            ssim_maps = [_ssim_channel(corrupted[:, c:c+1], clean[:, c:c+1]) for c in range(C)]
            ssim_per = torch.stack(ssim_maps, dim=1).mean(dim=(1, 2, 3)).cpu().tolist()

            for i in range(corrupted.shape[0]):
                total_l1 += l1_per[i]
                total_ssim += ssim_per[i]
                n += 1
                record = dataset.records[global_idx]
                key = (record["corruption"], record["severity"])
                per_group[key]["l1"].append(l1_per[i])
                per_group[key]["ssim"].append(ssim_per[i])
                global_idx += 1

    mean_l1 = total_l1 / n
    mean_ssim = total_ssim / n
    return {
        "overall": {
            "l1": mean_l1,
            "ssim": mean_ssim,
            "rank": ranking_score(mean_l1, mean_ssim),
            "n": n,
        },
        "by_group": {
            f"{c}/{s}": {
                "l1": float(np.mean(v["l1"])),
                "ssim": float(np.mean(v["ssim"])),
                "rank": ranking_score(
                    float(np.mean(v["l1"])), float(np.mean(v["ssim"]))
                ),
            }
            for (c, s), v in sorted(per_group.items())
        },
    }
# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def render_grid(
    indices: list[int],
    dataset: PetRestorationDataset,
    model: torch.nn.Module,
    device: str,
    out_path: Path,
    suptitle: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        corrupted, clean, _label = dataset[idx]
        with torch.no_grad():
            recon = model(corrupted.unsqueeze(0).to(device))[0].cpu()

        c = clean.permute(1, 2, 0).numpy()
        x = corrupted.permute(1, 2, 0).numpy()
        r = recon.permute(1, 2, 0).numpy()
        err = np.abs(c - r).mean(axis=2)

        record = dataset.records[idx]
        label = f"{record['corruption']}/{record['severity']}"
        rank_val = ranking_score(
            l1_loss(recon.unsqueeze(0), clean.unsqueeze(0)).item(),
            differentiable_ssim(recon.unsqueeze(0), clean.unsqueeze(0)).item(),
        )

        panels = [
            (c, "clean target", None),
            (x, f"input ({label})", None),
            (r, "reconstruction", None),
            (err, f"|error|  rank={rank_val:.3f}", "hot"),
        ]
        for col, (img, ttl, cmap) in enumerate(panels):
            axes[row, col].imshow(np.clip(img, 0, 1), cmap=cmap)
            axes[row, col].set_title(ttl, fontsize=9)
            axes[row, col].axis("off")

    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def pick_representative_indices(
    per_sample: list[dict], per_group: int = 1
) -> list[int]:
    """Pick one sample per (corruption, severity) group, sorted consistently."""
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in per_sample:
        by_group[(s["corruption"], s["severity"])].append(s)

    chosen = []
    for key in sorted(by_group.keys()):
        bucket = sorted(by_group[key], key=lambda s: s["rank"])
        # pick the median-ranked sample of each group (representative, not best)
        mid = len(bucket) // 2
        chosen.append(bucket[mid]["index"])
    return chosen


def pick_failure_indices(per_sample: list[dict], n: int = 4) -> list[int]:
    """Pick the worst sample per corruption type (diversity-constrained)."""
    by_corruption: dict[str, list[dict]] = defaultdict(list)
    for s in per_sample:
        by_corruption[s["corruption"]].append(s)

    failures = []
    for corruption in sorted(by_corruption.keys()):
        worst = max(by_corruption[corruption], key=lambda s: s["rank"])
        failures.append(worst)

    # if fewer than n corruption types (won't happen with 4 types) pad with global worst
    if len(failures) < n:
        remaining = sorted(per_sample, key=lambda s: s["rank"], reverse=True)
        for s in remaining:
            if s["index"] not in {f["index"] for f in failures}:
                failures.append(s)
                if len(failures) == n:
                    break

    return [f["index"] for f in failures[:n]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Task 1 model on the test manifest.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("/content/data"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model ----
    model, ckpt = load_model(args.checkpoint, device)
    print(f"Checkpoint epoch: {ckpt['epoch']}")
    print(f"Checkpoint ranking_score: {ckpt['ranking_score']:.4f}")
    print(f"Skip count: {ckpt['cfg']['skip_count']}")

    # ---- Data ----
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

    # ---- Per-sample metrics ----
    print("Running inference...")
    per_sample = evaluate_per_sample(model, loader, test_ds, device)
    agg = aggregate(per_sample)

    # ---- Copy baseline ----
    print("Computing copy baseline...")
    baseline = compute_copy_baseline(loader, test_ds)

    # ---- Print results ----
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    o = agg["overall"]
    print(f"OVERALL   L1={o['l1']['mean']:.4f}  SSIM={o['ssim']['mean']:.4f}  "
          f"rank={o['rank']['mean']:.4f}  (n={o['rank']['n']})")
    print(f"BASELINE  L1={baseline['overall']['l1']:.4f}  "
          f"SSIM={baseline['overall']['ssim']:.4f}  "
          f"rank={baseline['overall']['rank']:.4f}")

    print(f"\nModel beats copy baseline: "
          f"{'YES' if o['rank']['mean'] < baseline['overall']['rank'] else 'NO'}")

    print("\nPER-GROUP RESULTS:")
    print(f"{'corruption':<15} {'severity':<10} {'n':>5} "
          f"{'model_L1':>10} {'model_SSIM':>11} {'rank':>8}  "
          f"{'base_rank':>10}  {'Δ':>8}")
    for key, g in sorted(agg["by_group"].items()):
        base_key = key
        base_r = baseline["by_group"].get(base_key, {}).get("rank", float("nan"))
        delta = base_r - g["rank"]["mean"]
        print(f"{g['corruption']:<15} {g['severity']:<10} {g['rank']['n']:>5} "
              f"{g['l1']['mean']:>10.4f} {g['ssim']['mean']:>11.4f} "
              f"{g['rank']['mean']:>8.4f}  "
              f"{base_r:>10.4f}  {delta:>8.4f}")

    # ---- Save metrics ----
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(agg, indent=2)
    )
    (args.output_dir / "copy_baseline.json").write_text(
        json.dumps(baseline, indent=2)
    )

    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["index", "image_path", "corruption", "severity",
                           "l1", "ssim", "rank"],
        )
        w.writeheader()
        w.writerows(per_sample)

    # ---- Visual grids ----
    print("\nRendering representative grid...")
    rep_indices = pick_representative_indices(per_sample)
    render_grid(
        rep_indices, test_ds, model, device,
        args.output_dir / "grid_representative.png",
        "Representative test reconstructions (median sample per group)",
    )

    print("Rendering failure cases...")
    fail_indices = pick_failure_indices(per_sample, n=4)
    render_grid(
        fail_indices, test_ds, model, device,
        args.output_dir / "grid_failures.png",
        "Failure cases (worst sample per corruption type)",
    )

    # ---- Failure analysis ----
    failures = []
    for idx in fail_indices:
        s = per_sample[idx]
        failures.append({
            "index": idx,
            "image_path": s["image_path"],
            "corruption": s["corruption"],
            "severity": s["severity"],
            "l1": s["l1"],
            "ssim": s["ssim"],
            "rank": s["rank"],
            "copy_baseline_rank": baseline["by_group"]
                .get(f"{s['corruption']}/{s['severity']}", {})
                .get("rank"),
        })

    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2))

    print("\n" + "=" * 70)
    print("FAILURE CASES")
    print("=" * 70)
    for f in failures:
        base = f["copy_baseline_rank"]
        base_s = f"{base:.4f}" if base is not None else "n/a"
        print(f"  {f['image_path']}")
        print(f"    {f['corruption']}/{f['severity']}  "
              f"L1={f['l1']:.4f}  SSIM={f['ssim']:.4f}  rank={f['rank']:.4f}  "
              f"(baseline {base_s})")

    print(f"\nOutputs written to {args.output_dir}")
    print(f"  test_metrics.json")
    print(f"  copy_baseline.json")
    print(f"  per_sample_metrics.csv")
    print(f"  failures.json")
    print(f"  grid_representative.png")
    print(f"  grid_failures.png")


if __name__ == "__main__":
    main()
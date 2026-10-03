import sys
sys.path.insert(0, "/content/Generative_AI/src")

import json
from collections import Counter
from pathlib import Path

import torch
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STORAGE_DIR = Path("/content/drive/MyDrive/optuna")
OUTPUT_ROOT = Path("/content/drive/MyDrive/outputs/task1")
SKIP_COUNTS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Load each study and summarise
# ---------------------------------------------------------------------------
studies = {}
summary_rows = []

for skip in SKIP_COUNTS:
    db = STORAGE_DIR / f"task1_skip{skip}.db"
    if not db.exists():
        print(f"[skip {skip}] no DB at {db} — skipping")
        continue
    study = optuna.load_study(
        study_name=f"task1_skip{skip}",
        storage=f"sqlite:///{db}",
    )
    studies[skip] = study
    counts = Counter(t.state.name for t in study.trials)
    n_complete = counts.get("COMPLETE", 0)

    row = {
        "skip_count": skip,
        "n_trials": len(study.trials),
        "COMPLETE": n_complete,
        "PRUNED": counts.get("PRUNED", 0),
        "FAIL": counts.get("FAIL", 0),
        "RUNNING": counts.get("RUNNING", 0),
        "best_value": study.best_value if n_complete else None,
        "best_trial": study.best_trial.number if n_complete else None,
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("skip_count")
print("\n=== Study summary ===")
print(summary_df.to_string())

if not studies or all(r["COMPLETE"] == 0 for r in summary_rows):
    raise SystemExit("No completed trials in any study — nothing to compare.")

# ---------------------------------------------------------------------------
# 2. Full leaderboard for each study (for the appendix)
# ---------------------------------------------------------------------------
def trial_table(study, top_n=None):
    rows = []
    for t in study.trials:
        if t.state.name not in ("COMPLETE", "PRUNED"):
            continue
        rows.append({
            "trial": t.number,
            "state": t.state.name,
            "value": t.value,
            "base_channels": t.params.get("base_channels"),
            "latent_dim": t.params.get("latent_dim"),
            "batch_size": t.params.get("batch_size"),
            "dropout": round(t.params.get("dropout", 0.0), 3),
            "lr": round(t.params.get("lr", 0.0), 5),
            "weight_decay": round(t.params.get("weight_decay", 0.0), 6),
            "alpha": round(t.params.get("alpha", 0.0), 3),
            "epochs_run": max(t.intermediate_values.keys())
                          if t.intermediate_values else 0,
        })
    df = pd.DataFrame(rows).sort_values("value", na_position="last")
    return df.head(top_n) if top_n else df

for skip, study in studies.items():
    print(f"\n=== skip_count={skip} — leaderboard ===")
    print(trial_table(study, top_n=10).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Pick the global winner across skip counts
# ---------------------------------------------------------------------------
winner_skip = min(
    (s for s in studies if summary_df.loc[s, "COMPLETE"] > 0),
    key=lambda s: summary_df.loc[s, "best_value"],
)
winner_study = studies[winner_skip]
winner_trial = winner_study.best_trial

print("\n" + "=" * 60)
print(f"WINNER: skip_count={winner_skip}, trial #{winner_trial.number}")
print(f"  best_value = {winner_trial.value:.4f}")
print(f"  params     = {json.dumps(winner_trial.params, indent=4)}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 4. Load the winning checkpoint and extract stored val metrics
# ---------------------------------------------------------------------------
ckpt_path = OUTPUT_ROOT / f"skip{winner_skip}" / f"trial_{winner_trial.number}" / "best.pt"
if not ckpt_path.exists():
    print(f"\n[WARN] checkpoint missing at {ckpt_path}")
    print("       rerun with --output-root matching the study's original output dir")
else:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vm = ckpt["val_metrics"]
    print("\n=== Winning trial — validation metrics (best epoch) ===")
    print(f"  best_epoch        : {ckpt['epoch']}")
    print(f"  ranking_score     : {vm['ranking_score']:.4f}")
    print(f"  val_l1            : {vm['val_l1']:.4f}")
    print(f"  val_ssim          : {vm['val_ssim']:.4f}")
    print("\n  per-corruption:")
    for name in ("clean", "salt_pepper", "gaussian_blur", "occlusion"):
        print(f"    {name:15s}  L1={vm[f'{name}_l1']:.4f}  "
              f"(1−SSIM)={vm[f'{name}_ssim_loss']:.4f}")

# ---------------------------------------------------------------------------
# 5. Copy-input baseline (the floor every model must beat)
# ---------------------------------------------------------------------------
from torch.utils.data import DataLoader
from pet_restoration.dataset import PetRestorationDataset, CORRUPTION_LABELS
from pet_restoration.losses import l1_loss, differentiable_ssim
from task1.train import ranking_score

val_ds = PetRestorationDataset(
    "/content/data", "validation",
    manifest_path="/content/Generative_AI/data/manifests/validation_manifest.json",
)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

tot_l1 = tot_ssim = n = 0
per = {k: [0.0, 0.0, 0] for k in CORRUPTION_LABELS}

with torch.no_grad():
    for corr, clean, labels in val_loader:
        tot_l1   += l1_loss(corr, clean).item() * len(corr)
        tot_ssim += differentiable_ssim(corr, clean).item() * len(corr)
        n += len(corr)
        for name, idx in CORRUPTION_LABELS.items():
            m = labels == idx
            if m.any():
                per[name][0] += l1_loss(corr[m], clean[m]).item() * m.sum().item()
                per[name][1] += differentiable_ssim(corr[m], clean[m]).item() * m.sum().item()
                per[name][2] += m.sum().item()

baseline_l1 = tot_l1 / n
baseline_ssim = tot_ssim / n
baseline_rank = ranking_score(baseline_l1, baseline_ssim)

print("\n=== Copy-input baseline ===")
print(f"  L1={baseline_l1:.4f}  SSIM={baseline_ssim:.4f}  rank={baseline_rank:.4f}")
for name, (l, s, c) in per.items():
    print(f"    {name:15s}  L1={l/c:.4f}  SSIM={s/c:.4f}")

# ---------------------------------------------------------------------------
# 6. Skip-count ablation table (the report's headline)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SKIP-COUNT ABLATION")
print("=" * 60)

abl_rows = []
for skip, study in studies.items():
    if summary_df.loc[skip, "COMPLETE"] == 0:
        continue
    best = study.best_trial
    ck = OUTPUT_ROOT / f"skip{skip}" / f"trial_{best.number}" / "best.pt"
    row = {
        "skip_count": skip,
        "best_value": best.value,
        "beats_copy_baseline": best.value < baseline_rank,
        "base_channels": best.params.get("base_channels"),
        "latent_dim": best.params.get("latent_dim"),
        "batch_size": best.params.get("batch_size"),
        "alpha": round(best.params.get("alpha", 0.0), 3),
        "lr": round(best.params.get("lr", 0.0), 5),
    }
    if ck.exists():
        vm = torch.load(ck, map_location="cpu", weights_only=False)["val_metrics"]
        row.update({
            "clean_l1": round(vm["clean_l1"], 4),
            "salt_pepper_l1": round(vm["salt_pepper_l1"], 4),
            "gaussian_blur_l1": round(vm["gaussian_blur_l1"], 4),
            "occlusion_l1": round(vm["occlusion_l1"], 4),
            "val_ssim": round(vm["val_ssim"], 4),
        })
    abl_rows.append(row)

ablation_df = pd.DataFrame(abl_rows).set_index("skip_count")
print(ablation_df.to_string())
print(f"\nCopy baseline ranking_score = {baseline_rank:.4f}")
print("(any best_value above this means the model is worse than doing nothing)")

# ---------------------------------------------------------------------------
# 7. Save the report-ready bundle to disk
# ---------------------------------------------------------------------------
out_bundle = OUTPUT_ROOT / "report_bundle"
out_bundle.mkdir(parents=True, exist_ok=True)

summary_df.to_csv(out_bundle / "study_summary.csv")
ablation_df.to_csv(out_bundle / "skip_ablation.csv")
for skip, study in studies.items():
    trial_table(study).to_csv(out_bundle / f"leaderboard_skip{skip}.csv", index=False)

with open(out_bundle / "copy_baseline.json", "w") as f:
    json.dump({
        "overall_l1": baseline_l1,
        "overall_ssim": baseline_ssim,
        "ranking_score": baseline_rank,
        "per_corruption": {k: {"l1": v[0]/v[2], "ssim": v[1]/v[2]}
                           for k, v in per.items()},
    }, f, indent=2)

print(f"\nReport bundle written to {out_bundle}")
"""Final training with W&B logging."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb
from task1.train import TrainConfig, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-channels", type=int, required=True)
    p.add_argument("--latent-dim", type=int, required=True)
    p.add_argument("--dropout", type=float, required=True)
    p.add_argument("--skip-count", type=int, required=True, choices=[0,1,2])
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight-decay", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--wandb-project", default="GenAI-Assignment1")
    p.add_argument("--wandb-run-name", required=True)
    args = p.parse_args()

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        group=f"task1-final-skip{args.skip_count}",
        config=vars(args),
        reinit=True,
    )

    def log_epoch(epoch, tm, vm):
        wandb.log({
            "epoch": epoch,
            "train/l1": tm["train_l1"],
            "train/ssim": tm["train_ssim"],
            "train/combined_loss": tm["train_combined_loss"],
            "val/l1": vm["val_l1"],
            "val/ssim": vm["val_ssim"],
            "val/ranking_score": vm["ranking_score"],
            **{f"val/{k}": v for k, v in vm.items()
               if (k.endswith("_l1") or k.endswith("_ssim_loss"))
               and k != "val_l1"},
        })

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
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        log_epoch=log_epoch,
    )

    summary = train(cfg)
    wandb.log({f"final/{k}": v for k, v in summary.items()
               if isinstance(v, (int, float, str))})
    wandb.finish()
    print(summary)


if __name__ == "__main__":
    main()
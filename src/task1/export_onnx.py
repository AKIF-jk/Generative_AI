"""ONNX export and PyTorch-vs-ONNX verification for Task 1.

Usage:
    python -m task1.export_onnx \
        --checkpoint /path/to/best.pt \
        --output /path/to/model.onnx

The script:
  1. Loads the checkpoint (model + cfg)
  2. Rebuilds the UniversalRestorationAutoencoder from the stored cfg
  3. Exports to ONNX with a dynamic batch axis
  4. Runs a fixed random input through both PyTorch and onnxruntime
  5. Prints the max absolute difference (must be < 1e-4 for a valid export)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pet_restoration.models import UniversalRestorationAutoencoder


def load_model_from_checkpoint(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model with the checkpoint's architecture and load weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    model = UniversalRestorationAutoencoder(
        base_channels=cfg["base_channels"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
        skip_count=cfg["skip_count"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    *,
    opset: int = 17,
) -> None:
    """Export the model with a dynamic batch axis on input and output."""
    dummy = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )


def verify_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    *,
    batch_sizes: tuple[int, ...] = (1, 4),
    atol: float = 1e-4,
) -> dict:
    """Run identical inputs through PyTorch and ONNX; return max abs diff."""
    import onnxruntime as ort

    # Prefer CUDA if available, fall back to CPU
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    results = {}
    for bs in batch_sizes:
        torch.manual_seed(0)
        x = torch.rand(bs, 3, 128, 128, dtype=torch.float32)

        with torch.no_grad():
            y_torch = model(x).numpy()

        y_onnx = sess.run(None, {"input": x.numpy()})[0]

        max_diff = float(np.abs(y_torch - y_onnx).max())
        mean_diff = float(np.abs(y_torch - y_onnx).mean())
        results[bs] = {
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "pytorch_range": [float(y_torch.min()), float(y_torch.max())],
            "onnx_range": [float(y_onnx.min()), float(y_onnx.max())],
            "passed": max_diff < atol,
        }
        print(
            f"  batch={bs:2d}  max_diff={max_diff:.2e}  "
            f"mean_diff={mean_diff:.2e}  "
            f"range=[{float(y_onnx.min()):.3f}, {float(y_onnx.max()):.3f}]  "
            f"{'PASS' if max_diff < atol else 'FAIL'}"
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Export Task 1 model to ONNX and verify.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="optional path to save the verification summary as JSON",
    )
    args = p.parse_args()

    assert args.checkpoint.exists(), f"checkpoint not found: {args.checkpoint}"

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_model_from_checkpoint(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  architecture: base_channels={cfg['base_channels']}  "
        f"latent_dim={cfg['latent_dim']}  skip_count={cfg['skip_count']}"
    )
    print(f"  parameters:   {n_params:,}")
    print(f"  ranking_score stored in checkpoint: {cfg.get('ranking_score', 'n/a')}")

    print(f"\nExporting to ONNX (opset {args.opset}) -> {args.output}")
    export_onnx(model, args.output, opset=args.opset)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"  wrote {size_mb:.1f} MB")

    print("\nVerifying PyTorch vs ONNX consistency:")
    results = verify_onnx(model, args.output, batch_sizes=(1, 4, 8))

    all_passed = all(r["passed"] for r in results.values())
    print("\n" + ("=" * 60))
    print("VERIFICATION " + ("PASSED" if all_passed else "FAILED"))
    print("=" * 60)

    if args.report_json:
        summary = {
            "checkpoint": str(args.checkpoint),
            "onnx_path": str(args.output),
            "opset": args.opset,
            "n_parameters": n_params,
            "onnx_size_mb": round(size_mb, 2),
            "cfg": {
                "base_channels": cfg["base_channels"],
                "latent_dim": cfg["latent_dim"],
                "skip_count": cfg["skip_count"],
                "dropout": cfg["dropout"],
                "alpha": cfg.get("alpha"),
            },
            "verification": results,
            "all_passed": all_passed,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(summary, indent=2))
        print(f"\nReport written to {args.report_json}")


if __name__ == "__main__":
    main()
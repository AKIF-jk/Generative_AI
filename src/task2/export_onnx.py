"""ONNX export for Task 2: classifier + three specialist autoencoders.

Exports four models to ONNX and verifies that their outputs match the
PyTorch originals within a tight tolerance.

Usage
-----
    python -m task2.export_onnx \
        --classifier-ckpt output/task2/classifier/best_classifier.pt \
        --specialist-salt-ckpt output/task2/specialists/salt_pepper/best_salt_pepper.pt \
        --specialist-blur-ckpt output/task2/specialists/gaussian_blur/best_gaussian_blur.pt \
        --specialist-occlusion-ckpt output/task2/specialists/occlusion/best_occlusion.pt \
        --output-dir output/task2/onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from task2.router import load_classifier, load_specialist


def export_and_verify(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    out_path: Path,
    model_name: str,
    atol: float = 1e-4,
) -> None:
    import onnx
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        torch_out = model(dummy_input).cpu().numpy()

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(str(out_path))
    onnx_out = sess.run(None, {"input": dummy_input.cpu().numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    status = "OK" if max_diff < atol else "FAIL"
    print(f"  {model_name}: max_diff={max_diff:.2e}  [{status}]")

    if max_diff >= atol:
        raise ValueError(f"ONNX verification failed for {model_name}: max_diff={max_diff:.2e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export Task 2 models to ONNX")
    p.add_argument("--classifier-ckpt", type=Path, required=True)
    p.add_argument("--specialist-salt-ckpt", type=Path, required=True)
    p.add_argument("--specialist-blur-ckpt", type=Path, required=True)
    p.add_argument("--specialist-occlusion-ckpt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("output/task2/onnx"))
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"  # export on CPU for maximum portability

    dummy = torch.randn(args.batch_size, 3, 128, 128, device=device)

    print("Exporting classifier...")
    classifier = load_classifier(args.classifier_ckpt, device)
    export_and_verify(classifier, dummy, args.output_dir / "classifier.onnx", "classifier")

    print("Exporting salt_pepper specialist...")
    sp_salt = load_specialist(args.specialist_salt_ckpt, device)
    export_and_verify(sp_salt, dummy, args.output_dir / "specialist_salt_pepper.onnx", "salt_pepper_specialist")

    print("Exporting gaussian_blur specialist...")
    sp_blur = load_specialist(args.specialist_blur_ckpt, device)
    export_and_verify(sp_blur, dummy, args.output_dir / "specialist_gaussian_blur.onnx", "gaussian_blur_specialist")

    print("Exporting occlusion specialist...")
    sp_occ = load_specialist(args.specialist_occlusion_ckpt, device)
    export_and_verify(sp_occ, dummy, args.output_dir / "specialist_occlusion.onnx", "occlusion_specialist")

    print(f"\nAll four ONNX models saved to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Hard-routed restoration inference pipeline for Task 2.

Loads:
  - CorruptionClassifier  (classifier checkpoint)
  - Three specialist UniversalRestorationAutoencoders (one per corruption type)

Provides:
  - ``HardRouter``: wraps classifier + specialists in one nn.Module
  - ``route_batch``: run hard routing on a batch, return predictions + metadata
  - ``evaluate_hard_routing``: evaluate on test manifest in oracle or predicted mode
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pet_restoration.dataset import CORRUPTION_LABELS
from pet_restoration.models import UniversalRestorationAutoencoder
from task2.classifier import CorruptionClassifier

# Reverse mapping: int label → corruption name
LABEL_TO_NAME: dict[int, str] = {v: k for k, v in CORRUPTION_LABELS.items()}

# Corruption types handled by specialist (not identity)
SPECIALIST_TYPES = ("salt_pepper", "gaussian_blur", "occlusion")


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_classifier(ckpt_path: Path, device: str) -> CorruptionClassifier:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = CorruptionClassifier(
        base_channels=cfg["base_channels"],
        dropout=0.0,  # no dropout at inference
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


def load_specialist(ckpt_path: Path, device: str) -> UniversalRestorationAutoencoder:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = UniversalRestorationAutoencoder(
        base_channels=cfg["base_channels"],
        latent_dim=cfg["latent_dim"],
        dropout=0.0,
        skip_count=cfg["skip_count"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# HardRouter
# ---------------------------------------------------------------------------

class HardRouter(torch.nn.Module):
    """Classifier + three specialist experts in one module.

    Forward receives:
        x          : [B, 3, 128, 128] corrupted input
        routing    : 'predicted' | 'oracle'
        true_labels: [B] ground-truth labels (required for oracle routing)

    Returns:
        reconstructed : [B, 3, 128, 128]
        predicted_cls : [B] int tensor of classifier predictions
        probs         : [B, 4] softmax probabilities
    """

    def __init__(
        self,
        classifier: CorruptionClassifier,
        specialist_salt: UniversalRestorationAutoencoder,
        specialist_blur: UniversalRestorationAutoencoder,
        specialist_occlusion: UniversalRestorationAutoencoder,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.specialist_salt = specialist_salt
        self.specialist_blur = specialist_blur
        self.specialist_occlusion = specialist_occlusion

        # Map label index → specialist module
        self._specialists: dict[int, torch.nn.Module] = {
            CORRUPTION_LABELS["salt_pepper"]: specialist_salt,
            CORRUPTION_LABELS["gaussian_blur"]: specialist_blur,
            CORRUPTION_LABELS["occlusion"]: specialist_occlusion,
        }
        self._clean_label = CORRUPTION_LABELS["clean"]

    def forward(
        self,
        x: Tensor,
        routing: Literal["predicted", "oracle"] = "predicted",
        true_labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        import torch.nn.functional as F

        with torch.no_grad():
            logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        predicted_cls = logits.argmax(dim=1)

        if routing == "oracle":
            if true_labels is None:
                raise ValueError("true_labels required for oracle routing")
            route_labels = true_labels.to(x.device)
        else:
            route_labels = predicted_cls

        # Build output tensor — identity for clean, specialist otherwise
        out = x.clone()
        for label_idx, specialist in self._specialists.items():
            mask = route_labels == label_idx
            if mask.any():
                with torch.no_grad():
                    restored = specialist(x[mask])
                out[mask] = restored
        # Clean (label 0) → identity, already set by clone

        return out, predicted_cls, probs


# ---------------------------------------------------------------------------
# Batch routing helper (single image or batch, returns rich dict)
# ---------------------------------------------------------------------------

def route_batch(
    router: HardRouter,
    corrupted: Tensor,
    routing: Literal["predicted", "oracle"] = "predicted",
    true_labels: Tensor | None = None,
) -> dict:
    """Route a batch and return a rich metadata dict for logging / UI."""
    t0 = time.perf_counter()
    restored, predicted_cls, probs = router(corrupted, routing=routing, true_labels=true_labels)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    predicted_names = [LABEL_TO_NAME[int(c)] for c in predicted_cls.cpu()]
    true_names = (
        [LABEL_TO_NAME[int(l)] for l in true_labels.cpu()]
        if true_labels is not None else None
    )

    return {
        "restored": restored,
        "predicted_class_idx": predicted_cls.cpu(),
        "predicted_class_name": predicted_names,
        "true_class_name": true_names,
        "probabilities": probs.cpu(),
        "inference_ms": elapsed_ms,
    }

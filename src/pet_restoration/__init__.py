"""Shared Oxford-IIIT Pet restoration data pipeline for Tasks 1--3."""

__all__ = [
    "CORRUPTION_LABELS",
    "PetRestorationDataset",
    "UniversalRestorationAutoencoder",
    "LABEL_ORDER",
    "combined_loss",
    "l1_loss",
    "ssim_loss",
    "differentiable_ssim",
    "log_alpha_diagnostics",
    "sweep_alpha_per_corruption",
]


def __getattr__(name: str):
    # Keep each optional component importable independently.
    if name in {"CORRUPTION_LABELS", "PetRestorationDataset"}:
        from .dataset import CORRUPTION_LABELS, PetRestorationDataset

        return {"CORRUPTION_LABELS": CORRUPTION_LABELS, "PetRestorationDataset": PetRestorationDataset}[name]
    if name == "UniversalRestorationAutoencoder":
        from .models import UniversalRestorationAutoencoder

        return UniversalRestorationAutoencoder
    if name in {
        "LABEL_ORDER",
        "combined_loss",
        "l1_loss",
        "ssim_loss",
        "differentiable_ssim",
        "log_alpha_diagnostics",
        "sweep_alpha_per_corruption",
    }:
        from .losses import (
            LABEL_ORDER,
            combined_loss,
            differentiable_ssim,
            l1_loss,
            log_alpha_diagnostics,
            ssim_loss,
            sweep_alpha_per_corruption,
        )

        return {
            "LABEL_ORDER": LABEL_ORDER,
            "combined_loss": combined_loss,
            "l1_loss": l1_loss,
            "ssim_loss": ssim_loss,
            "differentiable_ssim": differentiable_ssim,
            "log_alpha_diagnostics": log_alpha_diagnostics,
            "sweep_alpha_per_corruption": sweep_alpha_per_corruption,
        }[name]
    raise AttributeError(name)

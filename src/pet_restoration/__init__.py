"""Shared Oxford-IIIT Pet restoration data pipeline for Tasks 1--3."""

__all__ = ["CORRUPTION_LABELS", "PetRestorationDataset", "UniversalRestorationAutoencoder"]


def __getattr__(name: str):
    # Keep each optional component importable independently.
    if name in {"CORRUPTION_LABELS", "PetRestorationDataset"}:
        from .dataset import CORRUPTION_LABELS, PetRestorationDataset

        return {"CORRUPTION_LABELS": CORRUPTION_LABELS, "PetRestorationDataset": PetRestorationDataset}[name]
    if name == "UniversalRestorationAutoencoder":
        from .models import UniversalRestorationAutoencoder

        return UniversalRestorationAutoencoder
    raise AttributeError(name)

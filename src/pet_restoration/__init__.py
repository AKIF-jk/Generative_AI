"""Shared Oxford-IIIT Pet restoration data pipeline for Tasks 1--3."""

__all__ = ["CORRUPTION_LABELS", "PetRestorationDataset"]


def __getattr__(name: str):
    # Keep manifest generation usable before the optional ML stack is installed.
    if name in __all__:
        from .dataset import CORRUPTION_LABELS, PetRestorationDataset

        return {"CORRUPTION_LABELS": CORRUPTION_LABELS, "PetRestorationDataset": PetRestorationDataset}[name]
    raise AttributeError(name)

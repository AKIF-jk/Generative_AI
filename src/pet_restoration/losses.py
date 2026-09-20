"""Loss functions for image restoration.

Implements the combined L1 + SSIM loss from Zhao et al. [11]:

    L = α · L1 + (1 − α) · (1 − SSIM)

α is a tunable hyperparameter (Optuna search range: continuous), not fixed at 0.8.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .dataset import CORRUPTION_LABELS as _LABEL_TO_IDX

# Ordered label names, consistent with the canonical dict in dataset.py.
LABEL_ORDER: tuple[str, ...] = tuple(sorted(_LABEL_TO_IDX, key=_LABEL_TO_IDX.get))


def _pad_reflect(x: Tensor, pad: int) -> Tensor:
    """Reflect-pad a 4-D tensor on spatial dimensions."""
    return F.pad(x, (pad, pad, pad, pad), mode="reflect")


def _ssim_channel(
    x: Tensor,
    y: Tensor,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> Tensor:
    """Compute per-pixel SSIM for a single channel over local patches.

    Uses uniform 11×11 averaging — a standard differentiable approximation of
    Wang et al.'s Gaussian window (σ = 1.5).  Reflect padding avoids the border
    bias that zero-padding would introduce.
    """
    x = _pad_reflect(x, 5)
    y = _pad_reflect(y, 5)

    mu_x = F.avg_pool2d(x, kernel_size=11, stride=1)
    mu_y = F.avg_pool2d(y, kernel_size=11, stride=1)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x * x, kernel_size=11, stride=1) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y * y, kernel_size=11, stride=1) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, kernel_size=11, stride=1) - mu_xy

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim_map


def differentiable_ssim(x: Tensor, y: Tensor) -> Tensor:
    """Mean structural similarity over the batch (differentiable).

    Expects *x*, *y* with shape ``[B, C, H, W]`` in ``[0, 1]``.
    Returns a scalar tensor in the range ``[−1, 1]`` (1 = identical).
    """
    if x.ndim != 4 or x.shape != y.shape:
        raise ValueError(f"x and y must have identical 4-D shape, got {x.shape} vs {y.shape}")
    channels = x.shape[1]
    ssim_sum = _ssim_channel(x[:, 0:1], y[:, 0:1])
    for c in range(1, channels):
        ssim_sum = ssim_sum + _ssim_channel(x[:, c : c + 1], y[:, c : c + 1])
    return ssim_sum.mean()


def combined_loss(
    prediction: Tensor,
    target: Tensor,
    alpha: float = 0.8,
) -> Tensor:
    """L = α · L1 + (1 − α) · (1 − SSIM).

    ``alpha`` controls the pixel-level vs. structural balance.
    Returns a scalar tensor.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    l1 = F.l1_loss(prediction, target)
    ssim_val = differentiable_ssim(prediction, target)
    return alpha * l1 + (1.0 - alpha) * (1.0 - ssim_val)


def l1_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Pixel-wise L1 loss (scalar)."""
    return F.l1_loss(prediction, target)


def ssim_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """1 − SSIM (scalar)."""
    return 1.0 - differentiable_ssim(prediction, target)


# ---------------------------------------------------------------------------
# α-by-corruption-type logging helpers
# ---------------------------------------------------------------------------

def _per_corruption_metrics(
    prediction: Tensor,
    target: Tensor,
    corruption_label: Tensor,
) -> dict[str, dict[str, float]]:
    """Compute per-corruption L1 and (1−SSIM), keyed by label name."""
    device = prediction.device
    labels = corruption_label.to(device)
    result: dict[str, dict[str, float]] = {}
    for name in LABEL_ORDER:
        idx = _LABEL_TO_IDX[name]
        mask = labels == idx
        if mask.any():
            pred_c = prediction[mask]
            tgt_c = target[mask]
            result[name] = {
                "l1": l1_loss(pred_c, tgt_c).item(),
                "ssim_loss": ssim_loss(pred_c, tgt_c).item(),
            }
        else:
            result[name] = {"l1": 0.0, "ssim_loss": 0.0}
    return result


def log_alpha_diagnostics(
    prediction: Tensor,
    target: Tensor,
    corruption_label: Tensor,
    alpha: float,
    *,
    return_per_sample: bool = False,
) -> dict[str, float] | list[dict[str, float]]:
    """Compute per-corruption L1 and (1−SSIM) for α analysis logging.

    Parameters
    ----------
    prediction : Tensor [B, 3, 128, 128]
    target : Tensor [B, 3, 128, 128]
    corruption_label : Tensor [B] with integer labels (0–3)
    alpha : float — the α used for this trial
    return_per_sample : bool — if True, return a list of per-sample dicts;
        otherwise return a single dict with batch means.

    Returns
    -------
    dict or list[dict] mapping ``"{label}_l1"`` and ``"{label}_ssim_loss"``
    to their mean (or per-sample) values, plus ``"alpha"`` itself.
    """
    if return_per_sample:
        B = prediction.shape[0]
        samples: list[dict[str, float]] = []
        for i in range(B):
            sample_metrics: dict[str, float] = {"alpha": alpha}
            for name in LABEL_ORDER:
                idx = _LABEL_TO_IDX[name]
                if corruption_label[i] == idx:
                    pred_i = prediction[i : i + 1]
                    tgt_i = target[i : i + 1]
                    sample_metrics[f"{name}_l1"] = l1_loss(pred_i, tgt_i).item()
                    sample_metrics[f"{name}_ssim_loss"] = ssim_loss(pred_i, tgt_i).item()
                else:
                    sample_metrics[f"{name}_l1"] = 0.0
                    sample_metrics[f"{name}_ssim_loss"] = 0.0
            samples.append(sample_metrics)
        return samples

    metrics: dict[str, float] = {"alpha": alpha}
    agg = _per_corruption_metrics(prediction, target, corruption_label)
    for name in LABEL_ORDER:
        metrics[f"{name}_l1"] = agg[name]["l1"]
        metrics[f"{name}_ssim_loss"] = agg[name]["ssim_loss"]
    return metrics


def sweep_alpha_per_corruption(
    prediction: Tensor,
    target: Tensor,
    corruption_label: Tensor,
    alphas: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> dict[str, tuple[float, float]]:
    """Sweep α values and report which minimises combined loss per corruption.

    Operates on a **single batch**.  For full validation-set results, iterate
    over the validation DataLoader and aggregate per-corruption losses in the
    caller before comparing.

    Returns a dict mapping corruption label name to ``(best_alpha, best_loss)``.
    The boundary values α=0.0 (pure SSIM) and α=1.0 (pure L1) establish whether
    the combination actually helps — key evidence for the "why combine?" section
    of the report.
    """
    results: dict[str, tuple[float, float]] = {}
    for name in LABEL_ORDER:
        idx = _LABEL_TO_IDX[name]
        mask = corruption_label == idx
        if not mask.any():
            continue
        pred_c = prediction[mask]
        tgt_c = target[mask]
        best_alpha, best_loss = alphas[0], float("inf")
        for a in alphas:
            loss = combined_loss(pred_c, tgt_c, alpha=a).item()
            if loss < best_loss:
                best_alpha, best_loss = a, loss
        results[name] = (best_alpha, best_loss)
    return results

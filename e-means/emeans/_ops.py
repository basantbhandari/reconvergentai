"""Shared low-level torch ops: device selection, distances, assignment, revival."""
from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "default_device",
    "sq_dists",
    "assign",
    "quantization_error",
    "revive_dead_clusters",
]

_TINY = 1e-10


def default_device(device: Optional[str] = None) -> torch.device:
    """MPS > CUDA > CPU."""
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sq_dists(X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """||a−b||² = ||a||² + ||b||² − 2aᵀb. Returns (n, k), clamped ≥ 0."""
    return (
        (X * X).sum(1, keepdim=True)
        + (C * C).sum(1).unsqueeze(0)
        - 2.0 * (X @ C.T)
    ).clamp_min_(0.0)


def assign(X: torch.Tensor, C: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """
    Nearest-centroid label per row, in row-chunks to bound peak memory.
    Peak alloc = chunk × k × 4 bytes — never materialises the full n×k matrix.
    """
    if len(X) <= chunk:
        return sq_dists(X, C).argmin(1)
    labels = torch.empty(len(X), dtype=torch.long, device=X.device)
    for s in range(0, len(X), chunk):
        labels[s : s + chunk] = sq_dists(X[s : s + chunk], C).argmin(1)
    return labels


def quantization_error(X: torch.Tensor, C: torch.Tensor, chunk: int = 1024) -> float:
    """(1/n) Σ_i min_j ||x_i − c_j||²."""
    total = X.new_zeros(())
    for s in range(0, len(X), chunk):
        total += sq_dists(X[s : s + chunk], C).min(1).values.sum()
    return (total / len(X)).item()


@torch.no_grad()
def revive_dead_clusters(
    N: torch.Tensor,   # (k,)   EMA counts  — modified in-place
    S: torch.Tensor,   # (k, d) EMA sums    — modified in-place
    tau: float,
    g: torch.Generator,
) -> None:
    """
    Split large live clusters into dead ones (in-place).

    Dead := N_j < τ · mean(N).
    Donor sampled ∝ N_j with replacement; each split perturbs the centroid
    ±1/128 along alternating dims and assigns half the EMA weight to each child.
    """
    k, d   = S.shape
    mean_N = N.mean()
    if mean_N <= 0.0:
        return

    dead = (N < tau * mean_N).nonzero(as_tuple=False).flatten()
    if dead.numel() == 0:
        return
    live = (N >= tau * mean_N).nonzero(as_tuple=False).flatten()
    if live.numel() == 0:
        return

    donors = live[torch.multinomial(N[live].clamp_min_(0.0), dead.numel(), replacement=True, generator=g)]
    signs  = torch.where(torch.arange(d, device=S.device) % 2 == 0, S.new_ones(d), -S.new_ones(d))

    for dead_i, donor_i in zip(dead.tolist(), donors.tolist()):
        N_d = N[donor_i].item()
        if N_d <= 0.0:
            continue
        center        = S[donor_i] / N_d
        half          = N_d * 0.5
        N[donor_i]    = half
        S[donor_i]    = center * (1.0 - signs / 128.0) * half
        N[dead_i]     = half
        S[dead_i]     = center * (1.0 + signs / 128.0) * half

"""Lloyd's K-Means: standard full-batch EM iterations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from ._ops import assign, default_device, quantization_error, revive_dead_clusters

__all__ = ["KMeans"]


@dataclass
class KMeans:
    """
    Lloyd's K-Means clustering.

    Parameters
    ----------
    n_clusters : int
    epochs : int
    seed : int
    revival_tau : float
        Clusters with N_j < tau * mean(N) are revived. Default 0.1.
    device : optional str
    dist_chunk : int
        Row-chunk size for distance matrix. None → auto (≤64 MB per chunk).
    """

    n_clusters: int
    epochs: int
    seed: int = 0
    revival_tau: float = 0.1
    device: Optional[str] = None
    dist_chunk: Optional[int] = None

    cluster_centers_: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    history_: List[float] = field(default_factory=list, init=False, repr=False)

    def fit(self, X: torch.Tensor) -> "KMeans":
        dev = default_device(self.device)
        X   = X.to(dev, dtype=torch.float32)
        n, d = X.shape
        k    = self.n_clusters

        chunk = self.dist_chunk or max(64, (64 * 1024 * 1024) // (k * 4))

        g = torch.Generator(device=dev)
        g.manual_seed(self.seed)

        idx = torch.randperm(n, generator=g, device=dev)[:k]
        C   = X[idx].clone()

        self.history_.clear()

        for _ in range(self.epochs):
            labels = assign(X, C, chunk)

            N = torch.bincount(labels, minlength=k).float()
            S = X.new_zeros(k, d).index_add_(0, labels, X)

            nonempty = N > 0

            dead = (~nonempty).nonzero(as_tuple=False).flatten()
            if dead.numel() > 0:
                revive_dead_clusters(N, S, self.revival_tau, g)

            C = torch.where(
                nonempty.unsqueeze(1),
                S / N.clamp_min(1.0).unsqueeze(1),
                C,
            )

            self.history_.append(quantization_error(X, C, chunk))

        self.cluster_centers_ = C
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self.cluster_centers_ is not None, "Call fit() first."
        dev   = self.cluster_centers_.device
        chunk = self.dist_chunk or max(64, (64 * 1024 * 1024) // (self.n_clusters * 4))
        return assign(X.to(dev, dtype=torch.float32), self.cluster_centers_, chunk)

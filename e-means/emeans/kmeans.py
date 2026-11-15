"""Lloyd's K-Means: standard full-batch EM iterations."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from ._ops import assign, default_device, quantization_error, revive_dead_clusters

__all__ = ["KMeans"]

log = logging.getLogger(__name__)


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

        log.info("fit start  n=%d  d=%d  k=%d  epochs=%d  chunk=%d  device=%s",
                 n, d, k, self.epochs, chunk, dev)

        g = torch.Generator(device=dev)
        g.manual_seed(self.seed)

        idx = torch.randperm(n, generator=g, device=dev)[:k]
        C   = X[idx].clone()

        self.history_.clear()
        t_start = time.perf_counter()

        for epoch in range(self.epochs):
            t_epoch = time.perf_counter()
            labels  = assign(X, C, chunk)

            N = torch.bincount(labels, minlength=k).float()
            S = X.new_zeros(k, d).index_add_(0, labels, X)

            nonempty = N > 0
            if (~nonempty).any():
                revive_dead_clusters(N, S, self.revival_tau, g)

            C = torch.where(
                nonempty.unsqueeze(1),
                S / N.clamp_min(1.0).unsqueeze(1),
                C,
            )

            mse = quantization_error(X, C, chunk)
            self.history_.append(mse)

            log.info("epoch %2d/%d  MSE=%.6f  elapsed=%.1fs",
                     epoch + 1, self.epochs, mse, time.perf_counter() - t_epoch)

        self.cluster_centers_ = C
        log.info("fit done  total=%.1fs  final_MSE=%.6f", time.perf_counter() - t_start, self.history_[-1])
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self.cluster_centers_ is not None, "Call fit() first."
        dev   = self.cluster_centers_.device
        chunk = self.dist_chunk or max(64, (64 * 1024 * 1024) // (self.n_clusters * 4))
        return assign(X.to(dev, dtype=torch.float32), self.cluster_centers_, chunk)

"""E-Means: mini-batch K-Means with EMA statistics and cosine-annealed step size."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .config import auto_alpha0, auto_batch_size
from ._ops import assign, default_device, quantization_error, revive_dead_clusters

__all__ = ["EMeans"]

log = logging.getLogger(__name__)

_TINY = 1e-10


@dataclass
class EMeans:
    """
    E-Means clustering (arXiv:2611.09501).

    Parameters
    ----------
    n_clusters : int
    epochs : int
    batch_size : optional int
        Mini-batch size. None → scaling law from paper Table 2.
    alpha0 : optional float
        Initial EMA step size. None → scaling law.
    seed : int
    revival_tau : float
        Clusters with N_j < tau * mean(N) are revived. Default 0.1.
    revival_every : int
        Dead-cluster check period (in mini-batches). Default 5.
    eps_schedule : float
        Cosine annealing floor: α_min = ε · α₀. Default 0.1.
    finalize : bool
        Replace EMA centroids with last-epoch batch means. Default True.
    device : optional str
    dist_chunk : int
        Row-chunk size for distance matrix. None → auto (≤64 MB per chunk).
    """

    n_clusters: int
    epochs: int
    batch_size: Optional[int] = None
    alpha0: Optional[float] = None
    seed: int = 0
    revival_tau: float = 0.1
    revival_every: int = 5
    eps_schedule: float = 0.1
    finalize: bool = True
    device: Optional[str] = None
    dist_chunk: Optional[int] = None

    cluster_centers_: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    history_: List[float] = field(default_factory=list, init=False, repr=False)
    batch_size_: Optional[int] = field(default=None, init=False, repr=False)
    alpha0_: Optional[float] = field(default=None, init=False, repr=False)

    def fit(self, X: torch.Tensor) -> "EMeans":
        dev = default_device(self.device)
        X   = X.to(dev, dtype=torch.float32)
        n, d = X.shape
        k    = self.n_clusters

        B  = self.batch_size or auto_batch_size(n, k, self.epochs)
        a0 = self.alpha0     or auto_alpha0(n, k, B, self.epochs)
        self.batch_size_ = B
        self.alpha0_     = a0

        chunk = self.dist_chunk or max(64, (64 * 1024 * 1024) // (k * 4))

        log.info("fit start  n=%d  d=%d  k=%d  epochs=%d  B=%d  α₀=%.4f  chunk=%d  device=%s",
                 n, d, k, self.epochs, B, a0, chunk, dev)

        g = torch.Generator(device=dev)
        g.manual_seed(self.seed)

        idx = torch.randperm(n, generator=g, device=dev)[:k]
        C0  = X[idx].clone()

        N = X.new_full((k,), _TINY)
        S = N.unsqueeze(1) * C0

        eps      = self.eps_schedule
        M        = self.epochs * math.ceil(n / B)
        step     = 0
        t_start  = time.perf_counter()

        self.history_.clear()
        last_epoch = self.epochs - 1

        for epoch in range(self.epochs):
            t_epoch = time.perf_counter()
            perm    = torch.randperm(n, generator=g, device=dev)
            is_last = epoch == last_epoch

            if is_last and self.finalize:
                fin_S = X.new_zeros(k, d)
                fin_N = X.new_zeros(k)

            for start in range(0, n, B):
                batch  = X[perm[start : start + B]]
                labels = assign(batch, S / N.clamp_min(_TINY).unsqueeze(1), chunk)

                n_j = torch.bincount(labels, minlength=k).float()
                s_j = X.new_zeros(k, d).index_add_(0, labels, batch)

                alpha = a0 * (eps + (1.0 - eps) * 0.5 * (1.0 + math.cos(math.pi * step / M)))
                gamma = 1.0 - alpha

                N.mul_(gamma).add_(n_j, alpha=alpha)
                S.mul_(gamma).add_(s_j, alpha=alpha)

                step += 1
                if step % self.revival_every == 0:
                    revive_dead_clusters(N, S, self.revival_tau, g)

                if is_last and self.finalize:
                    fin_N.add_(n_j)
                    fin_S.index_add_(0, labels, batch)

            with torch.no_grad():
                C_now = S / N.clamp_min(_TINY).unsqueeze(1)
            mse = quantization_error(X, C_now, chunk)
            self.history_.append(mse)

            log.info("epoch %2d/%d  MSE=%.6f  elapsed=%.1fs",
                     epoch + 1, self.epochs, mse, time.perf_counter() - t_epoch)

        if self.finalize:
            nonempty = fin_N > 0
            C_final  = S / N.clamp_min(_TINY).unsqueeze(1)
            C_final[nonempty] = fin_S[nonempty] / fin_N[nonempty].unsqueeze(1)
            self.cluster_centers_ = C_final
        else:
            self.cluster_centers_ = S / N.clamp_min(_TINY).unsqueeze(1)

        log.info("fit done  total=%.1fs  final_MSE=%.6f", time.perf_counter() - t_start, self.history_[-1])
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self.cluster_centers_ is not None, "Call fit() first."
        dev   = self.cluster_centers_.device
        chunk = self.dist_chunk or max(64, (64 * 1024 * 1024) // (self.n_clusters * 4))
        return assign(X.to(dev, dtype=torch.float32), self.cluster_centers_, chunk)

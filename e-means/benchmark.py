"""
Compare E-Means vs Lloyd's K-Means on the DEEP1M dataset.

Usage:
    python benchmark.py [--k K] [--em-epochs EM_EPOCHS] [--km-epochs KM_EPOCHS] [--n N]
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from emeans import EMeans, KMeans, load_deep
from emeans._ops import default_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--k",          type=int, default=1024,  help="number of clusters")
    p.add_argument("--em-epochs",  type=int, default=3,     help="E-Means epochs")
    p.add_argument("--km-epochs",  type=int, default=25,    help="Lloyd K-Means epochs")
    p.add_argument("--n",          type=int, default=None,  help="rows to load (default: all)")
    p.add_argument("--data",       type=str, default="data/deep1m.fbin")
    p.add_argument("--out",        type=str, default="output/benchmark_results.png")
    return p.parse_args()


def run_emeans(X, k, epochs, dev):
    em = EMeans(n_clusters=k, epochs=epochs, device=str(dev))
    t0 = time.perf_counter()
    em.fit(X)
    elapsed = time.perf_counter() - t0
    history = em.history_[:]
    del em
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()
    return history, elapsed


def run_kmeans(X, k, epochs, dev):
    km = KMeans(n_clusters=k, epochs=epochs, device=str(dev))
    t0 = time.perf_counter()
    km.fit(X)
    elapsed = time.perf_counter() - t0
    history = km.history_[:]
    del km
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()
    return history, elapsed


def main():
    args  = parse_args()
    dev   = default_device()
    data_path = Path(args.data)

    print(f"Device  : {dev}")
    print(f"Dataset : {data_path}")
    print(f"k       : {args.k}")
    print(f"Loading data...", flush=True)

    X = load_deep(data_path, n=args.n, device=dev)
    n, d = X.shape
    print(f"Loaded  : {n:,} × {d} float32  ({X.numel()*4/2**20:.1f} MB)")
    print()

    # E-Means
    print(f"Running E-Means  ({args.em_epochs} epochs)...", flush=True)
    em_hist, em_time = run_emeans(X, args.k, args.em_epochs, dev)
    print(f"  Done in {em_time:.1f}s")
    for ep, mse in enumerate(em_hist, 1):
        print(f"  epoch {ep:2d}: MSE = {mse:.6f}")

    # K-Means (Lloyd)
    print(f"\nRunning K-Means  ({args.km_epochs} epochs)...", flush=True)
    km_hist, km_time = run_kmeans(X, args.k, args.km_epochs, dev)
    print(f"  Done in {km_time:.1f}s")
    for ep, mse in enumerate(km_hist, 1):
        print(f"  epoch {ep:2d}: MSE = {mse:.6f}")

    # Summary table
    print("\n" + "=" * 55)
    print(f"{'':30s} {'E-Means':>10s}  {'K-Means':>10s}")
    print("=" * 55)
    print(f"{'epochs':30s} {args.em_epochs:>10d}  {args.km_epochs:>10d}")
    print(f"{'time (s)':30s} {em_time:>10.1f}  {km_time:>10.1f}")
    print(f"{'final MSE':30s} {em_hist[-1]:>10.6f}  {km_hist[-1]:>10.6f}")
    best_em_epoch = min(range(len(em_hist)), key=lambda i: em_hist[i]) + 1
    best_km_epoch = min(range(len(km_hist)), key=lambda i: km_hist[i]) + 1
    print(f"{'best MSE':30s} {min(em_hist):>10.6f}  {min(km_hist):>10.6f}")
    print(f"{'best epoch':30s} {best_em_epoch:>10d}  {best_km_epoch:>10d}")
    print(f"{'speed-up (time ratio)':30s} {km_time/em_time:>10.2f}x")
    print("=" * 55)

    # Convergence plot
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(em_hist) + 1), em_hist, "o-", label=f"E-Means ({args.em_epochs} ep)")
    ax.plot(range(1, len(km_hist) + 1), km_hist, "s--", label=f"K-Means ({args.km_epochs} ep)", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Quantization Error (MSE)")
    ax.set_title(f"E-Means vs K-Means  |  DEEP1M  |  k={args.k}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()

"""
Compare E-Means vs Lloyd's K-Means on the DEEP1M dataset.

Usage:
    python benchmark.py [--k K] [--em-epochs EM_EPOCHS] [--km-epochs KM_EPOCHS] [--n N]
"""
from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from emeans import EMeans, KMeans, load_deep
from emeans._ops import default_device

log = logging.getLogger("benchmark")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--k",          type=int,   default=1024,                       help="number of clusters")
    p.add_argument("--em-epochs",  type=int,   default=3,                          help="E-Means epochs")
    p.add_argument("--km-epochs",  type=int,   default=25,                         help="Lloyd K-Means epochs")
    p.add_argument("--n",          type=int,   default=None,                       help="rows to load (default: all)")
    p.add_argument("--data",       type=str,   default="data/deep1m.fbin")
    p.add_argument("--out",        type=str,   default="output/benchmark_results.png")
    p.add_argument("--log-level",  type=str,   default="INFO",                     help="DEBUG|INFO|WARNING")
    return p.parse_args()


def run_emeans(X, k, epochs, dev):
    log.info("--- E-Means  k=%d  epochs=%d ---", k, epochs)
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
    log.info("E-Means finished  time=%.1fs  final_MSE=%.6f", elapsed, history[-1])
    return history, elapsed


def run_kmeans(X, k, epochs, dev):
    log.info("--- K-Means  k=%d  epochs=%d ---", k, epochs)
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
    log.info("K-Means finished  time=%.1fs  final_MSE=%.6f", elapsed, history[-1])
    return history, elapsed


def main():
    args = parse_args()
    setup_logging(args.log_level)

    dev       = default_device()
    data_path = Path(args.data)

    log.info("device=%s  dataset=%s  k=%d", dev, data_path, args.k)
    log.info("loading data...")
    X    = load_deep(data_path, n=args.n, device=dev)
    n, d = X.shape
    log.info("loaded  n=%d  d=%d  size=%.1f MB", n, d, X.numel() * 4 / 2 ** 20)

    em_hist, em_time = run_emeans(X, args.k, args.em_epochs, dev)
    km_hist, km_time = run_kmeans(X, args.k, args.km_epochs, dev)

    best_em = min(range(len(em_hist)), key=lambda i: em_hist[i]) + 1
    best_km = min(range(len(km_hist)), key=lambda i: km_hist[i]) + 1

    sep = "=" * 55
    log.info(sep)
    log.info("%-30s %10s  %10s", "", "E-Means", "K-Means")
    log.info(sep)
    log.info("%-30s %10d  %10d", "epochs",      args.em_epochs,   args.km_epochs)
    log.info("%-30s %10.1f  %10.1f", "time (s)",   em_time,          km_time)
    log.info("%-30s %10.6f  %10.6f", "final MSE",  em_hist[-1],      km_hist[-1])
    log.info("%-30s %10.6f  %10.6f", "best MSE",   min(em_hist),     min(km_hist))
    log.info("%-30s %10d  %10d", "best epoch",  best_em,          best_km)
    log.info("%-30s %10.2fx", "speed-up",    km_time / em_time)
    log.info(sep)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(em_hist) + 1), em_hist, "o-",  label=f"E-Means ({args.em_epochs} ep)")
    ax.plot(range(1, len(km_hist) + 1), km_hist, "s--", label=f"K-Means ({args.km_epochs} ep)", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Quantization Error (MSE)")
    ax.set_title(f"E-Means vs K-Means  |  DEEP1M  |  k={args.k}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    log.info("plot saved → %s", args.out)


if __name__ == "__main__":
    main()

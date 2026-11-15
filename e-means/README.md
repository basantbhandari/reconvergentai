# E-Means

PyTorch reproduction of **E-Means: Fast Lloyd's Algorithm via Exponential Moving Averages**  
([arXiv:2611.09501](https://arxiv.org/abs/2611.09501) · [project page](https://e-means.github.io/))

## What is E-Means?

E-Means is a mini-batch K-Means variant that replaces the standard centroid update with an **exponential moving average (EMA)** over batch statistics, combined with a **cosine-annealed step size**. It converges to the same solution quality as Lloyd's algorithm in a fraction of the epochs and wall-clock time.

**Core update rule** (per mini-batch):

```
N_j ← (1−α)·N_j + α·n_j        # EMA count
S_j ← (1−α)·S_j + α·s_j        # EMA sum
c_j  = S_j / N_j                # centroid
```

**Step size schedule** (cosine annealing):

```
α(t) = α₀ · (ε + (1−ε) · ½ · (1 + cos(π·t/M)))
```

where `ε = 0.1`, `M` = total mini-batch steps, and `α₀` is derived from scaling laws fitted in the paper.

## Results on DEEP1M (k=1024)

| | E-Means (3 epochs) | K-Means (25 epochs) |
|---|---|---|
| Time | 8.2 s | 56.8 s |
| Final MSE | 0.4632 | 0.4639 |
| Speed-up | **6.9×** | — |

E-Means reaches lower quantization error in 3 epochs than Lloyd's algorithm does in 25.

## Project Structure

```
e-means/
├── emeans/
│   ├── __init__.py      public API
│   ├── config.py        scaling law coefficients (Table 2 of paper)
│   ├── _ops.py          low-level torch primitives
│   ├── data.py          dataset loader (.fbin / .npy)
│   ├── emeans.py        EMeans class
│   └── kmeans.py        KMeans class (Lloyd's algorithm)
├── output/              benchmark plots (gitignored)
├── benchmark.py         compare both algorithms on DEEP1M
└── pyproject.toml
```

## Quick Start

```python
from emeans import EMeans, KMeans, load_deep

X = load_deep("data/deep1m.fbin")   # (1_000_000, 96) float32

em = EMeans(n_clusters=1024, epochs=3).fit(X)
km = KMeans(n_clusters=1024, epochs=25).fit(X)

print(em.history_)   # MSE per epoch
print(km.history_)
```

Hyperparameters (`batch_size`, `alpha0`) are auto-derived from the paper's scaling laws when not specified.

## Running the Benchmark

Download the DEEP1M dataset first (1M × 96 float32 vectors, ~366 MB):

```bash
mkdir -p data
curl -r 0-384000007 "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin" \
     -o data/deep1m.fbin
```

Then run:

```bash
uv run python benchmark.py --k 1024 --em-epochs 3 --km-epochs 25
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--k` | 1024 | Number of clusters |
| `--em-epochs` | 3 | E-Means epochs |
| `--km-epochs` | 25 | K-Means epochs |
| `--n` | all | Rows to load |
| `--data` | `data/deep1m.fbin` | Dataset path |
| `--out` | `output/benchmark_results.png` | Plot output path |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |

## Logging

All components use Python's standard `logging` module under the `emeans.*` hierarchy. Each log line includes a timestamp, logger name, level, and message:

```
14:49:30  benchmark             INFO      device=mps  dataset=data/deep1m.fbin  k=1024
14:49:30  emeans.emeans         INFO      fit start  n=1000000  d=96  k=1024  epochs=3  B=32768  α₀=0.6960
14:49:31  emeans.emeans         INFO      epoch  1/3  MSE=0.469409  elapsed=1.1s
14:49:33  emeans.emeans         INFO      fit done  total=3.2s  final_MSE=0.463221
```

**Log levels:**

| Level | What you see |
|---|---|
| `WARNING` | Errors and warnings only |
| `INFO` (default) | Fit config, per-epoch MSE, timing, summary table |
| `DEBUG` | All of the above plus any fine-grained internal events |

To suppress per-epoch output and only see the summary:

```bash
uv run python benchmark.py --log-level WARNING
```

To integrate logging in your own code:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from emeans import EMeans
em = EMeans(n_clusters=1024, epochs=3).fit(X)
```

## Installation

Requires Python ≥ 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Key Implementation Details

- **Chunked distance computation** — `assign` processes rows in chunks so the full `n×k` distance matrix is never materialised in memory. Peak allocation is `chunk × k × 4` bytes.
- **Dead cluster revival** — every 5 mini-batches, clusters with `N_j < 0.1 · mean(N)` are revived by splitting a large donor cluster, perturbing the centroid ±1/128 along alternating dimensions.
- **Finalization** — in the last epoch, batch means replace EMA centroids for a sharper final solution.
- **Device selection** — automatically picks MPS (Apple Silicon) > CUDA > CPU.

## Reference

```bibtex
@article{emeans2024,
  title   = {E-Means: Fast Lloyd's Algorithm via Exponential Moving Averages},
  journal = {arXiv:2611.09501},
  year    = {2024}
}
```

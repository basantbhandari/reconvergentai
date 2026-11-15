# Perceptron

Reproduction of the **Rosenblatt Perceptron** (1957) — the original single-layer binary classifier.

## Algorithm

The perceptron learns a linear decision boundary via the online update rule applied to each misclassified sample:

```
ŷ   = step(w·x + b)
δ   = lr · (y − ŷ)
w  ← w + δ · x
b  ← b + δ
```

It is **guaranteed to converge** in finite steps if the data is linearly separable (Perceptron Convergence Theorem).

## Project Structure

```
perceptron/
├── perceptron/
│   ├── __init__.py      public API
│   ├── perceptron.py    Perceptron class
│   └── data.py          logic gates (AND, OR, NAND, NOR, XOR)
├── output/              benchmark plots (gitignored)
├── benchmark.py         compare scratch vs sklearn on gates + real data
└── pyproject.toml
```

## Quick Start

```python
from perceptron import Perceptron, gate

X, y = gate("AND")
p = Perceptron(learning_rate=0.1, epochs=20).fit(X, y)
print(p.score(X, y))      # 1.0
print(p.errors_)          # misclassifications per epoch
```

## Running the Benchmark

```bash
uv run python benchmark.py
```

Output covers:
- All 5 logic gates (AND, OR, NAND, NOR, XOR) — XOR expected to fail (not linearly separable)
- Scratch vs scikit-learn Perceptron on `make_classification` (1000 samples, 10 features)

Options:

| Flag | Default | Description |
|---|---|---|
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING` |
| `--out` | `output/benchmark_results.png` | Plot output path |

## Logging

Uses Python's `logging` module scoped to `perceptron.*` and `benchmark` — third-party libraries are always silent.

| Level | What you see |
|---|---|
| `WARNING` | Errors only |
| `INFO` (default) | Per-epoch misclassifications, accuracy, convergence |
| `DEBUG` | All of the above plus every weight update |

## Installation

Requires Python ≥ 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Linear Separability

| Gate | Separable | Converges |
|---|---|---|
| AND | yes | yes |
| OR | yes | yes |
| NAND | yes | yes |
| NOR | yes | yes |
| XOR | **no** | **no** |

A single perceptron cannot solve XOR — this motivated the development of multi-layer networks.

## Benchmark Results

**Logic Gates** (lr=0.1, max 20 epochs):

| Gate | Accuracy | Epochs to Converge | Separable |
|---|---|---|---|
| AND | 100% | 4 | yes |
| OR | 100% | 4 | yes |
| NAND | 100% | 6 | yes |
| NOR | 100% | 4 | yes |
| XOR | 50% | — (did not converge) | **no** |

**Real Dataset** — `make_classification` (1000 samples, 10 features, 80/20 split, lr=0.1, 50 epochs):

| | Test Accuracy |
|---|---|
| Scratch Perceptron | **87.00%** |
| Sklearn Perceptron | 80.00% |

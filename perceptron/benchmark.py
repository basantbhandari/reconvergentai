"""
Compare scratch Perceptron vs scikit-learn Perceptron on logic gates and a real dataset.

Usage:
    python benchmark.py [--log-level LOG_LEVEL]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from perceptron import Perceptron, GATES, gate

log = logging.getLogger("benchmark")


def setup_logging(level: str) -> None:
    lvl     = getattr(logging, level.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(name)-25s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    for name in ("benchmark", "perceptron"):
        logger = logging.getLogger(name)
        logger.setLevel(lvl)
        logger.addHandler(handler)
        logger.propagate = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log-level", type=str, default="INFO", help="DEBUG|INFO|WARNING")
    p.add_argument("--out",       type=str, default="output/benchmark_results.png")
    return p.parse_args()


def run_gates() -> dict[str, dict]:
    results = {}
    for name, (X, y) in GATES.items():
        p = Perceptron(learning_rate=0.1, epochs=20)
        p.fit(X, y)
        results[name] = {
            "acc":     p.score(X, y),
            "epochs":  len(p.errors_),
            "errors":  p.errors_,
            "weights": p.weights_.tolist(),
            "bias":    p.bias_,
        }
    return results


def run_real_dataset() -> dict:
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    log.info("--- Scratch Perceptron on make_classification ---")
    scratch = Perceptron(learning_rate=0.1, epochs=50)
    scratch.fit(X_train, y_train)
    scratch_acc = scratch.score(X_test, y_test)

    log.info("--- Sklearn Perceptron on make_classification ---")
    sk = SklearnPerceptron(eta0=0.1, max_iter=50, random_state=0)
    sk.fit(X_train, y_train)
    sk_acc = float(sk.score(X_test, y_test))

    return {
        "scratch_acc": scratch_acc,
        "sklearn_acc": sk_acc,
        "scratch_errors": scratch.errors_,
    }


def print_gate_table(gate_results: dict) -> None:
    sep = "=" * 58
    log.info(sep)
    log.info("%-8s  %10s  %8s  %s", "Gate", "Accuracy", "Epochs", "Separable")
    log.info(sep)
    for name, r in gate_results.items():
        separable = "yes" if r["acc"] == 1.0 else "no (expected)"
        log.info("%-8s  %9.0f%%  %8d  %s", name, r["acc"] * 100, r["epochs"], separable)
    log.info(sep)


def print_dataset_table(r: dict) -> None:
    sep = "=" * 46
    log.info(sep)
    log.info("%-30s  %10s", "make_classification (test)", "Accuracy")
    log.info(sep)
    log.info("%-30s  %9.2f%%", "Scratch Perceptron", r["scratch_acc"] * 100)
    log.info("%-30s  %9.2f%%", "Sklearn Perceptron", r["sklearn_acc"] * 100)
    log.info(sep)


def save_plot(gate_results: dict, dataset_result: dict, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gate convergence
    ax = axes[0]
    for name, r in gate_results.items():
        ax.plot(range(1, len(r["errors"]) + 1), r["errors"],
                marker="o", label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassifications")
    ax.set_title("Perceptron Convergence — Logic Gates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Real dataset error curve (scratch only)
    ax = axes[1]
    errs = dataset_result["scratch_errors"]
    ax.plot(range(1, len(errs) + 1), errs, "o-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassifications")
    ax.set_title("Scratch Perceptron — make_classification")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    log.info("plot saved → %s", out)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    log.info("=== Logic Gates ===")
    gate_results = run_gates()
    print_gate_table(gate_results)

    log.info("=== Real Dataset ===")
    dataset_result = run_real_dataset()
    print_dataset_table(dataset_result)

    save_plot(gate_results, dataset_result, args.out)


if __name__ == "__main__":
    main()

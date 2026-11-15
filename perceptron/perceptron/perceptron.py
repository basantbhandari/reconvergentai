"""Rosenblatt Perceptron: online learning with the perceptron update rule."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = ["Perceptron"]

log = logging.getLogger(__name__)


@dataclass
class Perceptron:
    """
    Rosenblatt Perceptron (1957).

    Learns a linear binary classifier via the online update rule:
        w ← w + lr · (y − ŷ) · x
        b ← b + lr · (y − ŷ)

    Converges in finite steps if the data is linearly separable.

    Parameters
    ----------
    learning_rate : float
    epochs : int
    seed : int
    """

    learning_rate: float = 0.1
    epochs: int = 100
    seed: int = 0

    weights_: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    bias_: float = field(default=0.0, init=False, repr=False)
    errors_: List[int] = field(default_factory=list, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        n, d = X.shape
        self.weights_ = np.zeros(d)
        self.bias_    = 0.0
        self.errors_  = []

        log.info("fit start  n=%d  d=%d  lr=%.4f  epochs=%d", n, d, self.learning_rate, self.epochs)
        t0 = time.perf_counter()

        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                yhat  = self._step(xi)
                delta = self.learning_rate * (yi - yhat)
                self.weights_ += delta * xi
                self.bias_    += delta
                if delta != 0:
                    errors += 1
                    log.debug("update  epoch=%d  delta=%.4f  w=%s  b=%.4f",
                              epoch + 1, delta, self.weights_, self.bias_)

            self.errors_.append(errors)
            log.info("epoch %3d/%d  errors=%d  acc=%.4f",
                     epoch + 1, self.epochs, errors, self.score(X, y))

            if errors == 0:
                log.info("converged at epoch %d", epoch + 1)
                break

        log.info("fit done  total=%.3fs  final_acc=%.4f", time.perf_counter() - t0, self.score(X, y))
        return self

    def _step(self, x: np.ndarray) -> int:
        return int(np.dot(x, self.weights_) + self.bias_ >= 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._step(xi) for xi in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

"""Dataset utilities: logic gates and sklearn dataset loaders."""
from __future__ import annotations

import numpy as np

__all__ = ["gate", "GATES"]

GATES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "AND": (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64),
        np.array([0, 0, 0, 1]),
    ),
    "OR": (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64),
        np.array([0, 1, 1, 1]),
    ),
    "NAND": (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64),
        np.array([1, 1, 1, 0]),
    ),
    "NOR": (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64),
        np.array([1, 0, 0, 0]),
    ),
    "XOR": (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64),
        np.array([0, 1, 1, 0]),
    ),
}


def gate(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a named logic gate. XOR is not linearly separable."""
    name = name.upper()
    if name not in GATES:
        raise ValueError(f"Unknown gate {name!r}. Choose from: {list(GATES)}")
    return GATES[name]

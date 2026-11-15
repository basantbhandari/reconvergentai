"""Load real vector datasets from binary files."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

__all__ = ["load_deep"]


def load_deep(
    path: str | Path,
    n: Optional[int] = None,
    device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    """
    Load vectors from a .fbin or .npy file and return a float32 torch.Tensor.

    .fbin layout (DEEP1B format): 4-byte int32 nrows, 4-byte int32 dim,
    then nrows × dim float32 values, row-major.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".fbin":
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(8), dtype=np.int32)
        _, d = int(header[0]), int(header[1])
        # Header may report full dataset size (e.g. 1B) even for a partial download.
        # Derive actual row count from file size so we never read past EOF.
        file_rows = (path.stat().st_size - 8) // (d * 4)
        load_n = min(n, file_rows) if n is not None else file_rows
        arr = np.fromfile(path, dtype=np.float32, count=load_n * d, offset=8).reshape(load_n, d)
    else:
        raise ValueError(f"Unsupported format: {suffix!r}. Expected .fbin or .npy")

    if n is not None:
        arr = arr[:n]

    t = torch.from_numpy(arr.copy()).float()
    if device is not None:
        t = t.to(device)
    return t

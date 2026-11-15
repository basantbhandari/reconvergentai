"""emeans — E-Means paper reproduction (arXiv:2611.09501)."""

from ._ops import assign, default_device, quantization_error, sq_dists
from .config import auto_alpha0, auto_batch_size
from .data import load_deep
from .emeans import EMeans
from .kmeans import KMeans

__all__ = [
    "EMeans",
    "KMeans",
    "load_deep",
    "auto_batch_size",
    "auto_alpha0",
    "default_device",
    "sq_dists",
    "assign",
    "quantization_error",
]

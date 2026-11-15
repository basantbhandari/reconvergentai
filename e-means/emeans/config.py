"""
Scaling law coefficients from Table 2 of the E-Means paper (arXiv:2611.09501).

Batch size law:   B  = c_B · k^b_k · n^b_n · T^b_T
                  rounded to nearest power of 2, clamped to [1, n]

Step size law:   α₀  = min(c_α · k^a_k · B^a_B · n^a_n · T^a_T, 1)

Both fitted via ordinary least squares in log-domain (R² ≈ 0.95–0.96).
"""

import math

__all__ = ["auto_batch_size", "auto_alpha0"]

_C_B   = 0.035   # default variant
_C_B_Q = 0.0044  # quality variant: smaller B → more gradient diversity → lower MSE
_B_K   = 0.13
_B_N   = 0.88
_B_T   = 0.81

_C_A = 0.824
_A_K = -0.46
_A_B =  0.90
_A_N = -0.45
_A_T = -0.11


def auto_batch_size(n: int, k: int, epochs: int, quality: bool = False) -> int:
    """Derive B from (n, k, T) via the scaling law. Rounded to nearest power of 2, clamped to [1, n]."""
    c = _C_B_Q if quality else _C_B
    B_raw = c * (k ** _B_K) * (n ** _B_N) * (epochs ** _B_T)
    B = int(2 ** round(math.log2(max(B_raw, 1.0))))
    return max(1, min(B, n))


def auto_alpha0(n: int, k: int, B: int, epochs: int) -> float:
    """Derive α₀ from (n, k, B, T) via the scaling law. Clamped to [0, 1]."""
    alpha = _C_A * (k ** _A_K) * (B ** _A_B) * (n ** _A_N) * (epochs ** _A_T)
    return float(min(max(alpha, 0.0), 1.0))

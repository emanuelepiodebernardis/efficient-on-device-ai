"""Symmetric quantization helpers."""
import numpy as np


def quantize(x, nbits, axis=None):
    """Per-tensor (axis=None) or per-axis symmetric quantization.

    Returns (q, s) with real value ~= q * s.  axis=1 over a (rows, cols) tensor
    gives a per-row scale of shape (rows, 1) (used for per-token activation quant).
    """
    qmax = (1 << (nbits - 1)) - 1
    if axis is None:
        s = np.max(np.abs(x)) / qmax
        s = s if s > 0 else 1e-12
        q = np.clip(np.round(x / s), -qmax - 1, qmax).astype(np.int64)
        return q, float(s)
    s = np.max(np.abs(x), axis=axis, keepdims=True) / qmax
    s = np.where(s > 0, s, 1e-12)
    q = np.clip(np.round(x / s), -qmax - 1, qmax).astype(np.int64)
    return q, s

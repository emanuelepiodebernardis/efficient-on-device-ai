"""
LLaMA-style transformer stack (NumPy) for the synthetic validation experiments.
Pre-norm: RMSNorm -> MHA[Softmax] -> residual ; RMSNorm -> SwiGLU[SiLU] -> residual.
Every matmul and non-linearity has an integer-only path (config-controlled).

Note: attention score matmuls (q@k, a@v) are kept in FP in BOTH paths, matching the
synthetic study scope; a fully-integer attention adds error there (see docs).
"""
import numpy as np
import math
from .operators_np import (int_linear, int_softmax, int_rmsnorm, int_silu,
                           fp_softmax, fp_rmsnorm, fp_silu)


class Config:
    def __init__(self, nbits=8, act_axis=None, smooth=False, alpha=0.5, ffn_out_mag=0.0):
        self.nbits = nbits
        self.act_axis = act_axis
        self.smooth = smooth
        self.alpha = alpha
        self.ffn_out_mag = ffn_out_mag


class Block:
    def __init__(self, d, h, dff, rng):
        self.d, self.h, self.dh = d, h, d // h
        sc = 1 / math.sqrt(d)
        self.g1 = rng.standard_normal(d) * 0.1 + 1
        self.g2 = rng.standard_normal(d) * 0.1 + 1
        self.Wq = rng.standard_normal((d, d)) * sc
        self.Wk = rng.standard_normal((d, d)) * sc
        self.Wv = rng.standard_normal((d, d)) * sc
        self.Wo = rng.standard_normal((d, d)) * sc
        self.Wg = rng.standard_normal((dff, d)) * sc
        self.Wu = rng.standard_normal((dff, d)) * sc
        self.Wd = rng.standard_normal((d, dff)) * sc
        self._oc = rng.choice(dff, size=3, replace=False)   # FFN-intermediate outlier channels
        self._os = np.sign(rng.standard_normal(3))

    def _lin(self, integer, cfg):
        if integer:
            return lambda a, W: int_linear(a, W, cfg.nbits, cfg.act_axis, cfg.smooth, cfg.alpha)
        return lambda a, W: a @ W.T

    def mha(self, x, integer, cfg):
        L = x.shape[0]
        lin = self._lin(integer, cfg)
        keep = np.tril(np.ones((L, L), dtype=bool))
        sm = (lambda s: int_softmax(s, cfg.nbits, keep=keep)) if integer else (lambda s: fp_softmax(s, keep=keep))
        q = lin(x, self.Wq).reshape(L, self.h, self.dh).transpose(1, 0, 2)
        k = lin(x, self.Wk).reshape(L, self.h, self.dh).transpose(1, 0, 2)
        v = lin(x, self.Wv).reshape(L, self.h, self.dh).transpose(1, 0, 2)
        sc = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.dh)
        a = sm(sc)
        o = (a @ v).transpose(1, 0, 2).reshape(L, self.d)
        return lin(o, self.Wo)

    def ffn(self, x, integer, cfg):
        lin = self._lin(integer, cfg)
        si = (lambda a: int_silu(a, cfg.nbits)) if integer else fp_silu
        h = si(lin(x, self.Wg)) * lin(x, self.Wu)
        if cfg.ffn_out_mag > 0:
            h = h.copy()
            h[:, self._oc] += self._os * cfg.ffn_out_mag
        return lin(h, self.Wd)

    def forward(self, x, integer, cfg):
        rms = (lambda a, g: int_rmsnorm(a, g, cfg.nbits)) if integer else fp_rmsnorm
        x = x + self.mha(rms(x, self.g1), integer, cfg)
        x = x + self.ffn(rms(x, self.g2), integer, cfg)
        return x


def build_stack(depth, d=256, h=4, dff=683, seed=0):
    rng = np.random.default_rng(7 + seed)
    return [Block(d, h, dff, rng) for _ in range(depth)]


def make_input(kind, L, d, seed=0, mag=80.0):
    rng = np.random.default_rng(1000 + seed)
    if kind == "gaussian":
        return rng.standard_normal((L, d))
    if kind == "heavy-tail":
        x = rng.standard_normal((L, d))
        m = rng.random((L, d)) < 0.03
        return x + m * rng.standard_normal((L, d)) * 12.0
    if kind == "massive":
        x = rng.standard_normal((L, d))
        ch = rng.choice(d, size=3, replace=False)
        x[:, ch] += rng.standard_normal((L, 3)) * mag + np.sign(rng.standard_normal(3)) * mag
        return x
    raise ValueError(kind)


def run_depth(stack, x0, cfg):
    """Run FP and integer forwards in parallel; return per-depth (relL2, cosine)."""
    xf = x0.copy(); xi = x0.copy(); errs = []
    for blk in stack:
        xf = blk.forward(xf, False, cfg)
        xi = blk.forward(xi, True, cfg)
        rel = np.linalg.norm(xi - xf) / np.linalg.norm(xf)
        cos = float(np.sum(xi * xf) / (np.linalg.norm(xi) * np.linalg.norm(xf) + 1e-12))
        errs.append((float(rel), cos))
    return errs

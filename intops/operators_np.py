"""
Integer-only operators (NumPy reference implementations).

At runtime these use ONLY integer arithmetic on the data (add/mul/shift, integer
division, integer sqrt). Constants are precomputed (bakeable). The float64 FP
references are provided for parity measurement.

Methodological choices baked in (learned during validation):
  * RMSNorm uses the "divide-large-by-large" form (divide by sqrt(denom) rather
    than representing the tiny rsqrt in fixed point) -> stable at any bit-width.
  * exp-based ops (softmax, silu) carry the input scale at high precision (2^SB)
    before reducing to 2^B -> no fixed-point underflow at high bit-widths.
  * Causal masking is applied AFTER exp (multiply by 0), never by adding -1e9
    before quantization (which would set the per-tensor scale to 1e9).
"""
import numpy as np
import math
from .quant import quantize

B = 16                 # fixed-point fractional bits
SCALE = 1 << B
SB = 30                # high-precision bits for carrying a (possibly tiny) input scale

LN2_FP = round(math.log(2) * SCALE)
A_FP = round(0.3585 * SCALE)   # I-BERT exp poly:  exp(p) ~ 0.3585(p+1.353)^2 + 0.344
Bp_FP = round(1.353 * SCALE)
C_FP = round(0.344 * SCALE)


def int_exp_fp(delta_fp):
    """exp of a non-positive fixed-point input (delta*2^B) -> exp*2^B, integer-only."""
    z = (-delta_fp) // LN2_FP
    p = delta_fp + z * LN2_FP
    pb = p + Bp_FP
    return ((((A_FP * ((pb * pb) >> B)) >> B) + C_FP) >> z)


def int_softmax(x, nbits, keep=None, out_bits=20):
    """Integer-only softmax over the last axis. `keep` (bool, same shape) applies a
    causal/padding mask AFTER exp (True = keep)."""
    q, s = quantize(x, nbits)
    s_hi = round(s * (1 << SB))
    if keep is None:
        qd = q - q.max(-1, keepdims=True)
    else:
        neg = np.where(keep, q, -10 ** 9)
        qd = q - neg.max(-1, keepdims=True)
    delta_fp = (qd * s_hi) >> (SB - B)
    delta_fp = np.minimum(delta_fp, 0)
    e = int_exp_fp(delta_fp)
    if keep is not None:
        e = e * keep
    ss = e.sum(-1, keepdims=True)
    ss = np.where(ss == 0, 1, ss)
    return ((e << out_bits) // ss).astype(np.float64) / (1 << out_bits)


def int_isqrt_arr(n):
    flat = n.reshape(-1)
    out = np.array([math.isqrt(int(v)) for v in flat], dtype=object)
    return out.reshape(n.shape)


def int_rmsnorm(x, gamma, nbits, eps=1e-6):
    """Integer-only RMSNorm over the last axis (divide-large-by-large form)."""
    q, s = quantize(x, nbits)
    n = x.shape[-1]
    sumsq = (q.astype(np.int64) ** 2).sum(-1, keepdims=True)
    denom = sumsq + int(round(eps * n / (s * s)))
    denom = np.where(denom == 0, 1, denom)
    sd = int_isqrt_arr(denom)
    sd = np.where(sd == 0, 1, sd)
    sqrtn = math.isqrt(n << (2 * B))
    g = np.round(gamma * SCALE).astype(np.int64).astype(object)
    return ((q.astype(object) * g * sqrtn) // sd).astype(np.float64) / (SCALE * SCALE)


def int_silu(x, nbits):
    """Integer-only SiLU = x * sigmoid(x), reusing int_exp_fp."""
    q, s = quantize(x, nbits)
    s_hi = round(s * (1 << SB))
    a = -((np.abs(q) * s_hi) >> (SB - B))      # -|x| * 2^B  (<= 0)
    e = int_exp_fp(a)
    denom = SCALE + e
    sig = np.where(q >= 0, (SCALE * SCALE) // denom, (e * SCALE) // denom)
    xfp = (q.astype(object) * s_hi) >> (SB - B)
    return ((xfp * sig) >> B).astype(np.float64) / SCALE


def int_linear(x, W, nbits, act_axis=None, smooth=False, alpha=0.5):
    """Integer matmul Y = x @ W.T  (W: (out, in)).
    act_axis: None=per-tensor activation quant, 1=per-token.
    smooth: SmoothQuant-style per-input-channel smoothing (migrate channel scale
            from activations into weights)."""
    if smooth:
        a_max = np.max(np.abs(x), axis=0) + 1e-9
        w_max = np.max(np.abs(W), axis=0) + 1e-9
        sm = (a_max ** alpha) / (w_max ** (1 - alpha))
        sm = np.clip(sm, 1e-3, 1e3)
        x = x / sm
        W = W * sm
    qx, sx = quantize(x, nbits, axis=act_axis)
    qw, sw = quantize(W, nbits, axis=1)
    acc = qx.astype(np.int64) @ qw.T.astype(np.int64)
    return acc.astype(np.float64) * sx * sw.T


# ---- FP references ----
def fp_softmax(x, keep=None):
    if keep is None:
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
    else:
        xm = np.where(keep, x, -np.inf)
        x = x - xm.max(-1, keepdims=True)
        e = np.exp(x) * keep
    return e / e.sum(-1, keepdims=True)


def fp_rmsnorm(x, gamma, eps=1e-6):
    return x / np.sqrt(np.mean(x ** 2, -1, keepdims=True) + eps) * gamma


def fp_silu(x):
    return x / (1 + np.exp(-x))

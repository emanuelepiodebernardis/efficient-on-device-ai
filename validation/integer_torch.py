"""
Torch integer-operator equivalents for REAL-MODEL validation.

These mirror the validated NumPy operators in intops/operators_np.py but operate on
torch tensors so they can be patched into a Hugging Face model's forward pass.
Each op computes the integer pipeline and returns a dequantized float tensor.

NOTE: integer matmuls are evaluated via float64 of the integer operands (exact for
our magnitudes: int8/int16 products summed over hidden < 2^53), avoiding backend
int-matmul limitations.

CANNOT be unit-tested in the authoring sandbox (no torch there). Run + debug in the
build environment; cross-check the per-operator parity against the NumPy spike.
"""
import math
import torch

B = 16
SCALE = 1 << B
SB = 30
LN2_FP = round(math.log(2) * SCALE)
A_FP = round(0.3585 * SCALE)
Bp_FP = round(1.353 * SCALE)
C_FP = round(0.344 * SCALE)


def quantize_t(x, nbits, per_token=False):
    qmax = (1 << (nbits - 1)) - 1
    if per_token:
        s = x.abs().amax(dim=-1, keepdim=True) / qmax
    else:
        s = x.abs().amax() / qmax
    s = s.clamp_min(1e-12)
    q = (x / s).round().clamp(-qmax - 1, qmax).to(torch.int64)
    return q, s


def quantize_perchannel_t(W, nbits):
    qmax = (1 << (nbits - 1)) - 1
    s = W.abs().amax(dim=1, keepdim=True) / qmax       # (out, 1)
    s = s.clamp_min(1e-12)
    q = (W / s).round().clamp(-qmax - 1, qmax).to(torch.int64)
    return q, s


def int_exp_fp_t(delta_fp):
    z = (-delta_fp) // LN2_FP
    p = delta_fp + z * LN2_FP
    pb = p + Bp_FP
    return ((((A_FP * ((pb * pb) >> B)) >> B) + C_FP) >> z)


def int_softmax_t(x, nbits, keep=None, out_bits=20):
    q, s = quantize_t(x, nbits)
    s_hi = int(round(float(s) * (1 << SB)))
    if keep is None:
        qd = q - q.amax(dim=-1, keepdim=True)
    else:
        neg = torch.where(keep, q, torch.full_like(q, -10 ** 9))
        qd = q - neg.amax(dim=-1, keepdim=True)
    delta_fp = (qd * s_hi) >> (SB - B)
    delta_fp = torch.clamp(delta_fp, max=0)
    e = int_exp_fp_t(delta_fp)
    if keep is not None:
        e = e * keep
    ss = e.sum(dim=-1, keepdim=True).clamp_min(1)
    return ((e << out_bits) // ss).double() / (1 << out_bits)


def int_rmsnorm_t(x, weight, nbits, eps=1e-6):
    """Integer RMSNorm over last dim (divide-large-by-large). x: (..., H)."""
    q, s = quantize_t(x, nbits)
    n = x.shape[-1]
    sumsq = (q.to(torch.int64) ** 2).sum(dim=-1, keepdim=True)
    denom = sumsq + int(round(eps * n / (float(s) * float(s))))
    denom = denom.clamp_min(1)
    sd = denom.double().sqrt().floor().to(torch.int64).clamp_min(1)   # integer isqrt (exact for our range)
    sqrtn = math.isqrt(n << (2 * B))
    g = (weight.double() * SCALE).round().to(torch.int64)
    out = (q.to(torch.int64) * g * sqrtn) // sd
    return out.double() / (SCALE * SCALE)


def int_silu_t(x, nbits):
    q, s = quantize_t(x, nbits)
    s_hi = int(round(float(s) * (1 << SB)))
    a = -((q.abs() * s_hi) >> (SB - B))
    e = int_exp_fp_t(a)
    denom = (SCALE + e).clamp_min(1)
    sig = torch.where(q >= 0, (SCALE * SCALE) // denom, (e * SCALE) // denom)
    xfp = (q.to(torch.int64) * s_hi) >> (SB - B)
    return ((xfp * sig) >> B).double() / SCALE


class QuantLinear(torch.nn.Module):
    """Drop-in replacement for nn.Linear: integer matmul with optional per-token
    activation quant and SmoothQuant-style per-channel smoothing.

    outlier_k=0 (default): identical behaviour to prior runs — no results change.
    outlier_k>0: LLM.int8-style isolation — top-k input channels by activation
    magnitude are kept in FP64; the rest are quantized as usual.
    """
    def __init__(self, lin, nbits=8, per_token=False, smooth=False, alpha=0.5,
                 outlier_k=0):
        super().__init__()
        self.weight = lin.weight.detach().clone()        # (out, in)
        self.bias = None if lin.bias is None else lin.bias.detach().clone()
        self.nbits = nbits
        self.per_token = per_token
        self.smooth = smooth
        self.alpha = alpha
        self.outlier_k = outlier_k
        self.last_outlier_idx = None

    def forward(self, x):
        orig_shape = x.shape
        xf = x.reshape(-1, orig_shape[-1]).to(torch.float64)
        W = self.weight.to(torch.float64)               # (out, in)

        if self.outlier_k > 0:
            n_in = xf.shape[1]
            k = min(self.outlier_k, n_in)
            mag = xf.abs().amax(dim=0)                  # (in,)
            idx = torch.topk(mag, k).indices             # (k,)
            mask = torch.ones(n_in, dtype=torch.bool, device=xf.device)
            mask[idx] = False
            x_rest = xf[:, mask]                        # (N, in-k)
            W_rest = W[:, mask]                         # (out, in-k)
            x_out  = xf[:, idx]                         # (N, k)
            W_out  = W[:, idx]                          # (out, k)
            qx, sx = quantize_t(x_rest, self.nbits, self.per_token)
            qw, sw = quantize_perchannel_t(W_rest, self.nbits)
            acc = qx.double() @ qw.double().t()         # (N, out)
            y = acc * sx * sw.t()
            y = y + x_out @ W_out.t()
            if self.bias is not None:
                y = y + self.bias.to(torch.float64)
            self.last_outlier_idx = idx.detach().cpu().tolist()
        else:
            if self.smooth:
                a_max = xf.abs().amax(dim=0) + 1e-9    # (in,)
                w_max = W.abs().amax(dim=0) + 1e-9      # (in,)
                sm = (a_max ** self.alpha) / (w_max ** (1 - self.alpha))
                sm = sm.clamp(1e-3, 1e3)
                xf = xf / sm
                W = W * sm
            qx, sx = quantize_t(xf, self.nbits, self.per_token)
            qw, sw = quantize_perchannel_t(W, self.nbits)
            acc = qx.double() @ qw.double().t()         # (N, out)
            y = acc * sx * sw.t()
            if self.bias is not None:
                y = y + self.bias.to(torch.float64)

        return y.reshape(*orig_shape[:-1], self.weight.shape[0]).to(x.dtype)


class QuantRMSNorm(torch.nn.Module):
    """Drop-in replacement for *RMSNorm modules (Llama/Qwen style)."""
    def __init__(self, rms, nbits=8):
        super().__init__()
        self.weight = rms.weight.detach().clone()
        self.eps = float(getattr(rms, "variance_epsilon", getattr(rms, "eps", 1e-6)))
        self.nbits = nbits

    def forward(self, x):
        out = int_rmsnorm_t(x, self.weight, self.nbits, self.eps)
        return out.to(x.dtype)

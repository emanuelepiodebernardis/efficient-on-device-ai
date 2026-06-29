import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Spike: integer operators in isolation vs FP, parity under gaussian + heavy-tail."""
import numpy as np
from intops.operators_np import int_softmax, int_rmsnorm, fp_softmax, fp_rmsnorm

rng = np.random.default_rng(0)


def inputs(kind, shape):
    if kind == "gaussian":
        return rng.standard_normal(shape)
    x = rng.standard_normal(shape)
    m = rng.random(shape) < 0.02
    return x + m * rng.standard_normal(shape) * 15.0


def sm_metrics(fp, it):
    mae = np.mean(np.abs(fp - it)); maxe = np.max(np.abs(fp - it))
    kl = np.sum(fp * (np.log(fp + 1e-12) - np.log(it + 1e-12)), -1).mean()
    am = np.mean(fp.argmax(-1) == it.argmax(-1))
    cos = np.mean(np.sum(fp * it, -1) / (np.linalg.norm(fp, axis=-1) * np.linalg.norm(it, axis=-1) + 1e-12))
    return mae, maxe, kl, am, cos


def rn_metrics(fp, it):
    rel = np.linalg.norm(fp - it) / np.linalg.norm(fp)
    cos = np.mean(np.sum(fp * it, -1) / (np.linalg.norm(fp, axis=-1) * np.linalg.norm(it, axis=-1) + 1e-12))
    return rel, cos


if __name__ == "__main__":
    print("SOFTMAX (64x128x128)")
    print(f"{'dist':10s}{'bit':>5s}{'argmax%':>9s}{'cos':>10s}{'KL':>10s}")
    for kind in ["gaussian", "heavy-tail"]:
        x = inputs(kind, (64, 128, 128)); fp = fp_softmax(x)
        for nb in [8, 16]:
            _, _, kl, am, cos = sm_metrics(fp, int_softmax(x, nb))
            print(f"{kind:10s}{nb:5d}{100*am:9.2f}{cos:10.6f}{kl:10.2e}")
    print("\nRMSNORM (512x2048)")
    print(f"{'dist':10s}{'bit':>5s}{'relL2':>10s}{'cos':>10s}")
    for kind in ["gaussian", "heavy-tail"]:
        x = inputs(kind, (512, 2048)); g = rng.standard_normal(2048) * 0.2 + 1
        fp = fp_rmsnorm(x, g)
        for nb in [8, 16]:
            rel, cos = rn_metrics(fp, int_rmsnorm(x, g, nb))
            print(f"{kind:10s}{nb:5d}{rel:10.2e}{cos:10.6f}")

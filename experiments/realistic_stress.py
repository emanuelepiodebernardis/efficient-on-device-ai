import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Realistic stress: massive activations + post-norm FFN outliers; per-tensor vs per-token."""
import numpy as np
from intops import Config, build_stack, make_input, run_depth


def endpoint(kind, nb, act_axis=None, ffn_out=0.0, seed=0):
    stack = build_stack(32, seed=seed)
    x0 = make_input("massive" if kind in ("massive", "post-norm") else kind, 48, 256, seed=seed)
    errs = run_depth(stack, x0, Config(nbits=nb, act_axis=act_axis, ffn_out_mag=ffn_out))
    return errs[-1]


if __name__ == "__main__":
    print("Depth-32 endpoint, relL2 (cosine)")
    rows = [("massive activations", "massive", 0.0), ("post-norm FFN outliers", "post-norm", 60.0)]
    cfgs = [("INT16", 16, None), ("INT8 per-tensor", 8, None), ("INT8 per-token", 8, 1)]
    for rname, kind, ffn in rows:
        print(f"\n[{rname}]")
        for cname, nb, ax in cfgs:
            rel, cos = endpoint(kind, nb, ax, ffn)
            print(f"  {cname:18s}: {rel:.4f} ({cos:.4f})")
    print("\nMulti-seed (3) depth-32, massive activations:")
    for cname, nb, ax in cfgs:
        vals = [endpoint("massive", nb, ax, 0.0, s)[0] for s in range(3)]
        print(f"  {cname:18s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

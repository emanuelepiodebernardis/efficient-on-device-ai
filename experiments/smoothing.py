import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""The INT8 fix: per-channel smoothing across all regimes (depth-32 endpoint)."""
from intops import Config, build_stack, make_input, run_depth

REGIMES = [("gaussian", "gaussian", 0.0), ("heavy-tail (scattered)", "heavy-tail", 0.0),
           ("massive activations", "massive", 0.0), ("post-norm FFN outliers", "massive", 60.0)]
CONFIGS = [("INT16", 16, None, False), ("INT8 naive", 8, None, False),
           ("INT8 per-token", 8, 1, False), ("INT8 smooth", 8, None, True),
           ("INT8 smooth+token", 8, 1, True)]

if __name__ == "__main__":
    print("relL2 @ depth32 (cosine)")
    for rname, kind, ffn in REGIMES:
        print(f"\n[{rname}]")
        for cname, nb, ax, sm in CONFIGS:
            stack = build_stack(32); x0 = make_input(kind, 48, 256)
            errs = run_depth(stack, x0, Config(nbits=nb, act_axis=ax, smooth=sm, ffn_out_mag=ffn))
            rel, cos = errs[-1]
            print(f"  {cname:18s}: {rel:.4f} ({cos:.4f})")

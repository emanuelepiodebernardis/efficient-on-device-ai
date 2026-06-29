import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""End-to-end error accumulation vs depth (synthetic, gaussian + heavy-tail)."""
from intops import Config, build_stack, make_input, run_depth

if __name__ == "__main__":
    print("Residual-stream error vs depth (relL2 | cosine)")
    sel = [1, 2, 4, 8, 16, 32]
    for kind in ["gaussian", "heavy-tail"]:
        for nb in [16, 8]:
            stack = build_stack(32); x0 = make_input(kind, 48, 256)
            errs = run_depth(stack, x0, Config(nbits=nb))
            print(f"\n[{kind:10s} INT{nb}]")
            print("  depth:  " + "  ".join(f"{d:>7d}" for d in sel))
            print("  relL2:  " + "  ".join(f"{errs[d-1][0]:7.4f}" for d in sel))
            print("  cosine: " + "  ".join(f"{errs[d-1][1]:7.4f}" for d in sel))

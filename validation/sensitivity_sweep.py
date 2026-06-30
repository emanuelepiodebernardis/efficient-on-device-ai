"""
Per-block sensitivity sweep for INT8 naive quantization of Qwen2.5-0.5B.
Pure measurement — no remedy, no mixed-precision design.

Regime fixed throughout: nbits=8, per_token=False, smooth=False, patch_lm_head=False.

Sweep B "protect-one" (primary ranker, baseline = full INT8):
    For each block l: skip_modules=[f"model.layers.{l}"] -> rest INT8.
    Also runs baseline with skip_modules=[].

Sweep A "degrade-one" (diagnostic contrast, baseline = full FP):
    For each block l: only block l goes INT8;
    skip_modules = [all other blocks] + ["model.norm"].

FP reference (hs_fp, ppl_fp) is computed ONCE and reused as denominator.
A fresh FP model is loaded with load("qwen2.5-0.5b") before each configuration
because patch_model acts in-place and is destructive.

Outputs:
    results/raw/sensitivity_protect_int8.json
    results/raw/sensitivity_degrade_int8.json

Run:
    python -m validation.sensitivity_sweep
"""
import gc
import json
import os
import sys

from .load_model import load
from .integer_patch import patch_model
from .error_vs_depth import layerwise_hidden, err, perplexity
from .calibration_prompts import CALIBRATION_PROMPTS, PERPLEXITY_TEXT

TEXT  = CALIBRATION_PROMPTS[0]
RAW   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "raw")
REGIME = dict(nbits=8, per_token=False, smooth=False, patch_lm_head=False)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run(skip_modules, hs_fp, ppl_fp, label=""):
    """Load a fresh FP model, patch it, measure full depth curve + ppl."""
    model, tok = load("qwen2.5-0.5b")
    patch_model(model, skip_modules=skip_modules, verbose=False, **REGIME)

    hs_q  = layerwise_hidden(model, tok, TEXT)
    ppl_q = perplexity(model, tok, PERPLEXITY_TEXT)

    del model
    gc.collect()

    layers = []
    for i, (hq, hf) in enumerate(zip(hs_q, hs_fp)):
        rel, cos = err(hq, hf)
        layers.append({"layer": i, "relL2": rel, "cosine": cos})

    final = layers[-1]
    return {
        "final_relL2":  final["relL2"],
        "final_cosine": final["cosine"],
        "ppl_int":      ppl_q,
        "ppl_ratio":    ppl_q / ppl_fp if ppl_fp > 0 else None,
        "layers":       layers,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    os.makedirs(RAW, exist_ok=True)

    # ---- FP reference (computed once) ------------------------------------
    print("Loading FP reference model ...")
    model_ref, tok_ref = load("qwen2.5-0.5b")
    hs_fp   = layerwise_hidden(model_ref, tok_ref, TEXT)
    ppl_fp  = perplexity(model_ref, tok_ref, PERPLEXITY_TEXT)
    N       = model_ref.config.num_hidden_layers
    del model_ref, tok_ref
    gc.collect()
    print(f"  ppl_fp = {ppl_fp:.3f}   num_hidden_layers = {N}")

    # ======================================================================
    # SWEEP B — protect-one
    # ======================================================================
    print(f"\n{'='*60}")
    print("SWEEP B — protect-one  (1 baseline + {N} block configs)".format(N=N))
    print(f"{'='*60}")

    # Baseline: full INT8
    print("  [B baseline] full INT8 ...")
    baseline = _run([], hs_fp, ppl_fp)
    baseline["skip_modules"] = []
    print(f"    final cosine={baseline['final_cosine']:.5f}  "
          f"ppl_ratio={baseline['ppl_ratio']:.3f}")

    b_blocks = []
    for l in range(N):
        skip = [f"model.layers.{l}"]
        print(f"  [B] block {l:02d}: {skip} ...")
        r = _run(skip, hs_fp, ppl_fp)
        r["block"]          = l
        r["skip_modules"]   = skip
        r["delta_cosine"]   = r["final_cosine"]  - baseline["final_cosine"]
        r["delta_ppl_ratio"] = baseline["ppl_ratio"] - r["ppl_ratio"]
        b_blocks.append(r)
        print(f"    cosine={r['final_cosine']:.5f}  "
              f"delta_cos={r['delta_cosine']:+.5f}  "
              f"ppl_ratio={r['ppl_ratio']:.3f}")

    b_json = {
        "sweep":               "B_protect_one",
        "model":               "qwen2.5-0.5b",
        "regime":              REGIME,
        "ppl_fp":              ppl_fp,
        "baseline_full_int8":  baseline,
        "blocks":              b_blocks,
    }
    path_b = os.path.join(RAW, "sensitivity_protect_int8.json")
    with open(path_b, "w") as f:
        json.dump(b_json, f, indent=2)
    print(f"\n  saved -> {path_b}")

    # ======================================================================
    # SWEEP A — degrade-one
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"SWEEP A — degrade-one  ({N} block configs)")
    print(f"{'='*60}")

    a_blocks = []
    for l in range(N):
        # Protect everything except block l and the final norm
        skip = [f"model.layers.{j}" for j in range(N) if j != l] + ["model.norm"]
        print(f"  [A] block {l:02d}: only block {l} INT8 ...")
        r = _run(skip, hs_fp, ppl_fp)
        r["block"] = l
        a_blocks.append(r)
        print(f"    final relL2={r['final_relL2']:.6f}  "
              f"cosine={r['final_cosine']:.5f}  "
              f"ppl_ratio={r['ppl_ratio']:.4f}")

    a_json = {
        "sweep":   "A_degrade_one",
        "model":   "qwen2.5-0.5b",
        "regime":  REGIME,
        "ppl_fp":  ppl_fp,
        "blocks":  a_blocks,
    }
    path_a = os.path.join(RAW, "sensitivity_degrade_int8.json")
    with open(path_a, "w") as f:
        json.dump(a_json, f, indent=2)
    print(f"\n  saved -> {path_a}")

    # ======================================================================
    # Tables
    # ======================================================================

    # Table B: sorted by delta_cosine descending (most recovery first)
    print("\n" + "=" * 74)
    print("SWEEP B — protect-one: blocks ranked by cosine recovery vs full-INT8 baseline")
    print(f"  baseline:  cosine={baseline['final_cosine']:.5f}  "
          f"ppl_ratio={baseline['ppl_ratio']:.3f}")
    print("=" * 74)
    print(f"{'rank':>4}  {'block':>5}  {'cosine_fin':>11}  {'delta_cos':>10}"
          f"  {'ppl_ratio':>10}  {'delta_ppl_r':>12}")
    print("-" * 74)
    for rank, r in enumerate(sorted(b_blocks, key=lambda x: -x["delta_cosine"]), 1):
        print(f"{rank:>4}  {r['block']:>5}  {r['final_cosine']:>11.5f}"
              f"  {r['delta_cosine']:>+10.5f}"
              f"  {r['ppl_ratio']:>10.3f}"
              f"  {r['delta_ppl_ratio']:>+12.3f}")

    # Table A: sorted by final_relL2 descending (most damage first)
    print("\n" + "=" * 66)
    print("SWEEP A — degrade-one: blocks ranked by final relL2 injected")
    print("=" * 66)
    print(f"{'rank':>4}  {'block':>5}  {'final_relL2':>12}  {'final_cosine':>13}"
          f"  {'ppl_ratio':>10}")
    print("-" * 66)
    for rank, r in enumerate(sorted(a_blocks, key=lambda x: -x["final_relL2"]), 1):
        print(f"{rank:>4}  {r['block']:>5}  {r['final_relL2']:>12.6f}"
              f"  {r['final_cosine']:>13.5f}"
              f"  {r['ppl_ratio']:>10.4f}")

    print(f"\nResults saved to:")
    print(f"  {path_b}")
    print(f"  {path_a}")


if __name__ == "__main__":
    main()

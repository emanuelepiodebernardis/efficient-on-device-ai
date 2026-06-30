"""
Smoothing-factorization diagnostic driver for Qwen2.5-0.5B.

Orchestration only: imports helpers from validation.error_vs_depth and
validation.integer_patch WITHOUT modifying them.  Uses the smooth and
per_token flags already implemented in QuantLinear (online smoothing)
as-is — no offline calibration, no skip_modules, no mixed-precision.

All 24 blocks always quantized to INT8 (nbits=8, patch_lm_head=False).

8 configs measured:
  1  naive:             per_token=False  smooth=False
  2  per-token only:    per_token=True   smooth=False
  3  smooth only 0.50:  per_token=False  smooth=True  alpha=0.50
  4  smooth only 0.75:  per_token=False  smooth=True  alpha=0.75
  5  smooth only 0.90:  per_token=False  smooth=True  alpha=0.90
  6  smooth+pt   0.50:  per_token=True   smooth=True  alpha=0.50
  7  smooth+pt   0.75:  per_token=True   smooth=True  alpha=0.75
  8  smooth+pt   0.90:  per_token=True   smooth=True  alpha=0.90

FP reference (hs_fp, ppl_fp, ppl_fp per calibration prompt) computed once
from a clean model and reused for all configs.  A fresh FP model is loaded
before every config because patch_model is destructive.

Run:
    python -m validation.smoothing_factorization

Output: results/raw/smoothing_factorization.json
"""
import gc
import json
import os
import sys

from .load_model import load
from .integer_patch import patch_model
from .error_vs_depth import layerwise_hidden, err, perplexity
from .calibration_prompts import CALIBRATION_PROMPTS, PERPLEXITY_TEXT

MODEL_NAME = "qwen2.5-0.5b"

RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "raw",
)

# Depth checkpoints for the depth table (layer indices; last = num_hidden_layers)
DEPTH_CHECKPOINTS = [0, 6, 12, 18, 21, 22, 24]

CONFIGS = [
    {"id": 1, "label": "naive",          "per_token": False, "smooth": False, "alpha": 0.5},
    {"id": 2, "label": "per-token",      "per_token": True,  "smooth": False, "alpha": 0.5},
    {"id": 3, "label": "smooth-0.50",    "per_token": False, "smooth": True,  "alpha": 0.50},
    {"id": 4, "label": "smooth-0.75",    "per_token": False, "smooth": True,  "alpha": 0.75},
    {"id": 5, "label": "smooth-0.90",    "per_token": False, "smooth": True,  "alpha": 0.90},
    {"id": 6, "label": "smooth+pt-0.50", "per_token": True,  "smooth": True,  "alpha": 0.50},
    {"id": 7, "label": "smooth+pt-0.75", "per_token": True,  "smooth": True,  "alpha": 0.75},
    {"id": 8, "label": "smooth+pt-0.90", "per_token": True,  "smooth": True,  "alpha": 0.90},
]


# ---------------------------------------------------------------------------
# Single-config measurement
# ---------------------------------------------------------------------------

def _measure(cfg, hs_fp, ppl_fp, ppl_fp_calib, tok):
    """Load fresh FP model, patch to INT8 with given flags, measure all metrics."""
    model, _ = load(MODEL_NAME)

    patch_model(
        model,
        nbits=8,
        per_token=cfg["per_token"],
        smooth=cfg["smooth"],
        alpha=cfg["alpha"],
        patch_lm_head=False,
        skip_modules=None,
    )

    hs_q = layerwise_hidden(model, tok, CALIBRATION_PROMPTS[0])
    ppl_q_text = perplexity(model, tok, PERPLEXITY_TEXT)

    ppl_ratios_calib = []
    for prompt, fp_val in zip(CALIBRATION_PROMPTS, ppl_fp_calib):
        q_val = perplexity(model, tok, prompt)
        ppl_ratios_calib.append(q_val / fp_val if fp_val > 0 else None)

    del model
    gc.collect()

    layers = []
    for i, (hq, hf) in enumerate(zip(hs_q, hs_fp)):
        rel, cos = err(hq, hf)
        layers.append({"layer": i, "relL2": rel, "cosine": cos})

    final = layers[-1]
    valid = [r for r in ppl_ratios_calib if r is not None]
    calib_mean = sum(valid) / len(valid) if valid else None

    return {
        "ppl_ratio_perplexitytext": ppl_q_text / ppl_fp if ppl_fp > 0 else None,
        "ppl_ratio_calibmean": calib_mean,
        "final_relL2": final["relL2"],
        "final_cosine": final["cosine"],
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _flag(val, threshold):
    if val is None:
        return "?"
    return "T" if val <= threshold else "F"


def _print_summary(results):
    """Table 1: 8 configs sorted by ppl_ratio_perplexitytext ascending."""
    print("\n" + "=" * 88)
    print("TABLE 1 — SUMMARY  (sorted by ppl_ratio on PERPLEXITY_TEXT, ascending)")
    print("=" * 88)
    header = (f"{'id':>2}  {'per_token':>9}  {'smooth':>6}  {'alpha':>5}  "
              f"{'ppl_ppltext':>12}  {'ppl_calibmean':>13}  {'cosine':>10}  "
              f"{'ppl<=2':>6}  {'ppl<=1.1':>8}")
    print(header)
    print("-" * 88)
    for r in sorted(results, key=lambda x: (x["ppl_ratio_perplexitytext"] or 1e18)):
        cfg = r["cfg"]
        ppl_t = r["ppl_ratio_perplexitytext"]
        ppl_c = r["ppl_ratio_calibmean"]
        cos   = r["final_cosine"]
        print(
            f"{cfg['id']:>2}  {str(cfg['per_token']):>9}  {str(cfg['smooth']):>6}"
            f"  {cfg['alpha']:>5.2f}  "
            f"{ppl_t:>12.4f}  {ppl_c:>13.4f}  {cos:>10.6f}  "
            f"{_flag(ppl_t,2.0):>6}  {_flag(ppl_t,1.1):>8}"
        )


def _print_depth(results, num_layers):
    """Table 2: relL2 at selected depth checkpoints for each config."""
    # Determine which layer indices to use (clamp last checkpoint to max)
    max_idx = num_layers  # hidden_states has num_layers+1 entries (0..num_layers)
    checkpoints = [c for c in DEPTH_CHECKPOINTS if c <= max_idx]

    print("\n" + "=" * 88)
    print("TABLE 2 — DEPTH  (relL2 at selected layer indices for each config)")
    print("  Layer indices: " + "  ".join(f"{c:>6}" for c in checkpoints))
    print("=" * 88)
    hdr = f"{'id':>2}  {'label':<15}" + "".join(f"  {'L'+str(c):>7}" for c in checkpoints)
    print(hdr)
    print("-" * 88)
    for r in results:
        cfg = r["cfg"]
        vals = []
        for c in checkpoints:
            if c < len(r["layers"]):
                vals.append(f"{r['layers'][c]['relL2']:>7.4f}")
            else:
                vals.append(f"{'N/A':>7}")
        print(f"{cfg['id']:>2}  {cfg['label']:<15}" + "".join(f"  {v}" for v in vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    os.makedirs(RAW, exist_ok=True)

    # ---- FP reference (computed once) --------------------------------------
    print(f"Loading FP reference model ({MODEL_NAME}) ...")
    model_ref, tok = load(MODEL_NAME)
    num_layers = model_ref.config.num_hidden_layers
    print(f"  num_hidden_layers = {num_layers}")

    print("Computing FP hidden states on CALIBRATION_PROMPTS[0] ...")
    hs_fp = layerwise_hidden(model_ref, tok, CALIBRATION_PROMPTS[0])

    print("Computing FP perplexity on PERPLEXITY_TEXT ...")
    ppl_fp = perplexity(model_ref, tok, PERPLEXITY_TEXT)

    print("Computing FP perplexity on all CALIBRATION_PROMPTS ...")
    ppl_fp_calib = [perplexity(model_ref, tok, p) for p in CALIBRATION_PROMPTS]

    del model_ref
    gc.collect()

    print(f"  ppl_fp (PERPLEXITY_TEXT) = {ppl_fp:.4f}")
    print(f"  ppl_fp (calib)           = {[round(v, 3) for v in ppl_fp_calib]}")

    # ---- Measure all 8 configs ---------------------------------------------
    results = []
    for cfg in CONFIGS:
        print(f"\n[config {cfg['id']}] {cfg['label']}"
              f"  per_token={cfg['per_token']}  smooth={cfg['smooth']}"
              f"  alpha={cfg['alpha']} ...")
        metrics = _measure(cfg, hs_fp, ppl_fp, ppl_fp_calib, tok)
        results.append({"cfg": cfg, **metrics})
        print(f"  ppl_ratio_ppltext={metrics['ppl_ratio_perplexitytext']:.4f}"
              f"  ppl_ratio_calibmean={metrics['ppl_ratio_calibmean']:.4f}"
              f"  cosine={metrics['final_cosine']:.6f}")

    # ---- Save JSON ---------------------------------------------------------
    output = {
        "model": MODEL_NAME,
        "ppl_fp_perplexitytext": ppl_fp,
        "ppl_fp_calibration_prompts": ppl_fp_calib,
        "configs": [
            {
                "id":                      r["cfg"]["id"],
                "label":                   r["cfg"]["label"],
                "per_token":               r["cfg"]["per_token"],
                "smooth":                  r["cfg"]["smooth"],
                "alpha":                   r["cfg"]["alpha"],
                "ppl_ratio_perplexitytext": r["ppl_ratio_perplexitytext"],
                "ppl_ratio_calibmean":     r["ppl_ratio_calibmean"],
                "final_relL2":             r["final_relL2"],
                "final_cosine":            r["final_cosine"],
                "layers":                  r["layers"],
            }
            for r in results
        ],
    }

    out_path = os.path.join(RAW, "smoothing_factorization.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # ---- Print tables ------------------------------------------------------
    _print_summary(results)
    _print_depth(results, num_layers)

    print(f"\nDone.  Results in {out_path}")


if __name__ == "__main__":
    main()

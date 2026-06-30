"""
Mixed-precision sweep driver for Qwen2.5-0.5B.

Orchestration only: imports helpers from validation.error_vs_depth and
validation.integer_patch WITHOUT modifying them.  Tests the rule "protect
a few blocks at INT16, rest INT8 naive" by orchestrating two sequential
patch_model calls with complementary skip_modules lists.

Regime fixed throughout: per_token=False, smooth=False, patch_lm_head=False.
Mixed precision obtained ONLY via the two-pass orchestration below; no
smoothing, no per-token scaling anywhere.

Three experiments
-----------------
1. Cumulative curve: top-k blocks (ranked by degrade-one ppl_ratio,
   descending) protected at INT16.  k in [0,1,2,3,4,5,6,8,10].
   k=0 is the full-INT8 baseline.

2. Reasoned sets: {21}, {16,17}, {16,17,4}, {16,17,21}, {16,17,4,21}.

3. Micro-diagnosis of block 4: only block 4 quantized to INT8 (rest FP,
   single pass); decompose final-layer residual error by channel, top-5.

FP reference (hs_fp, ppl_fp, ppl_fp per calibration prompt) computed ONCE
from a clean model and reused as the common denominator for all configs.
A fresh FP model is loaded before every individual config because
patch_model is destructive.

Run:
    python -m validation.mixed_precision_sweep

Output: results/raw/mixed_precision_tradeoff.json
"""
import gc
import json
import os
import sys

import torch

from .load_model import load
from .integer_patch import patch_model
from .error_vs_depth import layerwise_hidden, err, perplexity
from .calibration_prompts import CALIBRATION_PROMPTS, PERPLEXITY_TEXT

MODEL_NAME = "qwen2.5-0.5b"

RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "raw",
)
SENSITIVITY_PATH = os.path.join(RAW, "sensitivity_degrade_int8.json")

K_VALUES = [0, 1, 2, 3, 4, 5, 6, 8, 10]

REASONED_SETS = [
    {21},
    {16, 17},
    {16, 17, 4},
    {16, 17, 21},
    {16, 17, 4, 21},
]


# ---------------------------------------------------------------------------
# Two-pass mixed-precision patch
# ---------------------------------------------------------------------------

def _apply_mixed(model, protected, L):
    """
    Patch model in-place:
      pass 1 -> INT16 on protected blocks only
                (skip non-protected layers + model.norm)
      pass 2 -> INT8 on everything else
                (skip protected blocks; model.norm is patched here)
    """
    skip_for_int16 = (
        [f"model.layers.{j}" for j in range(L) if j not in protected]
        + ["model.norm"]
    )
    skip_for_int8 = [f"model.layers.{k}" for k in protected]

    patch_model(model, nbits=16, per_token=False, smooth=False,
                patch_lm_head=False, skip_modules=skip_for_int16)
    patch_model(model, nbits=8, per_token=False, smooth=False,
                patch_lm_head=False, skip_modules=skip_for_int8)


# ---------------------------------------------------------------------------
# Sanity check (first mixed config only)
# ---------------------------------------------------------------------------

def _sanity_check(model, protected, L):
    """
    Iterate over named modules; confirm protected blocks are INT16 and
    non-protected blocks are INT8.  Prints one line per block + model.norm.
    """
    from .integer_torch import QuantLinear, QuantRMSNorm

    print("\n[SANITY CHECK] first mixed config")
    print(f"  protected (INT16): blocks {sorted(protected)}")
    print(f"  {'block':>6}  {'expected':>8}  {'n_INT16':>7}  {'n_INT8':>6}  status")
    print(f"  {'-'*50}")

    all_ok = True
    for j in range(L):
        n16 = n8 = 0
        try:
            block = model.model.layers[j]
        except Exception:
            print(f"  block {j:02d}: cannot access")
            continue
        for _, m in block.named_modules():
            if isinstance(m, QuantLinear):
                if m.nbits == 16:
                    n16 += 1
                else:
                    n8 += 1
            elif isinstance(m, QuantRMSNorm):
                if m.nbits == 16:
                    n16 += 1
                else:
                    n8 += 1
        expected = "INT16" if j in protected else "INT8"
        ok = (
            (j in protected and n16 > 0 and n8 == 0) or
            (j not in protected and n8 > 0 and n16 == 0)
        )
        status = "OK" if ok else "WARN"
        if not ok:
            all_ok = False
        print(f"  block {j:02d}  {expected:>8}  {n16:>7}  {n8:>6}  {status}")

    # Check model.norm
    norm = getattr(getattr(model, "model", model), "norm", None)
    if isinstance(norm, QuantRMSNorm):
        norm_status = "OK" if norm.nbits == 8 else "WARN (expected INT8)"
        nbits_str = "INT" + str(norm.nbits)
        print(f"  model.norm       {nbits_str:>8}  {'':>7}  {'':>6}  {norm_status}")
    else:
        print(f"  model.norm       {'FP (not patched)':>8}  -- unexpected --")
    print(f"  overall: {'ALL OK' if all_ok else 'SEE WARN'}\n")


# ---------------------------------------------------------------------------
# Single-config measurement
# ---------------------------------------------------------------------------

def _measure_config(protected, L, hs_fp, ppl_fp, ppl_fp_calib, tok,
                    do_sanity=False):
    """
    Load a fresh FP model, apply mixed precision, measure ppl and
    residual-stream error.  Returns a flat dict with all metrics.
    """
    model, _ = load(MODEL_NAME)

    _apply_mixed(model, protected, L)

    if do_sanity and len(protected) > 0:
        _sanity_check(model, protected, L)

    hs_q = layerwise_hidden(model, tok, CALIBRATION_PROMPTS[0])
    ppl_q_text = perplexity(model, tok, PERPLEXITY_TEXT)

    ppl_ratios_calib = []
    for prompt, fp_val in zip(CALIBRATION_PROMPTS, ppl_fp_calib):
        q_val = perplexity(model, tok, prompt)
        ppl_ratios_calib.append(q_val / fp_val if fp_val > 0 else None)

    del model
    gc.collect()

    rel, cos = err(hs_q[-1], hs_fp[-1])

    valid = [r for r in ppl_ratios_calib if r is not None]
    calib_mean = sum(valid) / len(valid) if valid else None

    return {
        "ppl_ratio_perplexitytext": ppl_q_text / ppl_fp if ppl_fp > 0 else None,
        "ppl_ratio_calibmean": calib_mean,
        "final_relL2": rel,
        "final_cosine": cos,
    }


# ---------------------------------------------------------------------------
# Experiment 3: micro-diagnosis of block 4
# ---------------------------------------------------------------------------

def _block4_micro_diagnosis(hs_fp, tok, L, ppl_fp):
    """
    Quantize ONLY block 4 to INT8, rest stays FP (single patch_model pass).
    Decomposes final-layer residual-stream error by channel; returns top-5.
    """
    print("\n[EXP 3] micro-diagnosis of block 4 (INT8 only, rest FP) ...")
    model, _ = load(MODEL_NAME)

    skip = [f"model.layers.{j}" for j in range(L) if j != 4] + ["model.norm"]
    patch_model(model, nbits=8, per_token=False, smooth=False,
                patch_lm_head=False, skip_modules=skip)

    hs_q = layerwise_hidden(model, tok, CALIBRATION_PROMPTS[0])
    ppl_q = perplexity(model, tok, PERPLEXITY_TEXT)

    del model
    gc.collect()

    hq = hs_q[-1].to(torch.float64)
    hf = hs_fp[-1].to(torch.float64)

    diff_sq = (hq - hf).pow(2).sum(dim=(0, 1))   # (hidden_size,)
    total_sq = float(diff_sq.sum())

    k = min(5, diff_sq.numel())
    vals, idxs = torch.topk(diff_sq, k)

    top5 = [
        {
            "channel": int(idxs[i]),
            "sq_error": float(vals[i]),
            "pct_of_total": float(vals[i]) / total_sq * 100 if total_sq > 0 else None,
        }
        for i in range(k)
    ]

    rel, cos = err(hs_q[-1], hf)

    print(f"  ppl_ratio={ppl_q/ppl_fp:.4f}  final_relL2={rel:.6f}  cosine={cos:.6f}")
    print(f"  total_sq_error={total_sq:.6f}")
    print(f"  top-5 channels by squared error:")
    for t in top5:
        print(f"    channel {t['channel']:>4d}: sq={t['sq_error']:.6f}  "
              f"({t['pct_of_total']:.2f}% of total)")

    return {
        "description": (
            "Only block 4 quantized INT8; rest FP (single pass). "
            "Final residual-stream error decomposed by channel."
        ),
        "ppl_ratio": ppl_q / ppl_fp if ppl_fp > 0 else None,
        "final_relL2": rel,
        "final_cosine": cos,
        "total_sq_error": total_sq,
        "top5_channels_by_sq_error": top5,
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _row(ppl_text, ppl_calib, cosine):
    flag2 = ppl_text is not None and ppl_text <= 2.0
    flag11 = ppl_text is not None and ppl_text <= 1.1
    return (
        f"{ppl_text:>8.4f}" if ppl_text else f"{'None':>8}",
        f"{ppl_calib:>11.4f}" if ppl_calib else f"{'None':>11}",
        f"{cosine:>12.6f}" if cosine else f"{'None':>12}",
        f"{'T':>6}" if flag2 else f"{'F':>6}",
        f"{'T':>7}" if flag11 else f"{'F':>7}",
    )


def _print_tables(cumulative, reasoned, block4_diag):
    header = (f"{'ppl_ppl':>8}  {'ppl_calib':>11}  {'cosine':>12}"
              f"  {'ppl<=2':>6}  {'ppl<=1.1':>7}")

    # --- Table 1: cumulative curve ---
    print("\n" + "=" * 80)
    print("TABLE 1 — Cumulative curve (top-k blocks protected at INT16)")
    print("=" * 80)
    print(f"{'k':>3}  {'protected_blocks':<22}  {'n_INT16':>7}  " + header)
    print("-" * 80)
    for r in cumulative:
        blocks_str = str(r["protected"]) if r["protected"] else "[]"
        cols = _row(r["ppl_ratio_perplexitytext"],
                    r["ppl_ratio_calibmean"],
                    r["final_cosine"])
        print(f"{r['k']:>3}  {blocks_str:<22}  {r['n_protected']:>7}  "
              + "  ".join(cols))

    # --- Table 2: reasoned sets ---
    print("\n" + "=" * 80)
    print("TABLE 2 — Reasoned sets (decomposing contributions of key blocks)")
    print("=" * 80)
    print(f"{'set':<22}  {'n_INT16':>7}  " + header)
    print("-" * 80)
    for r in reasoned:
        set_str = str(r["protected"])
        cols = _row(r["ppl_ratio_perplexitytext"],
                    r["ppl_ratio_calibmean"],
                    r["final_cosine"])
        print(f"{set_str:<22}  {r['n_protected']:>7}  " + "  ".join(cols))

    # --- Block 4 micro-diagnosis summary ---
    print("\n" + "=" * 80)
    print("BLOCK 4 MICRO-DIAGNOSIS (INT8 only, rest FP)")
    print("=" * 80)
    d = block4_diag
    print(f"  ppl_ratio={d['ppl_ratio']:.4f}  "
          f"final_relL2={d['final_relL2']:.6f}  "
          f"final_cosine={d['final_cosine']:.6f}")
    print(f"  total_sq_error={d['total_sq_error']:.6f}")
    print(f"  top-5 channels by squared error:")
    for t in d["top5_channels_by_sq_error"]:
        print(f"    channel {t['channel']:>4d}: sq={t['sq_error']:.6f}  "
              f"({t['pct_of_total']:.2f}% of total)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    os.makedirs(RAW, exist_ok=True)

    # ---- FP reference (computed once, reused for all configs) --------------
    print(f"Loading FP reference model ({MODEL_NAME}) ...")
    model_ref, tok = load(MODEL_NAME)
    L = model_ref.config.num_hidden_layers
    print(f"  num_hidden_layers = {L}")

    print("Computing FP hidden states on CALIBRATION_PROMPTS[0] ...")
    hs_fp = layerwise_hidden(model_ref, tok, CALIBRATION_PROMPTS[0])

    print("Computing FP perplexity on PERPLEXITY_TEXT ...")
    ppl_fp = perplexity(model_ref, tok, PERPLEXITY_TEXT)

    print("Computing FP perplexity on all CALIBRATION_PROMPTS ...")
    ppl_fp_calib = [perplexity(model_ref, tok, p) for p in CALIBRATION_PROMPTS]

    del model_ref
    gc.collect()

    print(f"  ppl_fp (PERPLEXITY_TEXT) = {ppl_fp:.4f}")
    print(f"  ppl_fp (calib prompts)   = {[round(v,3) for v in ppl_fp_calib]}")

    # ---- Load degrade-one sensitivity ranking ------------------------------
    print(f"\nLoading sensitivity data from {SENSITIVITY_PATH} ...")
    with open(SENSITIVITY_PATH) as f:
        sens_data = json.load(f)

    blocks_sorted = sorted(sens_data["blocks"], key=lambda x: -x["ppl_ratio"])
    ranked_indices = [b["block"] for b in blocks_sorted]
    print("  Blocks ranked by degrade-one ppl_ratio (descending):")
    for rank, b in enumerate(blocks_sorted, 1):
        print(f"    rank {rank:>2d}: block {b['block']:>2d}  ppl_ratio={b['ppl_ratio']:.4f}")

    # ======================================================================
    # EXPERIMENT 1 — cumulative curve
    # ======================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 1 — cumulative curve")
    print(f"{'='*60}")

    cumulative = []
    first_mixed_sanity_done = False

    for k in K_VALUES:
        protected = set(ranked_indices[:k])
        do_sanity = (k > 0 and not first_mixed_sanity_done)
        print(f"\n  [k={k}] protected={sorted(protected)} ...")

        metrics = _measure_config(
            protected, L, hs_fp, ppl_fp, ppl_fp_calib, tok,
            do_sanity=do_sanity,
        )
        if do_sanity:
            first_mixed_sanity_done = True

        row = {
            "k": k,
            "protected": sorted(protected),
            "n_protected": len(protected),
            **metrics,
        }
        cumulative.append(row)
        print(f"    ppl_ratio_ppltext={row['ppl_ratio_perplexitytext']:.4f}  "
              f"ppl_ratio_calibmean={row['ppl_ratio_calibmean']:.4f}  "
              f"cosine={row['final_cosine']:.6f}")

    # ======================================================================
    # EXPERIMENT 2 — reasoned sets
    # ======================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 2 — reasoned sets")
    print(f"{'='*60}")

    reasoned = []
    for protected in REASONED_SETS:
        print(f"\n  [set={sorted(protected)}] ...")
        metrics = _measure_config(
            protected, L, hs_fp, ppl_fp, ppl_fp_calib, tok,
            do_sanity=False,
        )
        row = {
            "protected": sorted(protected),
            "n_protected": len(protected),
            **metrics,
        }
        reasoned.append(row)
        print(f"    ppl_ratio_ppltext={row['ppl_ratio_perplexitytext']:.4f}  "
              f"ppl_ratio_calibmean={row['ppl_ratio_calibmean']:.4f}  "
              f"cosine={row['final_cosine']:.6f}")

    # ======================================================================
    # EXPERIMENT 3 — micro-diagnosis of block 4
    # ======================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 3 — micro-diagnosis of block 4")
    print(f"{'='*60}")

    block4_diag = _block4_micro_diagnosis(hs_fp, tok, L, ppl_fp)

    # ======================================================================
    # Save JSON
    # ======================================================================
    result = {
        "model": MODEL_NAME,
        "regime": {
            "nbits_protected": 16,
            "nbits_rest": 8,
            "per_token": False,
            "smooth": False,
            "patch_lm_head": False,
        },
        "ppl_fp_perplexitytext": ppl_fp,
        "ppl_fp_calibration_prompts": ppl_fp_calib,
        "ranked_blocks_by_degrade_ppl_ratio": [
            {"rank": i + 1, "block": b["block"], "degrade_ppl_ratio": b["ppl_ratio"]}
            for i, b in enumerate(blocks_sorted)
        ],
        "baseline_full_int8": cumulative[0],
        "cumulative_curve": cumulative,
        "reasoned_sets": reasoned,
        "block4_micro_diagnosis": block4_diag,
    }

    out_path = os.path.join(RAW, "mixed_precision_tradeoff.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # ======================================================================
    # Print tables
    # ======================================================================
    _print_tables(cumulative, reasoned, block4_diag)

    print(f"\nDone.  Results in {out_path}")


if __name__ == "__main__":
    main()

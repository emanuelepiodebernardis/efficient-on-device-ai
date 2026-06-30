"""
Diagnostic driver: why does INT8 PPL remain catastrophic even when
smooth-0.5 recovers the residual stream (cosine@24=0.38, relL2@22=1.316)?

Four probes:
  1. Is model.norm the killer? 4 ppl_ratio variants (smooth05/naive x norm_FP/norm_INT8)
  2. Per-token NLL blow-up: concentrated on few tokens or uniform?
  3. Logit-level error for smooth05 (relL2, cosine, KL per token)
  4. Historical record: dump qwen2.5-0.5b_int8_pt_smooth.json

Constraints: nbits=8, patch_lm_head=False everywhere (unless indicated).
No skip_modules except for Sonda 1 model.norm variants.
Fresh model reload before every config.

Run:
    python -m validation.diagnose_ppl_collapse

Output: results/raw/diagnose_ppl_collapse.json
"""
import gc
import json
import os
import sys

import torch
import torch.nn.functional as F

from .load_model import load
from .integer_patch import patch_model
from .error_vs_depth import layerwise_hidden, err, perplexity
from .calibration_prompts import CALIBRATION_PROMPTS, PERPLEXITY_TEXT

MODEL_NAME = "qwen2.5-0.5b"

RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "raw",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_logits(model, tok, text):
    """Single forward pass -> logits (1, seq, vocab), CPU."""
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    return out.logits.detach().cpu()


def _per_token_nll(model, tok, text):
    """Per-token cross-entropy (seq_len - 1,) on CPU."""
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    shift_logits = out.logits[0, :-1, :].float()   # (seq-1, vocab)
    shift_labels = inp["input_ids"][0, 1:]           # (seq-1,)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    return loss_fn(shift_logits, shift_labels).detach().cpu()


# ---------------------------------------------------------------------------
# SONDA 1: is model.norm the killer?
# ---------------------------------------------------------------------------

def sonda1(ppl_fp, tok):
    variants = [
        {"label": "(a) smooth05_full",    "smooth": True,  "per_token": False, "alpha": 0.5, "skip": []},
        {"label": "(b) smooth05_norm_FP", "smooth": True,  "per_token": False, "alpha": 0.5, "skip": ["model.norm"]},
        {"label": "(c) naive_full",       "smooth": False, "per_token": False, "alpha": 0.5, "skip": []},
        {"label": "(d) naive_norm_FP",    "smooth": False, "per_token": False, "alpha": 0.5, "skip": ["model.norm"]},
    ]
    results = []
    for v in variants:
        print(f"  [S1] {v['label']} ...")
        model, _ = load(MODEL_NAME)
        patch_model(model, nbits=8, per_token=v["per_token"], smooth=v["smooth"],
                    alpha=v["alpha"], patch_lm_head=False, skip_modules=v["skip"])
        ppl_q = perplexity(model, tok, PERPLEXITY_TEXT)
        del model; gc.collect()
        ratio = ppl_q / ppl_fp if ppl_fp > 0 else None
        results.append({
            "label":     v["label"],
            "smooth":    v["smooth"],
            "norm_FP":   bool(v["skip"]),
            "ppl_int":   ppl_q,
            "ppl_ratio": ratio,
        })
        print(f"    ppl_int={ppl_q:.3f}  ppl_ratio={ratio:.4f}")
    return results


# ---------------------------------------------------------------------------
# SONDA 2: per-token NLL blow-up
# ---------------------------------------------------------------------------

def sonda2(nll_fp, tok):
    configs = [
        {"label": "naive",    "per_token": False, "smooth": False, "alpha": 0.5},
        {"label": "pertok",   "per_token": True,  "smooth": False, "alpha": 0.5},
        {"label": "smooth05", "per_token": False, "smooth": True,  "alpha": 0.5},
    ]
    results = []
    for cfg in configs:
        print(f"  [S2] {cfg['label']} ...")
        model, _ = load(MODEL_NAME)
        patch_model(model, nbits=8, per_token=cfg["per_token"], smooth=cfg["smooth"],
                    alpha=cfg["alpha"], patch_lm_head=False)
        nll_int = _per_token_nll(model, tok, PERPLEXITY_TEXT)
        del model; gc.collect()

        delta = nll_int - nll_fp                          # (seq-1,)
        total_delta = float(delta.sum())

        k = min(10, len(delta))
        top_vals, top_idx = torch.topk(delta, k)
        top10_sum = float(top_vals.sum())
        top10 = [
            {
                "position": int(top_idx[i]),
                "nll_fp":   float(nll_fp[top_idx[i]]),
                "nll_int":  float(nll_int[top_idx[i]]),
                "delta":    float(top_vals[i]),
            }
            for i in range(k)
        ]
        frac = top10_sum / total_delta if total_delta > 0 else None

        results.append({
            "label":                   cfg["label"],
            "nll_mean_fp":             float(nll_fp.mean()),
            "nll_mean_int":            float(nll_int.mean()),
            "total_delta":             total_delta,
            "top10_tokens":            top10,
            "top10_fraction_of_total": frac,
        })
        print(f"    nll_mean_fp={float(nll_fp.mean()):.4f}  "
              f"nll_mean_int={float(nll_int.mean()):.4f}  "
              f"top10_frac={frac:.4f}")
    return results


# ---------------------------------------------------------------------------
# SONDA 3: logit-level error for smooth05
# ---------------------------------------------------------------------------

def sonda3(logits_fp, tok):
    print("  [S3] smooth05 logit comparison ...")
    model, _ = load(MODEL_NAME)
    patch_model(model, nbits=8, per_token=False, smooth=True, alpha=0.5, patch_lm_head=False)
    logits_int = _get_logits(model, tok, PERPLEXITY_TEXT)
    del model; gc.collect()

    lf = logits_fp[0].float()    # (seq, vocab)
    li = logits_int[0].float()

    # Global relL2 and cosine on flattened logit tensor
    lf_d = lf.reshape(-1).double()
    li_d = li.reshape(-1).double()
    logit_relL2 = float((lf_d - li_d).norm() / (lf_d.norm() + 1e-12))
    logit_cosine = float((lf_d @ li_d) / (lf_d.norm() * li_d.norm() + 1e-12))

    # KL(P_fp || P_int) per token position
    log_p = F.log_softmax(lf.double(), dim=-1)   # (seq, vocab)
    log_q = F.log_softmax(li.double(), dim=-1)
    p = log_p.exp()
    kl_raw = (p * (log_p - log_q)).sum(dim=-1)   # (seq,)
    kl_per_token = torch.nan_to_num(kl_raw, nan=0.0, posinf=1e9, neginf=0.0)
    kl_mean = float(kl_per_token.mean())

    k5 = min(5, kl_per_token.numel())
    top5_vals, top5_idx = torch.topk(kl_per_token, k5)
    top5 = [{"position": int(top5_idx[i]), "kl": float(top5_vals[i])}
             for i in range(k5)]

    print(f"    logit relL2={logit_relL2:.6f}  cosine={logit_cosine:.6f}  "
          f"kl_mean={kl_mean:.6f}")

    return {
        "config":                 "smooth05 (per_token=False, smooth=True, alpha=0.5, patch_lm_head=False)",
        "logit_relL2":            logit_relL2,
        "logit_cosine":           logit_cosine,
        "kl_mean_fp_vs_int":      kl_mean,
        "top5_positions_by_kl":   top5,
    }


# ---------------------------------------------------------------------------
# SONDA 4: historical record
# ---------------------------------------------------------------------------

def sonda4():
    path = os.path.join(RAW, "qwen2.5-0.5b_int8_pt_smooth.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return {"file": os.path.basename(path), "found": True, "content": data}
    # Fallback: list any *smooth* JSON in RAW
    matches = [f for f in os.listdir(RAW) if "smooth" in f.lower() and f.endswith(".json")]
    return {"file": None, "found": False, "smooth_files_found": matches}


# ---------------------------------------------------------------------------
# Table printers
# ---------------------------------------------------------------------------

def _print_sonda1(results):
    print("\n" + "=" * 72)
    print("SONDA 1 — model.norm: is it the killer?")
    print("=" * 72)
    print(f"  {'label':<26}  {'smooth':>6}  {'norm_FP':>7}  {'ppl_int':>14}  {'ppl_ratio':>12}")
    print("-" * 72)
    for r in results:
        print(f"  {r['label']:<26}  {str(r['smooth']):>6}  {str(r['norm_FP']):>7}"
              f"  {r['ppl_int']:>14.3f}  {r['ppl_ratio']:>12.4f}")


def _print_sonda2(results):
    print("\n" + "=" * 80)
    print("SONDA 2 — per-token NLL blow-up")
    print("=" * 80)
    for r in results:
        print(f"\n  Config: {r['label']}")
        print(f"    nll_mean_fp   = {r['nll_mean_fp']:.6f}")
        print(f"    nll_mean_int  = {r['nll_mean_int']:.6f}")
        print(f"    total_delta   = {r['total_delta']:.6f}")
        frac = r["top10_fraction_of_total"]
        print(f"    top10_frac    = {frac:.6f}  ({frac*100:.2f}% of total blow-up)")
        print(f"    {'pos':>4}  {'nll_fp':>10}  {'nll_int':>14}  {'delta':>14}")
        print(f"    {'-'*48}")
        for t in r["top10_tokens"]:
            print(f"    {t['position']:>4}  {t['nll_fp']:>10.4f}  "
                  f"{t['nll_int']:>14.4f}  {t['delta']:>14.4f}")


def _print_sonda3(r):
    print("\n" + "=" * 70)
    print("SONDA 3 — logit-level error (smooth05)")
    print("=" * 70)
    print(f"  config    : {r['config']}")
    print(f"  relL2     : {r['logit_relL2']:.6f}")
    print(f"  cosine    : {r['logit_cosine']:.6f}")
    print(f"  KL(fp||int) mean : {r['kl_mean_fp_vs_int']:.6f}")
    print(f"  top-5 token positions by KL:")
    for t in r["top5_positions_by_kl"]:
        print(f"    pos {t['position']:>4}: KL = {t['kl']:.6f}")


def _print_sonda4(r):
    print("\n" + "=" * 70)
    print("SONDA 4 — historical record: qwen2.5-0.5b_int8_pt_smooth.json")
    print("=" * 70)
    if not r["found"]:
        print(f"  NOT FOUND.  Smooth-related files in results/raw/: {r['smooth_files_found']}")
        return
    c = r["content"]
    cfg = c.get("config", {})
    print(f"  file: {r['file']}")
    print(f"  config:")
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    print(f"  perplexity_fp    : {c.get('perplexity_fp')}")
    print(f"  perplexity_int   : {c.get('perplexity_int')}")
    print(f"  perplexity_ratio : {c.get('perplexity_ratio')}")
    fl = c.get("final_layer", {})
    print(f"  final_layer:")
    for k, v in fl.items():
        print(f"    {k}: {v}")


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
    L = model_ref.config.num_hidden_layers
    print(f"  num_hidden_layers = {L}")

    print("Computing FP hidden states (CALIBRATION_PROMPTS[0]) ...")
    hs_fp = layerwise_hidden(model_ref, tok, CALIBRATION_PROMPTS[0])

    print("Computing FP perplexity on PERPLEXITY_TEXT ...")
    ppl_fp = perplexity(model_ref, tok, PERPLEXITY_TEXT)

    print("Computing FP per-token NLL on PERPLEXITY_TEXT ...")
    nll_fp = _per_token_nll(model_ref, tok, PERPLEXITY_TEXT)

    print("Computing FP logits on PERPLEXITY_TEXT ...")
    logits_fp = _get_logits(model_ref, tok, PERPLEXITY_TEXT)

    del model_ref; gc.collect()
    print(f"  ppl_fp={ppl_fp:.4f}  nll_mean_fp={float(nll_fp.mean()):.4f}"
          f"  logits shape={list(logits_fp.shape)}")

    # ---- Probes ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SONDA 1 — model.norm variants")
    print("=" * 60)
    s1 = sonda1(ppl_fp, tok)

    print("\n" + "=" * 60)
    print("SONDA 2 — per-token NLL")
    print("=" * 60)
    s2 = sonda2(nll_fp, tok)

    print("\n" + "=" * 60)
    print("SONDA 3 — logit comparison smooth05")
    print("=" * 60)
    s3 = sonda3(logits_fp, tok)

    print("\n" + "=" * 60)
    print("SONDA 4 — historical record")
    print("=" * 60)
    s4 = sonda4()

    # ---- Save JSON ---------------------------------------------------------
    output = {
        "model":                   MODEL_NAME,
        "ppl_fp_perplexitytext":   ppl_fp,
        "nll_mean_fp":             float(nll_fp.mean()),
        "nll_fp_per_token":        nll_fp.tolist(),
        "sonda1_model_norm":       s1,
        "sonda2_per_token_nll":    s2,
        "sonda3_logit_smooth05":   s3,
        "sonda4_historical_record": s4,
    }
    out_path = os.path.join(RAW, "diagnose_ppl_collapse.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # ---- Print tables ------------------------------------------------------
    _print_sonda1(s1)
    _print_sonda2(s2)
    _print_sonda3(s3)
    _print_sonda4(s4)

    print(f"\nDone.  Results in {out_path}")


if __name__ == "__main__":
    main()

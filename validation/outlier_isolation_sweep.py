"""
Outlier-channel isolation sweep (Fork B).

Measures the effect of LLM.int8-style outlier isolation on perplexity and
residual-stream fidelity under naive INT8 (no smoothing, no per-token quant).
The only lever is outlier_k: how many input channels are kept in FP64.

Usage:
    python -m validation.outlier_isolation_sweep
"""
import json
import os
import torch
import torch.nn.functional as F

from .error_vs_depth import layerwise_hidden, err, perplexity
from .load_model import load
from .integer_patch import patch_model
from .calibration_prompts import CALIBRATION_PROMPTS, PERPLEXITY_TEXT

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")

MODEL_NAME = "qwen2.5-0.5b"
NBITS = 8
K_VALUES = [0, 1, 2, 4, 8, 16]

# Dotted attribute path → display label for diagnostic modules
DIAG_MODULES = [
    ("model.layers.2.mlp.down_proj",     "layers.2.mlp.down_proj"),
    ("model.layers.21.mlp.down_proj",    "layers.21.mlp.down_proj"),
    ("model.layers.18.self_attn.q_proj", "layers.18.self_attn.q_proj"),
    ("model.layers.18.self_attn.v_proj", "layers.18.self_attn.v_proj"),
]


def _get_module(model, dotpath):
    obj = model
    for part in dotpath.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _get_logits(model, tok, text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    return out.logits.detach()


def _logit_metrics(logit_fp, logit_int):
    lf = logit_fp.to(torch.float64).reshape(-1)
    li = logit_int.to(torch.float64).reshape(-1)
    cos = float((lf @ li) / (lf.norm() * li.norm() + 1e-12))
    p_fp   = F.softmax(logit_fp.to(torch.float64), dim=-1)     # (1, T, V)
    lp_int = F.log_softmax(logit_int.to(torch.float64), dim=-1)
    # KL(P_fp || P_int) per token, then mean across tokens
    kl = float(F.kl_div(lp_int, p_fp, reduction="none").sum(dim=-1).mean())
    return cos, kl


def _collect_diag(model, tok):
    inp = tok(CALIBRATION_PROMPTS[0], return_tensors="pt")
    with torch.no_grad():
        model(**inp)
    result = {}
    for dotpath, label in DIAG_MODULES:
        try:
            m = _get_module(model, dotpath)
            result[label] = getattr(m, "last_outlier_idx", None)
        except Exception as exc:
            result[label] = f"ERROR: {exc}"
    return result


def _print_table(results, piolo):
    rows = sorted([r for r in results if r["piolo"] == piolo], key=lambda r: r["K"])
    label = "FP norms (skip RMSNorm)" if piolo == 1 else "INT8 norms (all quantized)"
    print(f"\n=== PIOLO {piolo}: {label} ===")
    hdr = (f"{'K':>4}  {'ppl PERPLEXITY_TEXT':>19}  {'ppl calib mean':>14}  "
           f"{'logit cos':>9}  {'KL mean':>9}  {'cos residual':>12}  "
           f"{'ppl<=2':>6}  {'ppl<=1.1':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f2  = "YES" if r["ppl_ratio_perplexitytext"] <= 2.0   else "no"
        f11 = "YES" if r["ppl_ratio_perplexitytext"] <= 1.1   else "no"
        print(
            f"{r['K']:>4}  {r['ppl_ratio_perplexitytext']:>19.4f}  "
            f"{r['ppl_ratio_calibmean']:>14.4f}  "
            f"{r['logit_cosine']:>9.6f}  {r['kl_mean']:>9.4f}  "
            f"{r['final_cosine']:>12.6f}  {f2:>6}  {f11:>8}"
        )


def main():
    os.makedirs(RAW, exist_ok=True)

    # ── FP reference (computed once from a clean model) ───────────────────────
    print(f"Loading {MODEL_NAME} for FP reference ...")
    model_fp, tok = load(MODEL_NAME)

    norm_names = [
        name for name, m in model_fp.named_modules()
        if m.__class__.__name__.endswith("RMSNorm")
    ]
    print(f"  Found {len(norm_names)} RMSNorm modules (used for PIOLO 1 skip list)")

    print("  FP: layerwise hidden states on CALIBRATION_PROMPTS[0] ...")
    hs_fp = layerwise_hidden(model_fp, tok, CALIBRATION_PROMPTS[0])

    print("  FP: logits on PERPLEXITY_TEXT ...")
    logit_fp_ref = _get_logits(model_fp, tok, PERPLEXITY_TEXT)

    print("  FP: perplexities ...")
    ppl_fp       = perplexity(model_fp, tok, PERPLEXITY_TEXT)
    ppl_fp_calib = [perplexity(model_fp, tok, p) for p in CALIBRATION_PROMPTS]
    print(f"  ppl_fp (PERPLEXITY_TEXT)={ppl_fp:.4f}  "
          f"calib=[{', '.join(f'{v:.2f}' for v in ppl_fp_calib)}]")
    del model_fp

    # ── Sanity check: PIOLO 2, K=0 must reproduce naive INT8 ratio ───────────
    print(f"\nSanity check: PIOLO 2, K=0 (expected ppl_ratio ~61193) ...")
    model_s, _ = load(MODEL_NAME)
    patch_model(model_s, nbits=NBITS, per_token=False, smooth=False,
                patch_lm_head=False, outlier_k=0, verbose=False)
    ppl_sanity      = perplexity(model_s, tok, PERPLEXITY_TEXT)
    ratio_sanity    = ppl_sanity / ppl_fp
    sanity_ok       = abs(ratio_sanity - 61193) < 20000
    print(f"  ppl_int={ppl_sanity:.2f}  ppl_ratio={ratio_sanity:.2f}  "
          f"{'OK — matches expected ~61193' if sanity_ok else 'WARNING — differs from expected'}")
    del model_s

    # ── Main sweep ────────────────────────────────────────────────────────────
    results      = []
    diag_channels = None

    for piolo in [1, 2]:
        p_label = "FP norms" if piolo == 1 else "INT8 norms"
        print(f"\n{'='*62}\nPIOLO {piolo}: {p_label}\n{'='*62}")

        for K in K_VALUES:
            print(f"  PIOLO {piolo}, K={K} — loading fresh model ...")
            model, _ = load(MODEL_NAME)

            skip = norm_names if piolo == 1 else None
            patch_model(model, nbits=NBITS, per_token=False, smooth=False,
                        patch_lm_head=False, outlier_k=K,
                        skip_modules=skip, verbose=False)

            ppl_int          = perplexity(model, tok, PERPLEXITY_TEXT)
            ppl_ratio        = ppl_int / ppl_fp

            ppl_int_calib    = [perplexity(model, tok, p) for p in CALIBRATION_PROMPTS]
            ppl_ratio_calibmean = (
                sum(pi / pf for pi, pf in zip(ppl_int_calib, ppl_fp_calib))
                / len(CALIBRATION_PROMPTS)
            )

            logit_int        = _get_logits(model, tok, PERPLEXITY_TEXT)
            logit_cos, kl    = _logit_metrics(logit_fp_ref, logit_int)

            hs_q             = layerwise_hidden(model, tok, CALIBRATION_PROMPTS[0])
            final_rel, final_cos = err(hs_q[-1], hs_fp[-1])

            # Diagnostics: read last_outlier_idx right after the layerwise forward
            if piolo == 2 and K == 8 and diag_channels is None:
                print("    Collecting channel diagnostics ...")
                diag_channels = _collect_diag(model, tok)

            row = {
                "piolo": piolo,
                "K": K,
                "ppl_ratio_perplexitytext": ppl_ratio,
                "ppl_ratio_calibmean": ppl_ratio_calibmean,
                "logit_cosine": logit_cos,
                "kl_mean": kl,
                "final_relL2": final_rel,
                "final_cosine": final_cos,
            }
            results.append(row)
            print(f"    => ppl_ratio={ppl_ratio:.4f}  logit_cos={logit_cos:.6f}  "
                  f"kl={kl:.4f}  cos={final_cos:.6f}")
            del model

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "fp_reference": {
            "ppl_fp_perplexitytext": ppl_fp,
            "ppl_fp_calib": {f"prompt_{i}": v for i, v in enumerate(ppl_fp_calib)},
        },
        "sanity_piolo2_k0": {
            "ppl_int": ppl_sanity,
            "ppl_ratio": ratio_sanity,
            "ok": sanity_ok,
        },
        "configs": results,
        "diagnostics_piolo2_k8": diag_channels,
    }
    out_path = os.path.join(RAW, "outlier_isolation_sweep.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # ── Tables ────────────────────────────────────────────────────────────────
    _print_table(results, piolo=1)
    _print_table(results, piolo=2)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\n=== DIAGNOSTICA CANALI (PIOLO 2, K=8) ===")
    if diag_channels:
        for label, indices in diag_channels.items():
            print(f"  {label}: {indices}")
    else:
        print("  (not collected)")


if __name__ == "__main__":
    main()

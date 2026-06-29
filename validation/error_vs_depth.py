"""
REAL-MODEL VALIDATION DRIVER (Phase 0 protocol).

For a chosen config, measures:
  1. Per-layer residual-stream error (relL2, cosine) between FP and integer forward,
     across depth -> the real-model version of the synthetic accumulation curve.
  2. Perplexity (FP vs integer) on a fixed passage -> THE key question: does the
     residual-stream drift translate into task-level loss?

Usage (run in the build env, where HF is reachable):
    python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 16
    python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 8
    python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 8 --smooth --per-token

Outputs JSON to results/raw/ and prints a table.

Success criteria (from docs/findings): INT16 final-layer cosine >= 0.9999;
INT8 cosine >= 0.98 with characterized drift; perplexity ratio close to 1.0.
"""
import argparse
import json
import os
import math
import torch

from .load_model import load
from .integer_patch import patch_model
from .calibration_prompts import CALIBRATION_PROMPTS, PERPLEXITY_TEXT

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")


def layerwise_hidden(model, tok, text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    # tuple length n_layers+1: embeddings + each layer's output
    return [h.detach().to(torch.float64) for h in out.hidden_states]


def err(a, b):
    rel = (a - b).norm() / b.norm()
    cos = (a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-12)
    return float(rel), float(cos)


def perplexity(model, tok, text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, labels=inp["input_ids"])
    return float(torch.exp(out.loss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--per-token", action="store_true")
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--patch-lm-head", action="store_true")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    text = CALIBRATION_PROMPTS[0]

    print(f"Loading {args.model} ...")
    model, tok = load(args.model)

    print("FP reference forward ...")
    hs_fp = layerwise_hidden(model, tok, text)
    ppl_fp = perplexity(model, tok, PERPLEXITY_TEXT)

    print("Patching to integer ...")
    patch_model(model, nbits=args.nbits, per_token=args.per_token,
                smooth=args.smooth, alpha=args.alpha, patch_lm_head=args.patch_lm_head)

    print("Integer forward ...")
    hs_q = layerwise_hidden(model, tok, text)
    ppl_q = perplexity(model, tok, PERPLEXITY_TEXT)

    rows = []
    for i, (a, b) in enumerate(zip(hs_q, hs_fp)):
        rel, cos = err(a, b)
        rows.append({"layer": i, "relL2": rel, "cosine": cos})

    tag = f"{args.model}_int{args.nbits}" + ("_pt" if args.per_token else "") + ("_smooth" if args.smooth else "")
    result = {
        "config": vars(args),
        "perplexity_fp": ppl_fp,
        "perplexity_int": ppl_q,
        "perplexity_ratio": ppl_q / ppl_fp if ppl_fp else None,
        "layers": rows,
        "final_layer": rows[-1],
    }
    path = os.path.join(RAW, tag + ".json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {tag} ===")
    sel = [0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4, len(rows) - 1]
    print("  layer:  " + "  ".join(f"{rows[i]['layer']:>7d}" for i in sel))
    print("  relL2:  " + "  ".join(f"{rows[i]['relL2']:7.4f}" for i in sel))
    print("  cosine: " + "  ".join(f"{rows[i]['cosine']:7.4f}" for i in sel))
    print(f"  perplexity  FP={ppl_fp:.3f}  INT={ppl_q:.3f}  ratio={result['perplexity_ratio']:.4f}")
    print(f"  saved -> {path}")


if __name__ == "__main__":
    main()

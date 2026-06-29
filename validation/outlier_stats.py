"""
Capture real activation outlier statistics, to check whether the synthetic regimes
(massive activations, post-norm FFN outliers) cover the real model's behavior.

For each Linear input, records per-channel max/median ratio (the outlier-feature
signature). High, channel-localized ratios => SmoothQuant-style smoothing is the
right fix (as found in the synthetic study).

Usage:
    python -m validation.outlier_stats --model qwen2.5-0.5b
"""
import argparse
import os
import json
import torch
from .load_model import load
from .calibration_prompts import CALIBRATION_PROMPTS

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)

    model, tok = load(args.model)
    stats = {}
    hooks = []

    def mk_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).abs()
            chan_max = x.amax(dim=0)                 # (in,)
            med = x.median()
            ratio = float(chan_max.max() / (med + 1e-9))
            stats.setdefault(name, []).append(ratio)
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(mk_hook(name)))

    with torch.no_grad():
        for p in CALIBRATION_PROMPTS:
            model(**tok(p, return_tensors="pt"))

    for h in hooks:
        h.remove()

    summary = {k: {"mean_max_over_median": sum(v) / len(v), "n": len(v)} for k, v in stats.items()}
    worst = sorted(summary.items(), key=lambda kv: -kv[1]["mean_max_over_median"])[:10]

    path = os.path.join(RAW, f"outlier_stats_{args.model}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Top-10 linears by activation outlier ratio (max/median):")
    for name, s in worst:
        print(f"  {s['mean_max_over_median']:10.1f}x   {name}")
    print(f"\nInterpretation: ratios >> 10x that are channel-localized confirm the")
    print(f"channel-outlier regime -> smoothing is the right INT8 fix.")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()

"""
DIAGNOSTIC: Why does INT8 collapse at block 21-22 of Qwen2.5-0.5B?

This is a pure diagnosis script — no fix is applied, no documents are updated.

Steps executed:
  1. FP hidden-state statistics for layers 18-24: mean L2/token, max abs, top-5
     positions; check for massive-activation signature (Sun et al.: few fixed channels
     with persistently large values across tokens).
  2. At layer 22 (output of block 21): channel-wise and token-wise error decomposition
     for the INT8 naive forward; ablation of top-k channels to measure concentration.
  3. Selective ablation: keep subsets of modules in FP, measure cosine@24 + ppl_ratio.
  4. Print comparison: critical module from ablation vs outlier-maximum module.

Run:
    python -m validation.diagnose_block22
"""
import copy
import sys
import torch
from .load_model import load
from .integer_patch import patch_model
from .calibration_prompts import CALIBRATION_PROMPTS, PERPLEXITY_TEXT

TEXT = CALIBRATION_PROMPTS[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_hidden(model, tok, text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return [h.detach().to(torch.float64) for h in out.hidden_states]


def ppl(model, tok, text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, labels=inp["input_ids"])
    return float(torch.exp(out.loss))


def cosrel(a, b):
    rel = float((a - b).norm() / (b.norm() + 1e-12))
    cos = float((a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-12))
    return rel, cos


# ---------------------------------------------------------------------------
# Step 1 — FP hidden-state stats for layers 18-24
# ---------------------------------------------------------------------------

def step1_fp_stats(hs_fp, tok, text):
    tokens = tok.convert_ids_to_tokens(tok(text)["input_ids"])
    tokens_safe = [t.encode("ascii", "replace").decode("ascii") for t in tokens]
    T = hs_fp[0].shape[1]
    H = hs_fp[0].shape[2]

    print("\n" + "=" * 72)
    print("STEP 1 -- FP Hidden-State Statistics (layers 18-24)")
    print(f"  prompt tokens ({T}): {tokens_safe}")
    print(f"  hidden dim H = {H}")
    print("=" * 72)
    print(f"{'L':>4}  {'meanL2/tok':>12}  {'maxAbs':>10}  top-5 [abs, tok, chan]")
    print("-" * 72)

    chan_appearances = {}   # chan -> list of (layer_idx, tok_idx, raw_val)

    for li in range(18, 25):
        if li >= len(hs_fp):
            break
        h = hs_fp[li][0]                          # (T, H)
        mean_l2 = float(h.norm(dim=-1).mean())
        max_abs  = float(h.abs().max())

        flat_abs = h.abs()                         # (T, H)
        top_vals, top_idx = flat_abs.reshape(-1).topk(5)
        top5 = []
        for v, fi in zip(top_vals.tolist(), top_idx.tolist()):
            t_i = fi // H
            c_i = fi % H
            raw_v = float(h[t_i, c_i])
            top5.append((v, t_i, c_i, raw_v))
            chan_appearances.setdefault(c_i, []).append((li, t_i, raw_v))

        top5_str = "  ".join(f"[{v:.1f} t{t} c{c}]" for v, t, c, _ in top5)
        print(f"{li:>4}  {mean_l2:>12.4f}  {max_abs:>10.2f}  {top5_str}")

    # Massive-activation diagnosis: channels recurring in top-5 across layers 18-24
    print("\n--- Channel recurrence in top-5 (layers 18–24) ---")
    recurring = {c: occs for c, occs in chan_appearances.items() if len(occs) >= 2}
    if recurring:
        for c, occs in sorted(recurring.items(), key=lambda x: -len(x[1])):
            tok_set = sorted(set(t for _, t, _ in occs))
            print(f"  chan {c:>4}: appears {len(occs)}x across layers "
                  f"{sorted(set(l for l,_,_ in occs))}  "
                  f"tok_indices={tok_set}  "
                  f"vals=[{', '.join(f'{v:.1f}' for _,_,v in occs)}]")
        print(f"  -> MASSIVE ACTIVATION signature: {len(recurring)} channel(s) "
              f"reappear across multiple layers")
    else:
        print("  No channel reappears in top-5 across layers 18-24 -> values SCATTERED")

    # Per-layer: top channel concentration (what fraction of L2 energy sits in top-5 chan?)
    print("\n--- Top-5 channel energy concentration (fraction of total L2 energy) ---")
    print(f"{'L':>4}  {'top1_chan':>9}  {'top1%':>7}  {'top5%':>7}  {'top10%':>8}")
    for li in range(18, 25):
        if li >= len(hs_fp):
            break
        h = hs_fp[li][0]                          # (T, H)
        chan_energy = h.pow(2).sum(dim=0)          # (H,)
        total_e = float(chan_energy.sum())
        top10_idx = chan_energy.argsort(descending=True)[:10]
        top1_c = int(top10_idx[0])
        top1_pct  = 100 * float(chan_energy[top10_idx[0]])  / total_e
        top5_pct  = 100 * float(chan_energy[top10_idx[:5]].sum()) / total_e
        top10_pct = 100 * float(chan_energy[top10_idx].sum())      / total_e
        print(f"{li:>4}  {top1_c:>9}  {top1_pct:>7.3f}%  {top5_pct:>7.3f}%  {top10_pct:>8.3f}%")


# ---------------------------------------------------------------------------
# Step 2 — Error decomposition at layer 22 (INT8 naive)
# ---------------------------------------------------------------------------

def step2_error_decomp(hs_fp, hs_q8, layer_idx=22):
    print("\n" + "=" * 72)
    print(f"STEP 2 — Error Decomposition at Layer {layer_idx} (INT8 naive)")
    print("=" * 72)

    h_fp = hs_fp[layer_idx][0]    # (T, H)
    h_q  = hs_q8[layer_idx][0]    # (T, H)
    e    = h_q - h_fp             # (T, H)

    ref_sq  = float(h_fp.pow(2).sum())
    err_sq  = float(e.pow(2).sum())
    relL2   = (err_sq / ref_sq) ** 0.5
    _, cos  = cosrel(h_q, h_fp)

    print(f"  relL2  = {relL2:.6f}")
    print(f"  cosine = {cos:.6f}")
    print(f"  ref_norm² = {ref_sq:.4f}   err² = {err_sq:.4f}")

    # Per-channel squared error (sum over tokens)
    chan_err_sq = e.pow(2).sum(dim=0)   # (H,)
    top_chan    = chan_err_sq.argsort(descending=True)
    H = e.shape[1]

    print(f"\n--- Top-10 channels by Σ_token squared error ---")
    print(f"{'chan':>6}  {'errSq':>12}  {'%errSq':>8}  {'maxAbsErr':>10}  {'maxAbsFP':>10}")
    for i in range(min(10, H)):
        c    = int(top_chan[i])
        esq  = float(chan_err_sq[c])
        pct  = 100 * esq / (err_sq + 1e-12)
        mxe  = float(e[:, c].abs().max())
        mxfp = float(h_fp[:, c].abs().max())
        print(f"{c:>6}  {esq:>12.4f}  {pct:>8.3f}%  {mxe:>10.4f}  {mxfp:>10.4f}")

    # Ablation: if top-k channels were error-free, what's remaining relL2?
    print(f"\n--- RelL2 remaining if top-k channels were fixed (error set to 0) ---")
    print(f"{'k':>5}  {'relL2_remaining':>16}  {'%errSq_removed':>16}")
    for k in [1, 2, 5, 10, 20, 50, 100]:
        topk_esq   = float(chan_err_sq[top_chan[:k]].sum())
        rem_relL2  = max(0.0, (err_sq - topk_esq) / ref_sq) ** 0.5
        pct_gone   = 100 * topk_esq / (err_sq + 1e-12)
        print(f"{k:>5}  {rem_relL2:>16.6f}  {pct_gone:>15.3f}%")

    # Per-token breakdown
    tok_err_sq = e.pow(2).sum(dim=1)    # (T,)
    top_tok    = tok_err_sq.argsort(descending=True)
    T = e.shape[0]
    print(f"\n--- Top-5 tokens by Σ_channel squared error ---")
    print(f"{'tok':>5}  {'errSq':>12}  {'%errSq':>8}")
    for i in range(min(5, T)):
        t   = int(top_tok[i])
        esq = float(tok_err_sq[t])
        pct = 100 * esq / (err_sq + 1e-12)
        print(f"{t:>5}  {esq:>12.4f}  {pct:>8.3f}%")

    # Check if top error channels are also top FP channels (are the outliers the culprits?)
    top5_err_chan  = set(int(top_chan[i]) for i in range(5))
    fp_chan_energy = h_fp.pow(2).sum(dim=0).argsort(descending=True)
    top5_fp_chan   = set(int(fp_chan_energy[i]) for i in range(5))
    overlap = top5_err_chan & top5_fp_chan
    print(f"\n  top-5 error channels : {sorted(top5_err_chan)}")
    print(f"  top-5 FP energy chans: {sorted(top5_fp_chan)}")
    print(f"  overlap              : {sorted(overlap)} "
          f"({'YES - outlier channels drive error' if overlap else 'NO - error is in non-outlier channels'})")

    return relL2


# ---------------------------------------------------------------------------
# Step 3 — Ablation: selective FP modules
# ---------------------------------------------------------------------------

def ablation_run(model_fp, tok, skip_mods, hs_fp, ppl_fp):
    m = copy.deepcopy(model_fp)
    patch_model(m, nbits=8, per_token=False, smooth=False,
                skip_modules=skip_mods, verbose=False)
    hs_q = get_hidden(m, tok, TEXT)
    ppl_q = ppl(m, tok, PERPLEXITY_TEXT)
    del m

    rl22, cos22 = cosrel(hs_q[22], hs_fp[22])
    rl24, cos24 = cosrel(hs_q[24], hs_fp[24])
    ratio = ppl_q / (ppl_fp + 1e-9)
    return dict(relL2_22=rl22, cos22=cos22, relL2_24=rl24, cos24=cos24,
                ppl_q=ppl_q, ppl_ratio=ratio)


def step3_ablation(model_fp, tok, hs_fp, ppl_fp):
    n_layers = model_fp.config.num_hidden_layers  # 24 for Qwen2.5-0.5B

    ablations = [
        # baseline: everything INT8
        ([], "BASELINE — fully INT8"),
        # (a)
        (["model.layers.21.mlp.down_proj"],
         "(a) layers[21].mlp.down_proj in FP"),
        # (b)
        (["model.layers.21"],
         "(b) block 21 (attn+mlp) in FP"),
        # (c)
        (["model.layers.21", "model.layers.22"],
         "(c) blocks 21+22 in FP"),
        # (d)
        ([f"model.layers.{i}.mlp.down_proj" for i in range(n_layers)],
         "(d) ALL down_proj in FP"),
    ]

    print("\n" + "=" * 72)
    print("STEP 3 — Ablation: selective FP modules (everything else INT8)")
    print(f"  n_layers = {n_layers}")
    print("=" * 72)
    hdr = f"{'Config':<42}  {'cos@22':>8}  {'relL2@22':>10}  {'cos@24':>8}  {'ppl_ratio':>10}  {'collapse?':>9}"
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for skip_mods, label in ablations:
        r = ablation_run(model_fp, tok, skip_mods, hs_fp, ppl_fp)
        collapsed = "YES" if r["cos24"] < 0.9 else "no"
        print(f"{label:<42}  {r['cos22']:>8.5f}  {r['relL2_22']:>10.5f}"
              f"  {r['cos24']:>8.5f}  {r['ppl_ratio']:>10.3f}  {collapsed:>9}")
        results[label] = r

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Ensure UTF-8 output on Windows where the default console encoding may be cp1252
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading Qwen2.5-0.5B ...")
    model_fp, tok = load("qwen2.5-0.5b")
    ppl_fp = ppl(model_fp, tok, PERPLEXITY_TEXT)
    print(f"FP perplexity: {ppl_fp:.3f}")

    print("\nFP forward (output_hidden_states=True) ...")
    hs_fp = get_hidden(model_fp, tok, TEXT)
    n_hs  = len(hs_fp)
    T, H  = hs_fp[0].shape[1], hs_fp[0].shape[2]
    print(f"hidden_states: {n_hs} tensors  (embeddings + {n_hs-1} layers)  T={T}  H={H}")

    # ---- Step 1 ----
    step1_fp_stats(hs_fp, tok, TEXT)

    # ---- Build INT8 hidden states for Step 2 ----
    print("\nBuilding INT8 naive forward ...")
    model_q8 = copy.deepcopy(model_fp)
    patch_model(model_q8, nbits=8, per_token=False, smooth=False, verbose=True)
    hs_q8 = get_hidden(model_q8, tok, TEXT)
    del model_q8

    # Print INT8 relL2/cosine curve for reference
    print("\n--- INT8 naive: relL2 and cosine by layer (full) ---")
    print(f"{'L':>4}  {'relL2':>10}  {'cosine':>10}")
    for i in range(n_hs):
        rl, co = cosrel(hs_q8[i], hs_fp[i])
        print(f"{i:>4}  {rl:>10.6f}  {co:>10.6f}")

    # ---- Step 2 ----
    step2_error_decomp(hs_fp, hs_q8, layer_idx=22)

    # ---- Step 3 ----
    abl = step3_ablation(model_fp, tok, hs_fp, ppl_fp)

    # ---- Step 4 summary printed inline ----
    print("\n" + "=" * 72)
    print("STEP 4 — Cross-reference: critical module vs outlier-maximum module")
    print("=" * 72)
    print("  Known outlier ratios (max/median of linear INPUT, from outlier_stats):")
    print("    model.layers.2.mlp.down_proj  : 38983x  (max in model)")
    print("    model.layers.3.mlp.down_proj  : 21522x")
    print("    model.layers.21.mlp.down_proj : 13331x  <- block of collapse")
    print("    model.layers.21.mlp.gate_proj : 366x    <- hidden state entering block 21 MLP")
    print()
    print("  INT8 collapse location (from error_vs_depth):")
    print("    layer 21: relL2=0.1781  cosine=0.9896  (degraded but alive)")
    print("    layer 22: relL2=8.876   cosine=-0.016  <- CATASTROPHIC JUMP")
    print()
    print("  INT16 behaviour (same block):")
    print("    layer 21: relL2=0.00330  cosine=0.999995")
    print("    layer 22: relL2=0.02586  cosine=0.99967   <- 7.8x jump, still OK")
    print()
    print("  Ablation critical module -> see step 3 table above.")
    print("  If (a) alone restores cosine@24 > 0.9  => down_proj[21] is sufficient.")
    print("  If (b)/(c) needed      => residual-stream state entering block 21 also matters.")
    print("  If (d) also improves   => outlier accumulation across ALL down_proj is additive.")
    print()
    print("  HYPOTHESIS (to be confirmed by numbers):")
    print("  The collapse is NOT caused by the outlier magnitude alone (layers 2-3 have")
    print("  larger outliers and survive). It is a COMPOUND FAILURE:")
    print("    1. Accumulated INT8 drift in the residual stream (21 layers, relL2~0.18)")
    print("    2. Block 21 MLP intermediate activations have 13331x outlier in down_proj input")
    print("    3. INT8 (8 bits, range ±127) cannot represent a 13331x dynamic range:")
    print("       almost all non-outlier intermediate channels are zeroed by the scale factor")
    print("    4. down_proj output is dominated by a single badly-scaled channel")
    print("    5. This erroneous residual update, added to an already-drifted stream, tips")
    print("       the cosine below zero (full collapse)")
    print("  At layers 2-3 the same outlier pattern exists but the residual stream is still")
    print("  clean, AND the erroneous correction might partially cancel earlier error.")
    print("  The INT16 7.8x jump at layer 22 confirms the outlier IS hard even at 16 bits,")
    print("  but the absolute error (relL2=0.026) is small enough not to flip cosine.")
    print()
    print("  REMEDY HYPOTHESIS (not implemented):")
    print("  SmoothQuant applied to layers[21].mlp.down_proj (and possibly the other")
    print("  top-outlier down_projs at layers 2 and 3) should be sufficient, since the")
    print("  outlier is channel-localized (measurable via per-channel activation stats).")
    print("  The smoother redistributes the dynamic range between activations and weights,")
    print("  so that INT8 can represent the intermediate activations without catastrophic")
    print("  zeroing of non-outlier channels.")


if __name__ == "__main__":
    main()

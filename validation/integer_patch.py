"""
Patch a Hugging Face causal-LM in place: replace nn.Linear with QuantLinear and
*RMSNorm modules with QuantRMSNorm.

Scope of this first validation (honest, matches the synthetic study):
  * All Linear layers (q/k/v/o, gate/up/down, optionally lm_head) -> integer matmul.
  * All RMSNorm -> integer RMSNorm.
  * Attention SOFTMAX is left in FP for now (patching it requires model-specific
    surgery in the attention forward). int_softmax_t is provided in integer_torch.py
    for the advanced pass. This is the same scope limitation noted in the synthetic
    study (attention score matmuls were FP there too).
"""
import torch
from .integer_torch import QuantLinear, QuantRMSNorm


def _set_module(model, name, new):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def patch_model(model, nbits=8, per_token=False, smooth=False, alpha=0.5,
                patch_lm_head=False, verbose=True, skip_modules=None, outlier_k=0):
    """
    skip_modules: list of module names or name-prefixes to keep in FP (not patched).
    A module is skipped when its dotted name equals a skip entry exactly, or starts
    with '<entry>.' (i.e. is a sub-module of a skipped block).
    Example: skip_modules=["model.layers.21"]  keeps the whole block in FP;
             skip_modules=["model.layers.21.mlp.down_proj"]  keeps only that linear.
    """
    skip = list(skip_modules or [])

    def _is_skipped(name):
        return any(name == s or name.startswith(s + ".") for s in skip)

    n_lin = n_rms = 0
    for name, module in list(model.named_modules()):
        if _is_skipped(name):
            continue
        is_lm_head = name.endswith("lm_head")
        if isinstance(module, torch.nn.Linear) and (patch_lm_head or not is_lm_head):
            _set_module(model, name, QuantLinear(module, nbits, per_token, smooth, alpha, outlier_k))
            n_lin += 1
        elif module.__class__.__name__.endswith("RMSNorm"):
            _set_module(model, name, QuantRMSNorm(module, nbits))
            n_rms += 1
    if verbose:
        print(f"[patch] replaced {n_lin} Linear -> QuantLinear, {n_rms} RMSNorm -> QuantRMSNorm "
              f"(nbits={nbits}, per_token={per_token}, smooth={smooth})")
        if skip:
            print(f"[patch] kept in FP (skip_modules): {skip}")
    return model

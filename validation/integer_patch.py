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
                patch_lm_head=False, verbose=True):
    n_lin = n_rms = 0
    for name, module in list(model.named_modules()):
        is_lm_head = name.endswith("lm_head")
        if isinstance(module, torch.nn.Linear) and (patch_lm_head or not is_lm_head):
            _set_module(model, name, QuantLinear(module, nbits, per_token, smooth, alpha))
            n_lin += 1
        elif module.__class__.__name__.endswith("RMSNorm"):
            _set_module(model, name, QuantRMSNorm(module, nbits))
            n_rms += 1
    if verbose:
        print(f"[patch] replaced {n_lin} Linear -> QuantLinear, {n_rms} RMSNorm -> QuantRMSNorm "
              f"(nbits={nbits}, per_token={per_token}, smooth={smooth})")
    return model

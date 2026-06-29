"""Load a small causal-LM on CPU in float32 for FP-reference validation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Small, LLaMA-style, sub-2B models (RMSNorm + SiLU MLP). Pick one.
DEFAULT_MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def load(model_key="qwen2.5-0.5b"):
    name = DEFAULT_MODELS.get(model_key, model_key)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    return model, tok

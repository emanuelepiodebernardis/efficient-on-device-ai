"""
Edge benchmark orchestrator (Phase 0/1, runs ON the edge device).

Drives llama.cpp / bitnet.cpp baseline runs and collects the metrics in
metrics_schema.md into results/raw/. This is a SKELETON: fill in the exact
binary paths and flags for your build (see docs/piano_operativo_fase0_fase1.md).

Baselines to cover (defined by the paper positioning):
  B0 FP16 | B1 weight-only INT8 | B2 weight-only INT4 | B3 ternary BitNet
  M1 fully-integer INT16 | M2 fully-integer INT8 | M3 fully-integer INT8+smoothing
"""
import json
import os
import subprocess
import time

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")

# TODO (Claude Code on the Pi): set these to your built binaries and model paths.
LLAMA_BENCH = "./llama.cpp/build/bin/llama-bench"
LLAMA_CLI = "./llama.cpp/build/bin/llama-cli"


def run_llama_bench(model_path, threads=4):
    """Run llama-bench and parse tokens/sec. Returns dict (skeleton parser)."""
    cmd = [LLAMA_BENCH, "-m", model_path, "-t", str(threads)]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": " ".join(cmd), "wall_s": time.time() - t0,
            "stdout_tail": out.stdout[-2000:], "returncode": out.returncode}


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    print("Skeleton. Configure binary paths/flags, then implement the metric sweep")
    print("per docs/piano_operativo_fase0_fase1.md sections 3-4.")

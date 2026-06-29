# Prompt 04 — Baseline e misure su hardware edge (Fase 0/1, sul Raspberry Pi)

Da eseguire SUL dispositivo edge. Segui `docs/piano_operativo_fase0_fase1.md`.

1. Setup del Pi e build di llama.cpp e bitnet.cpp (comandi nel piano operativo, sezione 3).

2. Stabilisci il banco di misura: power meter che logga; implementa `read_power()` in `bench/energy.py` per il sensore disponibile.

3. Misura i baseline B0-B3 (FP16, weight-only INT8/INT4, ternario BitNet) con `bench/run_bench.py` (da completare con i path dei binari). Logga tutte le metriche di `bench/metrics_schema.md`.

4. Produci la prima Fig. 1 (Pareto accuratezza-vs-energia) dei baseline + un blog post.

5. (Fase 1) Porta la pipeline integer-only completa sul Pi e aggiungi M1-M3 alla tabella; misura il breakdown per-operatore (Fig. 2) e verifica empiricamente il collo di bottiglia non-GEMM su CPU.

Nota: questa fase richiede l'hardware; le fasi 02-03 (validazione su modello) si fanno prima, su una macchina con HF raggiungibile.

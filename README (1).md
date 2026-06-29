# Efficient On-Device AI

Inferenza LLM **interamente intera** (operatori non-lineari inclusi) su hardware edge economico — validazione, rimedio per l'INT8, e benchmarking edge. Progetto di ricerca finalizzato a un paper systems/measurement.

**Tesi:** quanto si può abbassare la precisione di un modello mantenendolo utile, e quanto costa davvero in latenza ed energia su un dispositivo da pochi euro?

## Stato

| Componente | Stato |
|---|---|
| Operatori interi (Softmax, RMSNorm, SiLU, linear) | ✅ validati vs FP (INT16 lossless) |
| Accumulo errore end-to-end | ✅ sub-lineare, non esplode |
| Rimedio INT8 (smoothing per-canale) | ✅ prototipato, recupera la parità |
| Validazione su modello reale | ⏳ harness pronto, da eseguire (HF) |
| Benchmark su edge (Pi) | ⏳ scheletro, da eseguire (hardware) |

Dettagli ed evidenza: `docs/findings_validazione.md`.

## Quick start

Esperimenti sintetici (solo NumPy):
```bash
pip install -r requirements.txt
python experiments/spike.py
python experiments/accumulation.py
python experiments/realistic_stress.py
python experiments/smoothing.py
```

Validazione su modello reale (richiede torch+transformers e accesso a Hugging Face):
```bash
python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 16
python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 8
python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 8 --per-token --smooth
python -m validation.outlier_stats --model qwen2.5-0.5b
```

## Usare Claude Code

Apri la cartella con Claude Code e incolla `prompts/00_master.md`. Poi procedi con i prompt numerati:
- `prompts/01_synthetic_repro.md` — sanity check degli esperimenti sintetici.
- `prompts/02_real_model_validation.md` — l'esperimento chiave (deriva INT8 → perplexity?).
- `prompts/03_smoothing_sweep.md` — sweep del rimedio su modello reale.
- `prompts/04_edge_bench.md` — baseline e misure sul Raspberry Pi.

`CLAUDE.md` dà a Claude Code il contesto completo automaticamente.

## Struttura
```
intops/        operatori interi validati + stack sintetico
experiments/   i quattro esperimenti sintetici
validation/    harness PyTorch per modelli reali
bench/         scheletro misure su edge
docs/          roadmap, piano operativo, posizionamento paper, findings
prompts/       prompt per Claude Code
results/       output degli esperimenti
```

## Documenti chiave
- `docs/paper_positioning.md` — gap, domande di ricerca, baseline, figure-target.
- `docs/findings_validazione.md` — tutta l'evidenza sperimentale.
- `docs/piano_operativo_fase0_fase1.md` — hardware, comandi, scheda-metriche per l'edge.
- `docs/roadmap.md` — il programma completo, dalla MVP alla frontiera.

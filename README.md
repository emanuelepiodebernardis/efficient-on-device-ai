# Efficient On-Device AI

Inferenza LLM **interamente intera** (operatori non-lineari inclusi) su hardware edge economico — validazione, rimedio per l'INT8, e benchmarking edge. Progetto di ricerca finalizzato a un paper systems/measurement.

**Tesi:** quanto si può abbassare la precisione di un modello mantenendolo utile, e quanto costa davvero in latenza ed energia su un dispositivo da pochi euro?

## Stato

| Componente | Stato |
|---|---|
| Operatori interi (Softmax, RMSNorm, SiLU, linear) | ✅ validati vs FP (INT16 lossless) |
| Accumulo errore end-to-end (sintetico) | ✅ sub-lineare, non esplode |
| Rimedio INT8 (smoothing per-canale, sintetico) | ✅ recupera la parità |
| **Validazione su modello reale (Qwen2.5-0.5B)** | ✅ **eseguita** — INT16 lossless (PPL 1.004×); INT8 collassa a un blocco specifico |
| **Diagnosi del collasso INT8 (layer 22)** | ✅ **causa identificata** (massive activation + drift accumulato) |
| Rimedio INT8 su modello reale | ⏳ da progettare (smoothing offline) |
| Benchmark su edge (Pi) | ⏳ scheletro, da eseguire (hardware) |

## Scoperte chiave dalla validazione reale

- **INT16 è lossless su un modello vero**: perplexity ratio 1.004× rispetto al floating-point. La via sicura del paper è confermata.
- **Gli outlier reali sono 2–3 ordini di grandezza oltre il sintetico**: ratio fino a 38.983× sui `mlp.down_proj`, contro i ~100–200× testati in sintetico. Questo è un finding metodologico: le valutazioni sintetiche di robustezza sottostimano il problema.
- **L'INT8 collassa a un blocco specifico (il 21→22), non per la sola magnitudine degli outlier**: la causa è *compound* — una massive activation (un singolo canale che concentra ~97% dell'energia del residual stream) che il blocco 21 ri-distribuisce, combinata con il drift accumulato dai layer precedenti. Dettagli in `docs/stato_progetto_post_validazione_reale.md`.

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
python -m validation.diagnose_block22          # diagnosi del collasso INT8
```

## Usare Claude Code

Apri la cartella con Claude Code e incolla `prompts/00_master.md`. `CLAUDE.md` dà a Claude Code il contesto completo automaticamente.

## Struttura
```
intops/        operatori interi validati + stack sintetico
experiments/   i quattro esperimenti sintetici
validation/    harness PyTorch per modelli reali + diagnosi
bench/         scheletro misure su edge
docs/          roadmap, piano operativo, posizionamento paper, findings, stato
prompts/       prompt per Claude Code
results/       output degli esperimenti (JSON di validazione versionati)
```

## Documenti chiave
- `docs/stato_progetto_post_validazione_reale.md` — stato attuale: cosa fatto, scoperto, e cosa manca.
- `docs/paper_positioning.md` — gap, domande di ricerca, baseline, figure-target.
- `docs/findings_validazione.md` — tutta l'evidenza sperimentale sintetica.
- `docs/piano_operativo_fase0_fase1.md` — hardware, comandi, scheda-metriche per l'edge.
- `docs/roadmap.md` — il programma completo, dalla MVP alla frontiera.

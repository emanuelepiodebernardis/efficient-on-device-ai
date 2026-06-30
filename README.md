# Efficient On-Device AI

Inferenza LLM con i **matmul interamente interi (INT8)** su hardware edge economico — progetto di ricerca finalizzato a un paper systems/measurement.

**Tesi:** quanto si può abbassare la precisione mantenendo il modello utile, e quanto costa in latenza/energia. La risposta, in breve: i matmul (calcolo dominante) possono stare in INT8; la precisione più alta serve solo sulle RMSNorm e su pochi canali outlier delle attivazioni.

## Stato e risultato principale

- **INT16 è lossless** (perplexity ratio ~1.0038 su Qwen2.5-0.5B) ed è la **rete sicura** del progetto.
- **L'INT8 naive collassa** perché la quantizzazione **per-tensor INT8 delle RMSNorm** non regge i **canali outlier** delle attivazioni (massive activations) — *non* per il cambio di base di un singolo blocco, come ipotizzato inizialmente. La severità del collasso scala con quanto l'outlier è concentrato.
- **La ricetta:** RMSNorm in alta precisione (**INT16**) + isolamento di **pochi canali outlier per Linear** (indici congelati in calibrazione, statici). I matmul restano INT8 per ≥99% del calcolo.
- **Generalizzazione (in corso):**
  - **lossless pulito** su Qwen2.5-0.5B (K≈4 canali) e Qwen2.5-1.5B (K≈8) — il numero di canali scala col modello;
  - su un modello **cross-famiglia a coda di outlier più grassa** (SmolLM2-1.7B) la ricetta recupera dal collasso ma **resta un gap residuo** (~1.1–1.4), tuttora in fase di diagnosi;
  - una regola automatica semplice per scegliere il numero di canali (`mag > T·mediana`) **non trasferisce bene** tra modelli.

> Stato: generalizzazione multi-modello **in corso**. L'INT16-lossless è acquisito; la ricetta INT8 è solida su Qwen e in caratterizzazione cross-famiglia.

## Riprodurre i numeri chiave

```bash
# INT16 lossless (rete del progetto)
python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 16

# Collasso INT8 naive
python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 8

# Ricetta deployable + sweep (norme INT16 + isolamento canali, statico vs dinamico)
python -m validation.static_recipe_calib                      # qwen2.5-0.5b
python -m validation.static_recipe_calib --model qwen2.5-1.5b
python -m validation.static_recipe_calib --model auto-cross   # cross-famiglia (Llama->SmolLM2->OLMo)
```

## Struttura del repo

```
validation/
  integer_torch.py            # operatori interi Torch (QuantLinear, QuantRMSNorm, ...)
  integer_patch.py            # patcher HF: parametri skip_modules, outlier_k, outlier_idx_map
  load_model.py               # caricamento modelli + fallback cross-famiglia
  error_vs_depth.py           # driver base: errore per strato + perplexity
  static_recipe_calib.py      # ricetta deployable, model-agnostic, sweep statico/auto-K/dinamico
  ...                         # altri driver di sweep e diagnosi
results/raw/                  # JSON dei risultati (evidenza numerica)
docs/                         # documenti di progetto, handoff, findings
```

## Documentazione

Per il percorso completo, la diagnosi dettagliata e il piano, vedi `docs/` — in particolare il file di handoff più recente.

## Caveat

La softmax dell'attention è ancora in FP nella passata corrente; l'integrazione intera è prevista come passata successiva. I matmul interi sono valutati via float64 degli operandi interi (esatti per le magnitudini in gioco).

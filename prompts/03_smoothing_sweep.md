# Prompt 03 — Sweep dello smoothing e dei bit su modello reale

Solo se il prompt 02 mostra che l'INT8 ha deriva non trascurabile sulla perplexity.

Obiettivo: trovare la config INT8 minima che recupera la perplexity entro una soglia (es. ratio <= 1.02) sul modello reale.

1. Sweep su alpha dello smoothing e granularità:
```bash
for a in 0.3 0.5 0.7; do
  python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 8 --smooth --alpha $a
  python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 8 --smooth --per-token --alpha $a
done
```

2. Ripeti su un secondo modello (qwen2.5-1.5b e/o llama3.2-1b) per generalità.

3. Estendi `validation/error_vs_depth.py` per loggare anche accuratezza su un piccolo task (es. un sottoinsieme di GSM8K o LAMBADA) oltre alla perplexity, se utile.

4. Produci la tabella finale config -> (cosine finale, perplexity_ratio, task_acc) e identifica la ricetta INT8 consigliata. Questo alimenta la Fig. 4 del paper (gap INT8 e suo recupero).

# Prompt 02 — Validazione su modello reale (l'esperimento chiave)

Obiettivo: rifare la curva errore-vs-profondità su un LLM addestrato reale e rispondere alla domanda che decide il paper forte: **la deriva INT8 del residual stream si traduce in perdita di perplexity?**

Passi:

1. Verifica accesso a Hugging Face. Se serve autenticazione per Llama, usa prima un modello aperto (Qwen2.5-0.5B-Instruct).

2. Esegui la validazione per le config principali:
```bash
python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 16
python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 8
python -m validation.error_vs_depth --model qwen2.5-0.5b --nbits 8 --per-token --smooth
```

3. Esegui anche le statistiche di outlier reali:
```bash
python -m validation.outlier_stats --model qwen2.5-0.5b
```

4. Aspettati e verifica (criteri da `docs/findings_validazione.md` §9):
   - INT16: cosine finale >= 0.9999, perplexity_ratio ~1.00 -> conferma "lossless".
   - INT8 naive: cosine >= 0.98, deriva caratterizzata; guarda il perplexity_ratio.
   - INT8 smooth+per-token: deve recuperare verso INT16.

5. NOTE / probabili aggiustamenti (è codice non testato in autoria):
   - I nomi dei moduli RMSNorm/Linear variano tra Qwen e Llama: il patch usa duck-typing (`*RMSNorm`, `nn.Linear`). Se un modello fallisce, ispeziona `model` e adatta `validation/integer_patch.py`.
   - La softmax dell'attention è lasciata in FP in questa prima passata (vedi commento nel patch). Se i risultati sono buoni, valuta una seconda passata che integra anche la softmax (usa `int_softmax_t`).
   - Se la memoria è insufficiente, usa il modello 0.5B.

6. Produci: una tabella relL2/cosine per layer per ogni config + i perplexity_ratio. Interpreta esplicitamente: **INT16 è confermato lossless end-to-end su modello reale? L'INT8 ha bisogno dello smoothing? Lo smoothing basta?** Aggiorna `docs/findings_validazione.md` con una sezione "Validazione su modello reale".

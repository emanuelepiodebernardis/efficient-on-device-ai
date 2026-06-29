# Prompt 01 — Riproduci la validazione sintetica (sanity check)

Esegui i quattro esperimenti sintetici e verifica che riproducano i risultati in `docs/findings_validazione.md`:

```bash
pip install -r requirements.txt
python experiments/spike.py
python experiments/accumulation.py
python experiments/realistic_stress.py
python experiments/smoothing.py
```

Attese (ordini di grandezza, vedi findings):
- spike: INT16 cosine ~0.9999+ su entrambe le distribuzioni; INT8 heavy-tail degrada (argmax ~90%, RMSNorm relL2 ~6-7%).
- accumulation: INT16 relL2 <0.3% a depth 32, cosine 1.0; INT8 cresce sub-linearmente.
- realistic_stress: INT16 lossless; per-token non risolve gli outlier per-canale.
- smoothing: sul caso "post-norm FFN outliers" lo smoothing porta il cosine a ~1.0.

Riporta una tabella con i tuoi numeri vs gli attesi. Se qualcosa devia molto, indaga (probabile differenza di versione NumPy o seed) prima di proseguire. Questo conferma che l'ambiente è sano.

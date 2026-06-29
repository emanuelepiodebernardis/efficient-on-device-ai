# Prompt master per Claude Code

Copia-incolla questo come primo messaggio a Claude Code dopo aver aperto la cartella del progetto. Poi usa i prompt numerati (01–04) per le singole fasi.

---

Sei in un progetto di ricerca chiamato **Efficient On-Device AI**. L'obiettivo è validare e misurare una pipeline di inferenza LLM *interamente intera* (operatori non-lineari inclusi) su hardware edge, e produrre i risultati per un paper systems/measurement.

Leggi prima questi file per il contesto completo, in quest'ordine:
1. `CLAUDE.md` — stato del progetto, cosa è già validato, cosa è aperto, convenzioni.
2. `docs/paper_positioning.md` — gap, domande di ricerca, baseline obbligatori, figure-target.
3. `docs/findings_validazione.md` — l'evidenza sperimentale già ottenuta (in sintetico).
4. `docs/roadmap.md` e `docs/piano_operativo_fase0_fase1.md` — il piano.

Cosa è già fatto e validato (in NumPy, regime sintetico):
- Operatori interi (Softmax, RMSNorm, SiLU, linear) con parità vs FP a INT16; degradazione INT8 sotto outlier diagnosticata.
- Accumulo end-to-end sub-lineare; INT16 lossless fino a 32 layer.
- Rimedio INT8: smoothing per-canale recupera la parità sul regime outlier realistico.

Cosa resta (richiede modelli reali, qui Hugging Face è raggiungibile):
- **Validazione su modello reale**: rifare la curva errore-vs-profondità su un LLM addestrato, e — la domanda chiave — verificare se la deriva INT8 si traduce in perdita di perplexity.
- Caratterizzare le statistiche di outlier reali.
- Sweep dello smoothing su modello reale.

Conferma di aver letto i file e riassumi in 5 righe lo stato del progetto e il primo esperimento che eseguiresti. Non scrivere codice finché non te lo chiedo con i prompt numerati.

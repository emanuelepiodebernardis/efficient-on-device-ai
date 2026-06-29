# CLAUDE.md — contesto per Claude Code

## Cos'è questo progetto
Programma di ricerca **Efficient On-Device AI**: inferenza LLM *interamente intera* (operatori non-lineari inclusi) su hardware edge economico, con validazione rigorosa e misure reali, finalizzato a un paper systems/measurement. Tesi: *quanto si può abbassare la precisione mantenendo il modello utile, e quanto costa davvero in latenza/energia su un dispositivo da pochi euro?*

## Struttura
- `intops/` — operatori interi validati (NumPy): `quantize`, `int_softmax`, `int_rmsnorm`, `int_silu`, `int_linear`, più lo stack LLaMA-style sintetico. **Codice di riferimento, non modificare la logica senza rieseguire gli esperimenti.**
- `experiments/` — i quattro esperimenti sintetici già validati (spike, accumulation, realistic_stress, smoothing).
- `validation/` — harness PyTorch per modelli reali (HF). **Codice non ancora eseguito in autoria**: va girato e rifinito qui dove HF è raggiungibile.
- `bench/` — scheletro per le misure su edge (Pi). Si completa sul dispositivo.
- `docs/` — roadmap, piano operativo Fase 0-1, posizionamento del paper, nota dei findings.
- `prompts/` — prompt numerati per eseguire le fasi di test.
- `results/raw/` — output JSON degli esperimenti.

## Cosa è GIÀ validato (sintetico, NumPy)
- Operatori interi: parità vs FP a INT16 anche sotto outlier; degradazione INT8 sotto outlier diagnosticata e quantificata.
- Accumulo end-to-end: **sub-lineare (~radice della profondità), non esplode**. INT16 lossless a 32 layer (cosine 1.0).
- Rimedio INT8: **smoothing per-canale** (SmoothQuant-style) recupera la parità sul regime outlier realistico (post-norm FFN); il per-token da solo non basta.
- Numeri precisi in `docs/findings_validazione.md`.

## Cosa è APERTO (richiede modelli reali / hardware)
1. **Validazione su modello reale** (prompt 02): rifare errore-vs-profondità su un LLM addestrato; **domanda chiave: la deriva INT8 si traduce in perdita di perplexity?**
2. Sweep dello smoothing su modello reale (prompt 03).
3. Baseline e misure su edge / Pi (prompt 04).

## Caveat noti (da rispettare e dichiarare)
- L'attention softmax è lasciata in FP nella prima passata di validazione (vedi `validation/integer_patch.py`). Integrarla è una seconda passata.
- I pattern sintetici non sostituiscono le attivazioni emergenti reali: per questo serve il prompt 02.
- `relL2` non è confrontabile *tra* regimi con norme diverse del residual stream; confrontare *dentro* lo stesso regime.

## Convenzioni
- Esperimenti sintetici: zero dipendenze oltre NumPy; eseguibili con `python experiments/<nome>.py`.
- Validazione reale: `python -m validation.<modulo> --args` dalla root.
- Lezioni metodologiche già incorporate (NON reintrodurre i bug): RMSNorm in forma "dividi grande per grande"; scala ad alta precisione (2^30) per gli operatori basati su exp; mascheramento causale applicato DOPO l'exp.
- Ogni nuovo risultato va salvato in `results/raw/` e riassunto in `docs/findings_validazione.md`.

## Criteri di successo (validazione reale)
INT16: cosine finale ≥ 0.9999 e perplexity_ratio ≈ 1.00. INT8: cosine ≥ 0.98 con deriva caratterizzata; obiettivo del rimedio: perplexity_ratio ≤ ~1.02.

## Come iniziare
Leggi `prompts/00_master.md`, poi esegui in ordine `prompts/01` (sanity), `prompts/02` (l'esperimento chiave). Prima di scrivere codice nuovo, leggi `docs/paper_positioning.md`.

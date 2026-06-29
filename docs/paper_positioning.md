# Posizionamento del paper + contratto sperimentale

Documento "indipendente" (non cambia coi risultati): definisce il gap, il contributo, i baseline obbligatori e le figure-target. È il contratto che rende il lavoro sperimentale economico e a prova di revisore. Aggiornato con i risultati dello spike di fattibilità.

---

## 1. Il gap, in una frase

> L'inferenza LLM *interamente intera* (operatori non-lineari inclusi) è stata dimostrata preservare l'accuratezza **solo su GPU da datacenter**, mentre i deployment su edge usano quantizzazione dei soli pesi con non-linearità in floating-point: nessuno ha **deployato e misurato con rigore** una pipeline interamente intera su hardware edge economico reale, né caratterizzato **dove** il divario di accuratezza a INT8 sotto attivazioni con outlier si paga in latenza ed energia.

Le due metà della letteratura non si parlano: gli integer-only (I-LLM, I-ViT, I-BERT, IntAttention, SoftmAP) validano su A100/A6000; gli edge-deployment (LLMPi, LiteRT-on-Pi) usano weight-only via llama.cpp; i ternari (BitNet, TeLLMe) restano su CPU generiche o FPGA. Il ponte misurato manca.

---

## 2. Cosa ci dice lo spike (premessa validata)

- **A INT16 la pipeline intera raggiunge parità praticamente perfetta col FP**, anche sotto outlier (Softmax: argmax-match 99,98%, cos 0,999999; RMSNorm: errore relativo L2 ~3e-4). → la premessa del paper è vera.
- **A INT8 la parità tiene su input gaussiani ma degrada sotto outlier** (Softmax argmax-match ~90%, RMSNorm relL2 ~7,5%). → questa degradazione **è il bersaglio scientifico**, non un difetto.

Conseguenza per il claim: il paper non rischia sull'accuratezza (INT16 è la rete di sicurezza), e ha una domanda di ricerca precisa e misurabile (il gap INT8-sotto-outlier).

---

## 3. Domande di ricerca (RQ)

- **RQ1 (deployment).** Una pipeline LLM interamente intera (non-lineari incluse) può girare su un dispositivo edge economico con parità di accuratezza rispetto al FP?
- **RQ2 (caratterizzazione).** Su CPU edge, *dove* vanno latenza ed energia — e gli operatori non-GEMM (Softmax, RMSNorm) diventano il collo di bottiglia quando i lineari sono quantizzati, come accade su GPU?
- **RQ3 (il gap INT8).** Quanto degrada la pipeline interamente intera a INT8 sotto attivazioni con outlier, e una tecnica di smoothing/quantizzazione per-token che resti *interamente intera* recupera la parità?
- **RQ4 (costo del rimedio).** Qual è il costo in latenza/energia su edge del rimedio di RQ3 rispetto al guadagno di accuratezza?

---

## 4. Contributo (a stadi, per de-rischiare)

**Paper minimo (sicuro):** RQ1 + RQ2.
1. Prima pipeline LLM interamente intera (operatori non-lineari interi) **deployata e misurata** su hardware edge economico reale, con latenza per-operatore ed energia per-token effettive.
2. Caratterizzazione integer-only vs weight-only su edge: dove va il tempo/energia, e localizzazione empirica del collo di bottiglia non-GEMM su CPU.

**Paper forte (ambizioso):** aggiunge RQ3 + RQ4.
3. Quantificazione del divario INT8-sotto-outlier nella pipeline intera, e un rimedio interamente intero (smoothing/per-token) che recupera la parità.
4. Misura del costo su edge del rimedio (il trade-off accuratezza/energia).

Regola di scope: si pubblica anche solo col paper minimo. RQ3–RQ4 sono estensione, non prerequisito.

---

## 5. Baseline obbligatori (definiti dallo spike)

Un revisore pretenderà tutti questi sullo stesso dispositivo, stesse metriche:

| # | Config | Ruolo |
|---|---|---|
| B0 | FP16 | riferimento accuratezza + baseline latenza/energia |
| B1 | Weight-only INT8 (llama.cpp) | "edge standard" odierno |
| B2 | Weight-only INT4 (llama.cpp) | edge aggressivo standard |
| B3 | Ternario nativo, BitNet b1.58 (bitnet.cpp) | estremo low-bit di riferimento |
| **M1** | **Fully-integer INT16 (nostro)** | target a parità garantita |
| **M2** | **Fully-integer INT8 naïve (nostro)** | mostra la degradazione |
| **M3** | **Fully-integer INT8 + smoothing (nostro)** | il rimedio (paper forte) |

B0–B3 esistono già a costo quasi zero (tooling pronto); M1–M3 sono il contributo.

---

## 6. Figure-target (suggerite dalle tabelle dello spike)

- **Fig. 1 — Pareto accuratezza-vs-energia** sul dispositivo edge, tutti i config B0–M3 su un unico grafico. È l'headline: mostra dove cadono le pipeline interamente intere rispetto a weight-only e ternario.
- **Fig. 2 — Breakdown per-operatore** di latenza ed energia, integer-only vs weight-only. Risponde a RQ2 e localizza il collo di bottiglia non-GEMM.
- **Fig. 3 — Parità per-operatore ed end-to-end** vs FP, per bit-width × distribuzione (gaussiano vs heavy-tail). È lo spike scalato alle attivazioni reali del modello: mostra la rete di sicurezza INT16 e l'apertura del gap INT8.
- **Fig. 4 (paper forte) — Il gap INT8 e il suo recupero**: degradazione naïve vs smoothing, con il costo edge del rimedio. Risponde a RQ3–RQ4.

---

## 7. Contratto sperimentale

**Modelli:** Qwen2.5 0.5B, Llama 3.2 1B, Qwen2.5 1.5B, BitNet b1.58 2B4T. Tutti sub-2B (regime in cui la quantizzazione aggressiva è difficile).

**Hardware:** Raspberry Pi 5 8GB (target primario, CPU Cortex-A76). Opzionale secondo dispositivo con NPU (RK3588 / AI HAT+ 2) per generalizzazione.

**Metriche (per ogni modello × config × seed):** perplexity; accuratezza su 1–2 task fissi; tokens/sec prefill e decode; latenza per-token; **Joule per token**; picco RAM; breakdown latenza per-operatore. Più i metadati di riproducibilità (temperatura, governor, thread, commit, firmware).

**Protocollo:** prompt fisso, lunghezza di generazione fissa, thread fissi; ripetizioni multi-seed con media e deviazione; controllo termico attivo.

**Misura accuratezza end-to-end:** validare la parità non solo per-operatore (come lo spike) ma sull'output del modello completo, per cogliere l'eventuale accumulo d'errore tra layer.

---

## 8. Minacce alla validità (da dichiarare nel paper)

- **Accumulo d'errore end-to-end:** l'errore per-operatore è piccolo, ma va verificato che non si amplifichi attraverso decine di layer. È il primo controllo da fare quando la pipeline è completa.
- **Efficienza non garantita dall'accuratezza:** la parità non implica speedup; il guadagno reale dipende dai kernel su ARM. Va misurato, non assunto.
- **Gestione di eps e delle costanti di scala:** lo spike mostra che la formulazione conta (la forma "dividi grande per grande" è stabile, altre underflowano). Documentare la scelta.
- **Generalità tra architetture:** risultati su modelli LLaMA-style; estendere ad altre famiglie è lavoro futuro.

---

## 9. Inquadramento e venue

Taglio: **systems/measurement paper**, non "nuovo algoritmo di quantizzazione". Il punto di forza per l'accettazione è ciò che gli altri non hanno: **numeri reali su hardware reale**, con artefatto riproducibile (l'harness G4).

Venue realistiche per singolo autore: workshop edge-AI/efficient-ML (EuroMLSys, workshop on-device/efficient-NLP di NeurIPS/ICML/EMNLP), o IEEE Access / MDPI per un primo lavoro. Preprint su arXiv appena le Fig. 1–3 esistono.

---

## 10. Il claim in due righe (da rifinire per l'abstract)

> Mostriamo la prima pipeline di inferenza LLM *interamente intera* (operatori non-lineari inclusi) deployata e misurata su hardware edge economico, raggiungendo parità di accuratezza col floating-point a INT16 e caratterizzando, su CPU edge reale, dove latenza ed energia si concentrano — incluso il divario di accuratezza a INT8 sotto attivazioni con outlier e un rimedio interamente intero che lo recupera.

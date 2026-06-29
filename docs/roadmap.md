# Roadmap — Efficient On-Device AI

**Programma di ricerca:** inferenza *integer-only* di small LLM su hardware edge economico, esteso poi alle KAN quantizzate, sotto un'unica tesi: *comprimere modelli perché girino su hardware minuscolo, misurando il costo reale invece di simularlo*.

**Tesi-ombrello (la frase che lega tutto):** "Quanto in basso si può spingere la precisione di un modello mantenendolo utile, e quanto costa davvero — in latenza ed energia — su un dispositivo da pochi euro?"

**Due pilastri:**
- **Pilastro LLM** — pipeline integer-only completa su edge reale (progetti G1+G2, harness G4, kernel G3 opzionale).
- **Pilastro KAN** — stesso framework di quantizzazione applicato alle Kolmogorov-Arnold Networks (A1), per coerenza di profilo e secondo paper.

---

## Principio guida: scope discipline

Il rischio numero uno di questo progetto è il *gonfiamento*. La regola: **ogni fase deve produrre un deliverable pubblicabile o dimostrabile da sola**. Non si passa alla fase successiva finché la precedente non "gira e si misura". Meglio un MVP pubblicato che un progetto perfetto mai finito.

---

## FASE 0 — Fondamenta e setup (settimane 1–2)

Obiettivo: avere un banco di misura affidabile *prima* di scrivere una riga di pipeline integer-only.

**Da fare (obbligatorio):**
- Procurarsi l'hardware: un **Raspberry Pi 5** (ARM Cortex-A76) come target primario, microSD veloce, alimentatore adeguato.
- Procurarsi un **power meter USB inline** reale (es. un misuratore di potenza tra alimentatore e board). L'energia misurata davvero è ciò che dà credibilità al paper — quasi nessuno la riporta con rigore.
- Toolchain di riferimento: `llama.cpp` e `bitnet.cpp` (per i baseline weight-only e ternari), un ambiente Python per la valutazione (perplexity, accuratezza su task standard), script di logging delle metriche.
- Definire la **scheda metriche** definitiva: perplexity + accuratezza su 1–2 task (es. un sottoinsieme di benchmark di language modeling), tokens/sec, latenza per-token (prefill e decode separati), **Joule per token**, picco di memoria, e — chiave — il **breakdown di latenza per-operatore** (lineari vs non-lineari).
- Scegliere i modelli: 2–3 piccoli e onesti. Proposta: **Llama 3.2 1B**, **Qwen2.5 1.5B**, e **BitNet b1.58 2B** (per coprire l'angolo ternario). Tutti sotto i 2B, dove la quantizzazione aggressiva è difficile e quindi interessante.

**Deliverable Fase 0:** un baseline riproducibile (modello weight-only INT4/INT8 via llama.cpp che gira sul Pi) con tutte le metriche già loggate. Questo valida il banco di misura — non è ancora ricerca, è infrastruttura.

---

## FASE 1 — MVP (G1 core): la pipeline integer-only gira su edge (settimane 3–6)

Obiettivo: il primo risultato che nessuno ha mostrato — un LLM con pipeline *interamente intera* (anche le non-lineari) che gira su un dispositivo edge reale.

**Da fare (obbligatorio):**
- Implementare/portare gli operatori non-lineari integer-only: **RMSNorm intera, Softmax intera (via bit-shift), GELU/SiLU intera**. Punto di partenza concettuale: i metodi tipo I-LLM (DI-Exp, DI-Norm, Softmax clippata) e I-ViT (Shiftmax/ShiftGELU). Tu parti avvantaggiato: hai già `integer_only_inference`.
- Far girare **un solo modello** completamente integer-only sul Pi, producendo output corretto.
- Confronto a tre vie: FP baseline vs weight-only quantizzato vs integer-only completo, su accuratezza.

**Deliverable Fase 1:** "it runs" + i primi numeri di accuratezza. Un **blog post / demo video** ("un LLM interamente in interi su una board da 60 euro"). Questo è già un segnale di mercato fortissimo, anche prima del paper.

---

## FASE 2 — Caratterizzazione (G2): lo studio di misura (settimane 6–10)

Obiettivo: trasformare l'MVP in un contributo scientifico — *dove va il tempo e l'energia*.

**Da fare (obbligatorio):**
- **Breakdown per-operatore** di latenza ed energia, integer-only vs weight-only, su hardware reale. Verificare empiricamente sul Pi il fenomeno noto su GPU: gli operatori non-GEMM (Softmax in primis) diventano collo di bottiglia quando i lineari sono accelerati. Su CPU edge il fenomeno può essere diverso — ed è proprio il dato nuovo.
- **Pareto accuratezza-vs-energia** su più bit-width (es. W8A8, W4A8, W2A8 e ternario).
- Rigore statistico: **multi-seed**, multi-modello, e se possibile un **secondo dispositivo** (es. una board con NPU come una Rockchip RK3588) per mostrare generalizzazione.

**Deliverable Fase 2:** le tabelle e figure centrali del paper (Pareto + breakdown). A questo punto hai il cuore di un workshop paper.

---

## FASE 3 — Harness open-source (G4): il sottoprodotto che fa rete (in parallelo a Fase 2)

Obiettivo: l'oggetto che genera visibilità e citazioni, e che diventa l'artefatto di riproducibilità del paper.

**Da fare (obbligatorio):**
- Ripulire il codice in un **harness di benchmark** documentato: dato un modello e una board, produce automaticamente tutte le metriche.
- README con numeri, un grafico, e istruzioni per riprodurre sul Pi passo-passo.
- Rilascio su GitHub con licenza permissiva; richiesta di inserimento nelle liste di riferimento del settore (awesome-list su edge-LLM / efficient-ML).

**Deliverable Fase 3:** repo pubblico + artefatto di riproducibilità. È anche il segnale "qualcuno lo usa davvero" che vale più di mille righe di CV.

---

## FASE 4 — Pubblicazione (settimane 10–16)

Obiettivo: il preprint e la submission.

**Da fare (obbligatorio):**
- Taglio del paper: **systems/measurement paper**, non "nuovo algoritmo". Claim: *"portiamo la pipeline integer-only completa su hardware edge reale e quantifichiamo il vero costo/beneficio (accuratezza, latenza per-operatore, Joule/token) rispetto al weight-only"*. Il punto di forza per l'accettazione: **numeri reali su hardware reale**, non simulazioni su GPU datacenter.
- Preprint su **arXiv**.
- Submission a un workshop edge-AI/efficient-ML (es. EuroMLSys, workshop on-device/efficient-NLP di NeurIPS/ICML/EMNLP) o a una rivista accessibile per un primo lavoro (IEEE Access / MDPI).
- Valutare di coinvolgere un co-autore (es. il contatto KuznetsovKarazin già nella tua rete) per trasformare il lavoro solitario in collaborazione.

**Deliverable Fase 4:** preprint pubblico + submission inviata.

---

## FASE 5 — Estensione di sistema (G3, opzionale ma ad alto valore)

Obiettivo: il flex ML-systems che pochissimi sanno fare.

**Da fare (se hai tempo/voglia):**
- Scrivere **kernel ternari ottimizzati per ARM** (NEON), stile XNOR+popcount o LUT-based (T-MAC). È una *sfida aperta esplicitamente dichiarata* in letteratura: i lavori ternari attuali usano kernel CUDA standard e ammettono che l'aritmetica ternaria ottimizzata per hardware è un problema irrisolto.
- Contribuire la PR a `bitnet.cpp` o `llama.cpp`.

**Deliverable Fase 5:** kernel + benchmark di speedup; potenziale **secondo paper** (systems) e una PR ad alto segnale in un progetto famoso.

---

## FASE 6 — Il ponte KAN (A1): unificare il programma

Obiettivo: chiudere il cerchio "efficient on-device AI" e produrre il secondo pilastro.

**Da fare:**
- Applicare lo **stesso framework di quantizzazione integer-only alle KAN**: operatori interi per la valutazione delle basi (spline/Chebyshev/Fourier) — terreno quasi vergine.
- Confronto diretto **KAN vs MLP/LLM sotto la stessa compressione su edge**: chi degrada meglio a bit ultra-bassi? È una domanda scientifica pulita e poco esplorata.

**Deliverable Fase 6:** terzo paper + la narrazione coerente che lega tutto il profilo.

---

## Frontiera scientifica — spingere il progetto ai limiti massimi

Queste sono le direzioni "all'avanguardia" da tenere come orizzonte. Non tutte vanno fatte; servono a sapere *dove può arrivare* il programma e a scegliere il claim più ambizioso quando il progetto base sarà solido.

**1. Lo stack ternario completo end-to-end.** Combinare pesi ternari (BitNet) con operatori non-lineari *interi* in un'unica pipeline edge: nessuno ha lo stack completo (oggi il ternario quantizza i pesi ma lascia FP altrove). Sarebbe il modello "tutto sub-byte" su hardware reale.

**2. KV-cache integer-only con allocazione adattiva dei bit su edge.** La KV-cache domina il decoding; allocare bit in base all'importanza del token, interamente in interi, su dispositivo, è un vuoto preciso.

**3. Co-design hardware-aware.** Sfruttare gli insiemi di istruzioni degli NPU edge; mpGEMM integer-only basato su lookup table (stile T-MAC) senza dequantizzazione. Qui il pilastro LLM e i kernel G3 convergono.

**4. Speculative decoding interamente integer su edge.** Un draft model piccolo, anch'esso integer-only, per accelerare il decode sul dispositivo — combinazione mai mostrata fully-integer su edge.

**5. Fine-tuning / continual learning in low-bit on-device.** Il backward pass integer-only è molto più difficile del forward: è una frontiera vera. Permetterebbe adattamento sul dispositivo senza cloud — enorme per privacy.

**6. KAN come blocco integer-only dentro un transformer.** Strati KAN-MLP o KAN-attention, interamente interi, dentro un piccolo LLM: è l'intersezione esatta dei tuoi due pilastri e non la possiede nessuno.

**7. Bound teorici di errore sugli operatori non-lineari interi.** Caratterizzare formalmente l'errore introdotto da Softmax/GELU intere e i limiti del Pareto energia-accuratezza: alza il lavoro da "misurativo" a "fondazionale".

**8. Scendere alla classe microcontrollore (Cortex-M).** Un LLM utile (anche minuscolo) su hardware senza MMU e con FP assente è TinyML estremo: massima difficoltà, massimo effetto-vetrina.

**9. Robustezza avversariale degli LLM integer-only su edge.** Chiude il cerchio con il tuo background di security: la quantizzazione cambia la superficie d'attacco. Riporta tutto il programma a casa tua.

---

## Mappatura sul mercato (cosa dimostra ogni fase ai recruiter)

- **Fase 1–2:** sai deployare e *misurare* ML su hardware reale — la skill systems più difficile da fingere.
- **Fase 3:** sai produrre software che altri usano (open-source con utenti).
- **Fase 4:** sai portare un lavoro fino alla pubblicazione (rigore, comunicazione).
- **Fase 5:** sai scrivere kernel a basso livello ottimizzati (profilo ML-systems raro).
- **Fase 6 + frontiera:** sei uno dei pochi che unisce ML + quantizzazione + systems + security in un programma coerente.

Il fronte on-device LLM è esattamente quello su cui Meta, Apple, Qualcomm, Arm, MediaTek stanno assumendo. Un candidato che mostra "LLM interamente integer-only su board da 60 euro, con Joule/token misurati" parla la loro lingua.

---

## Sequenza minima eseguibile (se vuoi partire subito)

1. Compra Pi 5 + power meter.
2. Fai girare un baseline weight-only via llama.cpp e logga le metriche (valida il banco).
3. Implementa Softmax + RMSNorm intere e porta **un** modello a integer-only completo.
4. Misura il breakdown per-operatore vs il baseline.
5. Scrivi blog post + apri il repo.
6. Estendi a 3 modelli + multi-seed → bozza paper.

Tutto il resto (G3, KAN, frontiera) è espansione successiva. La disciplina è: **non aprire una nuova fase finché la precedente non gira e si misura.**

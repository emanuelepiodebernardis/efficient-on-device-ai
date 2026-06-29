# Findings — validazione sperimentale della premessa

Nota di accompagnamento al posizionamento del paper. Consolida i risultati degli esperimenti eseguiti finora: spike degli operatori, accumulo end-to-end, stress test realistici. Tutti gli esperimenti usano un riferimento FP in float64 e operatori interamente interi (al runtime solo add/mul/shift/divisione/radice intere) su uno stack LLaMA-style (RMSNorm pre-norm + attention con Softmax + FFN SwiGLU con SiLU).

---

## 1. Sintesi in una riga

Su tutta la batteria di test, **l'errore di quantizzazione intera non esplode con la profondità**: cresce sub-linearmente (~radice della profondità) o va in plateau, perché la struttura residuo + RMSNorm fa da stabilizzatore. **INT16 è lossless end-to-end; INT8 naïve resta limitato e direzionalmente fedele.** La premessa del progetto è confermata; il rischio di accumulo end-to-end è sostanzialmente ritirato.

---

## 2. Spike — operatori in isolamento

Implementazioni interamente intere di Softmax e RMSNorm, parità misurata vs FP, inclusi input con outlier (regime difficile per gli LLM).

**Softmax intera** (input tipo punteggi di attention):

| dist | bit | argmax-match | cosine | KL |
|---|---|---|---|---|
| gaussiano | INT8 | 97.4% | 0.99994 | 1.0e-3 |
| gaussiano | INT16 | 99.99% | 0.999998 | 9.8e-4 |
| heavy-tail | INT8 | 89.6% | 0.996 | 7.3e-3 |
| heavy-tail | INT16 | 99.98% | 0.999999 | 2.2e-3 |

**RMSNorm intera** (hidden states):

| dist | bit | relL2 | cosine |
|---|---|---|---|
| gaussiano | INT8 | 1.1e-2 | 0.99994 |
| gaussiano | INT16 | 4.4e-5 | 1.0 |
| heavy-tail | INT8 | 7.5e-2 | 0.997 |
| heavy-tail | INT16 | 2.9e-4 | 1.0 |

Esito: a INT16 parità praticamente perfetta anche sotto outlier; a INT8 buona su gaussiano, degrada sotto outlier (il bersaglio del rimedio).

---

## 3. Accumulo end-to-end (depth 1 → 32)

Errore del residual stream vs FP, stack di profondità crescente:

| Config | d1 | d8 | d16 | d32 | cosine @32 |
|---|---|---|---|---|---|
| INT16 gaussiano | 0.0006 | 0.0016 | 0.0021 | 0.0029 | 1.0000 |
| INT16 heavy-tail | 0.0003 | 0.0009 | 0.0013 | 0.0018 | 1.0000 |
| INT8 gaussiano | 0.029 | 0.074 | 0.098 | 0.128 | 0.9919 |
| INT8 heavy-tail | 0.032 | 0.088 | 0.120 | 0.155 | 0.9880 |

Crescita ~depth^0.45 (sub-lineare, tipo radice): firma di accumulo in quadratura con ri-normalizzazione che impedisce l'amplificazione. 32 layer coprono la profondità dei modelli sub-2B reali (Llama 3.2 1B ~16, Qwen2.5 1.5B ~28).

---

## 4. Stress test realistici — massive activations

Pochi canali fissi con valori enormi e persistenti su tutti i token (struttura documentata da Sun et al. e dalla letteratura outlier-feature). Depth 32:

| Config | relL2 @32 | cosine @32 |
|---|---|---|
| INT16 per-tensor | 0.0004 | 1.0000 |
| INT8 per-tensor (naïve) | 0.064 | 0.998 |
| INT8 per-token (anteprima rimedio) | 0.060 | 0.998 |

Multi-seed (3 semi), endpoint depth-32: INT16 0.0004 ± 0.0000; INT8 per-tensor 0.0583 ± 0.0059; INT8 per-token 0.0558 ± 0.0047.

Due risultati chiave:
- La **pre-norm neutralizza** gli outlier del residual stream prima di ogni quantizzazione → l'INT8 sotto massive activations va perfino *meglio* del rumore sparso.
- Il **per-token quasi non aiuta** (0.060 vs 0.064): gli outlier sono per-canale, non per-token → il rimedio corretto è per-canale/smoothing, non per-token.

---

## 5. Caso peggiore — outlier post-normalizzazione nella FFN

Outlier iniettati (identici in entrambi i percorsi) nell'intermedio della FFN, la dimensione *non* ri-normalizzata prima della down-projection: il punto in cui gli outlier reali mordono di più. Depth 32:

| Config | relL2 @32 | cosine @32 |
|---|---|---|
| INT16 (post-norm outliers) | 0.0002 | 1.0000 |
| INT8 per-tensor (post-norm outliers) | 0.033 | 0.9995 |

L'INT8 va in **plateau** (0.033 già a depth 8, stabile a depth 32): i canali outlier, essendo grandi, dominano l'output del matmul e sono ben rappresentati vicino a qmax, mentre i canali piccoli contribuiscono poco. Errore limitato e non crescente.

---

## 6. Risultato controintuitivo da evidenziare

Il caso più duro per l'INT8 **non** sono gli outlier strutturati "realistici" (0.064 / 0.033), ma il **rumore sparso** (0.155) — perché gli outlier strutturati e persistenti vengono normalizzati o dominano in modo *prevedibile*, mentre il rumore sparso inietta errore fresco e non strutturato ovunque a ogni strato. Implicazione: la robustezza dipende dalla *struttura* degli outlier, non solo dalla loro magnitudine.

---

## 7. Due lezioni metodologiche (per evitare artefatti)

1. **Underflow della scala in virgola fissa ai bit alti.** Rappresentare una scala `s` piccola come `round(s·2^B)` perde bit a INT16. Soluzioni adottate: per la RMSNorm la forma "dividi grande per grande" (divisione per `sqrt(denom)` invece di rappresentare il rsqrt minuscolo); per gli operatori basati su exp (Softmax, SiLU) il trasporto della scala a precisione più alta (2^30) prima di ridurla a 2^B. Senza questi accorgimenti, l'errore a INT16 risultava *peggiore* che a INT8 — un artefatto, non un limite dell'operatore.
2. **Il mascheramento causale va applicato dopo l'exp.** Aggiungere `-1e9` ai punteggi *prima* della quantizzazione per-tensore fissa la scala su quel valore e azzera tutti i punteggi reali. Corretto: mascherare moltiplicando per 0 *dopo* l'exp. Questo da solo spiegava un errore spurio del 30%+ per blocco.

Entrambe vanno documentate nel paper come scelte di design.

---

## 8. Caveat residui (da dichiarare, non chiusi)

1. **Pattern di attivazione sintetici.** I test riproducono fedelmente i *meccanismi* documentati (massive activations, outlier-feature, rumore heavy-tail), ma non i pattern *emergenti* di uno specifico modello addestrato. La verifica definitiva richiede pesi reali.
2. **Matmul dei punteggi di attention in FP.** Nei test, q·k e a·v sono in FP in entrambi i percorsi; una attention interamente intera aggiungerebbe errore lì. Va incluso quando la pipeline è completa.

---

## 9. Protocollo di validazione su modello reale (Fase 0)

Da eseguire nell'ambiente di build (dove Hugging Face è raggiungibile), per chiudere il caveat #1.

1. Scaricare un piccolo LLM addestrato (Llama 3.2 1B e/o Qwen2.5 1.5B).
2. Eseguire un forward FP (riferimento) e un forward con gli operatori interi, su prompt di calibrazione reali (per attivazioni realistiche, non sintetiche).
3. Catturare l'errore del residual stream vs FP **a ogni layer**: relL2 e cosine, esattamente come qui.
4. Misurare le statistiche reali di outlier (rapporto max/mediana per canale, per layer) e confrontarle con le assunzioni sintetiche — verifica se il regime testato copre quello reale.
5. Confermare che INT16 resti lossless end-to-end; caratterizzare l'INT8 sotto outlier reali.
6. Solo se l'INT8 conferma la deriva: progettare il rimedio di smoothing/per-canale e rimisurare.

Criterio di successo: INT16 cosine ≥ 0.9999 a fine modello; INT8 cosine ≥ 0.98 con deriva caratterizzata.

---

## 10. Implicazioni per il progetto

- **Via sicura (paper minimo):** la pipeline INT16 fully-integer non ha rischi residui di accumulo. Procedere al deployment e alla misura su edge.
- **Via forte (paper ambizioso):** il problema INT8 è circoscritto e diagnosticato (outlier per-canale → smoothing). Il per-token è già escluso come rimedio sufficiente.
- **Prossimo controllo prima dell'hardware:** il protocollo di §9 su un modello reale. È l'unico passo che resta tra "validato in sintetico" e "validato".

---

## 11. Il rimedio INT8 — smoothing per-canale (prototipo)

Prototipo di smoothing stile SmoothQuant nei layer lineari: si migra la scala per-canale dalle attivazioni ai pesi (`x' = x/s`, `W' = W·s`, prodotto invariato), con `s_j = max|x_j|^α / max|W_j|^(1-α)`, α=0.5. Confronto a profondità 32, relL2 (cosine):

| Regime | INT16 | INT8 naïve | INT8 per-token | INT8 smoothing | INT8 smooth+token |
|---|---|---|---|---|---|
| gaussiano | 0.0027 (1.0000) | 0.143 (0.9899) | 0.110 (0.9940) | 0.107 (0.9943) | 0.096 (0.9954) |
| heavy-tail sparso | 0.0016 (1.0000) | 0.158 (0.9875) | 0.133 (0.9912) | 0.128 (0.9919) | 0.122 (0.9926) |
| massive activations | 0.0004 (1.0000) | 0.065 (0.9979) | 0.061 (0.9982) | 0.060 (0.9982) | 0.060 (0.9982) |
| outlier FFN post-norm | 0.0002 (1.0000) | 0.033 (0.9995) | 0.032 (0.9995) | **0.0098 (1.0000)** | **0.0093 (1.0000)** |

Esiti:
- **Lo smoothing recupera la parità dove deve.** Sul caso outlier post-normalizzazione (bersaglio genuino: feature strutturate per-canale nell'intermedio non normalizzato) l'errore cala di 3.3× e il cosine torna a 1.0000 — qualità INT16. Meccanismo validato.
- **Aiuta poco altrove, per ragioni comprese.** Sul rumore sparso non c'è struttura per-canale da lisciare (regime sintetico poco realistico). Sul gaussiano l'errore residuo (~0.10) è il pavimento irriducibile di arrotondamento INT8 su ~200 matmul, non un problema di outlier.
- **Ricetta INT8 consigliata:** smoothing + per-token (sistematicamente la migliore).

**Caveat metodologico:** la relL2 non è confrontabile *tra* regimi diversi (i casi con outlier hanno residual stream di norma maggiore che deflaziona l'errore relativo). Il confronto valido è *dentro* lo stesso regime.

**Prossimo passo per il rimedio:** la calibrazione di `s` qui usa le statistiche del batch corrente (oracolo). In Fase 0/1 va sostituita con una calibrazione su prompt reali, e va misurato il costo su edge del rimedio (RQ4 del posizionamento). Su modello reale, verificare se la deriva gaussiana ~0.10 a INT8 si traduce o no in perdita di perplexity/accuratezza sul task — è ciò che conta davvero, non la relL2 del residual stream.

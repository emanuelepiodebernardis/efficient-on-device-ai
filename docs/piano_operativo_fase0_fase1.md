# Piano operativo — Fase 0 e Fase 1

Documento eseguibile per avviare il progetto *Efficient On-Device AI* senza dover decidere nulla a freddo. Copre: hardware da comprare, comandi concreti per i baseline, struttura del repo, scheda-metriche, e i criteri di "fatto" (Definition of Done) per ciascuna fase.

> Nota di contesto (giugno 2026): è in corso una crisi dei prezzi della DRAM che ha fatto salire molto i Raspberry Pi. I prezzi qui sotto sono indicativi e vanno riverificati al momento dell'acquisto.

---

## 1. Hardware — lista della spesa

### Target primario (consigliato)
- **Raspberry Pi 5, 8GB** (~130 $/€). È il punto di equilibrio: la CPU Cortex-A76 è il bersaglio "puro" su cui far girare i tuoi kernel integer-only, e 8GB bastano con margine per modelli sub-2B quantizzati + KV-cache + OS.
- **Alimentatore ufficiale 27W USB-C PD** (obbligatorio: il Pi 5 è esigente, e un'alimentazione instabile falsa le misure).
- **microSD A2 da 64GB** (o, meglio, un **SSD NVMe via HAT M.2** se vuoi I/O serio — utile ma non indispensabile in Fase 0–1).
- **Dissipatore attivo ufficiale** (obbligatorio: senza, il throttling termico inquina le misure di latenza/energia).

### Variante budget
- **Raspberry Pi 5, 4GB** (~80 $/€): sufficiente per modelli sub-2B a 4-bit. Resta sotto memoria solo se provi modelli più grossi o context lunghi.

### Per la misura dell'energia (la parte che dà credibilità)
Tre livelli di rigore crescente — scegli in base a quanto vuoi spingere il paper:
- **Base:** un **misuratore di potenza USB-C inline con logging** (registra V/A/W nel tempo tra alimentatore e Pi). Economico, sufficiente per un primo lavoro.
- **Intermedio:** una **presa smart con misura di consumo** + log, oppure un power meter da banco.
- **Rigoroso (consigliato per pubblicare):** un sensore **INA219/INA260** inline sull'alimentazione DC, letto da un microcontrollore/secondo Pi, per campionare a frequenza alta e calcolare i **Joule per token** in modo difendibile. È il dettaglio che distingue un measurement paper serio.

### Secondo dispositivo (opzionale, per la generalizzazione in Fase 2)
- Una board con **NPU** per confrontare CPU integer-only vs acceleratore commerciale: es. una **Rockchip RK3588** (Orange Pi 5 / NanoPC-T6, NPU ~6 TOPS), oppure l'**AI HAT+ 2** per Pi (40 TOPS, 8GB dedicati, pensato per LLM generativi 1–1.5B). Non serve in Fase 0–1; tienilo come estensione.

---

## 2. Modelli da usare

Scala piccola e onesta, tutti sotto i 2B, dove la quantizzazione aggressiva è difficile e quindi scientificamente interessante:

| Modello | Ruolo | Formato edge |
|---|---|---|
| Qwen2.5 0.5B | ancora "minuscola" | GGUF Q4/Q8 |
| Llama 3.2 1B | riferimento mainstream | GGUF Q4/Q8 |
| Qwen2.5 1.5B | taglia media | GGUF Q4/Q8 |
| **BitNet b1.58 2B4T** | angolo ternario nativo | GGUF `i2_s` (~1.2GB) |

BitNet è chiave: è un modello **nativamente ternario** ({-1, 0, +1}) con attivazioni a 8 bit, quindi è già "quasi integer-only" e ti dà il riferimento estremo del Pareto. Microsoft riporta ~0.028 J per inferenza contro ~0.347 J di Qwen2.5 a parità di scala — esattamente il tipo di numero che tu misurerai *tu stesso* sul Pi.

---

## 3. Setup e baseline — comandi concreti

> I flag esatti possono cambiare: verifica sempre il README dei due repo. Questi comandi riflettono lo stato a metà 2026.

### 3a. Preparare il Pi
```bash
# aggiorna firmware (rilevante: gli update recenti danno +performance gratis)
sudo rpi-eeprom-update -a && sudo reboot

# toolchain di build
sudo apt update && sudo apt install -y build-essential cmake git python3-pip python3-venv libcurl4-openssl-dev
```

### 3b. Baseline weight-only con llama.cpp
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# scarica un modello in GGUF (esempio: Llama 3.2 1B Instruct, variante Q4_K_M)
# usa huggingface-cli per scaricare la GGUF desiderata in ./models

# benchmark integrato (tokens/sec, prefill e decode separati)
./build/bin/llama-bench -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -t 4

# run singolo per misurare latenza/energia su un prompt fisso
./build/bin/llama-cli -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "Spiega la fotosintesi in tre frasi." -n 128 -t 4
```
Ripeti `llama-bench` per ogni bit-width (Q8_0, Q4_K_M, Q2_K) e per ogni modello, registrando le metriche della sezione 4.

### 3c. Baseline ternario con bitnet.cpp
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"

huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
  --local-dir models/BitNet-b1.58-2B-4T

# build + setup (kernel ternari LUT, basati su T-MAC)
python3 setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# se compare un errore di compilazione su "int8_t * y_col":
# sed -i 's/^\([[:space:]]*\)int8_t \* y_col/\1const int8_t * y_col/' src/ggml-bitnet-mad.cpp

# inferenza
python3 run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Spiega la fotosintesi in tre frasi." -cnv -t 4 -n 128
```

### 3d. Valutazione di qualità (perplexity / accuratezza)
- Perplexity: `llama.cpp` ha `llama-perplexity` su un corpus di testo standard (es. wikitext).
- Accuratezza su task: valuta su 1–2 benchmark leggeri (es. un sottoinsieme di GSM8K per il ragionamento, dove BitNet è notoriamente competitivo). Tieni il set piccolo e fisso per riproducibilità.

---

## 4. Scheda-metriche (cosa loggare, sempre, per ogni run)

Una riga di tabella per ogni combinazione (modello × bit-width × dispositivo × seed):

- **Qualità:** perplexity; accuratezza sul task scelto.
- **Velocità:** tokens/sec in *prefill*; tokens/sec in *decode*; latenza per-token (ms).
- **Energia:** **Joule per token** (la metrica-faro); potenza media (W); potenza di picco.
- **Memoria:** picco RAM (MB); dimensione del modello su disco (MB).
- **Per-operatore (chiave per la Fase 2):** frazione di tempo in operatori lineari (GEMM) vs non-lineari (Softmax, RMSNorm, GELU). In Fase 0 puoi stimarlo con un profiler; diventa centrale quando introdurrai i tuoi operatori interi.
- **Contesto:** temperatura ambiente, governor della CPU, numero di thread, versione del firmware/OS, commit dei repo. Senza questi, le misure non sono riproducibili.

Regola d'oro: **fissa il prompt, la lunghezza di generazione e il numero di thread**, e ripeti ogni misura su più seed/ripetizioni riportando media e deviazione.

---

## 5. Struttura del repo (l'harness G4 nasce qui)

```
efficient-on-device-ai/
├── README.md                 # claim, numeri principali, un grafico, come riprodurre
├── LICENSE                   # permissiva (MIT/Apache-2.0)
├── env/
│   └── setup_pi.sh           # installa toolchain + build llama.cpp/bitnet.cpp
├── models/
│   └── download.sh           # scarica le GGUF dei modelli scelti
├── bench/
│   ├── run_bench.py          # orchestratore: lancia i run e raccoglie le metriche
│   ├── energy.py             # lettura del power meter / INA219, calcolo J/token
│   ├── profile_ops.py        # breakdown latenza per-operatore
│   └── metrics_schema.md     # definizione esatta di ogni metrica (sezione 4)
├── intops/                   # FASE 1: i tuoi operatori integer-only
│   ├── int_softmax.py
│   ├── int_rmsnorm.py
│   └── int_gelu.py
├── results/
│   ├── raw/                  # log grezzi per run (csv/json)
│   └── tables/               # tabelle e figure aggregate
└── paper/
    └── notes.md              # claim, related work, gap, struttura del paper
```

---

## 6. Definition of Done

### Fase 0 — "il banco di misura è affidabile"
- [ ] Pi 5 configurato, firmware aggiornato, dissipatore attivo, alimentazione stabile.
- [ ] Power meter che logga e da cui sai calcolare i Joule per token.
- [ ] `llama.cpp` e `bitnet.cpp` compilati e funzionanti sul Pi.
- [ ] Tabella baseline completa: ogni modello × {Q8, Q4, Q2, ternario} con tutte le metriche della sezione 4, su più ripetizioni.
- [ ] Lo script `run_bench.py` rigenera la tabella da zero con un comando.

**Deliverable:** un primo grafico Pareto accuratezza-vs-energia dei baseline + un **blog post**. È già materiale mostrabile.

### Fase 1 — "la pipeline integer-only completa gira su edge"
- [ ] Implementati e validati (parità di accuratezza vs FP, scarto entro soglia) almeno **Softmax intera** e **RMSNorm intera**; idealmente anche GELU/SiLU intera.
- [ ] Integrati nel forward di **un** modello piccolo che produce output corretto.
- [ ] Il modello integer-only completo gira sul Pi e ne misuri qualità, latenza, J/token.
- [ ] Confronto a tre vie (FP vs weight-only vs integer-only completo) nella tabella.

**Deliverable:** il primo risultato che **nessuno ha mostrato** — pipeline interamente intera (non-lineari incluse) su un dispositivo edge reale, con numeri. È il nucleo del paper e il demo che ferma i recruiter.

---

## 7. Trappole da evitare

- **Misurare con throttling termico:** senza dissipatore attivo le latenze sono inaffidabili. Verifica `vcgencmd measure_temp` e fissa il governor.
- **Confondere weight-only e integer-only:** llama.cpp/bitnet.cpp quantizzano i pesi ma non danno la pipeline *interamente* intera. La novità della Fase 1 sono i tuoi operatori non-lineari interi — non darli per scontati nei baseline.
- **Energia non riproducibile:** misura sempre allo stesso punto (ingresso USB-C/DC), a temperatura stabile, con lo stesso prompt e la stessa lunghezza di generazione.
- **Scope creep:** non aprire la Fase 2 (caratterizzazione completa, multi-device) finché la Fase 1 non gira e si misura. Un modello, due operatori interi, numeri puliti: quello è l'MVP.

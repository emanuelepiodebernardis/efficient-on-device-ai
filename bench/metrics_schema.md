# Scheda-metriche (cosa loggare per ogni run)

Una riga per combinazione (modello × config × seed/ripetizione):

**Qualità:** perplexity; accuratezza su 1-2 task fissi.
**Velocità:** tokens/sec prefill; tokens/sec decode; latenza per-token (ms).
**Energia:** Joule per token (metrica-faro); potenza media (W); potenza di picco.
**Memoria:** picco RAM (MB); dimensione modello su disco (MB).
**Per-operatore:** frazione di tempo in GEMM vs non-GEMM (Softmax, RMSNorm, SiLU).
**Riproducibilità:** temperatura, CPU governor, n. thread, commit dei repo, versione firmware/OS.

Regola: prompt fisso, lunghezza di generazione fissa, thread fissi; ripetere su più seed; riportare media e deviazione.

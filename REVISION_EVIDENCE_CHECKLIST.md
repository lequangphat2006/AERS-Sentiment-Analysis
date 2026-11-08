# Revision Evidence Checklist

## Done
*   [x] N-B statistics, TOST, Holm.
*   [x] Calibration 6-row table (TS/Platt).
*   [x] Ablation (Lexicon, POS, Class weight, Base learners).
*   [x] Baselines: Tuned LR (0.8501), Tuned LinearSVC (0.8497).
*   [x] Split audit (sklearn 1.6.1 vs 1.9.1).
*   [x] Near-duplicate sensitivity (proxy).
*   [x] PII masking (phones/URLs).
*   [x] T01 Kappa: 0.803 (see 	01_annotation/).

## Remaining / Limitations
*   [ ] **A2 (Calibration split-half):** Not performed. Relying on Table 6 as indirect evidence.
*   [ ] **RDRSegmenter:** Not tested. Underthesea used. PhoBERT may be under-evaluated.
*   [ ] **RAM measurement:** Not measured for deployment.
*   [ ] **Optuna:** Not reproducible across environments (best_params differ 10/10 seeds). Original run 0.8665 is the headline; re-runs 0.8625 are for sensitivity.

## Final Note on Results (08/11)
*   The per-sample logits and JSON provided in this repository correspond to the **5-epoch run** (F1 = 0.8805). 
*   As recommended by the reviewer, because the 8-epoch logits were not fully preserved, the 5-epoch run is now used as the primary result for F1, p-values (p = 0.081), and all calibration plots. 
*   The 8-epoch result (F1 = 0.8834) is reported only as a supplementary sensitivity check.
*   All figures (including the fixed Reliability Diagram) have been regenerated using the 5-epoch logits.

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

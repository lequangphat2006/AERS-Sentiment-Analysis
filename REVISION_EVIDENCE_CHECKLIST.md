# Revision Evidence Checklist (QNUJS-B2662)

This document maps each reviewer request to the evidence in this repository and lists the items that are still pending. Paths are relative to the repository root.

## 1. Statistical inference (R2-1, R1-3.4)
- [x] Nadeau-Bengio corrected paired t-test, TOST at several margins, Holm adjustment: `tables/tests_nadeau_bengio_8ep.csv` (8-epoch PhoBERT, main) and `tables/tests_nadeau_bengio_5ep.csv` (5-epoch, sensitivity).
- [x] F1-macro difference: PhoBERT 0.8834 vs AERS-Pro 0.8665; corrected 95% CI [-0.0337, -0.0002], p = 0.048 (not adjusted for multiple comparisons; about 0.10 after Holm over the four primary metrics, to be confirmed with `scripts/stats.py`). Equivalence is not established at margins 0.01, 0.015, 0.02, 0.03.

## 2. Fair calibration comparison (R2-7)
- [x] Raw ECE: AERS-Pro 0.0241 vs PhoBERT 0.1069.
- [x] After post-hoc calibration of PhoBERT (temperature scaling 0.0314, 3x3 Platt 0.0225) the ECE and Brier differences are not significant: `tables/calibration_6rows_8ep.csv`, `tables/calibration_6rows_5ep.csv`.

## 3. Baselines and ablation (R2-5)
- [x] AERS-Pro (stacking) 0.8665; B2 (LinearSVC on TF-IDF + hashed character features) 0.8454; B3 (TF-IDF only) 0.8252; B4 (hashed character features only) 0.7972: `tables/baseline_b3_b4_results.csv`.
- [x] Ablation (stacking vs LR, vs probability averaging, without lexicon / POS / class weight, calibration variants): `tables/ablation_table.csv`, `tables/ablation_table_per_seed.csv`. Lexicon and POS blocks show no measurable contribution.
- [ ] Per-seed files and run scripts for B2/B3/B4.
- [ ] Individual base learners (SVM, LightGBM) reported separately.

## 4. Figures
- [x] Confusion matrix, seed 42: `figures_t05/fig_confusion_matrix.png`
- [x] Paired F1 across 10 seeds: `figures_t05/fig_f1_per_seed.png`
- [x] Reliability diagram: `figures_t05/fig_reliability_diagram.png`

## 5. Deployment (R2-11)
- [x] Model size: AERS-Pro 7.44 MB mean over 10 seeds (5.95-8.27 MB) vs PhoBERT 515-517 MB (about 69x). Size is the total of the files needed for inference (`t10/inference_benchmark.json`).
- [x] Batch of 32 on CPU, per sentence: AERS-Pro 24.4 ms vs PhoBERT 280 ms.
- [x] Single sentence on CPU (median / p95): AERS-Pro 171 / 511 ms vs PhoBERT 87 / 194 ms; AERS-Pro is slower for single requests.
- Limits: one seed and one CPU type; the PhoBERT timing excludes word segmentation and uses a randomly initialised classification head.

## 6. Documentation (R2-3)
- [x] Pseudo-code (Algorithm 1) and hyperparameter table: `docs/AERS-Pro_Pseudocode_and_Hyperparameters.docx`.
- [x] Split audit for 10 seeds x 3 parts (sizes, groups, class proportions) under scikit-learn 1.6.1 and 1.9.1: `evidence/split_audit_sklearn_1.6.1.csv`, `evidence/split_audit_sklearn_1.9.1.csv`.
- [x] scikit-learn 1.8 release note read; `REPRODUCIBILITY.md` section 1 rewritten accordingly (fix of shuffle behaviour in `StratifiedGroupKFold`, PR #32540).

## 7. Reproducibility and data protection
- [x] scikit-learn pinned: `requirements.txt` (1.6.1).
- [x] Raw corpus is not distributed; PII-masked corpus: `data/dataset_cleaned_masked.csv`, produced by `scripts/mask_pii.py` (output contains 5 [PHONE], 53 [URL], 17 [SOCIAL] including 2 @handles, 3 [REFCODE] and 0 e-mails; 71 of 25,289 rows changed; the 19 phone-pattern matches counted before masking overlap with social-media handles; a re-scan finds no residual phone numbers, @handles, referral codes or e-mails). Personal names and addresses were not screened.
- [ ] Release rebuilt from a repository whose git history contains no raw corpus; new DOI recorded here and in `README.md` (the DOI currently in `README.md`, (old record, withdrawn), belongs to the previous repository).

## 8. Statistical caveats to carry into the paper
- The Nadeau-Bengio correction is a conservative approximation: the test sets of different seeds overlap.
- The calibration advantage of AERS-Pro holds against raw PhoBERT only.
- Near-duplicate sensitivity was measured in a word-TF-IDF proxy space.

## 9. Pending (not yet done)
- [ ] Second annotator and inter-annotator agreement (R1-1i, R1-3.2, R2-4): tools in `scripts/make_label_sample.py` and `scripts/compute_kappa.py`.
- [ ] Name and version of the LLM used for label review, its prompt, and the number of labels changed (R2-4).
- [ ] Counts for the two data-collection stages (R2-4).
- [ ] PhoBERT word segmentation with RDRSegmenter, or an explicitly scoped claim (R2-6).
- [ ] Justification for choosing PhoBERT over ViSoBERT (R1-3.3).
- [ ] Sensitivity of the single calibration split used for several selection roles (R1-1iii, R2-7).
- [ ] Funding statement, Shopee Terms of Service and personal-data-protection statement (R1-3.6).


Zenodo DOI of the clean release: https://doi.org/10.5281/zenodo.23797912

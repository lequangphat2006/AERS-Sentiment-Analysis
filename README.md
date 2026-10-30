[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23797912.svg)](https://doi.org/10.5281/zenodo.23797912)


# AERS-Pro: A Hybrid Stacking Framework for Vietnamese E-commerce Sentiment Analysis

AERS-Pro (v40) is a hybrid machine learning framework for three-class (negative / neutral / positive) sentiment classification of Vietnamese e-commerce reviews. It fuses TF-IDF and character-hashing text representations with handcrafted Vietnamese-specific linguistic features, combines several heterogeneous base learners through out-of-fold stacking, and applies a conditional probability-calibration stage on top.

## Why This Exists

Deep transformer fine-tuning is not always available or necessary: it requires a GPU, is slower to iterate on, and is harder to inspect. AERS-Pro targets the same three-class sentiment task using only classical ML components plus a lexicon and rule layer tuned specifically for Vietnamese e-commerce text.

## Key Findings (Revision)

All numbers are means over 10 seeds on the same group-aware test partitions, compared with the Nadeau-Bengio corrected paired t-test. PhoBERT is `vinai/phobert-base-v2`, learning rate 3e-5, up to 8 epochs with early stopping (patience 2), checkpoint selected by F1-macro on the calibration split. The learning rate was chosen on the calibration split, never on the test split (sweep over 1e-5, 2e-5, 3e-5, 5e-5; see `results/lr_sweep_calibration_f1.csv`).

**Accuracy.** A tuned PhoBERT baseline reaches an F1-macro of **0.8834**, which is **0.0169** (1.69 percentage points) higher than AERS-Pro (**0.8665**); the corrected 95% CI of the difference is [-0.0337, -0.0002] (p = 0.048). AERS-Pro wins on 1 of 10 seeds. This p-value is not adjusted for multiple comparisons; a Holm adjustment over the four primary metrics (F1, Accuracy, ECE, Brier) would put the F1 p-value near 0.10. Equivalence (TOST) is not established at margins of 0.01, 0.015, 0.02 or 0.03. The gap is concentrated in the negative and positive classes; the neutral class is tied (F1 0.8426 vs 0.8415, p = 0.94). No subgroup (negation, hedging, contrast, short text) shows a significant difference. Results are essentially unchanged when residual near-duplicate pairs are removed (measured in a word-TF-IDF proxy space, not the full feature space). With the earlier 5-epoch PhoBERT (F1 0.8805) the gap is 0.0140 (p = 0.081).

**Calibration.** AERS-Pro's raw probabilities are well calibrated (ECE **0.0241**) while raw PhoBERT is not (ECE **0.1069**). This advantage disappears once PhoBERT is calibrated post hoc: with temperature scaling (ECE 0.0314) or 3x3 Platt scaling (ECE 0.0225), the ECE and Brier differences to AERS-Pro are not statistically significant.

**Cost.**

| Aspect | AERS-Pro | PhoBERT |
|---|---|---|
| F1-macro | 0.8665 | **0.8834** (8 epochs) |
| Raw ECE | **0.0241** | 0.1069 |
| ECE after temperature scaling / Platt | n/a (already calibrated) | 0.0314 / 0.0225 |
| Model size | **7.44 MB** (5.95-8.27 MB across seeds) | 515-517 MB (about **69x** larger) |
| Training hardware | CPU only (4 vCPU) | GPU (2 x T4) |
| Training time per seed | about 2,080 s (30 Optuna trials) | about 1,660 s (5 epochs) |
| Single-sentence CPU latency (median; one seed, one CPU type) | 171 ms | 87 ms (excludes word segmentation) |
| Batch of 32 on CPU, per sentence | **24.4 ms** | 280 ms |

AERS-Pro is therefore lighter to store and deploy and does not need a GPU, and it processes batches much faster on CPU. It is not faster to train and is slower for single-sentence requests. See `RESULTS_SUMMARY.md` and the paper for the full analysis.

**Component ablation (10 seeds).** Stacking is about 0.013 F1 above a tuned logistic regression on the fused features (9 of 10 seeds, but not significant after the Nadeau-Bengio correction, p = 0.12). Removing the lexicon block, the POS block or class weighting changes F1 by less than 0.002 (all p > 0.7).

## Architecture Overview

The pipeline is implemented in `src/train_v40.py` and runs in the following stages.

### 1. Text Preprocessing (`TextPreprocessor`)

- **Teencode / abbreviation normalization**: `ko`, `k`, `dc`, `vs`, `sp`, ... are mapped to standard Vietnamese.
- **Emoji-to-sentiment-word mapping.**
- **Clause-boundary-aware negation handling**:
  - Negation scope is tracked per clause rather than with a fixed token window.
  - Even/odd negation counts drive sign flipping, which handles double negation.
  - Lookahead rules avoid misreading tag-questions (`...không?`, `...không vậy?`) and locational "đâu" as negation.
- **Contrast-marker reweighting** ("nhưng", "tuy nhiên", "mà", ...): the clause before the marker is downweighted (0.5x) and the clause after keeps full weight, implemented as a single buffer-and-flush pass over clauses.
- **Punctuation-safe tokenization**: lexicon and multi-word phrase matching (compound terms joined with `_`) does not silently fail on punctuation-glued tokens such as `"tốt,"` or `"hỏng!"`.

### 2. Manual (Lexicon) Feature Extraction (`TextPreprocessor.extract_features`, `MetaFeatureExtractor`)

- Positive/negative lexicon hit counts and `sentiment_ratio`.
- `is_neutral` flag, whose threshold **T** is tuned on the Calibration split by maximizing mutual information with the true label (`tune_neutral_threshold`) rather than being fixed by hand.
- Relative position of the last sentiment-bearing word, `is_question`, `is_tag_question`, and interaction columns (`pos_s_x_tagq`, `neg_s_x_tagq`), so the linear meta-learner can discount sentiment scores on tag-questions without discounting genuine rhetorical questions.
- POS-tag ratios (adjective / verb) when `underthesea` is available.

### 3. Feature Pipeline (`FeaturePipeline`)

- Word-level TF-IDF (1-2, 1-3 or 1-4 grams, selected by Optuna; sublinear TF optional).
- Hashed character n-gram features (`HashingVectorizer`, `char_wb` 2-4; 2^14 or 2^15 buckets selected by Optuna), a fixed-size, stateless complement to the word TF-IDF block. This block is hashed counts, not TF-IDF.
- Manual lexicon and POS-ratio features, each min-max scaled and clipped to `[0, 1]` at inference time to guard against out-of-range extrapolation on unseen data.
- All feature blocks are concatenated into a single sparse matrix.

### 4. Base Learner Pool (`BaseLearnersPool`)

- Logistic Regression (`class_weight='balanced'`).
- LinearSVC wrapped in `CalibratedClassifierCV` (Platt/sigmoid). When a grouping column is supplied, calibration uses group-aware folds (`StratifiedGroupKFold`) so near-duplicate reviews never leak across the internal calibration split.
- LightGBM (optional, used when installed).

Learners are rebuilt fresh for every fold rather than cloned, because the SVM's internal calibration folds depend on the exact `(y, groups)` of the data currently being fit.

### 5. Stacking Meta-Learner (`StackingMetaLearner`, `MetaClusterer`)

- Out-of-fold (OOF) predicted probabilities from each base learner are stacked as meta-features.
- A one-hot K-Means cluster indicator (default `k=4`) over a meta-feature space, together with the manual lexicon features, is concatenated **directly** into the meta-learner's input. This lets the linear meta-learner use lexicon signal without it being diluted inside the TF-IDF space.
- The final stacking model is a regularized multinomial Logistic Regression.

### 6. Conditional Probability Calibration (`ProbabilityCalibrator`)

- A small Platt-style calibration head (logistic regression over the logit-transformed stacked probabilities) is fit on the held-out Calibration split.
- It is applied **only if** it improves calibration-split ECE **and** costs no more than 0.5 percentage points of calibration-split F1-macro. Otherwise it is discarded and the raw stacked probabilities are used. In the original 10-seed run it was kept for 4 seeds (123, 2026, 8080, 12321); in the re-run for 6 seeds.

### 7. Hyperparameter Search

Optuna (TPE sampler, 30 trials per seed) tunes five hyperparameters (`tfidf_max_feat`, `ngram_range`, `lr_C`, `hash_features`, `sublinear_tf`) through a logistic regression on the fused features, with a group-aware cross-validation objective on the Train split; the SVM and LightGBM base learners are not part of the search objective. Optuna's choices are not reproducible across software environments (see "Reproducibility" below); the selected values of the original run are stored in `results_v40_multiseed.json` (`best_params`).

### 8. Data Splitting

A roughly 60/20/20 Train/Calibration/Test split is produced by `StratifiedGroupKFold` (5-fold, then 4-fold within the training part, `shuffle=True`) over a near-duplicate group id. Near-duplicate reviews (see `dedup_clustering.py`) never appear on both sides of a split. Group-aware splitting does not give exactly stratified test sets; test sizes range from 4,877 to 5,315 across seeds.

## Repository Layout

```text
.
├── src/
│   ├── train_v40.py                    # Full pipeline: preprocessing, features, base learners, stacking, calibration
│   ├── train_phobert_v2.py             # PhoBERT baseline: hyperparameter sweep, multi-seed runner, logits export
│   ├── seeds_config.py                 # SEEDS: single source of truth for the 10-seed protocol
│   ├── dedup_clustering.py             # Generates near_dup_group_id
│   ├── extract_lexicon_v40.py          # Dumps the lexicon / teencode / emoji tables
│   ├── finish_analysis.py              # Unified subgroup tagging + figures
│   ├── v40_finish.py                   # Environment / version-drift diagnostics
│   └── reference_lexicon_v40.json      # Serialized lexicon data
├── data/
│   └── dataset_cleaned_masked.csv      # Training data (25,289 rows, PII-masked)
├── scripts/
│   ├── run_ablation.py                 # Original ablation (cluster / neutral-flag variants)
│   ├── run_ablation_fixed.py           # Revision ablation (lexicon / POS / class-weight variants)
│   ├── verify_identical_partitions.py  # Confirms identical train/cal/test partitions across pipelines
│   ├── mask_pii.py                     # Masks phone numbers, URLs, e-mails and social handles; re-scans after masking
│   ├── make_label_sample.py            # Draws the stratified sample for the second annotator (writes a private key file)
│   └── compute_kappa.py                # Inter-annotator agreement (Cohen's kappa, per-class, confusion matrix)
├── results/
│   ├── results_v40_multiseed.json      # AERS-Pro aggregated metrics (mean ± std), original run
│   ├── results_phobert_8ep.json        # PhoBERT 8-epoch aggregated metrics
│   ├── ablation_table.csv              # Ablation results (mean)
│   ├── ablation_table_per_seed.csv     # Ablation results (per-seed)
│   ├── lr_sweep_calibration_f1.csv     # PhoBERT learning-rate sweep (calibration F1)
│   ├── phobert_logits/                 # Per-sample logits for PhoBERT (10 seeds)
│   └── v40_probs/                      # Per-sample probabilities for AERS-Pro (10 seeds)
├── evidence/                           # Split audits (scikit-learn 1.6.1 and 1.9.1) and training logs
├── tables/                             # Statistical tests, calibration, per-class, subgroup and ablation tables
├── t10/                                # Inference benchmark (latency, memory, model size)
├── REVISION_EVIDENCE_CHECKLIST.md      # Reviewer request -> evidence map, including items still pending
├── REPRODUCIBILITY.md                  # Environment pinning and known reproducibility caveats
├── RESULTS_SUMMARY.md                  # Full tables behind the numbers in this README
└── README.md                           # This file
```

## Setup

```bash
pip install -r requirements.txt
```

> **Critical reproducibility note**
> `scikit-learn` **must** be pinned to exactly `1.6.1`. Other versions can produce different train/calibration/test partitions from the same seed. The scikit-learn 1.8 release notes list a fix to the shuffle behaviour of `StratifiedGroupKFold` (stratification is now preserved when `shuffle=True`); with 1.9.1 we observed 15,173 / 5,058 / 5,058 partitions for every seed, instead of the 14,849 / 5,125 / 5,315 obtained for seed 42 under 1.6.1. With any other version, the paired statistical tests reported in the paper cannot be reproduced exactly. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for details.

## Usage

**1. Generate near-duplicate groups** (required once, before any group-aware training or splitting):

```bash
python src/dedup_clustering.py --input_dir data --output_dir data_deduped --threshold 0.95
```

**2. Run the full multi-seed protocol for AERS-Pro** (the 10 seeds in `seeds_config.py`):

```bash
python src/train_v40.py
```

**3. Run the PhoBERT baseline (8 epochs):**

```bash
python src/train_phobert_v2.py --lr 3e-5 --epochs 8 --result_path results/results_phobert_8ep.json
```

**4. Run the ablation study** (revision variants: `full`, `no_pos`, `no_lex`, `no_both`, `no_cw`):

```bash
python scripts/run_ablation_fixed.py <cache_file> <out_dir> <seeds> <variants>
```

The original cluster / neutral-flag ablation is still available as `scripts/run_ablation.py --input_dir data --group_col near_dup_group_id`.

**5. Mask personal data** (only needed if you hold the raw corpus; the raw file is not distributed):

```bash
python scripts/mask_pii.py --src data/dataset_cleaned.csv --dst data/dataset_cleaned_masked.csv
```

**6. Verify that two pipelines produce identical partitions** (e.g., this system vs. the PhoBERT baseline):

```bash
python scripts/verify_identical_partitions.py --data data/dataset_cleaned_masked.csv --group_col near_dup_group_id
```

**7. Second-annotator sample and agreement** (the key file `annotation_key_PRIVATE.csv` must never be committed):

```bash
python scripts/make_label_sample.py --src data/dataset_cleaned_masked.csv --out label_round2 --seed 20261101
python scripts/compute_kappa.py --key label_round2/annotation_key_PRIVATE.csv --ann label_round2/annotation_done.csv --out label_round2/agreement
```

## Evaluation Protocol

- **Seeds:** 10 fixed seeds (`src/seeds_config.py`), shared with the PhoBERT baseline to enable paired statistical testing.
- **Splits:** roughly 60/20/20 group-aware Train/Calibration/Test split per seed (for seed 42: Train 14,849 / Calibration 5,125 / Test 5,315).
- **Metrics:** F1-macro, Accuracy, Expected Calibration Error (ECE, 10 bins; 15 and 20 bins as sensitivity checks), multiclass Brier score (un-normalized, range [0, 2]), training time, and model size.
- **Significance testing:** Nadeau-Bengio corrected paired t-test (variance factor 1/J + n_test/n_train, about 0.43 here, accounting for overlapping training sets across seeds), TOST for equivalence at several margins, and Holm-Bonferroni correction within each family of comparisons. Wilcoxon signed-rank is reported only as a robustness check.
- **Post-hoc calibration of the baseline:** temperature scaling (fit on the calibration logits) and 3x3 Platt scaling, so that AERS-Pro is compared with a calibrated PhoBERT as well as a raw one.

## Reproducibility

- Figures in the paper come from the original run (`results/results_v40_multiseed.json`, `results/v40_probs/`, `results/phobert_logits/`).
- A full re-run of all 10 seeds in a new environment (Kaggle CPU, scikit-learn 1.6.1, underthesea 9.5.0) gave a mean F1 of 0.8625 (original 0.8665; per-seed differences from -0.019 to +0.011) and the same mean ECE (0.0242 vs 0.0241). The Optuna-selected hyperparameters differed from the original run for all 10 seeds, so exact numeric reproduction of the hyperparameter search should not be expected across environments. The trained models of the re-run (10 seeds) and its environment files are archived with the release.


## Data use and privacy

`data/dataset_cleaned_masked.csv` contains review texts with phone numbers, URLs, e-mail addresses, social-media handles and referral codes replaced by placeholders (`[PHONE]`, `[URL]`, `[EMAIL]`, `[SOCIAL]`, `[REFCODE]`). Personal names and postal addresses were not screened. The corpus is provided for non-commercial research use only; if you find residual personal information, please open an issue so it can be removed. All experiments were run on the unmasked corpus; the released file differs from it in 71 of 25,289 rows (same row order, labels and near-duplicate groups). Per-sample prediction files contain `row_index` and numeric outputs only; review text can be recovered from the masked corpus through `row_index`.
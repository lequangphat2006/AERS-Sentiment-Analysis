# Reproducibility Notes

This document collects the environment details and reproducibility caveats that were originally reported as Section 5.6 ("Reproducibility notes") of the accompanying paper. It is intended as a companion to the code in this repository: read it before re-running any of the multi-seed experiments, and especially before comparing new numbers against the ones reported in the paper.

## Reference environment

All results reported in the paper were produced on Kaggle Notebooks with the following configuration:

- **scikit-learn**: 1.6.1
- **CPU**: 4 vCPUs, Intel Xeon @ 2.20 GHz
- **RAM**: 31 GB
- **Accelerator**: none (CPU only)
- **Container image**: `kaggle_thxJ1vmUVWo8x8mcXPnotWkgU4913GLeqjOrTXxzqE-342271493-webtier`
- **OS**: Ubuntu 22.04.5 LTS
- **Python**: 3.12.13

Three observations about reproducibility emerged while assembling these results. Each one bounds how the per-seed figures in the paper should be read.

## 1. The group-aware split is version-sensitive

All reported results use `scikit-learn==1.6.1` with `StratifiedGroupKFold(shuffle=True, random_state=seed)` over `near_dup_group_id` (outer 5-fold, then inner 4-fold within the training part). Shuffling is also enabled in the Optuna cross-validation, the out-of-fold stacking and the internal calibration folds.

The scikit-learn 1.8 release notes list one change to `StratifiedGroupKFold`: a fix of the shuffle behaviour, so that stratification among folds is also preserved when `shuffle=True` (PR #32540). The release notes do not describe any other change to how folds are built. Because every split in this project uses `shuffle=True`, partitions produced with scikit-learn 1.8 or later can differ from those produced with 1.6.1.

Our split audit (`evidence/split_audit_sklearn_1.6.1.csv`, `evidence/split_audit_sklearn_1.9.1.csv`; 10 seeds, three parts each) shows the difference empirically:

| | scikit-learn 1.6.1 | scikit-learn 1.9.1 |
|---|---|---|
| Train / calibration / test sizes | vary by seed (train 14,849-15,273; calibration 4,939-5,174; test 4,877-5,315) | 15,173 / 5,058 / 5,058 for every seed |
| Largest absolute deviation of a class proportion from the corpus proportion | about 0.037 (calibration), 0.024 (test) | at most 0.0002 |

For seed 42 the sizes are 14,849 / 5,125 / 5,315 under 1.6.1. The 1.6.1 partitions are therefore not exactly stratified, which is a limitation of the evaluation protocol. We did not retrain either system on the 1.9.1 partitions, so the effect of this difference on the reported F1 and calibration results has not been measured.

**Implication:** the scikit-learn version is part of the experimental specification, not an implementation detail. Pin `scikit-learn==1.6.1` to reproduce the reported partitions. Identical split hashes across the two systems show that both used the same partitions; they do not by themselves show that the partitions are well stratified, which is why the class-proportion tables in `evidence/` are provided.

## 2. Optuna search is the only unstable stage across machines

An earlier complete run of the same code over the same ten seeds, on a different and less well documented machine, allows the pipeline to be checked for stability across environments — and locates the one component that isn't stable.

- Everything upstream of hyperparameter search is bit-identical across the two machines: the partitions agree for all ten seeds, and the neutral-region threshold (with its mutual-information score) agrees to seventeen significant digits in every seed.
- Divergence enters at exactly one point: the **Optuna search**. The cross-validation objective is evaluated with a degree of parallelism derived from the available core count, so the order of floating-point accumulation differs between machines, and the Tree-structured Parzen Estimator explores a different trial sequence despite a fixed seed.

The consequence is visible **per seed** but washes out **in aggregate**:

| Metric | This run (mean, 10 seeds) | Earlier run (mean, 10 seeds) |
|---|---|---|
| F1-macro | 0.8665 | 0.8667 |
| Accuracy | 0.8692 | 0.8697 |
| ECE | 0.0241 | 0.0253 |
| Brier | 0.1946 | 0.1952 |

Individual seeds can move substantially even though the means agree — e.g. ECE at seed 42 was 0.0311 here vs. 0.0109 in the earlier run; at seed 2026, 0.0196 vs. 0.0332. The per-class F1 pattern of Table 3 also holds in both runs (−0.0141 / +0.0226 / −0.0152 here vs. −0.0120 / +0.0200 / −0.0150 there).

One difference is worth stating explicitly: in the earlier run, the Accuracy gap of Table 2 did not reach significance on its own (p = 0.152, vs. p = 0.038 here) — a further reason equivalence is claimed for F1-macro only.

**Implication:** treat per-seed figures from this pipeline as draws from a distribution, not fixed quantities. This is the argument for reporting ten seeds with paired tests rather than a single split, and for not over-interpreting any single seed's numbers.

## 3. Unrecorded environments cost timing data

The earlier run (see point 2) also illustrates the cost of not recording an environment: its wall-clock training time was **2.6× shorter** than the figures in Table 2, for identical code and data. The difference was concentrated entirely in the two compute-bound stages (hyperparameter search and out-of-fold generation) rather than in preprocessing, indicating a machine with more available cores — but that machine's configuration was never logged and cannot now be reconstructed.

As a result, that earlier run cannot contribute a timing figure to the paper and is used only for the agreement check in point 2. **Training time is reported solely from the reference environment documented at the top of this file.**

**Recommendation:** any training script should log host core count, memory, and library versions at the start of a run — a step this pipeline did not originally take.

## 4. Two design decisions are narrow enough to flip under this variation

Two mechanisms in the framework are decided by comparisons whose margins are narrow enough to flip under the machine-to-machine variation described in point 2. Both should be read as design limitations rather than settled choices:

- **Conditional calibrator** (Section 3.1): retained only if it improves calibration-split ECE *and* does not reduce calibration-split F1-macro by more than 0.5 percentage points. Both conditions fail at the margin in some seeds — in one seed the two ECE values differ by only 0.000033; in another, the calibrator improved ECE by 0.0127 yet was discarded because F1-macro fell by 0.0051 against a 0.0050 tolerance. A tolerance band, or selection on a single combined objective, would be a more stable design.
- **Hyperparameter search bounds**: the upper limit on TF-IDF vocabulary size was selected in 4 of 10 seeds (the lower limit, in none); the regularization strength reached 9.77 within a range ending at 10; the widest n-gram setting offered was never selected. A wider upper range for the first two, and a narrower categorical set for the third, would be the appropriate correction.

## 5. Word-segmentation availability changes inference results silently

The framework's inference path depends on whether a Vietnamese word-segmentation package (`underthesea`) is importable at runtime, because preprocessing falls back to whitespace tokenization when it is absent.

Running inference from a saved model in an environment where the package **is** present, while the model was fitted where it was **not**, cost **1.4 percentage points of accuracy** — without raising any error.

**Recommendation:** a saved model should record its preprocessing environment (whether word segmentation was available at fit time) and refuse to run — or at least warn loudly — under a different one.

## Verification standard used for this paper

Every figure and subgroup statistic reported in the paper was required to reproduce its seed's logged F1-macro and accuracy from an independent per-sample export before being used. All ten seeds passed this check.

## Practical checklist before reproducing results

- [ ] Pin `scikit-learn==1.6.1` (already pinned in `requirements.txt`).
- [ ] Use the seed list in `seeds_config.py` (`SEEDS`) — do not regenerate it.
- [ ] Run `dedup_clustering.py` before training, so `near_dup_group_id` exists in `input_data/*.csv`.
- [ ] Do not compare wall-clock training time across machines with different core counts; only compare it against a run on the reference environment above.
- [ ] Confirm whether `underthesea` is importable in both the training and inference environments before trusting an exported model's predictions.
- [ ] Treat single-seed metrics as illustrative only; use the multi-seed summary (mean ± std, paired tests) for any claim of equivalence or improvement.

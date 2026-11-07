# T01: Inter-Annotator Agreement

*   **Total Samples:** 500 (stratified by annotator 1's label; 40% neutral)
*   **Cohen's Kappa (Overall):** 0.8026
*   **Cohen's Kappa (Linear Weighted):** 0.8355
*   **Per-class Kappa:** Negative 0.7373 | Neutral 0.7405 | Positive 0.9426
*   **Arbitration:** 24 labels were changed by the first author after reviewing the 48 conflicting neutral cases.
*   **Note:** The models were trained on the original corpus. The arbitration is reported for transparency but does not affect the test results.

## Disagreement Matrix (Original vs Annotator 2)
| True \ Predicted | negative | neutral | positive |
| :--- | :--- | :--- | :--- |
| **negative** | 143 | 6 | 1 |
| **neutral** | 48 | 148 | 4 |
| **positive** | 5 | 2 | 143 |

# Analysis run

Input SHA-256: `7e0e0421398c31ef23f8fe97a8a3479d5dfaf6735a6afe7ea6b1c5035654660f`.

Validation mode: full-shape source (provenance requires review). Rows analyzed: 253,680. Statistical tests executed: 25; skipped: 0.

Review quality.json and cleaning.json before interpreting eda/observations.md and hypotheses.csv. Blank inferential fields are explained in status, notes, or ci_method. A manifest is written only after all four stages finish.

## Source and endpoint

This run uses the uploaded **three-class** survey table. The explicit combined endpoint maps 0 to negative and both 1 (prediabetes) and 2 (diabetes) to positive. The source contained 213,703 / 4,631 / 35,346 respondents in classes 0 / 1 / 2. Combining classes produces 39,977 positives (15.76%); this derived file is not claimed to be the official binary release.

No missing or invalid feature values were found. All rows were retained, including 23,968 identical profiles, because there are no respondent IDs to establish accidental duplicates. All 25 tests executed, and nine figures were generated.

For example, high blood pressure has an unadjusted odds ratio of 4.782 (95% CI 4.668–4.898) for the combined endpoint. This is an exploratory association, subject to confounding and unavailable survey design; it is not a causal or diagnostic estimate. Very small p-values may underflow to numeric zero. See the methods document before interpreting the table.

The row-level `cleaned.csv` referenced by the workflow is intentionally excluded from Git; reproduce it with the README commands. Checked-in CSVs here contain aggregate summaries only. The provided ZIP and derived input remain local ignored data, with hashes and mapping in `source_provenance.json`.

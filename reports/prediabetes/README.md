> **Secondary (sensitivity) report.** Prediabetes versus no diabetes, existing diabetes excluded. The [primary report](../uploaded-combined/README.md) uses the combined endpoint.

# Analysis run

Input SHA-256: `33018d988b7abcc627bc03717a0dfb8a2c7685814776a83182e7bbf16e7978e3`.

Endpoint: prediabetes. Validation mode: full-shape source (provenance requires review). Rows analyzed: 218,334. Statistical tests executed: 25; skipped: 0.

Review quality.json and cleaning.json before interpreting eda/observations.md and hypotheses.csv. Blank inferential fields are explained in status, notes, or ci_method. A manifest is written only after all four stages finish.

## Prediabetes-only endpoint

The source has 253,680 respondents. Original label 2 (existing diabetes) is excluded: 35,346 respondents. Original labels 0 and 1 remain unchanged as negative and positive, respectively. The analysis therefore includes 218,334 respondents: 213,703 without diabetes and 4,631 with self-reported prediabetes (2.12%).

This eligibility rule is recorded in the source provenance, along with the exact uploaded/derived hashes. All eligible respondents were retained, including 23,650 identical profiles; there were no missing or invalid feature values. All 25 planned association tests executed, with zero skips. The report contains nine figures.

The target is current self-reported prediabetes. This survey has no follow-up outcome to evaluate future diabetes onset, and existing diabetes is not treated as a negative control. Associations may reflect confounding or survey design. They are not causal, diagnostic, or predictive-performance estimates.

The cleaned row-level table stays locally under `outputs/prediabetes/cleaned.csv` and is excluded from Git. Reproduce the run from the uploaded ZIP using the root README. Stages 5–8 have not been implemented.

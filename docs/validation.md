# Verification record

Validated in the cloud checkout with Python 3.12.14 and `requirements.lock`.

- Unit and integration suite: 27 passing tests; source statement coverage above 98%, with an enforced minimum of 80%.
- `pip check`: no broken dependency requirements.
- Packaging: source distribution and wheel built successfully with `python -m build`.
- CLI integration: a deterministic small synthetic fixture exercised all four stages, checked outputs, and verified explicit failure and overwrite protection.
- Scale check: a separate, explicitly synthetic 253,680-row × 22-column fixture completed all four stages, produced nine figures, and executed all 25 planned tests with zero skips. Output shape, synthetic provenance label, and adjusted p-value completeness were checked. This validates workflow execution at the requested scale, not actual BRFSS findings.
- Synthetic test data uses fixed seeds; production analyses contain no random sampling. The source hash and installed package versions are captured in each successful manifest.

## Remaining real-data validation

The public UCI and Kaggle endpoints returned HTTP 403 through the cloud egress proxy. The required domains were saved in the environment configuration draft, but draft persistence does not activate network access. No real source CSV was available in this checkout. The downloader's archive handling is unit tested; a live download and the actual source-cohort analysis remain unverified. There are no claimed BRFSS prevalence estimates, real-data charts, or empirical association findings in this commit.

Enable `archive.ics.uci.edu` in environment settings and apply the network change, or place the requested Kaggle CSV locally. Then follow the README download/run commands without `--synthetic-rows`. Review:

1. `quality.json`: source hash, 253,680 rows, 22 columns, target counts, invalid/missing codes, duplicate count.
2. `cleaning.json`: default duplicate policy, unchanged row count and original codes, BMI/health-day summaries.
3. `eda/observations.md`, charts and CSVs: actual table fraction and target-group comparisons.
4. `hypotheses.csv`: 25 planned tests, execution status, raw/BH p-values, effect direction, interval methods, sparse-cell notes.
5. `manifest.json`: this exists only after successful completion; check source provenance and package versions.

The test suite reports upstream Seaborn/Matplotlib deprecation warnings; they do not fail the tests or suppress checks. CI is configured to repeat verification on pushes/PRs, but a remote CI result is not claimed here. No work on roadmap stages 5–8 has been included.

# Source and research boundaries

Use `diabetes_binary_health_indicators_BRFSS2015.csv`, with 253,680 records and 22 columns. Do not substitute the balanced 70,692-row sample or the three-class target. The binary positive class combines self-reported prediabetes and diabetes; it is not lab-confirmed diabetes and cannot measure future incident disease.

- Kaggle, Alex Teboul: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- UCI mirror: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- CDC annual survey files: https://www.cdc.gov/brfss/annual_data/annual_data.htm

The supplied roadmap identifies the derived file as BRFSS 2015. UCI's landing-page description references 2014. Preserve that discrepancy in citations rather than treating the landing page as verification of the year. The pipeline records the exact input SHA-256 and selected member filename. Check the upstream derivation before making claims about the original survey year or exclusions.

This derived table has no survey weights, strata, or clusters. Unweighted prevalence describes this table, not population prevalence. Tests assume independent observations and cannot reproduce complex-survey inference. Associations are exploratory, unadjusted, and noncausal. Ordinal codes are ordered categories, not exact ages, incomes, or equal measurement intervals. NHANES lab-based labels and newer raw BRFSS releases are separate datasets and are outside stages 1–4.

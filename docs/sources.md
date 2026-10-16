# Source and research boundaries

Use `diabetes_binary_health_indicators_BRFSS2015.csv`, with 253,680 records and 22 columns. Do not substitute the balanced 70,692-row sample or the three-class target. The requested combined screening endpoint combines self-reported prediabetes and diabetes; it is not lab-confirmed diabetes and cannot measure future incident disease.

- Kaggle, Alex Teboul: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- UCI mirror: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- CDC annual survey files: https://www.cdc.gov/brfss/annual_data/annual_data.htm

The supplied roadmap identifies the derived file as BRFSS 2015. UCI's landing-page description references 2014. Preserve that discrepancy in citations rather than treating the landing page as verification of the year. The pipeline records the exact input SHA-256 and selected member filename. Check the upstream derivation before making claims about the original survey year or exclusions.

This derived table has no survey weights, strata, or clusters. Unweighted prevalence describes this table, not population prevalence. Tests assume independent observations and cannot reproduce complex-survey inference. Associations are exploratory, unadjusted, and noncausal. Ordinal codes are ordered categories, not exact ages, incomes, or equal measurement intervals. NHANES lab-based labels and newer raw BRFSS releases are separate datasets and are outside stages 1–4.

## Uploaded three-class source

The supplied `diabetes_012_health_indicators_BRFSS2015.csv.zip` contains one three-class CSV: 213,703 no-diabetes records, 4,631 prediabetes records, and 35,346 diabetes records. For the stated combined screening endpoint, the preparation command explicitly maps 0→0, 1→1 and 2→1. The resulting 39,977 positives (15.76%) are derived from the upload; they are not an independently downloaded upstream binary file. Input hashes identify the exact upload but do not independently authenticate the original CDC derivation. The supplied archive is the source for this run; external download restrictions no longer block its analysis.

Do not infer the upstream binary file's label mapping from its filename or shape. Verify its source codebook before comparing endpoints. The archived overview's approximate 17% prevalence was not a measured result and is superseded by the uploaded-data report.

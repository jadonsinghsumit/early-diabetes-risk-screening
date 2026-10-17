# Exploratory observations

Analysis cohort: 253,680 rows; positive fraction: 15.76%.
Binary endpoint: combined.
This is the observed table fraction, not a survey-weighted population estimate.

- BMI: median 27 in group 0; 31 in group 1.
- Age: median 8 in group 0; 10 in group 1.
- MentHlth: median 0 in group 0; 0 in group 1.
- PhysHlth: median 0 in group 0; 1 in group 1.
- GenHlth: median 2 in group 0; 3 in group 1.

Age and general health are ordinal codes. Correlations depend on coding; use Spearman as an ordered-rank complement to Pearson. Group differences can reflect age, healthcare access, and other confounding. No causal or diagnostic conclusions follow. Later modeling must keep hold-out data isolated; this full-cohort exploration is descriptive and must not be used to tune test-set decisions.

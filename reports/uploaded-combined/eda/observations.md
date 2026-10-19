# Exploratory observations

Analysis cohort: 253,680 rows; positive fraction: 15.76%.
Binary endpoint: combined.
This is the observed table fraction, not a survey-weighted population estimate.

- BMI: median 27 in group 0; 31 in group 1.
- Age: median 8 in group 0; 10 in group 1.
- MentHlth: median 0 in group 0; 0 in group 1.
- PhysHlth: median 0 in group 0; 1 in group 1.
- GenHlth: median 2 in group 0; 3 in group 1.

Strongest Spearman correlations with the outcome (descriptive; coded variables): GenHlth +0.296; HighBP +0.270; BMI +0.235; DiffWalk +0.222; HighChol +0.210.

Largest unadjusted prevalence ratios among binary factors (code 1 vs code 0; not odds ratios): CholCheck 5.08x (16.25% vs 3.20%); HighBP 3.76x (27.12% vs 7.22%); DiffWalk 2.79x (33.76% vs 12.12%); HighChol 2.69x (24.69% vs 9.18%); HeartDiseaseorAttack 2.61x (35.75% vs 13.68%).

Age and general health are ordinal codes. Correlations depend on coding; use Spearman as an ordered-rank complement to Pearson. Group differences can reflect age, healthcare access, and other confounding. No causal or diagnostic conclusions follow. Later modeling must keep hold-out data isolated; this full-cohort exploration is descriptive and must not be used to tune test-set decisions.

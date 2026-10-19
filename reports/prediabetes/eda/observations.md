# Exploratory observations

Analysis cohort: 218,334 rows; positive fraction: 2.12%.
Binary endpoint: prediabetes.
This is the observed table fraction, not a survey-weighted population estimate.

- BMI: median 27 in group 0; 30 in group 1.
- Age: median 8 in group 0; 9 in group 1.
- MentHlth: median 0 in group 0; 0 in group 1.
- PhysHlth: median 0 in group 0; 0 in group 1.
- GenHlth: median 2 in group 0; 3 in group 1.

Strongest Spearman correlations with the outcome (descriptive; coded variables): GenHlth +0.084; HighBP +0.077; BMI +0.073; HighChol +0.072; DiffWalk +0.061.

Largest unadjusted prevalence ratios among binary factors (code 1 vs code 0; not odds ratios): CholCheck 3.25x (2.19% vs 0.67%); HighBP 2.81x (3.54% vs 1.26%); HighChol 2.62x (3.43% vs 1.31%); DiffWalk 2.45x (4.35% vs 1.77%); HeartDiseaseorAttack 2.11x (4.15% vs 1.96%).

Age and general health are ordinal codes. Correlations depend on coding; use Spearman as an ordered-rank complement to Pearson. Group differences can reflect age, healthcare access, and other confounding. No causal or diagnostic conclusions follow. Later modeling must keep hold-out data isolated; this full-cohort exploration is descriptive and must not be used to tune test-set decisions.

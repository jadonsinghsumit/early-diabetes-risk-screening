# Stage 2 decisions

Default: retain duplicate rows. There is no respondent identifier, and identical coded profiles need not represent accidental duplicates. `--duplicates drop` is an explicit sensitivity option, not the primary analysis. The report records before/after counts. Either way, repeated profiles and survey design limit the independence assumptions of later tests.

No balancing, winsorizing, scaling, or automatic imputation occurs in stages 1–4. Missing, nonnumeric, fractional or out-of-range codes stop cleaning with a concrete error. Valid source codes are stored as integers. BMI's 12–98 source bounds identify unexpected inputs, not universal clinical plausibility thresholds. Age, education, income, and general health retain their ordinal coding. Health-day variables stay between 0 and 30. Quantiles and distributions expose unusual but valid observations rather than deleting them.

The data dictionary is exported with each run. A failed cleaning attempt retains the stage 1 quality report for diagnosis, but does not produce a cleaned analysis table.

The primary prediabetes cohort intentionally excludes original target code 2 (existing diabetes) before feature cleaning. Provenance records 35,346 excluded respondents from the 253,680-row source. This scientific eligibility rule is distinct from missing-value or duplicate handling; all 218,334 eligible respondents are retained by default.

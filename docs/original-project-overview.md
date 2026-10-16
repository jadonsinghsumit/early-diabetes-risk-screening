> Archived pre-migration overview. This preserves existing project context; current README, source provenance, and measured results supersede its initial target/prevalence and implementation assumptions.

# Early Population Diabetes Risk Screening via CDC Survey Indicators

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

## Project Overview

This repository contains the end-to-end machine learning pipeline and statistical analysis for **AAI-500 Final Project (Group 3)**. 

Undiagnosed diabetes and prediabetes present a major public health challenge, leading to severe long-term cardiovascular and renal complications alongside escalating healthcare expenditures. Leveraging the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset**, this project develops, validates, and evaluates predictive screening models to identify high-risk individuals using self-reported health and lifestyle indicators.

Our primary objective is to optimize predictive models for **Sensitivity/Recall** and **PR-AUC**, ensuring high-risk individuals are effectively flagged for early preventative clinical evaluation.

---

## Key Features & Dataset Specifications

- **Data Source:** CDC BRFSS 2015 (`diabetes_binary_health_indicators_BRFSS2015.csv`)
- **Dataset Scale:** 253,680 rows × 22 columns (~18.3 MB CSV)
- **Target Variable:** `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes/Diabetes)
- **Class Imbalance:** ~17% positive class prevalence (imbalanced medical screening scenario)
- **Primary Feature Categories:**
  - *Clinical Metrics:* `HighBP`, `HighChol`, `BMI`, `Stroke`, `HeartDiseaseorAttack`
  - *Lifestyle Indicators:* `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`
  - *Demographics & Socioeconomic:* `Age`, `Sex`, `Education`, `Income`, `GenHlth`

---

## 🛠️ Technical Pipeline Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     TOP LAYER: DATA PREPARATION & EDA                                  │
├───────────────────────────────────────┬───────────────────────────────────┬────────────────────────────┤
│ [ PHASE 1: INGESTION & QC ]           │ [ PHASE 2: PREPROCESSING ]        │ [ PHASE 3: EDA ]           │
│ • CDC BRFSS 2015 (253,680 x 22)       │ • Null audit & deduplication      │ • Class balance (~17%)     │
│ • Schema & data type validation       │ • Ordinal survey scaling          │ • Distribution plots       │
│ • Memory check (~38 MB CSV)           │ • BMI continuous skewness check   │ • Correlation heatmaps     │
│ 🛠️ Python 3.10+, Pandas, NumPy        │ 🛠️ Pandas, NumPy, Scikit-Learn    │ 🛠️ Matplotlib, Seaborn     │
└───────────────────────────────────────┴───────────────────────────────────┴─────────────┬──────────────┘
                                                                                          │
                                                                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     BOTTOM LAYER: STATS, ML & EVALUATION                               │
├───────────────────────────────────────┬───────────────────────────────────┬────────────────────────────┤
│ [ PHASE 4: HYPOTHESIS TESTING ]       │ [ PHASE 5: MODEL TRAINING ]       │ [ PHASE 6: EVALUATION ]    │
│ • Chi-Square (χ²) categorical tests   │ • Stratified 80/20 train/test     │ • Optimize Recall / PR-AUC │
│ • Odds Ratios (OR) calculation        │ • Baseline Logistic Regression    │ • Threshold shift (0.35)   │
│ • Statsmodels inference & p-values    │ • XGBoost & Random Forest         │ • Final matrix validation  │
│ 🛠️ SciPy, Statsmodels                 │ 🛠️ Scikit-Learn, XGBoost          │ 🛠️ Scikit-Learn Metrics    │
└───────────────────────────────────────┴───────────────────────────────────┴────────────────────────────┘


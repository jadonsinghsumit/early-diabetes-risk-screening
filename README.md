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

## Technical Pipeline Architecture

```mermaid
flowchart TD
    A["<b>PHASE 1: Data Ingestion & Quality Control</b><br/>• CDC BRFSS 2015 Dataset (253,680 rows × 22 cols)<br/>• Schema validation & memory optimization (~38 MB)<br/>• <i>Tech Stack: Python 3.10+, Pandas, NumPy</i>"] 
    --> B["<b>PHASE 2: Data Preprocessing & Cleaning</b><br/>• Missing value audit & duplicate checks<br/>• Variable encoding & ordinal survey scaling<br/>• Continuous feature inspection & BMI scaling<br/>• <i>Tech Stack: Pandas, NumPy, Scikit-Learn</i>"]

    B --> C["<b>PHASE 3: Exploratory Data Analysis (EDA)</b><br/>• Verification of ~17% positive class prevalence<br/>• Distribution histograms, KDE plots & heatmaps<br/>• <i>Tech Stack: Matplotlib, Seaborn</i>"]

    C --> D["<b>PHASE 4: Hypothesis & Statistical Testing</b><br/>• Chi-Square (χ²) tests for categorical factors<br/>• Odds Ratios (OR) for clinical risk factors<br/>• Logistic Regression p-values & statsmodels<br/>• <i>Tech Stack: SciPy (scipy.stats), Statsmodels</i>"]

    D --> E["<b>PHASE 5: Model Training & Threshold Optimization</b><br/>• Stratified 80/20 Train/Test Split<br/>• Logistic Regression, Random Forest & XGBoost<br/>• Imbalance handling (scale_pos_weight / class_weight)<br/>• Threshold adjustment (0.50 → 0.35) for max Sensitivity<br/>• <i>Tech Stack: Scikit-Learn, XGBoost</i>"]

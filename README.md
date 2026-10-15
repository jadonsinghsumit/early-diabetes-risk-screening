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

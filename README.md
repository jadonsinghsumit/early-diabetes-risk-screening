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

## 🛠️ Technical Pipeline Architecture

> **Pipeline Overview:** Left-to-right data ingestion, statistical testing, advanced machine learning, and decision threshold optimization.

---

### 🔄 End-to-End Workflow Diagram

```mermaid
flowchart LR
    %% Row 1: Left to Right
    subgraph Row1 [" "]
        direction LR
        P1["<b>1. Ingestion & QC</b><br/>• CDC BRFSS 2015<br/>• 253,680 × 22 rows<br/>• Schema validation<br/><i>Pandas • NumPy</i>"] 
        --> 
        P2["<b>2. Preprocessing</b><br/>• Missing value audit<br/>• Ordinal feature scaling<br/>• BMI skewness scaling<br/><i>Scikit-Learn</i>"]
        --> 
        P3["<b>3. Exploratory Data Analysis</b><br/>• Class prevalence (~17%)<br/>• Feature distributions<br/>• Correlation heatmaps<br/><i>Seaborn • Matplotlib</i>"]
    end

    %% Transition Row 1 to Row 2
    P3 --> P4

    %% Row 2: Right to Left / Bottom Layer
    subgraph Row2 [" "]
        direction LR
        P4["<b>4. Hypothesis Testing</b><br/>• Chi-Square (χ²) tests<br/>• Odds Ratios (OR)<br/>• Baseline Logistic p-values<br/><i>SciPy • Statsmodels</i>"]
        --> 
        P5["<b>5. Modeling & Tuning</b><br/>• Stratified 80/20 split<br/>• Random Forest & XGBoost<br/>• Threshold shift (0.50 → 0.35)<br/><i>XGBoost • Scikit-Learn</i>"]
    end

    %% Styling
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px,font-size:11px;
    classDef header fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,font-size:11px;
    class P1,P2,P3,P4,P5 header;

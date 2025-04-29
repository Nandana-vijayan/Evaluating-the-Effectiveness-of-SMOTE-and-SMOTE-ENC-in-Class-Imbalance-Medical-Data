# Evaluating-the-Effectiveness-of-SMOTE-ENC-in-Class-Imbalance-Medical-Data


## 📋 Project Overview

This project investigates how Synthetic Minority Over-sampling Techniques (SMOTE) and its extended version SMOTE-ENC impact machine learning model performance on **imbalanced medical datasets**. Specifically, it focuses on classifying Chronic Obstructive Pulmonary Disease (COPD) cases using the **ExaSens** dataset, which contains both **nominal** and **continuous** features.

The findings show that **SMOTE-ENC**, an enhancement over traditional SMOTE, significantly improves the classification performance for decision trees, a critical insight for medical diagnostics with imbalanced mixed-type data.

---

## 🧠 Problem Statement

Medical datasets often exhibit **severe class imbalance**, leading to biased machine learning predictions — critical minority conditions like COPD can be overlooked. Moreover, when datasets contain **both continuous and nominal features**, standard oversampling methods underperform.

This project aims to:
- Evaluate SMOTE and SMOTE-ENC for balancing mixed-type imbalanced datasets.
- Analyze classification performance across Decision Trees, Random Forests, and Support Vector Machines (SVM).

---

## 🎯 Research Question

> How does SMOTE and SMOTE-ENC perform in balancing mixed nominal and continuous feature datasets for COPD classification using the ExaSens dataset?

---

## 🎯 Objectives
- Conduct a **literature review** on data imbalance challenges.
- **Preprocess** the ExaSens dataset (handle missing values, encode features).
- Apply **SMOTE** and **SMOTE-ENC** separately.
- Train and evaluate **Decision Tree**, **Random Forest**, and **SVM** models.
- Compare model performance using metrics: Accuracy, Precision, Recall, F1-Score, AUC-PR, MCC, and Balanced Accuracy.

---

## 📂 Dataset

- **Name**: ExaSens Dataset
- **Source**: [IEEE DataPort](https://ieee-dataport.org/open-access/exasens-novel-dataset-classification-saliva-samples-copd-patients)
- **Contents**:
  - Demographics: Age, Gender, Smoking Status
  - Saliva Permittivity Measurements (Real and Imaginary Parts)
- **Classes**:
  - COPD
  - Asthma
  - Infected (Respiratory Infection)
  - Healthy Controls (HC)

---

## 🏗️ Methodology

1. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables.
   - Feature scaling (continuous variables).

2. **Data Balancing**:
   - Apply **SMOTE** on continuous features.
   - Apply **SMOTE-ENC** on mixed nominal + continuous features.

3. **Model Training and Evaluation**:
   - Algorithms used: **Decision Tree**, **Random Forest**, **Support Vector Machine**.
   - Evaluation Metrics: Accuracy, F1-Score, Recall, Precision, AUC-PR, MCC, Balanced Accuracy.

4. **Result Analysis**:
   - Compare performance before and after balancing.
   - Identify the best oversampling technique + classifier combination.

---

## 📊 Key Findings

| Model | Oversampling | F1-Score | Accuracy | AUC-PR | MCC |
|------|--------------|---------|----------|--------|-----|
| Decision Tree | SMOTE | 0.64 | 70% | 0.56 | 0.58 |
| Decision Tree | SMOTE-ENC | **0.72** | **75%** | 0.61 | **0.65** |
| Random Forest | Both | 0.64 | 70% | 0.84 | 0.58 |
| SVM | Both | 0.62 | 65% | ~0.69-0.75 | 0.53 |

✅ **Best model**: Decision Tree with SMOTE-ENC  
✅ **SMOTE-ENC** improves minority class predictions significantly  
✅ Random Forest and SVM models remained relatively stable

---

## 🛠️ Technologies and Libraries Used

- Python 3.10
- Scikit-learn
- imbalanced-learn
- Pandas
- Matplotlib
- Seaborn

---

## 📜 Installation and Setup

```bash
# Clone this repository
git clone https://github.com/Nandana-vijayan/Evaluating-the-Effectiveness-of-SMOTE-and-SMOTE-ENC-in-Class-Imbalance-Medical-Data.git

# Move into the project directory
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
```

> Requirements include: `scikit-learn`, `imbalanced-learn`, `pandas`, `matplotlib`, `seaborn`.

---

## 🚀 How to Run

1. Download the **ExaSens** dataset from [IEEE DataPort](https://ieee-dataport.org/open-access/exasens-novel-dataset-classification-saliva-samples-copd-patients).
2. Preprocess the data by running the scripts provided under 'Saliva_samples of COPD patients.ipynb'
3. Use the provided python files to:
   - Train models with SMOTE.('2-SMOTE balanced Exasens.ipynb')
   - Train models with SMOTE-ENC.
   - Then Evaluate and compare performance.
4. Optionally, run **manual prediction tests** using trained Decision Tree models.



## ✨ Future Work

- Extend evaluations with **XGBoost**, **LightGBM**, and **deep learning** methods.
- Explore **hybrid oversampling** strategies.
- Apply **feature selection** methods to improve model performance.
- Collect **larger medical datasets** for better model generalization.

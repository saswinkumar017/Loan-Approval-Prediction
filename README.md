# 🏦 Loan Approval Prediction

<p align="center">
  <strong>Intern ID:</strong> CITS5452
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-1.3%2B-150458?logo=pandas">
  <img src="https://img.shields.io/badge/Status-Completed-success">
</p>

A **binary classification** project that automates loan approval decisions using **Logistic Regression**. The model predicts whether a loan applicant will repay (`Approved`) or default (`Rejected`) based on financial and demographic features.

---

## 📋 Table of Contents

- [🏦 Loan Approval Prediction](#-loan-approval-prediction)
  - [📋 Table of Contents](#-table-of-contents)
  - [📊 Model Performance](#-model-performance)
  - [🔄 How It Works](#-how-it-works)
    - [End-to-End Flow for a New Applicant](#end-to-end-flow-for-a-new-applicant)
  - [⚡ Quick Start](#-quick-start)
    - [Prerequisites](#prerequisites)
    - [Run the Full Pipeline](#run-the-full-pipeline)
    - [Expected Output](#expected-output)
  - [📁 Project Structure](#-project-structure)
  - [🛠 Tech Stack](#-tech-stack)
  - [📊 Dataset Overview](#-dataset-overview)
  - [📈 Evaluation Plots](#-evaluation-plots)

---

## 📊 Model Performance

| Metric | Score | Meaning |
|--------|-------|---------|
| **Accuracy** | **80.5%** | Overall correct predictions |
| **Precision** | **79.8%** | Of approved loans, 79.8% were correct |
| **Recall** | **93.8%** | Of good applicants, 93.8% were caught |
| **F1-Score** | **86.2%** | Harmonic mean of precision & recall |
| **ROC-AUC** | **82.3%** | 82.3% chance model ranks a good applicant above a bad one |

> ⚡ **Key Insight:** Credit History is the strongest predictor. Applicants with good credit history are far more likely to be approved.

---

## 🔄 How It Works

```
  loan_data.csv ──► Clean ──► Feature Engineer ──► Encode ──► Scale ──► Train ──► Evaluate ──► Save Model
       (614 rows)      │           │                   │          │         │           │             │
                       ▼           ▼                   ▼          ▼         ▼           ▼             ▼
                  fill NaN    log1p income      LabelEncoder   Standard   Logistic    Conf Matrix   loan_model.pkl
                  median/mode  EMI, ratio       Gender→0/1     Scaler     Regression  ROC-AUC       scaler.pkl
```

### End-to-End Flow for a New Applicant

```
Raw Input ──► Encode text ──► Engineer features ──► Scale ──► sigmoid(w·x + b) ──► Probability ──► Decision
 {dict}         Male→1        Log_TotalIncome       z-score    σ(z) = 1/(1+e⁻ᶻ)     p = 0.92       ✅ Approved
```

---

## ⚡ Quick Start

### Prerequisites

```powershell
pip install pandas numpy matplotlib scikit-learn joblib
```

### Run the Full Pipeline

```powershell
# Step 1 — Create synthetic loan dataset
python src/generate_data.py

# Step 2 — Train and evaluate the model
python src/train.py

# Step 3 — Predict on a new applicant
python src/predict.py
```

### Expected Output

```
Dataset created: 614 rows, 12 columns
...
Test Accuracy:  0.8049 | ROC-AUC: 0.8233
...
Prediction: Approved
Approval Probability: 92.31%
```

---

## 📁 Project Structure

```
loan-approval/
│
├── 📂 data/                        # Dataset and evaluation artifacts
│   ├── loan_data.csv               # 614 applicants, 12 columns
│   ├── confusion_matrix.png        # Visual confusion matrix
│   └── roc_curve.png               # ROC curve with AUC score
│
├── 📂 models/                      # Serialized trained objects
│   ├── loan_model.pkl              # Trained Logistic Regression (weights + bias)
│   ├── scaler.pkl                  # StandardScaler (mean + std per feature)
│   ├── label_encoders.pkl          # Category → integer mappings
│   └── feature_columns.pkl         # Feature names & order for inference
│
├── 📂 src/                         # Source code
│   ├── generate_data.py            # Creates realistic synthetic dataset
│   ├── train.py                    # Full ML pipeline (clean → train → save)
│   └── predict.py                  # Load artifacts & predict on new data
│
├── 📂 notebooks/                   # Jupyter notebooks (for EDA)
│
├── DOCUMENTATION.md                # 18-section complete technical reference
└── README.md                       # This file
```

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3** | Core programming language |
| **Pandas** | Data loading, cleaning, manipulation |
| **NumPy** | Numerical operations, log transforms |
| **Matplotlib** | Confusion matrix & ROC curve plotting |
| **Scikit-Learn** | Train/test split, LabelEncoder, StandardScaler, LogisticRegression, metrics |
| **Joblib** | Serialize/deserialize model artifacts (`.pkl` files) |

---

## 📊 Dataset Overview

| Feature | Type | Values/Range | Role |
|---------|------|-------------|------|
| `Gender` | Categorical | Male, Female | Demographic |
| `Married` | Categorical | Yes, No | Demographic |
| `Dependents` | Categorical | 0, 1, 2, 3+ | Demographic |
| `Education` | Categorical | Graduate, Not Graduate | Socioeconomic |
| `Self_Employed` | Categorical | Yes, No | Employment |
| `ApplicantIncome` | Numerical | 1,500 – 50,000 | Financial |
| `CoapplicantIncome` | Numerical | 0 – 30,000 | Financial |
| `LoanAmount` | Numerical | 9 – 700 (thousands) | Loan |
| `Loan_Amount_Term` | Numerical | 12 – 480 (months) | Loan |
| `Credit_History` | Numerical | 0 or 1 | **Most important** |
| `Property_Area` | Categorical | Urban, Semiurban, Rural | Geographic |
| `Loan_Status` | **Target** | 0 (Rejected), 1 (Approved) | **Prediction** |

**Class Distribution:** ~65% Approved, ~35% Rejected (moderately balanced)

---

## 📈 Evaluation Plots

The training script generates two plots in the `data/` folder:

- **confusion_matrix.png** — Shows TP=75, TN=24, FP=19, FN=5
- **roc_curve.png** — Shows ROC curve with AUC = 0.823

---




<p align="center">
  <sub>Intern ID: CITS5452</sub>
</p>

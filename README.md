# Loan Approval Prediction

**Intern ID:** CITS5452

A binary classification project that predicts loan approval status (Approved/Rejected) using Logistic Regression.

| Metric | Score |
|--------|-------|
| Accuracy | 80.5% |
| Precision | 79.8% |
| Recall | 93.8% |
| F1-Score | 86.2% |
| ROC-AUC | 82.3% |

## Quick Start

```powershell
pip install pandas numpy matplotlib scikit-learn joblib
python src/generate_data.py   # create dataset
python src/train.py            # train model
python src/predict.py          # test prediction
```

## Project Structure

```
├── data/          ← dataset + evaluation plots
├── models/        ← trained model + scaler + encoders
├── src/           ← source code (generate, train, predict)
└── DOCUMENTATION.md  ← complete technical reference
```

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Scikit-Learn · Joblib

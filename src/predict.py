import pandas as pd
import numpy as np
import joblib
import sys

MODEL_PATH = 'models/loan_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODER_PATH = 'models/label_encoders.pkl'
FEATURES_PATH = 'models/feature_columns.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

def preprocess_single(row: dict) -> pd.DataFrame:
    raw = pd.DataFrame([row])

    for col in ['Gender', 'Married', 'Self_Employed', 'Education']:
        raw[col] = label_encoders[col].transform(raw[col])
    for col in ['Dependents', 'Property_Area']:
        raw[col] = label_encoders[col].transform(raw[col].astype(str))
    raw['Credit_History'] = float(row.get('Credit_History', 1))

    total_income = row['ApplicantIncome'] + row['CoapplicantIncome']
    raw['Log_TotalIncome'] = np.log1p(total_income)
    raw['LoanAmount_Log'] = np.log1p(row['LoanAmount'])
    raw['EMI'] = row['LoanAmount'] / (row['Loan_Amount_Term'] / 12)

    dep = str(row.get('Dependents', '0'))
    dep_num = 3 if dep == '3+' else int(dep)
    raw['Income_per_Person'] = total_income / (dep_num + 1)

    raw = raw[feature_columns]
    return raw

def predict_applicant(row: dict):
    processed = preprocess_single(row)
    scaled = scaler.transform(processed)
    prob = model.predict_proba(scaled)[0, 1]
    pred = model.predict(scaled)[0]

    return {
        'prediction': 'Approved' if pred == 1 else 'Rejected',
        'probability': round(prob, 4),
        'confidence': round(max(prob, 1 - prob), 4)
    }

if __name__ == '__main__':
    sample = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '2',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 'Urban',
    }

    result = predict_applicant(sample)
    print(f"Prediction: {result['prediction']}")
    print(f"Approval Probability: {result['probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")

import numpy as np
import pandas as pd

np.random.seed(42)
n = 614

gender = np.random.choice(['Male', 'Female'], n, p=[0.8, 0.2])
married = np.random.choice(['Yes', 'No'], n, p=[0.65, 0.35])
dependents = np.random.choice(['0', '1', '2', '3+'], n, p=[0.55, 0.2, 0.15, 0.1])
education = np.random.choice(['Graduate', 'Not Graduate'], n, p=[0.75, 0.25])
self_employed = np.random.choice(['Yes', 'No'], n, p=[0.15, 0.85])

applicant_income = np.random.lognormal(mean=7.5, sigma=0.6, size=n).astype(int)
applicant_income = np.clip(applicant_income, 1500, 50000)

coapplicant_income = np.where(
    np.random.random(n) < 0.6,
    np.random.lognormal(mean=7.0, sigma=0.7, size=n).astype(int),
    0
)
coapplicant_income = np.clip(coapplicant_income, 0, 30000)

loan_amount = np.where(
    np.random.random(n) < 0.9,
    np.random.lognormal(mean=4.5, sigma=0.5, size=n),
    np.nan
)
loan_amount = np.clip(loan_amount, 9, 700)

loan_amount_term = np.where(
    np.random.random(n) < 0.05,
    np.nan,
    np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360, 480], n)
)

credit_history = np.where(
    np.random.random(n) < 0.05,
    np.nan,
    np.random.choice([0, 1], n, p=[0.15, 0.85])
)

property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], n, p=[0.35, 0.35, 0.30])

income_high = (applicant_income + coapplicant_income) > 5000
good_credit = credit_history == 1
high_loan = loan_amount > 200
short_term = loan_amount_term < 180
semiurban = property_area == 'Semiurban'
graduate = education == 'Graduate'

logit = (
    2.0 * good_credit
    - 1.2 * (education == 'Not Graduate')
    + 0.8 * semiurban
    + 0.5 * income_high
    - 0.6 * high_loan
    + 0.4 * (property_area == 'Urban')
    - 0.5 * (self_employed == 'Yes')
    + 0.3 * (married == 'Yes')
    + np.random.normal(0, 0.7, n)
)

prob = 1 / (1 + np.exp(-logit))

offset = np.percentile(prob, 35)
loan_status = (prob > offset).astype(int)

df = pd.DataFrame({
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': np.round(loan_amount, 1),
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': property_area,
    'Loan_Status': loan_status,
})

df.to_csv('data/loan_data.csv', index=False)
print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Loan_Status distribution:\n{df['Loan_Status'].value_counts()}")
print(f"Missing values:\n{df.isnull().sum()}")

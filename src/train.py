import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, roc_auc_score
)
import joblib
import os

# -------------------------------------------------------------------
# 1. Load dataset
# -------------------------------------------------------------------
DATA_PATH = 'data/loan_data.csv'
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape}")
print(f"Target distribution:\n{df['Loan_Status'].value_counts()}\n")

# -------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# -------------------------------------------------------------------
print("--- Basic Info ---")
df.info()
print(f"\n--- Null Counts ---\n{df.isnull().sum()}\n")
print(f"--- Describe ---\n{df.describe(include='all')}\n")

# -------------------------------------------------------------------
# 3. Data Cleaning — handle missing values
# -------------------------------------------------------------------
numeric_cols = ['LoanAmount', 'Loan_Amount_Term']
categorical_cols_for_mode = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols_for_mode:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Missing after cleaning:\n{df.isnull().sum()}\n")

# -------------------------------------------------------------------
# 4. Feature Engineering
# -------------------------------------------------------------------
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Log_TotalIncome'] = np.log1p(df['TotalIncome'])

df['LoanAmount_Log'] = np.log1p(df['LoanAmount'])

df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] / 12)
df['Income_per_Person'] = df['TotalIncome'] / (df['Dependents'].replace({'3+': 3}).astype(int) + 1)

df.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'TotalIncome'], axis=1, inplace=True)

# -------------------------------------------------------------------
# 5. Encode categorical variables
# -------------------------------------------------------------------
binary_cols = ['Gender', 'Married', 'Self_Employed', 'Education']
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

multi_cols = ['Dependents', 'Property_Area']
for col in multi_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print(f"Columns after encoding:\n{df.dtypes}\n")
print(f"Sample:\n{df.head()}\n")

# -------------------------------------------------------------------
# 6. Split features and target
# -------------------------------------------------------------------
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# -------------------------------------------------------------------
# 7. Feature scaling
# -------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Scaler mean (first 5): {scaler.mean_[:5]}")
print(f"Scaler std  (first 5): {np.sqrt(scaler.var_)[:5]}")

# -------------------------------------------------------------------
# 8. Train Logistic Regression
# -------------------------------------------------------------------
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print(f"\nModel coefficients:\n{model.coef_}")
print(f"Model intercept: {model.intercept_}")

# -------------------------------------------------------------------
# 9. Evaluate
# -------------------------------------------------------------------
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n===== TRAIN METRICS =====")
print(f"Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Recall:    {recall_score(y_train, y_train_pred):.4f}")
print(f"F1-Score:  {f1_score(y_train, y_train_pred):.4f}")

print("\n===== TEST METRICS =====")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_prob):.4f}")

print(f"\nClassification Report (Test):\n{classification_report(y_test, y_test_pred)}")

# -------------------------------------------------------------------
# 10. Confusion Matrix
# -------------------------------------------------------------------
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix — Loan Approval')
plt.savefig('data/confusion_matrix.png', dpi=100)
print(f"\nConfusion Matrix:\n{cm}")

# -------------------------------------------------------------------
# 11. ROC Curve
# -------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Loan Approval')
plt.legend()
plt.savefig('data/roc_curve.png', dpi=100)

# -------------------------------------------------------------------
# 12. Save artifacts
# -------------------------------------------------------------------
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/loan_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(list(X.columns), 'models/feature_columns.pkl')

print("\nAll artifacts saved to models/")
print("Project complete.")

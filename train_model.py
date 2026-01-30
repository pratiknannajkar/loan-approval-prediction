import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "bin", "model.pkl")

print("📂 Data path:", DATA_PATH)
print("📂 Model will be saved to:", MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print("✅ RandomForest model trained and saved successfully")

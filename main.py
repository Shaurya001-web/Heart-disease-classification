import pandas as pd
import matplotlib.pyplot as plt
from pycaret.classification import setup, create_model, predict_model

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("heart_disease_uci.csv")

print("Columns:", df.columns)
print("\nData types before processing:")
print(df.dtypes)

# Rename target column
if 'num' in df.columns:
    df.rename(columns={'num': 'target'}, inplace=True)

# Drop unnecessary columns
df = df.drop(columns=['id', 'dataset'], errors='ignore')

# =========================
# 2. Encode Categorical Data
# =========================
print("\nString columns found:")
string_cols = df.select_dtypes(include=['object']).columns
print(string_cols.tolist())

for col in string_cols:
    if col != 'target':
        unique_vals = df[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)
        print(f"Encoded {col}: {mapping}")

# Convert target to binary
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print("\nData types after processing:")
print(df.dtypes)

# =========================
# 3. 📊 BASIC GRAPHS (ONLY 3)
# =========================

# 🔹 1. Target Distribution
plt.figure()
df['target'].value_counts().plot(kind='bar')
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 🔹 2. Age Distribution
plt.figure()
plt.hist(df['age'])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 🔹 3. Cholesterol vs Target
plt.figure()
plt.scatter(df['chol'], df['target'])
plt.title("Cholesterol vs Disease")
plt.xlabel("Cholesterol")
plt.ylabel("Target")
plt.show()

# =========================
# 4. ML MODEL
# =========================

print("\nStarting model setup...")

clf = setup(
    data=df,
    target='target',
    session_id=42,
    verbose=False,
    html=False
)

model = create_model('lr')

print("Model training completed successfully!")

# =========================
# 5. TESTING / PREDICTION
# =========================

# Take one sample patient
sample_patient = df.drop('target', axis=1).iloc[0:1]

print("\nSample Patient Data:")
print(sample_patient)

# Predict using PyCaret
prediction = predict_model(model, data=sample_patient)

print("\nPrediction Output:")
print(prediction[['prediction_label']])

# Clean result
result = "Heart Disease" if prediction['prediction_label'].values[0] == 1 else "No Heart Disease"
print("\nFinal Result:", result)

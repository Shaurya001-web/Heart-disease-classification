import pandas as pd
from pycaret.classification import setup, create_model

# Load dataset
df = pd.read_csv("heart_disease_uci.csv")

# Check column names (optional)
print("Columns:", df.columns)
print("\nData types before processing:")
print(df.dtypes)

# If target column is named 'num', rename it
if 'num' in df.columns:
    df.rename(columns={'num': 'target'}, inplace=True)

# Drop non-useful columns like 'id' and 'dataset'
df = df.drop(columns=['id', 'dataset'], errors='ignore')

# Identify all object (string) columns and encode them
print("\nString columns found:")
string_cols = df.select_dtypes(include=['object']).columns
print(string_cols.tolist())

# Encode string columns to numeric
for col in string_cols:
    if col != 'target':
        # Create a mapping of unique values to integers
        unique_vals = df[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)
        print(f"Encoded {col}: {mapping}")

print("\nData types after processing:")
print(df.dtypes)

# Setup PyCaret environment
clf = setup(data=df, target='target', session_id=42, verbose=False)

# Train Logistic Regression model
model = create_model('lr')

print("Model training completed successfully!")

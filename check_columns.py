import pandas as pd
import joblib

# Load data and pipeline
df = pd.read_csv('data/raw/telco.csv')
pipeline = joblib.load('models/churn_pipeline.pkl')

# Drop leaked/unnecessary columns
cols_to_drop = ['Customer ID', 'Country', 'State', 'City', 'Zip Code', 
                'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason', 'Customer Status']
remaining_cols = [c for c in df.columns if c not in cols_to_drop]

print(f"Total columns needed: {len(remaining_cols)}\n")
print("Columns needed for prediction:")
for c in remaining_cols:
    print(f"  - {c}")

# Try a test prediction
test_row = df[remaining_cols].iloc[0:1]
try:
    pred = pipeline.predict_proba(test_row)
    print(f"\n✓ Pipeline works! Prediction: {pred[0][1]:.2%}")
except Exception as e:
    print(f"\n✗ Error: {e}")

import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load('models/churn_pipeline.pkl')

# Test case 1: Low risk customer (long tenure, low charge, high satisfaction)
test1 = {
    'Gender': 'Male',
    'Age': 35,
    'Under 30': 'No',
    'Senior Citizen': 'No',
    'Married': 'No',
    'Dependents': 'No',
    'Number of Dependents': 0,
    'Latitude': 40.7128,
    'Longitude': -74.0060,
    'Population': 100000,
    'Quarter': 'Q1',
    'Referred a Friend': 'No',
    'Number of Referrals': 5,
    'Tenure in Months': 60,  # Long tenure
    'Offer': 'None',
    'Phone Service': 'Yes',
    'Avg Monthly Long Distance Charges': 5.0,
    'Multiple Lines': 'No',
    'Internet Service': 'Yes',
    'Internet Type': 'Fiber Optic',
    'Avg Monthly GB Download': 50.0,
    'Online Security': 'Yes',
    'Online Backup': 'Yes',
    'Device Protection Plan': 'Yes',
    'Premium Tech Support': 'Yes',
    'Streaming TV': 'Yes',
    'Streaming Movies': 'Yes',
    'Streaming Music': 'Yes',
    'Unlimited Data': 'Yes',
    'Contract': 'Two year',
    'Paperless Billing': 'Yes',
    'Payment Method': 'Bank transfer',
    'Monthly Charge': 50.0,  # Low charge
    'Total Charges': 3000.0,
    'Total Refunds': 0.0,
    'Total Extra Data Charges': 0.0,
    'Total Long Distance Charges': 50.0,
    'Total Revenue': 3000.0,
    'Satisfaction Score': 5,  # High satisfaction
    'CLTV': 1800.0
}

# Test case 2: Medium risk customer (short tenure, medium charge, medium satisfaction, month-to-month)
test2 = {
    'Gender': 'Female',
    'Age': 40,
    'Under 30': 'No',
    'Senior Citizen': 'No',
    'Married': 'Yes',
    'Dependents': 'No',
    'Number of Dependents': 0,
    'Latitude': 40.7128,
    'Longitude': -74.0060,
    'Population': 100000,
    'Quarter': 'Q1',
    'Referred a Friend': 'No',
    'Number of Referrals': 0,
    'Tenure in Months': 6,  # New customer
    'Offer': 'None',
    'Phone Service': 'Yes',
    'Avg Monthly Long Distance Charges': 5.0,
    'Multiple Lines': 'No',
    'Internet Service': 'Yes',
    'Internet Type': 'Cable',
    'Avg Monthly GB Download': 50.0,
    'Online Security': 'No',
    'Online Backup': 'No',
    'Device Protection Plan': 'No',
    'Premium Tech Support': 'No',
    'Streaming TV': 'No',
    'Streaming Movies': 'No',
    'Streaming Music': 'No',
    'Unlimited Data': 'No',
    'Contract': 'Month-to-month',  # Risky
    'Paperless Billing': 'Yes',
    'Payment Method': 'Electronic check',
    'Monthly Charge': 65.0,  # Medium-high charge
    'Total Charges': 390.0,
    'Total Refunds': 0.0,
    'Total Extra Data Charges': 0.0,
    'Total Long Distance Charges': 50.0,
    'Total Revenue': 390.0,
    'Satisfaction Score': 3,  # Medium satisfaction
    'CLTV': 2340.0
}

# Test case 3: High risk customer (short tenure, high charge, low satisfaction, month-to-month)
test3 = {
    'Gender': 'Male',
    'Age': 50,
    'Under 30': 'No',
    'Senior Citizen': 'Yes',
    'Married': 'No',
    'Dependents': 'No',
    'Number of Dependents': 0,
    'Latitude': 40.7128,
    'Longitude': -74.0060,
    'Population': 100000,
    'Quarter': 'Q1',
    'Referred a Friend': 'No',
    'Number of Referrals': 0,
    'Tenure in Months': 2,  # Very new
    'Offer': 'None',
    'Phone Service': 'Yes',
    'Avg Monthly Long Distance Charges': 5.0,
    'Multiple Lines': 'Yes',
    'Internet Service': 'Yes',
    'Internet Type': 'Fiber Optic',
    'Avg Monthly GB Download': 50.0,
    'Online Security': 'No',
    'Online Backup': 'No',
    'Device Protection Plan': 'No',
    'Premium Tech Support': 'No',
    'Streaming TV': 'No',
    'Streaming Movies': 'No',
    'Streaming Music': 'No',
    'Unlimited Data': 'No',
    'Contract': 'Month-to-month',
    'Paperless Billing': 'No',
    'Payment Method': 'Electronic check',
    'Monthly Charge': 95.0,  # High charge
    'Total Charges': 190.0,
    'Total Refunds': 10.0,
    'Total Extra Data Charges': 50.0,
    'Total Long Distance Charges': 100.0,
    'Total Revenue': 240.0,
    'Satisfaction Score': 2,  # Low satisfaction
    'CLTV': 3420.0
}

df_test1 = pd.DataFrame([test1])
df_test2 = pd.DataFrame([test2])
df_test3 = pd.DataFrame([test3])

print("="*70)
print("PREDICTION TEST RESULTS")
print("="*70)

prob1 = pipeline.predict_proba(df_test1)[0][1]
print(f"\n✓ TEST 1 (LOW RISK EXPECTED)")
print(f"  Profile: 60mo tenure, $50/mo, 5/5 satisfaction, 2yr contract, all services")
print(f"  Prediction: {prob1:.1%} churn probability")
print(f"  Risk Level: {'🟢 LOW' if prob1 < 0.4 else '🟡 MEDIUM' if prob1 < 0.7 else '🔴 HIGH'}")
if prob1 < 0.4:
    print("  ✅ CORRECT - Should be low risk")
else:
    print("  ❌ WRONG - Should be low risk")

prob2 = pipeline.predict_proba(df_test2)[0][1]
print(f"\n✓ TEST 2 (MEDIUM RISK EXPECTED)")
print(f"  Profile: 6mo tenure, $65/mo, 3/5 satisfaction, month-to-month, no services")
print(f"  Prediction: {prob2:.1%} churn probability")
print(f"  Risk Level: {'🟢 LOW' if prob2 < 0.4 else '🟡 MEDIUM' if prob2 < 0.7 else '🔴 HIGH'}")
if 0.4 <= prob2 < 0.7:
    print("  ✅ CORRECT - Should be medium risk")
else:
    print("  ⚠️  Different than expected - check if model is calibrated differently")

prob3 = pipeline.predict_proba(df_test3)[0][1]
print(f"\n✓ TEST 3 (HIGH RISK EXPECTED)")
print(f"  Profile: 2mo tenure, $95/mo, 2/5 satisfaction, month-to-month, no services")
print(f"  Prediction: {prob3:.1%} churn probability")
print(f"  Risk Level: {'🟢 LOW' if prob3 < 0.4 else '🟡 MEDIUM' if prob3 < 0.7 else '🔴 HIGH'}")
if prob3 >= 0.7:
    print("  ✅ CORRECT - Should be high risk")
else:
    print("  ⚠️  Different than expected - check model calibration")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Low risk customer:    {prob1:.1%}")
print(f"Medium risk customer: {prob2:.1%}")
print(f"High risk customer:   {prob3:.1%}")
print("\nAll predictions look reasonable? Check above ✅ or ⚠️")
print("="*70)

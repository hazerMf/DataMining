import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load original (raw) dataset
print("=" * 70)
print("CREATING SCALERS FOR KNN MODELS")
print("=" * 70)

raw_data_path = r'E:\SourceCode\DataMining\dataset\PPG-BP dataset(cardiovascular dataset).csv'
normalized_data_path = r'E:\SourceCode\DataMining\dataset\ppg_bp_normalized_standard_with_categories.csv'

print("\n📂 Loading datasets...")
df_raw = pd.read_csv(raw_data_path)
df_normalized = pd.read_csv(normalized_data_path)

print(f"✅ Raw dataset: {df_raw.shape}")
print(f"✅ Normalized dataset: {df_normalized.shape}")

# Map column names (raw -> normalized)
column_mapping = {
    'sex': 'Sex',
    'age': 'Age', 
    'height': 'Height',
    'weight': 'Weight',
    'ap_hi': 'Systolic_BP',
    'ap_lo': 'Diastolic_BP',
    'heart_rate': 'Heart_Rate'
}

df_raw_mapped = df_raw.rename(columns=column_mapping)

# Encode Sex: Female=0, Male=1 (same as normalized dataset)
df_raw_mapped['Sex'] = df_raw_mapped['Sex'].map({'Female': 0, 'Male': 1})

# Calculate BMI from raw data
df_raw_mapped['BMI'] = df_raw_mapped['Weight'] / ((df_raw_mapped['Height'] / 100) ** 2)

print("\n🔧 Column mapping and BMI calculation completed")

# ========================================================================
# SCALER 1: For Systolic BP Model (includes Diastolic_BP as feature)
# ========================================================================

print("\n" + "=" * 70)
print("SCALER 1: SYSTOLIC BP MODEL")
print("=" * 70)

# Features for SBP model (15 features including Diastolic_BP)
# Drop: Num, subject_ID, Hypertension, Systolic_BP
columns_to_drop_sbp = ['Num', 'subject_ID', 'Hypertension', 'Systolic_BP']
df_sbp_normalized = df_normalized.drop(columns=columns_to_drop_sbp, errors='ignore')

print(f"📊 SBP Model Features: {df_sbp_normalized.shape[1]} columns")
print(f"   Features: {df_sbp_normalized.columns.tolist()}")

# Only numeric features need scaling (7 features)
# Sex is binary (0/1), one-hot encoded features are already 0/1
numeric_features_sbp = ['Age', 'Height', 'Weight', 'Diastolic_BP', 'Heart_Rate', 'BMI']

# Add Systolic_BP to raw data if exists (for other features)
# But we need all features that will be in X_sbp
raw_features_sbp = df_raw_mapped[numeric_features_sbp].copy()

print(f"\n📋 Numeric features to scale: {numeric_features_sbp}")
print(f"   Total: {len(numeric_features_sbp)} features")

# Create and fit scaler on raw data
scaler_sbp = StandardScaler()
scaler_sbp.fit(raw_features_sbp)

print(f"\n✅ Scaler SBP fitted on raw data:")
print(f"   Mean: {scaler_sbp.mean_[:3]}... (first 3)")
print(f"   Std:  {scaler_sbp.scale_[:3]}... (first 3)")

# Verify scaler accuracy
scaled_verification_sbp = scaler_sbp.transform(raw_features_sbp)
normalized_values_sbp = df_normalized[numeric_features_sbp].values

max_diff_sbp = np.abs(scaled_verification_sbp - normalized_values_sbp).max()
print(f"\n🔍 Verification - Max difference: {max_diff_sbp:.2e}")

if max_diff_sbp < 1e-10:
    print("   ✅ PERFECT MATCH!")
elif max_diff_sbp < 1e-5:
    print("   ✅ Excellent match (acceptable precision)")
else:
    print(f"   ⚠️ WARNING: Large difference detected!")

# Save scaler
output_dir = r'E:\SourceCode\DataMining\BE\models\knn'
os.makedirs(output_dir, exist_ok=True)

scaler_sbp_path = os.path.join(output_dir, 'scaler_systolic_bp.pkl')
joblib.dump(scaler_sbp, scaler_sbp_path)
print(f"\n💾 Scaler saved: {scaler_sbp_path}")

# ========================================================================
# SCALER 2: For Diastolic BP Model (includes Systolic_BP as feature)
# ========================================================================

print("\n" + "=" * 70)
print("SCALER 2: DIASTOLIC BP MODEL")
print("=" * 70)

# Features for DBP model (15 features including Systolic_BP)
# Drop: Num, subject_ID, Hypertension, Diastolic_BP
columns_to_drop_dbp = ['Num', 'subject_ID', 'Hypertension', 'Diastolic_BP']
df_dbp_normalized = df_normalized.drop(columns=columns_to_drop_dbp, errors='ignore')

print(f"📊 DBP Model Features: {df_dbp_normalized.shape[1]} columns")
print(f"   Features: {df_dbp_normalized.columns.tolist()}")

# Only numeric features need scaling (7 features, but different from SBP!)
numeric_features_dbp = ['Age', 'Height', 'Weight', 'Systolic_BP', 'Heart_Rate', 'BMI']

raw_features_dbp = df_raw_mapped[numeric_features_dbp].copy()

print(f"\n📋 Numeric features to scale: {numeric_features_dbp}")
print(f"   Total: {len(numeric_features_dbp)} features")

# Create and fit scaler on raw data
scaler_dbp = StandardScaler()
scaler_dbp.fit(raw_features_dbp)

print(f"\n✅ Scaler DBP fitted on raw data:")
print(f"   Mean: {scaler_dbp.mean_[:3]}... (first 3)")
print(f"   Std:  {scaler_dbp.scale_[:3]}... (first 3)")

# Verify scaler accuracy
scaled_verification_dbp = scaler_dbp.transform(raw_features_dbp)
normalized_values_dbp = df_normalized[numeric_features_dbp].values

max_diff_dbp = np.abs(scaled_verification_dbp - normalized_values_dbp).max()
print(f"\n🔍 Verification - Max difference: {max_diff_dbp:.2e}")

if max_diff_dbp < 1e-10:
    print("   ✅ PERFECT MATCH!")
elif max_diff_dbp < 1e-5:
    print("   ✅ Excellent match (acceptable precision)")
else:
    print(f"   ⚠️ WARNING: Large difference detected!")

# Save scaler
scaler_dbp_path = os.path.join(output_dir, 'scaler_diastolic_bp.pkl')
joblib.dump(scaler_dbp, scaler_dbp_path)
print(f"\n💾 Scaler saved: {scaler_dbp_path}")

# ========================================================================
# SUMMARY
# ========================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n✅ Created 2 separate scalers:")
print(f"   1. Systolic BP Scaler:  {len(numeric_features_sbp)} features")
print(f"      - {scaler_sbp_path}")
print(f"      - Features: {numeric_features_sbp}")
print(f"      - Max diff: {max_diff_sbp:.2e}")
print(f"\n   2. Diastolic BP Scaler: {len(numeric_features_dbp)} features") 
print(f"      - {scaler_dbp_path}")
print(f"      - Features: {numeric_features_dbp}")
print(f"      - Max diff: {max_diff_dbp:.2e}")

print("\n📝 NOTE:")
print("   - SBP model includes Diastolic_BP as feature")
print("   - DBP model includes Systolic_BP as feature")
print("   - Cannot share scaler between models (different features!)")
print("   - Sex column NOT scaled (binary 0/1)")
print("   - One-hot encoded columns NOT scaled (already 0/1)")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)

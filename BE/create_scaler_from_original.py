import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


def create_scaler():
    """Tạo scaler từ dataset GỐC"""

    # 1. Load dataset GỐC (chưa normalize)
    original_path = (
        Path(__file__).parent.parent
        / "dataset"
        / "PPG-BP dataset(cardiovascular dataset).csv"
    )
    # Skip first row (header description)
    df_original = pd.read_csv(original_path, skiprows=1)

    print("Dataset GỐC (Original):")
    print(f"Shape: {df_original.shape}")
    print(f"Columns: {df_original.columns.tolist()}\n")
    print("First few rows:")
    print(df_original.head(3))

    # 2. Load dataset ĐÃ NORMALIZE (để verify)
    normalized_path = (
        Path(__file__).parent.parent
        / "dataset"
        / "ppg_bp_normalized_standard_with_categories.csv"
    )
    df_normalized = pd.read_csv(normalized_path)

    print("Dataset NORMALIZED:")
    print(f"Shape: {df_normalized.shape}")
    print(f"Columns: {df_normalized.columns.tolist()}\n")

    # 3. Xác định mapping giữa column names
    # Original dataset columns (with full names)
    original_col_map = {
        "Sex(M/F)": "Sex",
        "Age(year)": "Age",
        "Height(cm)": "Height",
        "Weight(kg)": "Weight",
        "Systolic Blood Pressure(mmHg)": "Systolic_BP",
        "Diastolic Blood Pressure(mmHg)": "Diastolic_BP",
        "Heart Rate(b/m)": "Heart_Rate",
        "BMI(kg/m^2)": "BMI",
    }

    # Rename columns trong original dataset
    df_original_renamed = df_original.rename(columns=original_col_map)

    # Convert Sex to numeric (Female=0, Male=1)
    df_original_renamed["Sex"] = df_original_renamed["Sex"].map(
        {"Female": 0, "Male": 1}
    )

    numeric_features_to_scale = [
        "Age",
        "Height",
        "Weight",
        "Systolic_BP",
        "Diastolic_BP",
        "Heart_Rate",
        "BMI",
    ]
    all_features = ["Sex"] + numeric_features_to_scale

    # 4. Kiểm tra tên cột sau khi rename
    print("Checking renamed columns:")
    for feat in numeric_features_to_scale:
        if feat in df_original_renamed.columns:
            print(f"   {feat}")
        else:
            print(f"   {feat} - NOT FOUND!")

    # 5. Lấy dữ liệu gốc
    X_original = df_original_renamed[numeric_features_to_scale]

    print(f"\nOriginal data statistics (features to scale):")
    print(X_original.describe())

    # 6. Tạo và fit scaler
    scaler = StandardScaler()
    scaler.fit(X_original)

    print(f"\nScaler fitted!")
    print(f"Features: {numeric_features_to_scale}")
    print(f"Mean: {scaler.mean_}")
    print(f"Std:  {scaler.scale_}")

    # 7. Verify: Transform và so sánh với normalized dataset
    X_transformed = scaler.transform(X_original)
    X_normalized_values = df_normalized[numeric_features_to_scale].values

    print(f"\nVerification:")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Normalized shape:  {X_normalized_values.shape}")

    # So sánh một vài dòng đầu
    print(f"\nFirst row comparison:")
    print(f"Transformed: {X_transformed[0]}")
    print(f"Normalized:  {X_normalized_values[0]}")
    print(f"Difference:  {np.abs(X_transformed[0] - X_normalized_values[0])}")

    # Kiểm tra độ chính xác
    max_diff = np.max(np.abs(X_transformed - X_normalized_values))
    print(f"\nMax difference: {max_diff}")

    if max_diff < 0.01:
        print("✅ Scaler matches the normalized dataset")
    elif max_diff < 0.1:
        print("⚠️  Close but not exact")
    else:
        print("❌ Large difference")

    # 8. Save scaler
    output_path = Path(__file__).parent / "models" / "random_forest" / "scaler.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)

    print(f"\nScaler saved to: {output_path}")

    # 9. Test với raw data thực tế
    print(f"\nTest transformation:")
    test_raw = pd.DataFrame(
        [
            {
                "Age": df_original_renamed["Age"].iloc[0],
                "Height": df_original_renamed["Height"].iloc[0],
                "Weight": df_original_renamed["Weight"].iloc[0],
                "Systolic_BP": df_original_renamed["Systolic_BP"].iloc[0],
                "Diastolic_BP": df_original_renamed["Diastolic_BP"].iloc[0],
                "Heart_Rate": df_original_renamed["Heart_Rate"].iloc[0],
                "BMI": df_original_renamed["BMI"].iloc[0],
            }
        ]
    )

    test_normalized = scaler.transform(test_raw)

    print(f"Raw input (first row from original, excluding Sex):")
    print(test_raw.iloc[0].to_dict())
    print(f"Sex (not scaled): {df_original_renamed['Sex'].iloc[0]}")
    print(f"\nNormalized output:")
    print(test_normalized[0])
    print(f"\nExpected (from normalized dataset, excluding Sex):")
    print(df_normalized[numeric_features_to_scale].iloc[0].values)

    return scaler


if __name__ == "__main__":
    try:
        scaler = create_scaler()
        print("\nSUCCESS! Scaler created and saved!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()

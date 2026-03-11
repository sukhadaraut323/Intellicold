"""
IntelliCold - Retrain Scaler (For YOUR CSV with 29 columns)
"""

import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

print("="*70)
print("RETRAINING SCALER - USING YOUR ACTUAL CSV COLUMNS")
print("="*70)

data_path = "ml_model/data/features_dataset.csv"

if not os.path.exists(data_path):
    print(f"❌ Error: {data_path} not found!")
    exit(1)

print(f"\n📂 Loading {data_path}...")
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# These are the features from YOUR CSV (without spoilage_rate_per_hr!)
features_to_scale = [
    'temperature_C',
    'humidity_percent',
    'safe_temp_low_C',
    'safe_temp_mid_C',
    'safe_temp_high_C',
    'humidity_low_percent',
    'humidity_mid_percent',
    'humidity_high_percent',
    'exposure_hours',
    'ethylene_ppm',
    'co2_ppm',
    'nh3_ppm',
    'h2s_ppm',
    'temp_deviation',
    'temp_deviation_degree_hr',
    'cumulative_damage_index',
    'humidity_deviation',
    'spoilage_probability',
    'remaining_shelf_life_hours',
    'category_encoded',
    'product_name_encoded',
    'risk_encoded',
    'action_encoded',
]

print(f"\n📊 Features to scale ({len(features_to_scale)}):")
for i, feat in enumerate(features_to_scale, 1):
    print(f"  {i:2d}. {feat}")

# Verify all exist
missing = [f for f in features_to_scale if f not in df.columns]
if missing:
    print(f"\n❌ Missing features: {missing}")
    exit(1)

print("\n✅ All features found!")

# Create feature matrix
X = df[features_to_scale]

print(f"\n🔧 Training StandardScaler...")
scaler = StandardScaler()
scaler.fit(X)

print(f"✅ Scaler fitted with {scaler.n_features_in_} features")

# Save
scaler_path = "ml_model/models/scaler.pkl"
os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
pickle.dump(scaler, open(scaler_path, "wb"))

print(f"\n💾 Saved to: {scaler_path}")

# Test
scaled = scaler.transform(X.iloc[0:1])
print(f"\n🧪 Test: {X.shape} -> {scaled.shape}")
print(f"✅ SUCCESS!")

print("\n" + "="*70)
print("✅ SCALER RETRAINED FOR YOUR CSV!")
print("="*70)
print("\nNext: Use predict_FINAL_FOR_YOUR_CSV.py")
print("="*70)
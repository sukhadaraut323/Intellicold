"""
Complete Reset and Test Script
This does EVERYTHING in one go
"""

import pandas as pd
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

print("="*70)
print("INTELLICOLD - COMPLETE RESET & FIX")
print("="*70)

# Step 1: Load CSV
print("\n[1/5] Loading CSV...")
df = pd.read_csv("ml_model/data/features_dataset.csv")
df.columns = df.columns.str.strip()
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# Step 2: Create and save scaler
print("\n[2/5] Training new scaler...")

features_to_scale = [
    'temperature_C', 'humidity_percent', 'safe_temp_low_C', 'safe_temp_mid_C',
    'safe_temp_high_C', 'humidity_low_percent', 'humidity_mid_percent',
    'humidity_high_percent', 'exposure_hours', 'ethylene_ppm', 'co2_ppm',
    'nh3_ppm', 'h2s_ppm', 'temp_deviation', 'temp_deviation_degree_hr',
    'cumulative_damage_index', 'humidity_deviation', 'spoilage_probability',
    'remaining_shelf_life_hours', 'category_encoded', 'product_name_encoded',
    'risk_encoded', 'action_encoded',
]

X = df[features_to_scale]
scaler = StandardScaler()
scaler.fit(X)

scaler_path = "ml_model/models/scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

print(f"✅ Scaler saved with {scaler.n_features_in_} features")

# Step 3: Test the scaler immediately
print("\n[3/5] Testing scaler...")
test_data = OrderedDict([
    ('temperature_C', 5.0),
    ('humidity_percent', 70.0),
    ('safe_temp_low_C', 2.0),
    ('safe_temp_mid_C', 4.0),
    ('safe_temp_high_C', 6.0),
    ('humidity_low_percent', 60.0),
    ('humidity_mid_percent', 75.0),
    ('humidity_high_percent', 85.0),
    ('exposure_hours', 12.0),
    ('ethylene_ppm', 5.0),
    ('co2_ppm', 500.0),
    ('nh3_ppm', 2.0),
    ('h2s_ppm', 0.2),
    ('temp_deviation', 1.0),
    ('temp_deviation_degree_hr', 12.0),
    ('cumulative_damage_index', 0.12),
    ('humidity_deviation', -5.0),
    ('spoilage_probability', 0.12),
    ('remaining_shelf_life_hours', 88.0),
    ('category_encoded', 0),
    ('product_name_encoded', 0),
    ('risk_encoded', 0),
    ('action_encoded', 0),
])

test_df = pd.DataFrame([test_data])
scaled = scaler.transform(test_df)
print(f"✅ Test passed! Input: {test_df.shape}, Output: {scaled.shape}")

# Step 4: Test prediction
print("\n[4/5] Testing prediction module...")
sys.path.insert(0, 'ml_model')

# Force reload if already imported
if 'predict' in sys.modules:
    del sys.modules['predict']

from ml_model.predict import predict

result = predict({
    'avg_temp_c': 5.0,
    'humidity_percent': 70.0,
    'transport_duration_hr': 12.0,
    'product_type': 'milk'
})

print(f"✅ Prediction successful!")
print(f"   Quality: {result['quality_remaining']}%")
print(f"   Risk: {result['risk_level']}")
print(f"   Hours to spoilage: {result['hours_to_spoilage']}")

# Step 5: Final verification
print("\n[5/5] Final verification...")
print(f"✅ Scaler features: {list(scaler.feature_names_in_)[:3]}... ({len(scaler.feature_names_in_)} total)")
print(f"✅ Prediction module: Working")

print("\n" + "="*70)
print("✅ ALL CHECKS PASSED!")
print("="*70)
print("\nYour backend is ready!")
print("Start server: python app.py")
print("Test: http://localhost:5000/api/health")
print("="*70)
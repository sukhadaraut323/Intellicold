"""
Check what the retrained scaler actually expects
"""

import pickle
import os

scaler_path = "ml_model/models/scaler.pkl"

if not os.path.exists(scaler_path):
    print("❌ Scaler not found!")
    exit(1)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print("="*70)
print("SCALER FEATURE ANALYSIS")
print("="*70)

print(f"\nScaler expects {scaler.n_features_in_} features:")
print("-"*70)

for i, feat in enumerate(scaler.feature_names_in_, 1):
    print(f"{i:2d}. {feat}")

print("\n" + "="*70)
print("Copy this list for predict.py:")
print("="*70)

print("\nfeatures_for_scaler = [")
for feat in scaler.feature_names_in_:
    print(f"    '{feat}',")
print("]")

print("\n" + "="*70)
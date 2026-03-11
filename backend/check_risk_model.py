"""
Check what the RISK MODEL expects
"""

import joblib

risk_model_path = "ml_model/models/risk_model.pkl"

print("="*70)
print("RISK MODEL FEATURE ANALYSIS")
print("="*70)

risk_model = joblib.load(risk_model_path)

print(f"\nRisk model expects {risk_model.n_features_in_} features:")
print("-"*70)

if hasattr(risk_model, 'feature_names_in_'):
    for i, feat in enumerate(risk_model.feature_names_in_, 1):
        print(f"{i:2d}. {feat}")
else:
    print("❌ Risk model doesn't have feature names stored")
    print(f"   But it expects {risk_model.n_features_in_} features")

print("\n" + "="*70)
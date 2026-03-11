"""
IntelliCold - FINAL CORRECT Prediction Module

SCALER: 23 features (including spoilage_probability, remaining_shelf_life_hours, risk_encoded, action_encoded)
RISK MODEL: 20 features (NO spoilage/shelf_life/risk/action, but HAS temp_danger_flag)
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any
from collections import OrderedDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, 'models')

print(f"Loading models from: {_MODELS_DIR}")

try:
    risk_model = joblib.load(os.path.join(_MODELS_DIR, 'risk_model.pkl'))
    scaler = joblib.load(os.path.join(_MODELS_DIR, 'scaler.pkl'))
    
    try:
        package = joblib.load(os.path.join(_MODELS_DIR, 'intellicold_model_package.pkl'))
        if isinstance(package, dict) and 'model' in package:
            quality_model = package['model']
            time_model = package['model']
            action_model = risk_model
        else:
            quality_model = risk_model
            time_model = risk_model
            action_model = risk_model
    except:
        quality_model = risk_model
        time_model = risk_model
        action_model = risk_model
    
    print("[OK] Models loaded!")
    
except Exception as e:
    print(f"[ERROR] {e}")
    raise

RISK_LEVELS = ['Low', 'Medium', 'High', 'Critical']

ACTION_MAP = {
    0: '[OK] Maintain current cooling settings - Product is stable',
    1: '[WARNING] Increase cooling intensity - Temperature trending upward',
    2: '[ALERT] Expedite delivery - Quality degrading faster than expected',
    3: '[CRITICAL] CRITICAL: Activate emergency protocols - Immediate intervention required'
}

PRODUCT_SAFE_TEMPS = {
    'milk': {'low': 2, 'mid': 4, 'high': 6},
    'meat': {'low': 0, 'mid': 2, 'high': 4},
    'vegetables': {'low': 8, 'mid': 10, 'high': 12},
    'fish': {'low': -2, 'mid': 0, 'high': 2},
    'fruits': {'low': 6, 'mid': 8, 'high': 10},
    'dairy': {'low': 2, 'mid': 4, 'high': 6},
    'vaccines': {'low': 2, 'mid': 4, 'high': 8},
}

CATEGORY_ENCODING = {
    'dairy': 0, 'meat': 1, 'produce': 2, 'seafood': 3, 'pharmaceutical': 4, 'frozen': 5,
}

PRODUCT_ENCODING = {
    'milk': 0, 'cheese': 1, 'yogurt': 2, 'butter': 3, 'beef': 4, 'chicken': 5,
    'pork': 6, 'lamb': 7, 'meat': 1, 'vegetables': 8, 'fruits': 9, 'fish': 10,
    'shrimp': 11, 'salmon': 12, 'vaccines': 13, 'medicines': 14, 'ice_cream': 15,
    'frozen_vegetables': 16,
}

def engineer_features(backend_data: Dict[str, Any]) -> tuple:
    temperature = float(backend_data.get('avg_temp_c', 5.0))
    humidity = float(backend_data.get('humidity_percent', 70.0))
    exposure_hrs = float(backend_data.get('transport_duration_hr', 12.0))
    product_type = backend_data.get('product_type', 'milk').lower()
    
    category_map = {
        'milk': 'dairy', 'cheese': 'dairy', 'yogurt': 'dairy', 'butter': 'dairy',
        'beef': 'meat', 'chicken': 'meat', 'pork': 'meat', 'lamb': 'meat', 'meat': 'meat',
        'vegetables': 'produce', 'fruits': 'produce',
        'fish': 'seafood', 'shrimp': 'seafood', 'salmon': 'seafood',
        'vaccines': 'pharmaceutical', 'medicines': 'pharmaceutical',
        'ice_cream': 'frozen', 'frozen_vegetables': 'frozen',
    }
    category = category_map.get(product_type, 'dairy')
    
    safe_temps = PRODUCT_SAFE_TEMPS.get(product_type, PRODUCT_SAFE_TEMPS['milk'])
    safe_temp_low = safe_temps['low']
    safe_temp_mid = safe_temps['mid']
    safe_temp_high = safe_temps['high']
    
    ethylene_ppm = float(backend_data.get('ethylene_ppm', 5.0))
    co2_ppm = float(backend_data.get('co2_ppm', 500.0))
    nh3_ppm = float(backend_data.get('nh3_ppm', 2.0))
    h2s_ppm = float(backend_data.get('h2s_ppm', 0.2))
    
    temp_deviation = temperature - safe_temp_mid
    temp_deviation_degree_hr = temp_deviation * exposure_hrs
    humidity_deviation = humidity - 75.0
    cumulative_damage_index = abs(temp_deviation_degree_hr) / 100.0
    
    spoilage_probability = min(1.0, abs(temp_deviation * 0.01) * exposure_hrs)
    remaining_shelf_life_hours = max(0, (1.0 - spoilage_probability) * 100.0)
    
    temp_danger_flag = 1 if temperature > safe_temp_high else 0
    
    category_encoded = CATEGORY_ENCODING.get(category, 0)
    product_name_encoded = PRODUCT_ENCODING.get(product_type, 0)
    
    # 23 features for SCALER (exact order)
    features_for_scaler = OrderedDict([
        ('temperature_C', temperature),
        ('humidity_percent', humidity),
        ('safe_temp_low_C', safe_temp_low),
        ('safe_temp_mid_C', safe_temp_mid),
        ('safe_temp_high_C', safe_temp_high),
        ('humidity_low_percent', 60.0),
        ('humidity_mid_percent', 75.0),
        ('humidity_high_percent', 85.0),
        ('exposure_hours', exposure_hrs),
        ('ethylene_ppm', ethylene_ppm),
        ('co2_ppm', co2_ppm),
        ('nh3_ppm', nh3_ppm),
        ('h2s_ppm', h2s_ppm),
        ('temp_deviation', temp_deviation),
        ('temp_deviation_degree_hr', temp_deviation_degree_hr),
        ('cumulative_damage_index', cumulative_damage_index),
        ('humidity_deviation', humidity_deviation),
        ('spoilage_probability', spoilage_probability),
        ('remaining_shelf_life_hours', remaining_shelf_life_hours),
        ('category_encoded', category_encoded),
        ('product_name_encoded', product_name_encoded),
        ('risk_encoded', 0),
        ('action_encoded', 0),
    ])
    
    return pd.DataFrame([features_for_scaler]), temp_danger_flag, spoilage_probability, remaining_shelf_life_hours

def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    # Generate 23 features for scaler
    X_for_scaler, temp_danger_flag, spoilage_prob, shelf_life = engineer_features(features)
    
    # Scale the 23 features
    X_scaled = scaler.transform(X_for_scaler)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_for_scaler.columns)
    
    # Create 20 features for RISK MODEL (exact order model expects)
    model_features = OrderedDict([
        ('temperature_C', X_scaled_df['temperature_C'].values[0]),
        ('humidity_percent', X_scaled_df['humidity_percent'].values[0]),
        ('safe_temp_low_C', X_scaled_df['safe_temp_low_C'].values[0]),
        ('safe_temp_mid_C', X_scaled_df['safe_temp_mid_C'].values[0]),
        ('safe_temp_high_C', X_scaled_df['safe_temp_high_C'].values[0]),
        ('humidity_low_percent', X_scaled_df['humidity_low_percent'].values[0]),
        ('humidity_mid_percent', X_scaled_df['humidity_mid_percent'].values[0]),
        ('humidity_high_percent', X_scaled_df['humidity_high_percent'].values[0]),
        ('exposure_hours', X_scaled_df['exposure_hours'].values[0]),
        ('ethylene_ppm', X_scaled_df['ethylene_ppm'].values[0]),
        ('co2_ppm', X_scaled_df['co2_ppm'].values[0]),
        ('nh3_ppm', X_scaled_df['nh3_ppm'].values[0]),
        ('h2s_ppm', X_scaled_df['h2s_ppm'].values[0]),
        ('temp_deviation', X_scaled_df['temp_deviation'].values[0]),
        ('temp_deviation_degree_hr', X_scaled_df['temp_deviation_degree_hr'].values[0]),
        ('cumulative_damage_index', X_scaled_df['cumulative_damage_index'].values[0]),
        ('humidity_deviation', X_scaled_df['humidity_deviation'].values[0]),
        ('temp_danger_flag', temp_danger_flag),  # NOT SCALED!
        ('category_encoded', X_scaled_df['category_encoded'].values[0]),
        ('product_name_encoded', X_scaled_df['product_name_encoded'].values[0]),
    ])
    
    model_input = pd.DataFrame([model_features])
    
    # Predict
    risk_prediction = risk_model.predict(model_input)[0]
    
    # Model returns string labels, not indices
    if isinstance(risk_prediction, str):
        risk_level = risk_prediction
        risk_idx = RISK_LEVELS.index(risk_level) if risk_level in RISK_LEVELS else 0
    else:
        risk_idx = int(risk_prediction)
        risk_idx = max(0, min(3, risk_idx))
        risk_level = RISK_LEVELS[risk_idx]
    
    risk_probs = risk_model.predict_proba(model_input)[0]
    
    quality_remaining = (1.0 - spoilage_prob) * 100.0
    hours_to_spoilage = shelf_life
    
    # ⚠️ QUALITY-BASED RISK ADJUSTMENT (Risk inversely proportional to quality)
    # If quality is critically low, risk MUST be elevated!
    if quality_remaining < 10:
        # Quality < 10% → CRITICAL risk
        risk_idx = max(risk_idx, 3)
        risk_level = 'Critical'
    elif quality_remaining < 30:
        # Quality < 30% → HIGH risk
        risk_idx = max(risk_idx, 2)
        risk_level = 'High'
    elif quality_remaining < 50:
        # Quality < 50% → MEDIUM risk
        risk_idx = max(risk_idx, 1)
        risk_level = 'Medium'
    # If quality > 50%, keep the model's risk prediction
    
    action_idx = min(risk_idx, 3)
    quality_remaining = max(0.0, min(100.0, quality_remaining))
    
    return {
        'quality_remaining': round(quality_remaining, 1),
        'risk_level': risk_level,
        'risk_index': risk_idx,
        'hours_to_spoilage': round(max(0, hours_to_spoilage), 1),
        'recommended_action': ACTION_MAP.get(action_idx, ACTION_MAP[0]),
        'risk_probabilities': {
            'Low': round(float(risk_probs[0]), 3),
            'Medium': round(float(risk_probs[1]), 3),
            'High': round(float(risk_probs[2]), 3),
            'Critical': round(float(risk_probs[3]), 3),
        }
    }

if __name__ == "__main__":
    test = {'avg_temp_c': 5.0, 'humidity_percent': 70.0, 'transport_duration_hr': 12.0, 'product_type': 'milk'}
    result = predict(test)
    print("\n✅ Prediction successful!")
    for k, v in result.items():
        if k != 'risk_probabilities':
            print(f"  {k}: {v}")
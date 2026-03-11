"""
IntelliCold API Test Script (FIXED for S001, S002, S003)
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    print("\n🧪 Test 1: Health Check")
    print("-"*70)
    r = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    return r.status_code == 200

def test_predict():
    print("\n🧪 Test 2: General Prediction")
    print("-"*70)
    data = {
        "avg_temp_c": 7.0,
        "humidity_percent": 70.0,
        "transport_duration_hr": 57.0,
        "product_type": "milk"
    }
    r = requests.post(f"{BASE_URL}/api/predict", json=data)
    print(f"Status: {r.status_code}")
    result = r.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return r.status_code == 200

def test_shipment_predict():
    print("\n🧪 Test 3: Shipment Prediction (S001)")
    print("-"*70)
    r = requests.get(f"{BASE_URL}/api/shipments/S001/predict")
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    return r.status_code == 200

def test_monitor():
    print("\n🧪 Test 4: Monitor Update (S002)")
    print("-"*70)
    data = {
        "temperature": 8.5,
        "humidity": 82.0
    }
    r = requests.post(f"{BASE_URL}/api/shipments/S002/monitor", json=data)
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    return r.status_code == 200

def test_all_shipments():
    print("\n🧪 Test 5: Get All Shipments")
    print("-"*70)
    r = requests.get(f"{BASE_URL}/api/shipments")
    print(f"Status: {r.status_code}")
    shipments = r.json()
    print(f"Found {len(shipments)} shipments:")
    for s in shipments:
        print(f"  - {s['id']}: {s['name']} | Risk: {s['risk_level']} | Quality: {s['quality_remaining']}%")
    return r.status_code == 200 and len(shipments) == 3

if __name__ == "__main__":
    print("="*70)
    print("INTELLICOLD API TESTS (FIXED)")
    print("="*70)
    
    results = {
        "Health Check": test_health(),
        "General Prediction": test_predict(),
        "Shipment Prediction": test_shipment_predict(),
        "Monitor Update": test_monitor(),
        "All Shipments": test_all_shipments(),
    }
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed")
    print("="*70)
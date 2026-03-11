"""
Diagnostic Script - Check CSV Columns
Run this to see EXACTLY what columns are in your CSV
"""

import pandas as pd
import os

csv_path = "ml_model/data/features_dataset.csv"

print("="*70)
print("CSV COLUMN DIAGNOSTIC")
print("="*70)

if not os.path.exists(csv_path):
    print(f"\n❌ File not found: {csv_path}")
    print("\nSearching for CSV files...")
    for root, dirs, files in os.walk("ml_model"):
        for file in files:
            if file.endswith('.csv'):
                print(f"   Found: {os.path.join(root, file)}")
    exit(1)

print(f"\n📂 Loading: {csv_path}")
df = pd.read_csv(csv_path)

# Clean column names
df.columns = df.columns.str.strip()

print(f"\n✅ Loaded successfully!")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")

print(f"\n📋 ALL COLUMNS ({len(df.columns)}):")
print("="*70)
for i, col in enumerate(df.columns, 1):
    # Show any hidden characters
    col_repr = repr(col)
    print(f"{i:2d}. {col:40s} (repr: {col_repr})")

print("\n" + "="*70)
print("CHECKING FOR SPOILAGE COLUMNS:")
print("="*70)
for col in df.columns:
    if 'spoilage' in col.lower():
        print(f"✅ Found: '{col}'")
        print(f"   Length: {len(col)} characters")
        print(f"   Repr: {repr(col)}")

print("\n" + "="*70)
print("COPY THESE EXACT COLUMN NAMES:")
print("="*70)
print("features_to_scale = [")
for col in df.columns:
    if col not in ['shipment_id', 'product_name', 'category', 'temp_danger_flag', 
                   'risk_category', 'recommended_action']:
        print(f"    '{col}',")
print("]")

print("\n" + "="*70)
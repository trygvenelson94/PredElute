#!/usr/bin/env python3
"""
Convert retention volumes to NaCl concentrations for SP Sepharose HP data

Formula: NaCl (M) = 0.022 + (Retention Volume / Column Volume) × 0.040

Where:
- Starting buffer: 22 mM phosphate (0.022 M)
- Gradient: 40 mM NaCl per column volume
- Column volume: ~1 mL (50 × 5 mm)
"""

import pandas as pd
import numpy as np

# Load elution data
df = pd.read_csv('sp_sepharose_hp_elution_dataset.csv', header=4, usecols=[0,1,2,3,4,5,6], nrows=17)
df.columns = ['pdb_id', 'protein_name', 'pH8', 'pH7', 'pH6', 'pH5', 'pH4']

print("="*100)
print("CONVERTING RETENTION VOLUMES TO NaCl CONCENTRATIONS")
print("="*100)
print("\nFormula: NaCl (M) = 0.022 + (Retention Volume in mL) × 0.040")
print("  - Starting buffer: 22 mM phosphate")
print("  - Gradient: 40 mM NaCl per column volume")
print("  - Column volume: 1 mL")
print("="*100)

# Conversion parameters
starting_ionic_strength = 0.022  # M
gradient_slope = 0.040  # M/CV
column_volume = 1.0  # mL

# Convert retention volumes to NaCl concentrations
df_nacl = df.copy()

for ph_col in ['pH8', 'pH7', 'pH6', 'pH5', 'pH4']:
    df_nacl[ph_col] = starting_ionic_strength + (df[ph_col] / column_volume) * gradient_slope

# Display results
print("\nOriginal Retention Volumes (mL):")
print(df[['protein_name', 'pH8', 'pH7', 'pH6', 'pH5', 'pH4']].to_string(index=False))

print("\n" + "="*100)
print("Converted NaCl Concentrations (M):")
print("="*100)
print(df_nacl[['protein_name', 'pH8', 'pH7', 'pH6', 'pH5', 'pH4']].to_string(index=False))

# Save to CSV
output_file = 'sp_sepharose_hp_nacl_concentrations.csv'
df_nacl.to_csv(output_file, index=False)

print("\n" + "="*100)
print(f"[OK] Saved NaCl concentrations to: {output_file}")
print("="*100)

# Summary statistics
print("\nNaCl Concentration Range by pH:")
for ph_col in ['pH4', 'pH5', 'pH6', 'pH7', 'pH8']:
    values = df_nacl[ph_col].dropna()
    if len(values) > 0:
        print(f"  {ph_col}: {values.min():.3f} - {values.max():.3f} M  (n={len(values)})")

print("\n" + "="*100)

#!/usr/bin/env python3
"""
Create combined ProDes + Schrodinger descriptors for modifier training set
"""

import pandas as pd
import numpy as np

print("="*100)
print("CREATING COMBINED DESCRIPTORS FOR MODIFIER TRAINING SET")
print("="*100)

# Load ProDes descriptors
print("\n[1/3] Loading ProDes descriptors...")
df_prodes = pd.read_csv('prodes_descriptors_complete.csv', index_col=0)
df_prodes.index = df_prodes.index.str.lower().str.strip()
print(f"  ProDes: {df_prodes.shape}")
print(f"  Proteins: {len(df_prodes)}")

# Load Schrodinger descriptors
print("\n[2/3] Loading Schrodinger descriptors...")
df_schrodinger = pd.read_csv('elution_with_modifiers_protein_descriptors.csv')
df_schrodinger['Name'] = df_schrodinger['Name'].str.lower().str.strip().str.replace('_-_prepared', '')

# Map PDB codes to protein names
pdb_to_name = {
    '1bsy': 'aprotinin',
    '1rav': 'ribonuclease a',
    '1yge': 'conalbumin',
    '1yph': 'lysozyme',
    '1z6s': 'alpha-chymotrypsinogen a',
    '2cds': 'carbonic anhydrase',
    '2cga': 'alpha-chymotrypsin',
    '2pel': 'peanut lectin',
    '2zcc': 'papain',
    '3etg': 'avidin',
    '3u1j': 'l-glutamic dehydrogenase',
    '3wc8': 'lipoxidase',
    '4f5s': 'bovine serum albumin',
    '4luf': 'human serum albumin',
    '5ezt': 'beta-lactoglobulin b',
    '5ntb': 'sheep albumin',
    '7vr0': 'ubiquitin',
    '8fei': 'horse cytochrome c',
    'beta-chymotrypsin': 'beta-chymotrypsin',
    'bovine_trypsin': 'bovine trypsin',
    'porcine_albumin': 'porcine albumin',
    'porcine_trypsin': 'porcine trypsin',
    'trypsinogen': 'trypsinogen'
}

df_schrodinger['Name'] = df_schrodinger['Name'].map(lambda x: pdb_to_name.get(x, x))
df_schrodinger = df_schrodinger.set_index('Name')

# Drop non-descriptor columns
if 'Experiment' in df_schrodinger.columns:
    df_schrodinger = df_schrodinger.drop(columns=['Experiment'])
print(f"  Schrodinger: {df_schrodinger.shape}")
print(f"  Proteins: {len(df_schrodinger)}")

# Find common proteins
print("\n[3/3] Merging descriptors...")
common_proteins = sorted(set(df_prodes.index) & set(df_schrodinger.index))
print(f"  Common proteins: {len(common_proteins)}")

# Exclude proteins without Schrodinger descriptors
excluded = sorted(set(df_prodes.index) - set(df_schrodinger.index))
if excluded:
    print(f"  Excluded (no Schrodinger): {excluded}")

# Merge
df_prodes_common = df_prodes.loc[common_proteins]
df_schrodinger_common = df_schrodinger.loc[common_proteins]
df_combined = pd.concat([df_prodes_common, df_schrodinger_common], axis=1)

# Clean
df_combined = df_combined.replace([np.inf, -np.inf], np.nan)
cols_with_nan = df_combined.columns[df_combined.isna().any()].tolist()
if cols_with_nan:
    print(f"  Dropping {len(cols_with_nan)} columns with NaN")
    df_combined = df_combined.drop(columns=cols_with_nan)

print(f"\n  Final shape: {df_combined.shape}")
print(f"    ProDes features: {df_prodes_common.shape[1]}")
print(f"    Schrodinger features: {df_schrodinger_common.shape[1]}")
print(f"    Total features: {df_combined.shape[1]}")
print(f"    NaN values: {df_combined.isna().sum().sum()}")

# Save
output_file = 'training_modifiers_combined_descriptors.csv'
df_combined.to_csv(output_file)
print(f"\n[OK] Saved: {output_file}")

# Show protein list
print(f"\nProteins included ({len(common_proteins)}):")
for i, prot in enumerate(common_proteins, 1):
    print(f"  {i:2d}. {prot}")

print("\n" + "="*100)
print("COMPLETE")
print("="*100)

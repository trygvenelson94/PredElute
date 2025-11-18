#!/usr/bin/env python3
"""
Create combined descriptor files for both training sets:
1. pH-dependent elution model (ProDes + Schrodinger)
2. Elution with modifiers model (ProDes only, but save for consistency)
"""

import pandas as pd
import numpy as np

print("="*100)
print("CREATING COMBINED DESCRIPTOR FILES FOR TRAINING SETS")
print("="*100)

# ============================================================================
# 1. pH-DEPENDENT ELUTION MODEL (ProDes + Schrodinger)
# ============================================================================
print("\n[1/2] Creating combined descriptors for pH-dependent elution model...")

# Load ProDes descriptors
df_prodes_ph = pd.read_csv('sp_sepharose_hp_descriptors_complete.csv', index_col=0)
print(f"  ProDes: {df_prodes_ph.shape}")

# Load Schrodinger descriptors
df_schrodinger = pd.read_csv('schrodinger_descriptors.csv')
df_schrodinger['Name'] = df_schrodinger['Name'].str.lower().str.strip().str.replace(' - prepared', '')
df_schrodinger = df_schrodinger.set_index('Name')
print(f"  Schrodinger: {df_schrodinger.shape}")

# Normalize indices
df_prodes_ph.index = df_prodes_ph.index.str.lower().str.strip()

# Find common proteins
common_proteins_ph = sorted(set(df_prodes_ph.index) & set(df_schrodinger.index))
print(f"  Common proteins: {len(common_proteins_ph)}")

# Merge
df_prodes_ph_common = df_prodes_ph.loc[common_proteins_ph]
df_schrodinger_common = df_schrodinger.loc[common_proteins_ph]
df_combined_ph = pd.concat([df_prodes_ph_common, df_schrodinger_common], axis=1)

# Clean
df_combined_ph = df_combined_ph.replace([np.inf, -np.inf], np.nan)
cols_with_nan = df_combined_ph.columns[df_combined_ph.isna().any()].tolist()
if cols_with_nan:
    print(f"  Dropping {len(cols_with_nan)} columns with NaN")
    df_combined_ph = df_combined_ph.drop(columns=cols_with_nan)

print(f"  Final shape: {df_combined_ph.shape}")
print(f"    ProDes features: {df_prodes_ph_common.shape[1]}")
print(f"    Schrodinger features: {df_schrodinger_common.shape[1]}")
print(f"    Total features: {df_combined_ph.shape[1]}")

# Save
output_file_ph = 'training_ph_combined_descriptors.csv'
df_combined_ph.to_csv(output_file_ph)
print(f"  [OK] Saved: {output_file_ph}")

# ============================================================================
# 2. ELUTION WITH MODIFIERS MODEL (ProDes only)
# ============================================================================
print("\n[2/2] Creating descriptors for elution with modifiers model...")

# Load ProDes descriptors
df_prodes_mod = pd.read_csv('prodes_descriptors_complete.csv', index_col=0)
df_prodes_mod.index = df_prodes_mod.index.str.lower().str.strip()

print(f"  ProDes: {df_prodes_mod.shape}")
print(f"  Proteins: {len(df_prodes_mod)}")

# Clean
df_prodes_mod = df_prodes_mod.replace([np.inf, -np.inf], np.nan)
cols_with_nan = df_prodes_mod.columns[df_prodes_mod.isna().any()].tolist()
if cols_with_nan:
    print(f"  Dropping {len(cols_with_nan)} columns with NaN")
    df_prodes_mod = df_prodes_mod.drop(columns=cols_with_nan)

print(f"  Final shape: {df_prodes_mod.shape}")

# Save
output_file_mod = 'training_modifiers_descriptors.csv'
df_prodes_mod.to_csv(output_file_mod)
print(f"  [OK] Saved: {output_file_mod}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print(f"\nPh-dependent elution model:")
print(f"  File: {output_file_ph}")
print(f"  Proteins: {len(df_combined_ph)}")
print(f"  Features: {df_combined_ph.shape[1]} (ProDes + Schrodinger)")
print(f"  Proteins: {', '.join(df_combined_ph.index[:5].tolist())}...")

print(f"\nElution with modifiers model:")
print(f"  File: {output_file_mod}")
print(f"  Proteins: {len(df_prodes_mod)}")
print(f"  Features: {df_prodes_mod.shape[1]} (ProDes only)")
print(f"  Proteins: {', '.join(df_prodes_mod.index[:5].tolist())}...")

print("\n" + "="*100)
print("COMPLETE")
print("="*100)

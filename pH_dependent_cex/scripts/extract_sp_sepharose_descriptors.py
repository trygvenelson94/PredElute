#!/usr/bin/env python3
"""
Extract SP Sepharose HP proteins from existing training descriptors
"""

import pandas as pd

# Load existing training descriptors
df_training = pd.read_csv('prodes_descriptors_training_only.csv', index_col=0)

# SP Sepharose HP proteins from your folder
sp_proteins_map = {
    'alpha chymotrypsin': 'alpha-chymotrypsin',
    'alpha chymotrypsinogen A': 'alpha-chymotrypsinogen a',
    'aprotinin': 'aprotinin',
    'avidin': 'avidin',
    'bovine trypsin': 'bovine trypsin',
    'carbonic anhydrase': 'carbonic anhydrase',
    'conalbumin': 'conalbumin',
    'horse cytochrome c': 'horse cytochrome c',
    'lysozyme': 'lysozyme',
    'ribonuclease a': 'ribonuclease a',
    'ribonuclease b': 'ribonuclease b',
    'trypsinogen': 'trypsinogen'
}

# Proteins NOT in training set (need to calculate separately)
missing_proteins = [
    'bee phospholipase a2',
    'bovine cytochrome c',
    'elastase',
    'lactoferrin',
    'pyruvate kinase'
]

print("="*100)
print("EXTRACTING SP SEPHAROSE HP DESCRIPTORS")
print("="*100)

# Extract proteins that exist
found_proteins = []
for new_name, training_name in sp_proteins_map.items():
    if training_name in df_training.index:
        found_proteins.append(training_name)
        print(f"[OK] {new_name} -> {training_name}")
    else:
        print(f"[MISSING] {new_name}")

# Create subset
df_sp = df_training.loc[found_proteins]

# Save
output_file = 'sp_sepharose_hp_descriptors.csv'
df_sp.to_csv(output_file)

print(f"\n[OK] Saved {len(df_sp)} proteins to {output_file}")
print(f"\nProteins included:")
for p in df_sp.index:
    print(f"  - {p}")

print(f"\n{'='*100}")
print(f"MISSING PROTEINS (need to calculate separately):")
print(f"{'='*100}")
for p in missing_proteins:
    print(f"  - {p}")
print(f"\nThese {len(missing_proteins)} proteins are not in the training set.")
print(f"ProDes is hanging on the Schrodinger-prepared PDBs.")
print(f"You may need to find alternative PDB files or skip these proteins.")
print("="*100)

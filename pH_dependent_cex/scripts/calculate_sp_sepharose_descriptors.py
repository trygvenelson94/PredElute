#!/usr/bin/env python3
"""
Calculate ProDes descriptors for all PDB files in sp_sepharose_hp_training_pdb folder
"""

import os
import subprocess
import pandas as pd
from glob import glob

pdb_folder = r"C:\Users\tryg.nelson\predelute\sp_sepharose_hp_training_pdb"
output_csv = r"C:\Users\tryg.nelson\predelute\sp_sepharose_hp_descriptors.csv"

print("="*100)
print("CALCULATING PRODES DESCRIPTORS FOR SP SEPHAROSE HP TRAINING SET")
print("="*100)

# Get all PDB files
pdb_files = glob(os.path.join(pdb_folder, "*.pdb"))
print(f"\nFound {len(pdb_files)} PDB files")

# Remove duplicates (files ending in _1.pdb)
pdb_files = [f for f in pdb_files if not f.endswith('_1.pdb')]
print(f"After removing duplicates: {len(pdb_files)} files\n")

all_descriptors = []

for i, pdb_file in enumerate(sorted(pdb_files), 1):
    protein_name = os.path.basename(pdb_file).replace('_-_prepared.pdb', '').replace('_', ' ')
    temp_output = f"temp_{i}.csv"
    
    print(f"[{i}/{len(pdb_files)}] {protein_name}...")
    
    # Run ProDes
    cmd = f'python -m prodes "{pdb_file}" "{temp_output}" --ph 6.0'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(temp_output):
            # Read the output
            df = pd.read_csv(temp_output, index_col=0)
            df.index = [protein_name]
            all_descriptors.append(df)
            print(f"  [OK] Success - {len(df.columns)} descriptors")
            
            # Clean up temp file
            os.remove(temp_output)
        else:
            print(f"  [FAIL] Failed")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
    
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Exceeded 10 minutes - skipping")
        if os.path.exists(temp_output):
            os.remove(temp_output)
    except Exception as e:
        print(f"  [ERROR] {str(e)[:100]}")

# Combine all descriptors
if all_descriptors:
    print("\n" + "="*100)
    print("COMBINING RESULTS")
    print("="*100)
    
    df_combined = pd.concat(all_descriptors, axis=0)
    df_combined.to_csv(output_csv)
    
    print(f"\n[OK] Saved {len(df_combined)} proteins with {len(df_combined.columns)} descriptors")
    print(f"  Output: {output_csv}")
    
    # Show summary
    print("\nProteins:")
    for protein in df_combined.index:
        print(f"  - {protein}")
    
    print("\n" + "="*100)
else:
    print("\n[FAIL] No descriptors calculated")

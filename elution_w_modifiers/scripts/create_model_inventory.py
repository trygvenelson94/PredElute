#!/usr/bin/env python3
"""
Create model inventory from trained_models_final directory
"""

import os
import pickle
import pandas as pd

model_dir = 'trained_models_final'

# Experiment names mapping
exp_names = {
    1: 'Arginine on CEX (pH 6)',
    2: 'Arginine on Capto MMC (pH 6)',
    3: 'Guanidine on CEX (pH 6)',
    4: 'Guanidine on Capto MMC (pH 6)',
    5: 'Different Resins (pH 6)',
    6: 'Different Resins (pH 7)',
    7: 'Glycols on SP Sepharose (pH 6)',
    8: 'NaCl on Capto MMC (pH 6)',
    9: 'NaCl on Capto MMC (pH 7)'
}

inventory = []

for filename in os.listdir(model_dir):
    if not filename.endswith('.pkl'):
        continue
    
    filepath = os.path.join(model_dir, filename)
    
    # Parse filename
    # Format: exp{N}_{condition}.pkl
    parts = filename.replace('.pkl', '').split('_', 1)
    exp_num = int(parts[0].replace('exp', ''))
    condition = parts[1] if len(parts) > 1 else 'unknown'
    
    # Load model to get R² and MAE
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        r2 = model_data.get('r2_loo', model_data.get('r2', 'N/A'))
        mae = model_data.get('mae_loo', model_data.get('mae', 'N/A'))
        dataset = model_data.get('dataset_used', 'unknown')
        
        inventory.append({
            'experiment': exp_num,
            'experiment_name': exp_names.get(exp_num, f'Experiment {exp_num}'),
            'condition': condition,
            'model_file': filename,
            'r2_used': r2,
            'mae_used': mae,
            'dataset_used': dataset
        })
        
        r2_str = f"{r2:.4f}" if isinstance(r2, float) else str(r2)
        mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
        print(f"[OK] {filename}: R2={r2_str}, MAE={mae_str}")
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

# Create DataFrame and save
df = pd.DataFrame(inventory)
df = df.sort_values(['experiment', 'condition'])

output_path = os.path.join(model_dir, 'model_inventory.csv')
df.to_csv(output_path, index=False)

print(f"\n[OK] Created inventory: {output_path}")
print(f"  Total models: {len(inventory)}")
print(f"\nModels by R2:")
if len(df) > 0:
    numeric_r2 = pd.to_numeric(df['r2_used'], errors='coerce')
    print(f"  R2 > 0.8: {(numeric_r2 > 0.8).sum()}")
    print(f"  R2 > 0.6: {(numeric_r2 > 0.6).sum()}")
    print(f"  R2 > 0.4: {(numeric_r2 > 0.4).sum()}")
    print(f"  Missing R2: {numeric_r2.isna().sum()}")

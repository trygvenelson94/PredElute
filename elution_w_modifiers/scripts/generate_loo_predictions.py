#!/usr/bin/env python3
"""
Generate predicted vs actual values for all 27 models using LOO validation
Output CSV for creating predicted vs actual plots
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

print("="*120)
print("GENERATING LOO PREDICTIONS FOR ALL 27 MODELS")
print("="*120)

# Load data
df_raw = pd.read_csv('multi_qspr.csv')
df_prodes = pd.read_csv('prodes_descriptors_training_only.csv', index_col=0)

# Parse protein names
protein_names = df_raw.iloc[4:28, 0].values
protein_names = [str(name).strip().lower() for name in protein_names if pd.notna(name) and str(name) not in ['Column1', 'Name', 'nan']]

# Add interaction features
print("\n[1/3] Adding interaction features to ProDes descriptors...")

def add_interaction_features(df_prodes):
    """Add the 11 interaction features"""
    df_aug = df_prodes.copy()
    
    # Safe division function
    def safe_divide(a, b, default=0):
        return np.where(b != 0, a / b, default)
    
    # 1. Charge balance: (ARG+LYS+HIS)/(ASP+GLU)
    df_aug['INT_charge_balance'] = safe_divide(
        df_aug.get('ARGSurfFrac', 0) + df_aug.get('LYSSurfFrac', 0) + df_aug.get('HISSurfFrac', 0),
        df_aug.get('ASPSurfFrac', 0) + df_aug.get('GLUSurfFrac', 0),
        default=1.0
    )
    
    # 2. ARG/LYS ratio
    df_aug['INT_arg_lys_ratio'] = safe_divide(df_aug.get('ARGSurfFrac', 0), df_aug.get('LYSSurfFrac', 0), default=1.0)
    
    # 3. Aromatic sum
    df_aug['INT_aromatic_sum'] = df_aug.get('TRPSurfFrac', 0) + df_aug.get('TYRSurfFrac', 0) + df_aug.get('PHESurfFrac', 0)
    
    # 4. TRP * TYR
    df_aug['INT_trp_tyr'] = df_aug.get('TRPSurfFrac', 0) * df_aug.get('TYRSurfFrac', 0)
    
    # 5. (TRP+TYR) * hydrophobicity
    df_aug['INT_aromatic_hydrophobic'] = (df_aug.get('TRPSurfFrac', 0) + df_aug.get('TYRSurfFrac', 0)) * df_aug.get('SurfMhpMean', 0)
    
    # 6. Aliphatic cluster: ILE*LEU*VAL
    df_aug['INT_aliphatic_cluster'] = df_aug.get('ILESurfFrac', 0) * df_aug.get('LEUSurfFrac', 0) * df_aug.get('VALSurfFrac', 0)
    
    # 7. Polar-positive: (SER+THR)*(ARG+LYS)
    df_aug['INT_polar_positive'] = (df_aug.get('SERSurfFrac', 0) + df_aug.get('THRSurfFrac', 0)) * (df_aug.get('ARGSurfFrac', 0) + df_aug.get('LYSSurfFrac', 0))
    
    # 8. GLN * charge
    df_aug['INT_gln_charge'] = df_aug.get('GLNSurfFrac', 0) * df_aug.get('SurfEpPosSumAverage', 0)
    
    # 9. Shape * ARG
    df_aug['INT_shape_arg'] = df_aug.get('Shape max', 0) * df_aug.get('ARGSurfFrac', 0)
    
    # 10. Pocket * charge
    df_aug['INT_pocket_charge'] = df_aug.get('Shape min', 0) * df_aug.get('SurfEpPosSumAverage', 0)
    
    # 11. pI * charge
    df_aug['INT_pi_charge'] = df_aug.get('Isoelectric point', 0) * df_aug.get('SurfEpPosSumAverage', 0)
    
    return df_aug

df_prodes_aug = add_interaction_features(df_prodes)
print(f"  Original features: {df_prodes.shape[1]}")
print(f"  With interactions: {df_prodes_aug.shape[1]}")

# Experiments
experiments = {
    1: {'name': 'Arginine on CEX (pH 6)', 'conditions': ['0M', '0.025M', '0.1M'], 'columns': [2, 3, 4]},
    2: {'name': 'Arginine on Capto MMC (pH 6)', 'conditions': ['0M', '0.025M', '0.1M'], 'columns': [6, 7, 8]},
    3: {'name': 'Guanidine on CEX (pH 6)', 'conditions': ['0M', '0.025M', '0.1M'], 'columns': [10, 11, 12]},
    4: {'name': 'Guanidine on Capto MMC (pH 6)', 'conditions': ['0M', '0.025M', '0.1M'], 'columns': [14, 15, 16]},
    5: {'name': 'pH 5 - Different resins', 'conditions': ['Capto MMC', 'CM Seph', 'SP Seph'], 'columns': [18, 19, 20]},
    6: {'name': 'pH 6 - Different resins', 'conditions': ['Capto MMC', 'CM Seph', 'SP Seph'], 'columns': [22, 23, 24]},
    7: {'name': 'pH 6 - Modifiers', 'conditions': ['No modifier', '20% ethylene glycol', '20% propylene glycol'], 'columns': [26, 27, 28]},
    8: {'name': 'pH 6 - Arginine on CEX (low conc)', 'conditions': ['0M', '0.01M', '0.025M'], 'columns': [30, 31, 32]},
    9: {'name': 'pH 6 - Guanidine on Capto MMC (low conc)', 'conditions': ['0M', '0.01M', '0.025M'], 'columns': [34, 35, 36]}
}

print("\n[2/3] Running LOO cross-validation for all models...")
print("-"*120)

all_predictions = []

for exp_num, exp_data in experiments.items():
    for cond_idx, condition in enumerate(exp_data['conditions']):
        col_idx = exp_data['columns'][cond_idx]
        model_name = f"exp{exp_num}_{condition.replace(' ', '_').replace('%', 'pct')}"
        
        # Extract elution data
        elution_values = df_raw.iloc[4:28, col_idx].values
        elution_values = [float(x) if pd.notna(x) and str(x) not in ['', 'nan', 'Column1', 'Name'] else np.nan 
                         for x in elution_values]
        
        valid_indices = [i for i, (prot, elut) in enumerate(zip(protein_names, elution_values)) 
                        if not np.isnan(elut) and prot in df_prodes_aug.index]
        
        if len(valid_indices) < 5:
            continue
        
        valid_proteins = [protein_names[i] for i in valid_indices]
        y_true = np.array([elution_values[i] for i in valid_indices])
        X_full = df_prodes_aug.loc[valid_proteins].values
        
        # Remove NaN columns
        nan_mask = ~np.isnan(X_full).any(axis=0)
        X_clean = X_full[:, nan_mask]
        
        if X_clean.shape[1] == 0:
            continue
        
        # LOO CV - store predictions for each protein
        for loo_idx in range(len(valid_proteins)):
            train_mask = np.ones(len(valid_proteins), dtype=bool)
            train_mask[loo_idx] = False
            
            X_train = X_clean[train_mask]
            y_train = y_true[train_mask]
            X_test = X_clean[~train_mask]
            y_test_val = y_true[loo_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # PCA
            n_components = min(10, len(y_train)-1, X_train_scaled.shape[1])
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            
            # Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_pca, y_train)
            y_pred_val = ridge.predict(X_test_pca)[0]
            
            # Store result
            all_predictions.append({
                'model': model_name,
                'experiment': exp_num,
                'experiment_name': exp_data['name'],
                'condition': condition,
                'protein': valid_proteins[loo_idx],
                'actual': y_test_val,
                'predicted': y_pred_val,
                'residual': y_test_val - y_pred_val,
                'abs_error': abs(y_test_val - y_pred_val)
            })
        
        print(f"  {model_name:<35} | {len(valid_proteins)} proteins")

# Create DataFrame
df_predictions = pd.DataFrame(all_predictions)

print("\n[3/3] Saving results...")
output_file = 'loo_predictions_all_models.csv'
df_predictions.to_csv(output_file, index=False)

print(f"\n[OK] Saved to: {output_file}")
print(f"Total predictions: {len(df_predictions)}")
print(f"Models: {df_predictions['model'].nunique()}")
print(f"Proteins: {df_predictions['protein'].nunique()}")

# Summary statistics
print("\n" + "="*120)
print("SUMMARY STATISTICS BY MODEL:")
print("="*120)
print(f"{'Model':<35} {'N':<5} {'R2':<8} {'MAE':<8} {'RMSE':<8}")
print("-"*120)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

for model in df_predictions['model'].unique():
    model_data = df_predictions[df_predictions['model'] == model]
    r2 = r2_score(model_data['actual'], model_data['predicted'])
    mae = mean_absolute_error(model_data['actual'], model_data['predicted'])
    rmse = np.sqrt(mean_squared_error(model_data['actual'], model_data['predicted']))
    n = len(model_data)
    print(f"{model:<35} {n:<5} {r2:<8.3f} {mae:<8.3f} {rmse:<8.3f}")

print("="*120)
print(f"\nYou can now create predicted vs actual plots for each model!")
print(f"Example: Filter by 'model' column and plot 'actual' vs 'predicted'")
print("="*120)

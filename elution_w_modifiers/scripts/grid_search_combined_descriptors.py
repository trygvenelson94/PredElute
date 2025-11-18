#!/usr/bin/env python3
"""
Comprehensive grid search for combined Schrodinger + ProDes descriptors
Includes standardization, PCA, Ridge regression, and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("GRID SEARCH WITH COMBINED SCHRODINGER + PRODES DESCRIPTORS")
print("="*100)

# Load data
print("\n[1/7] Loading data...")
df_prodes = pd.read_csv('prodes_descriptors_training_only.csv', index_col=0)
df_schrodinger = pd.read_csv('schrodinger_descriptors.csv')

# Load elution data
df_multi = pd.read_csv('multi_qspr.csv', header=[0,1,2,3,4])
protein_names = df_multi.iloc[:, 0].values
df_elution = df_multi.iloc[:, 1:37]
df_elution.index = protein_names
df_elution.columns = [f'col_{i}' for i in range(len(df_elution.columns))]
df_elution = df_elution.apply(pd.to_numeric, errors='coerce')

print(f"  ProDes: {df_prodes.shape}")
print(f"  Schrodinger: {df_schrodinger.shape}")
print(f"  Elution: {df_elution.shape}")

# Standardize protein names
print("\n[2/7] Matching protein names...")
df_schrodinger['Name'] = df_schrodinger['Name'].str.lower().str.strip()
df_schrodinger['Name'] = df_schrodinger['Name'].str.replace(' - prepared', '')
df_schrodinger = df_schrodinger.set_index('Name')
df_prodes.index = df_prodes.index.str.lower().str.strip()
df_elution.index = df_elution.index.str.lower().str.strip()

# Find common proteins
common_proteins = set(df_prodes.index) & set(df_schrodinger.index) & set(df_elution.index)
common_proteins_sorted = sorted(common_proteins)

print(f"  Common proteins: {len(common_proteins)}")

# Merge descriptors
print("\n[3/7] Merging descriptors...")
df_prodes_common = df_prodes.loc[common_proteins_sorted]
df_schrodinger_common = df_schrodinger.loc[common_proteins_sorted]
df_elution_common = df_elution.loc[common_proteins_sorted]

df_combined = pd.concat([df_prodes_common, df_schrodinger_common], axis=1)

# Clean data
df_combined = df_combined.replace([np.inf, -np.inf], np.nan)
cols_with_nan = df_combined.columns[df_combined.isna().any()].tolist()
if cols_with_nan:
    print(f"  Removing {len(cols_with_nan)} columns with NaN/inf")
    df_combined = df_combined.drop(columns=cols_with_nan)

print(f"  Final features: {len(df_combined.columns)}")

# Grid search parameters
print("\n[4/7] Setting up grid search...")
n_components_range = [3, 5, 7, 10, 15, 20]
alpha_range = [0.01, 0.1, 1.0, 10.0, 100.0]

print(f"  PCA components: {n_components_range}")
print(f"  Ridge alpha: {alpha_range}")
print(f"  Total combinations: {len(n_components_range) * len(alpha_range)}")

# Train models
print("\n[5/7] Training models with grid search...")
print("="*100)

output_dir = 'trained_models_combined_grid_search'
os.makedirs(output_dir, exist_ok=True)

all_results = []
loo = LeaveOneOut()

for col_idx, column in enumerate(df_elution_common.columns, 1):
    print(f"\n[{col_idx}/{len(df_elution_common.columns)}] {column}")
    print("-"*100)
    
    y = df_elution_common[column].values
    X = df_combined.values
    
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 5:
        print(f"  [SKIP] Only {valid_mask.sum()} valid samples")
        continue
    
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    proteins_valid = [common_proteins_sorted[i] for i in range(len(valid_mask)) if valid_mask[i]]
    
    print(f"  Samples: {len(y_valid)}")
    
    # Grid search
    best_score = -np.inf
    best_params = {}
    best_predictions = None
    
    grid_results = []
    
    for n_comp in n_components_range:
        if n_comp > len(X_valid) or n_comp > X_valid.shape[1]:
            continue
            
        for alpha in alpha_range:
            # LOO cross-validation
            y_pred_loo = np.zeros(len(y_valid))
            
            for train_idx, test_idx in loo.split(X_valid):
                X_train, X_test = X_valid[train_idx], X_valid[test_idx]
                y_train, y_test = y_valid[train_idx], y_valid[test_idx]
                
                # Standardize
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # PCA
                pca = PCA(n_components=n_comp)
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                
                # Ridge
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_train_pca, y_train)
                
                y_pred_loo[test_idx] = ridge.predict(X_test_pca)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred_loo))
            r2 = r2_score(y_valid, y_pred_loo)
            
            grid_results.append({
                'n_components': n_comp,
                'alpha': alpha,
                'rmse': rmse,
                'r2': r2
            })
            
            # Track best
            if r2 > best_score:
                best_score = r2
                best_params = {'n_components': n_comp, 'alpha': alpha}
                best_predictions = y_pred_loo.copy()
    
    # Display grid search results
    df_grid = pd.DataFrame(grid_results)
    print(f"\n  Grid search complete: {len(grid_results)} combinations tested")
    print(f"  Best params: n_components={best_params['n_components']}, alpha={best_params['alpha']}")
    print(f"  Best R²: {best_score:.4f}")
    
    # Train final model with best params
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X_valid)
    
    pca_final = PCA(n_components=best_params['n_components'])
    X_pca_final = pca_final.fit_transform(X_scaled_final)
    
    ridge_final = Ridge(alpha=best_params['alpha'])
    ridge_final.fit(X_pca_final, y_valid)
    
    # Calculate final metrics
    rmse_final = np.sqrt(mean_squared_error(y_valid, best_predictions))
    r2_final = r2_score(y_valid, best_predictions)
    
    print(f"  Final RMSE: {rmse_final:.4f}")
    print(f"  Final R²: {r2_final:.4f}")
    
    # Save model
    model_data = {
        'model': ridge_final,
        'scaler': scaler_final,
        'pca': pca_final,
        'feature_names': df_combined.columns.tolist(),
        'proteins': proteins_valid,
        'y_true': y_valid,
        'y_pred_loo': best_predictions,
        'rmse_loo': rmse_final,
        'r2_loo': r2_final,
        'best_params': best_params,
        'grid_results': df_grid,
        'column_name': column
    }
    
    model_file = os.path.join(output_dir, f'{column}.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    all_results.append({
        'column': column,
        'n_samples': len(y_valid),
        'n_features': X_valid.shape[1],
        'best_n_components': best_params['n_components'],
        'best_alpha': best_params['alpha'],
        'rmse_loo': rmse_final,
        'r2_loo': r2_final
    })

# Save results
print("\n" + "="*100)
print("[6/7] SUMMARY")
print("="*100)

df_results = pd.DataFrame(all_results)
df_results.to_csv('model_performance_combined_grid_search.csv', index=False)

print(f"\nTrained {len(all_results)} models")
print(f"Saved to: {output_dir}/")

print(f"\nOverall performance:")
print(f"  Mean RMSE: {df_results['rmse_loo'].mean():.4f} ± {df_results['rmse_loo'].std():.4f}")
print(f"  Mean R²: {df_results['r2_loo'].mean():.4f} ± {df_results['r2_loo'].std():.4f}")

print(f"\nBest performing models (by R²):")
for idx, row in df_results.nlargest(5, 'r2_loo').iterrows():
    print(f"  {row['column']:15s} R²={row['r2_loo']:.4f}  RMSE={row['rmse_loo']:.4f}  " +
          f"(n_comp={row['best_n_components']}, alpha={row['best_alpha']})")

print(f"\nHyperparameter distribution:")
print(f"  PCA components:")
for n_comp in sorted(df_results['best_n_components'].unique()):
    count = (df_results['best_n_components'] == n_comp).sum()
    print(f"    {n_comp:2d}: {count:2d} models")

print(f"\n  Ridge alpha:")
for alpha in sorted(df_results['best_alpha'].unique()):
    count = (df_results['best_alpha'] == alpha).sum()
    print(f"    {alpha:6.2f}: {count:2d} models")

print("\n" + "="*100)
print("[7/7] CREATING OPTIMAL MODELS FILE")
print("="*100)

# Create optimal models summary (like optimal_models_v2_with_interactions.csv)
optimal_data = []
for idx, row in df_results.iterrows():
    optimal_data.append({
        'Column': row['column'],
        'Best_n_components': row['best_n_components'],
        'Best_alpha': row['best_alpha'],
        'LOO_RMSE': row['rmse_loo'],
        'LOO_R2': row['r2_loo'],
        'n_samples': row['n_samples'],
        'n_features': row['n_features']
    })

df_optimal = pd.DataFrame(optimal_data)
df_optimal.to_csv('optimal_models_combined_descriptors.csv', index=False)

print(f"Saved optimal parameters to: optimal_models_combined_descriptors.csv")

print("\n" + "="*100)
print("COMPLETE")
print("="*100)
print(f"Models saved to: {output_dir}/")
print(f"Performance summary: model_performance_combined_grid_search.csv")
print(f"Optimal parameters: optimal_models_combined_descriptors.csv")
print("="*100)

#!/usr/bin/env python3
"""
Extract top 10 features for each of the 27 models (with interaction features)
and create a feature importance matrix
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load training data
df_prodes = pd.read_csv('prodes_descriptors_training_only.csv', index_col=0)
df_qspr = pd.read_csv('training_data.csv')

# Add interaction features
def add_interaction_features(df_prodes):
    """Add the 11 interaction features"""
    df_aug = df_prodes.copy()
    
    # Safe division function
    def safe_divide(a, b, default=0):
        b = np.array(b)
        a = np.array(a)
        result = np.full_like(a, default, dtype=float)
        mask = b != 0
        result[mask] = a[mask] / b[mask]
        return result
    
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
    
    # 7. Shape * charge
    df_aug['INT_shape_charge'] = df_aug.get('Shape_max', 0) * df_aug.get('FormalCharge', 0)
    
    # 8. pI * positive charge fraction
    df_aug['INT_pi_positive'] = df_aug.get('IsoelectricPoint', 0) * (
        df_aug.get('ARGSurfFrac', 0) + df_aug.get('LYSSurfFrac', 0) + df_aug.get('HISSurfFrac', 0)
    )
    
    # 9. pI^2 (non-linear pI effect)
    df_aug['INT_pi_squared'] = df_aug.get('IsoelectricPoint', 0) ** 2
    
    # 10. Hydrophobic * aromatic
    df_aug['INT_hydrophobic_aromatic'] = df_aug.get('SurfMhpMean', 0) * (
        df_aug.get('TRPSurfFrac', 0) + df_aug.get('PHESurfFrac', 0)
    )
    
    # 11. Charge density: FormalCharge / SurfaceArea
    df_aug['INT_charge_density'] = safe_divide(df_aug.get('FormalCharge', 0), df_aug.get('SurfaceArea', 0), default=0)
    
    return df_aug

# Add interaction features
df_prodes_aug = add_interaction_features(df_prodes)

print("="*100)
print("EXTRACTING TOP 10 FEATURES FOR EACH MODEL (WITH INTERACTION FEATURES)")
print("="*100)
print(f"Total features: {len(df_prodes_aug.columns)} (105 original + 11 interactions)")
print()

# Get list of experiments and conditions
experiments = df_qspr['Experiment'].unique()
conditions_map = {}
for exp in experiments:
    conditions = df_qspr[df_qspr['Experiment'] == exp]['Condition'].unique()
    conditions_map[exp] = sorted(conditions)

# Store all feature importances
all_importances = {}
all_top_features = []

model_count = 0
for exp in sorted(experiments):
    for condition in conditions_map[exp]:
        model_count += 1
        model_name = f"exp{exp}_{condition}"
        
        print(f"[{model_count}/27] {model_name}")
        
        # Get elution data for this condition
        mask = (df_qspr['Experiment'] == exp) & (df_qspr['Condition'] == condition)
        elution_data = df_qspr[mask].set_index('Protein')['Elution (M NaCl)']
        
        # Align proteins
        common_proteins = df_prodes_aug.index.intersection(elution_data.index)
        X = df_prodes_aug.loc[common_proteins]
        y = elution_data.loc[common_proteins]
        
        # Remove NaN columns
        X = X.dropna(axis=1)
        
        # Scale and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=min(10, len(common_proteins)-1))
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate feature importance from PCA loadings
        # Weight by explained variance
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        feature_importance = np.abs(loadings).sum(axis=1)
        
        # Normalize
        feature_importance = feature_importance / feature_importance.sum()
        
        # Get top 10 features
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        top_features = X.columns[top_indices].tolist()
        top_scores = feature_importance[top_indices]
        
        all_importances[model_name] = dict(zip(X.columns, feature_importance))
        all_top_features.extend(top_features)
        
        print(f"  Top 10 features:")
        for i, (feat, score) in enumerate(zip(top_features, top_scores), 1):
            print(f"    {i:2d}. {feat:40s} {score:.4f}")
        print()

# Get unique features from all top features
unique_features = sorted(set(all_top_features))

print("="*100)
print("CREATING FEATURE IMPORTANCE MATRIX")
print("="*100)
print(f"Total unique features in top 10 across all models: {len(unique_features)}")
print()

# Create matrix
matrix_data = []
for model_name in sorted(all_importances.keys()):
    row = {'model': model_name}
    for feat in unique_features:
        row[feat] = all_importances[model_name].get(feat, 0)
    matrix_data.append(row)

df_matrix = pd.DataFrame(matrix_data)

# Save to CSV
output_file = 'feature_importance_matrix_with_interactions.csv'
df_matrix.to_csv(output_file, index=False)

print(f"[OK] Saved feature importance matrix to {output_file}")
print(f"  Dimensions: {df_matrix.shape[0]} models x {df_matrix.shape[1]-1} features")

# Summary statistics
print("\n" + "="*100)
print("FEATURE STATISTICS")
print("="*100)

# Count how many models each feature appears in top 10
feature_counts = {}
for feat in unique_features:
    count = sum(1 for model in all_importances.values() if feat in sorted(model, key=model.get, reverse=True)[:10])
    feature_counts[feat] = count

# Sort by frequency
sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

print("\nMost frequently appearing features in top 10:")
for feat, count in sorted_features[:20]:
    is_interaction = 'INT_' in feat
    marker = '[INTERACTION]' if is_interaction else ''
    print(f"  {feat:40s} appears in {count:2d}/27 models {marker}")

# Count interaction features
interaction_features = [f for f in unique_features if 'INT_' in f]
print(f"\n[OK] {len(interaction_features)} interaction features in top 10 pool")
print(f"[OK] {len(unique_features) - len(interaction_features)} original features in top 10 pool")

print("\n" + "="*100)

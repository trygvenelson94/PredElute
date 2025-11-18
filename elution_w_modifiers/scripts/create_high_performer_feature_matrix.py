#!/usr/bin/env python3
"""
Create feature importance matrix for high-performing models (R² > 0.7)
Only includes features from the components actually used in each model
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def get_feature_importance_from_model(model_path, n_components_used):
    """
    Get weighted feature importance for only the components used in the model
    """
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract PCA object
    if isinstance(model_data, dict):
        pca = model_data.get('pca')
        if 'feature_names' in model_data:
            feature_names = model_data['feature_names']
        elif 'features' in model_data:
            feature_names = model_data['features']
        else:
            feature_names = [f"Feature_{i}" for i in range(pca.n_features_in_)]
    else:
        return None, None
    
    # Only use the components actually in the model
    n_components_used = min(n_components_used, pca.n_components_)
    
    # Calculate weighted importance (weighted by variance explained)
    weighted_importance = np.zeros(len(feature_names))
    for pc_idx in range(n_components_used):
        loadings = np.abs(pca.components_[pc_idx])
        variance = pca.explained_variance_ratio_[pc_idx]
        weighted_importance += loadings * variance
    
    # Create dictionary mapping feature to importance
    importance_dict = {feat: imp for feat, imp in zip(feature_names, weighted_importance)}
    
    return importance_dict, feature_names


# Load model inventory
print("="*100)
print("CREATING FEATURE IMPORTANCE MATRIX FOR HIGH-PERFORMING MODELS")
print("="*100)

inventory = pd.read_csv('trained_models_final/model_inventory.csv')

# Filter for R² > 0.7
high_performers = inventory[inventory['r2_used'] > 0.7].copy()

print(f"\nFound {len(high_performers)} models with R² > 0.7\n")

# Collect all feature importances
all_importances = {}
all_features = set()

for idx, row in high_performers.iterrows():
    model_file = f"trained_models_final/{row['model_file']}"
    model_name = row['model_file'].replace('.pkl', '')
    n_components = row['n_components']
    
    print(f"Processing: {model_name} (R² = {row['r2_used']:.3f}, {n_components} components)")
    
    try:
        importance_dict, feature_names = get_feature_importance_from_model(model_file, n_components)
        
        if importance_dict is not None:
            # Create full model name with experiment and condition
            full_name = f"{row['experiment_name']} - {row['condition']}"
            all_importances[full_name] = importance_dict
            all_features.update(importance_dict.keys())
            print(f"  [OK] {len(importance_dict)} features extracted")
        else:
            print(f"  [FAIL] Failed to extract features")
            
    except Exception as e:
        print(f"  [FAIL] Error: {e}")

# Create matrix
print(f"\n{'='*100}")
print(f"Creating matrix with {len(all_features)} unique features across {len(all_importances)} models")
print(f"{'='*100}\n")

# Sort features alphabetically
sorted_features = sorted(all_features)

# Create DataFrame
matrix_data = []

for feature in sorted_features:
    row = {'Feature': feature}
    for model_name, importance_dict in all_importances.items():
        # Use importance if feature exists in this model, otherwise NaN
        row[model_name] = importance_dict.get(feature, np.nan)
    matrix_data.append(row)

df_matrix = pd.DataFrame(matrix_data)

# Calculate summary statistics
df_matrix['Mean'] = df_matrix.iloc[:, 1:].mean(axis=1)
df_matrix['Std'] = df_matrix.iloc[:, 1:-1].std(axis=1)
df_matrix['Min'] = df_matrix.iloc[:, 1:-2].min(axis=1)
df_matrix['Max'] = df_matrix.iloc[:, 1:-3].max(axis=1)
df_matrix['Count_NonZero'] = df_matrix.iloc[:, 1:-4].notna().sum(axis=1)

# Sort by mean importance
df_matrix = df_matrix.sort_values('Mean', ascending=False)

# Save full matrix
df_matrix.to_csv('feature_importance_matrix_high_performers.csv', index=False)
print(f"[OK] Full matrix saved to: feature_importance_matrix_high_performers.csv")

# Create top 50 features version
df_top50 = df_matrix.head(50).copy()
df_top50.to_csv('feature_importance_matrix_high_performers_top50.csv', index=False)
print(f"[OK] Top 50 features saved to: feature_importance_matrix_high_performers_top50.csv")

# Create summary
print(f"\n{'='*100}")
print("TOP 20 FEATURES BY AVERAGE IMPORTANCE")
print(f"{'='*100}\n")

print(f"{'Rank':<6} {'Feature':<45} {'Mean':<12} {'Std':<12} {'Models':<10}")
print("-"*100)

for rank, (idx, row) in enumerate(df_matrix.head(20).iterrows(), 1):
    print(f"{rank:<6} {row['Feature']:<45} {row['Mean']:<12.4f} {row['Std']:<12.4f} {int(row['Count_NonZero'])}/{len(all_importances)}")

# Create a transposed version (models as rows, features as columns) for easier viewing
df_transposed = df_matrix.set_index('Feature').iloc[:, :-5].T  # Exclude summary stats
df_transposed.to_csv('feature_importance_matrix_high_performers_transposed.csv')
print(f"\n[OK] Transposed matrix saved to: feature_importance_matrix_high_performers_transposed.csv")

# Create a heatmap-ready version (top 30 features only)
df_heatmap = df_matrix.head(30).set_index('Feature').iloc[:, :-5]  # Top 30, exclude summary stats
df_heatmap.to_csv('feature_importance_matrix_high_performers_heatmap.csv')
print(f"[OK] Heatmap version (top 30) saved to: feature_importance_matrix_high_performers_heatmap.csv")

print(f"\n{'='*100}")
print("COMPLETE")
print(f"{'='*100}")

print("\nFiles created:")
print("  1. feature_importance_matrix_high_performers.csv - Full matrix (all features)")
print("  2. feature_importance_matrix_high_performers_top50.csv - Top 50 features")
print("  3. feature_importance_matrix_high_performers_transposed.csv - Models as rows")
print("  4. feature_importance_matrix_high_performers_heatmap.csv - Top 30 for visualization")

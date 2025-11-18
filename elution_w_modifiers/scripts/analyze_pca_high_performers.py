#!/usr/bin/env python3
"""
Analyze PCA loadings for only the components used in high-performing models (R² > 0.7)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def analyze_model_components(model_path, n_components_used, feature_names=None, top_n=15):
    """
    Analyze only the PCA components actually used in the model
    """
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract PCA object
    if isinstance(model_data, dict):
        if 'pca' in model_data:
            pca = model_data['pca']
        else:
            return None
    elif hasattr(model_data, 'named_steps') and 'pca' in model_data.named_steps:
        pca = model_data.named_steps['pca']
    else:
        return None
    
    # Get feature names
    if feature_names is None:
        if 'feature_names' in model_data:
            feature_names = model_data['feature_names']
        elif 'features' in model_data:
            feature_names = model_data['features']
        else:
            feature_names = [f"Feature_{i}" for i in range(pca.n_features_in_)]
    
    # Only analyze the components actually used
    n_components_used = min(n_components_used, pca.n_components_)
    
    # Collect results
    results = []
    
    for pc_idx in range(n_components_used):
        loadings = pca.components_[pc_idx]
        explained_var = pca.explained_variance_ratio_[pc_idx]
        
        # Get top features by absolute loading
        abs_loadings = np.abs(loadings)
        top_indices = np.argsort(abs_loadings)[-top_n:][::-1]
        
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'PC': f'PC{pc_idx + 1}',
                'PC_variance': explained_var,
                'Rank': rank,
                'Feature': feature_names[idx],
                'Loading': loadings[idx],
                'Abs_Loading': abs_loadings[idx]
            })
    
    # Calculate overall importance (weighted by variance, only for used components)
    weighted_importance = np.zeros(len(feature_names))
    for pc_idx in range(n_components_used):
        loadings = np.abs(pca.components_[pc_idx])
        variance = pca.explained_variance_ratio_[pc_idx]
        weighted_importance += loadings * variance
    
    return pd.DataFrame(results), weighted_importance, feature_names


# Load model inventory
print("="*100)
print("ANALYZING HIGH-PERFORMING MODELS (R² > 0.7)")
print("="*100)

inventory = pd.read_csv('trained_models_final/model_inventory.csv')

# Filter for R² > 0.7
high_performers = inventory[inventory['r2_used'] > 0.7].copy()

print(f"\nFound {len(high_performers)} models with R² > 0.7")
print(f"\n{'Model':<40} {'R²':<8} {'Components Used':<18} {'Total Variance':<15}")
print("-"*100)

all_results = []
all_importances = {}

for idx, row in high_performers.iterrows():
    model_file = f"trained_models_final/{row['model_file']}"
    n_components = row['n_components']
    r2 = row['r2_used']
    
    print(f"{row['model_file']:<40} {r2:<8.3f} {n_components:<18}", end='')
    
    try:
        df_results, weighted_importance, feature_names = analyze_model_components(
            model_file, n_components, top_n=15
        )
        
        # Calculate total variance explained by used components
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        if isinstance(model_data, dict):
            pca = model_data.get('pca')
        else:
            pca = model_data.named_steps.get('pca')
        total_var = pca.explained_variance_ratio_[:n_components].sum()
        
        print(f"{total_var:<15.2%}")
        
        df_results['Model'] = row['model_file'].replace('.pkl', '')
        df_results['Experiment'] = row['experiment_name']
        df_results['Condition'] = row['condition']
        df_results['R2'] = r2
        
        all_results.append(df_results)
        all_importances[row['model_file']] = {
            'importance': weighted_importance,
            'features': feature_names,
            'n_components': n_components,
            'r2': r2,
            'experiment': row['experiment_name'],
            'condition': row['condition']
        }
        
    except Exception as e:
        print(f"ERROR: {e}")

# Combine all results
combined_df = pd.concat(all_results, ignore_index=True)
combined_df.to_csv('pca_loadings_high_performers_only.csv', index=False)

print(f"\n[OK] Detailed loadings saved to: pca_loadings_high_performers_only.csv")

# Create summary of most important features across all high-performing models
print("\n" + "="*100)
print("OVERALL TOP FEATURES ACROSS ALL HIGH-PERFORMING MODELS")
print("="*100)

# Collect all unique features and their importances
feature_importance_dict = {}

for model_name, data in all_importances.items():
    for feat_idx, feat_name in enumerate(data['features']):
        if feat_name not in feature_importance_dict:
            feature_importance_dict[feat_name] = []
        feature_importance_dict[feat_name].append(data['importance'][feat_idx])

# Average importance for each feature (across models that have it)
avg_feature_importance = {
    feat: np.mean(imps) for feat, imps in feature_importance_dict.items()
}

# Sort by average importance
sorted_features = sorted(avg_feature_importance.items(), key=lambda x: x[1], reverse=True)
top_overall = sorted_features[:30]

print(f"\n{'Rank':<6} {'Feature':<45} {'Avg Weighted Importance':<25} {'Appears in Top 5 (models)':<10}")
print("-"*100)

for rank, (feat_name, avg_imp) in enumerate(top_overall, 1):
    # Count how many models have this feature in top 5
    top5_count = 0
    for model_name, data in all_importances.items():
        if feat_name in data['features']:
            feat_idx = list(data['features']).index(feat_name)
            model_top5_indices = np.argsort(data['importance'])[-5:]
            if feat_idx in model_top5_indices:
                top5_count += 1
    
    print(f"{rank:<6} {feat_name:<45} {avg_imp:<25.4f} {top5_count}/{len(all_importances)}")

# Save summary
summary_df = pd.DataFrame({
    'Rank': range(1, len(top_overall) + 1),
    'Feature': [feat for feat, _ in top_overall],
    'Avg_Weighted_Importance': [imp for _, imp in top_overall]
})
summary_df.to_csv('top_features_high_performers.csv', index=False)

print(f"\n[OK] Summary saved to: top_features_high_performers.csv")

# Create per-model summary
print("\n" + "="*100)
print("TOP 10 FEATURES PER HIGH-PERFORMING MODEL")
print("="*100)

per_model_summary = []

for model_name, data in sorted(all_importances.items(), key=lambda x: x[1]['r2'], reverse=True):
    print(f"\n{model_name.replace('.pkl', '')} (R² = {data['r2']:.3f}, {data['n_components']} components)")
    print(f"  {data['experiment']} - {data['condition']}")
    print(f"  {'-'*80}")
    
    top10 = np.argsort(data['importance'])[-10:][::-1]
    
    for rank, idx in enumerate(top10, 1):
        feat_name = data['features'][idx]
        imp = data['importance'][idx]
        print(f"  {rank:2d}. {feat_name:<40} {imp:.4f}")
        
        per_model_summary.append({
            'Model': model_name.replace('.pkl', ''),
            'Experiment': data['experiment'],
            'Condition': data['condition'],
            'R2': data['r2'],
            'N_Components': data['n_components'],
            'Rank': rank,
            'Feature': feat_name,
            'Weighted_Importance': imp
        })

per_model_df = pd.DataFrame(per_model_summary)
per_model_df.to_csv('top_features_per_model_high_performers.csv', index=False)

print(f"\n[OK] Per-model summary saved to: top_features_per_model_high_performers.csv")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print("\nFiles created:")
print("  1. pca_loadings_high_performers_only.csv - Detailed loadings for used components")
print("  2. top_features_high_performers.csv - Overall top 30 features")
print("  3. top_features_per_model_high_performers.csv - Top 10 per model")

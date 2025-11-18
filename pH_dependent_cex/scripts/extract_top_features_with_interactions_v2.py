#!/usr/bin/env python3
"""
Extract top 10 features from each trained model with interactions
based on coefficient magnitude
"""

import os
import pickle
import pandas as pd
import numpy as np
from glob import glob

model_dir = 'trained_models_with_interactions'
output_file = 'top_features_by_model_with_interactions.csv'
matrix_file = 'feature_importance_matrix_with_interactions.csv'

print("="*100)
print("EXTRACTING TOP 10 FEATURES FROM TRAINED MODELS (WITH INTERACTIONS)")
print("="*100)

# Find all model files
model_files = glob(os.path.join(model_dir, '*.pkl'))
print(f"\nFound {len(model_files)} model files\n")

results = []
all_importances = {}
all_top_features = []

for model_path in sorted(model_files):
    model_name = os.path.basename(model_path).replace('.pkl', '')
    
    print(f"Processing: {model_name}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract information
    ridge_model = model_data['model']
    pca = model_data['pca']
    feature_names = model_data.get('feature_names', None)
    
    if feature_names is None:
        print(f"  [WARNING] No feature names saved, skipping")
        continue
    
    # Get Ridge coefficients (after PCA)
    ridge_coefs = ridge_model.coef_
    
    # Get PCA components (maps PC space back to original features)
    pca_components = pca.components_  # Shape: (n_components, n_features)
    
    # Calculate feature importance by projecting Ridge coefficients back through PCA
    # Feature importance = sum of (PC coefficient × PCA loading for that feature)
    feature_importance = np.abs(ridge_coefs @ pca_components)
    
    # Normalize to sum to 1
    feature_importance = feature_importance / feature_importance.sum()
    
    # Store all importances
    all_importances[model_name] = dict(zip(feature_names, feature_importance))
    
    # Get top 10 features
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = feature_importance[top_indices]
    
    all_top_features.extend(top_features)
    
    print(f"  Top 10 features:")
    for i, (feat, score) in enumerate(zip(top_features, top_scores), 1):
        is_interaction = 'INT_' in feat
        marker = '[INT]' if is_interaction else ''
        print(f"    {i:2d}. {feat:45s} {score:.4f} {marker}")
    
    # Store for CSV
    row = {'model': model_name}
    for i, (feat, score) in enumerate(zip(top_features, top_scores), 1):
        row[f'feature_{i}'] = feat
        row[f'importance_{i}'] = score
    results.append(row)
    print()

# Save individual model results
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)
print(f"[OK] Saved top features to {output_file}")

# Create feature importance matrix
print("\n" + "="*100)
print("CREATING FEATURE IMPORTANCE MATRIX")
print("="*100)

unique_features = sorted(set(all_top_features))
print(f"Total unique features in top 10 across all models: {len(unique_features)}\n")

# Create matrix
matrix_data = []
for model_name in sorted(all_importances.keys()):
    row = {'model': model_name}
    for feat in unique_features:
        row[feat] = all_importances[model_name].get(feat, 0)
    matrix_data.append(row)

df_matrix = pd.DataFrame(matrix_data)
df_matrix.to_csv(matrix_file, index=False)

print(f"[OK] Saved feature importance matrix to {matrix_file}")
print(f"  Dimensions: {df_matrix.shape[0]} models x {df_matrix.shape[1]-1} features")

# Summary statistics
print("\n" + "="*100)
print("FEATURE STATISTICS")
print("="*100)

# Count how many models each feature appears in top 10
feature_counts = {}
for feat in unique_features:
    count = sum(1 for model in all_importances.values() 
                if feat in sorted(model, key=model.get, reverse=True)[:10])
    feature_counts[feat] = count

# Sort by frequency
sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

print("\nMost frequently appearing features in top 10:")
for feat, count in sorted_features[:25]:
    is_interaction = 'INT_' in feat
    marker = '[INTERACTION]' if is_interaction else ''
    print(f"  {feat:45s} appears in {count:2d}/27 models {marker}")

# Count interaction features
interaction_features = [f for f in unique_features if 'INT_' in f]
original_features = [f for f in unique_features if 'INT_' not in f]

print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")
print(f"Total unique features in top 10 pool: {len(unique_features)}")
print(f"  - Interaction features: {len(interaction_features)}")
print(f"  - Original features: {len(original_features)}")

if interaction_features:
    print(f"\nInteraction features that made it to top 10:")
    for feat in sorted(interaction_features):
        count = feature_counts[feat]
        print(f"  {feat:45s} ({count} models)")

print("\n" + "="*100)

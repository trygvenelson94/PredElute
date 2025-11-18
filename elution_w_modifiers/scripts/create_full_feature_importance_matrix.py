#!/usr/bin/env python3
"""
Create comprehensive feature importance matrix
Rows = All unique features across all models
Columns = All 27 models
Values = Feature importance for each feature in each model
"""

import os
import pickle
import pandas as pd
import numpy as np
from glob import glob

model_dir = 'trained_models'
output_file = 'feature_importance_matrix.csv'

print("="*100)
print("CREATING COMPREHENSIVE FEATURE IMPORTANCE MATRIX")
print("="*100)

# Find all model files
model_files = glob(os.path.join(model_dir, '*.pkl'))
print(f"\nFound {len(model_files)} model files")

# First pass: collect all unique features across all models
all_features = set()
model_data_cache = {}

for model_path in sorted(model_files):
    model_name = os.path.basename(model_path).replace('.pkl', '')
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    feature_names = model_data.get('feature_names', None)
    
    if feature_names is None:
        print(f"[WARNING] {model_name}: No feature names, skipping")
        continue
    
    # Cache the model data
    model_data_cache[model_name] = model_data
    
    # Add features to the set
    all_features.update(feature_names)

all_features = sorted(all_features)
print(f"\nTotal unique features across all models: {len(all_features)}")

# Second pass: calculate importance for all features in each model
importance_matrix = {}

for model_name, model_data in model_data_cache.items():
    ridge_model = model_data['model']
    pca = model_data['pca']
    feature_names = model_data['feature_names']
    
    # Get Ridge coefficients
    ridge_coefs = ridge_model.coef_
    
    # Get PCA components
    pca_components = pca.components_
    
    # Calculate feature importance by projecting back through PCA
    feature_importance = np.abs(ridge_coefs @ pca_components)
    
    # Create mapping from feature name to importance
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    
    # For features not in this model (filtered out during cleaning), set to 0 or NaN
    # Using 0 makes more sense - they weren't important enough to keep
    model_importance = {}
    for feature in all_features:
        model_importance[feature] = feature_importance_dict.get(feature, 0.0)
    
    importance_matrix[model_name] = model_importance
    
    print(f"  {model_name}: Calculated importance for {len(all_features)} features")

# Create DataFrame
df_matrix = pd.DataFrame(importance_matrix)

# Sort by sum of importance across all models (most important features first)
df_matrix['total_importance'] = df_matrix.sum(axis=1)
df_matrix = df_matrix.sort_values('total_importance', ascending=False)
df_matrix = df_matrix.drop(columns=['total_importance'])

# Reorder columns (models) by experiment number
model_cols = sorted(df_matrix.columns, key=lambda x: (int(x.split('_')[0].replace('exp', '')), x))
df_matrix = df_matrix[model_cols]

# Save to CSV
df_matrix.to_csv(output_file)

print(f"\n[OK] Feature importance matrix saved to: {output_file}")
print(f"  Dimensions: {df_matrix.shape[0]} features × {df_matrix.shape[1]} models")
print("="*100)

# Summary statistics
print("\nTOP 20 FEATURES (by total importance across all models):")
print("-"*100)
top_20 = df_matrix.sum(axis=1).sort_values(ascending=False).head(20)
for i, (feature, total) in enumerate(top_20.items(), 1):
    # Count how many models have this in top 12
    top_12_count = sum(1 for col in df_matrix.columns if df_matrix.loc[feature, col] > 0)
    avg_importance = df_matrix.loc[feature].mean()
    max_importance = df_matrix.loc[feature].max()
    
    print(f"{i:2d}. {feature:30s} | Total: {total:7.3f} | Avg: {avg_importance:6.4f} | "
          f"Max: {max_importance:6.4f} | In {top_12_count}/27 models")

print("\n" + "="*100)
print("BOTTOM 20 FEATURES (least important):")
print("-"*100)
bottom_20 = df_matrix.sum(axis=1).sort_values(ascending=True).head(20)
for i, (feature, total) in enumerate(bottom_20.items(), 1):
    models_with_feature = sum(1 for col in df_matrix.columns if df_matrix.loc[feature, col] > 0)
    print(f"{i:2d}. {feature:30s} | Total: {total:7.3f} | In {models_with_feature}/27 models")

print("="*100)

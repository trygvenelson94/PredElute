#!/usr/bin/env python3
"""
Export predicted vs actual values from the pH-dependent model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import VarianceThreshold

# Optimal parameters from grid search
VAR_THRESH = 0.1
CORR_THRESH = 0.90
N_COMPONENTS = 15
ALPHA = 1.0
SEED = 42

print("="*120)
print("EXPORTING PREDICTED VS ACTUAL VALUES")
print("="*120)
print(f"\nUsing optimal parameters:")
print(f"  var_thresh = {VAR_THRESH}")
print(f"  corr_thresh = {CORR_THRESH}")
print(f"  n_components = {N_COMPONENTS}")
print(f"  alpha = {ALPHA}")

# Load data
print("\n[1/3] Loading data...")
df_prodes = pd.read_csv('sp_sepharose_hp_descriptors_complete.csv', index_col=0)
df_schrodinger = pd.read_csv('schrodinger_descriptors.csv')
df_nacl = pd.read_csv('sp_sepharose_hp_nacl_concentrations.csv')

# Merge descriptors
df_schrodinger['Name'] = df_schrodinger['Name'].str.lower().str.strip().str.replace(' - prepared', '')
df_schrodinger = df_schrodinger.set_index('Name')
df_prodes.index = df_prodes.index.str.lower().str.strip()

common_desc_proteins = sorted(set(df_prodes.index) & set(df_schrodinger.index))
df_prodes_common = df_prodes.loc[common_desc_proteins]
df_schrodinger_common = df_schrodinger.loc[common_desc_proteins]
df_descriptors = pd.concat([df_prodes_common, df_schrodinger_common], axis=1)

# Prepare elution data
df_nacl['protein_name'] = df_nacl['protein_name'].str.lower().str.strip()
df_nacl = df_nacl.set_index('protein_name')
for col in ['pH4', 'pH5', 'pH6', 'pH7', 'pH8']:
    if col in df_nacl.columns:
        df_nacl[col] = pd.to_numeric(df_nacl[col], errors='coerce')

# Find common proteins
common_proteins = sorted(set(df_descriptors.index) & set(df_nacl.index))
df_descriptors = df_descriptors.loc[common_proteins]
df_nacl = df_nacl.loc[common_proteins]

print(f"  Proteins: {len(common_proteins)}")
print(f"  Initial descriptors: {df_descriptors.shape[1]}")

# Apply filtering
print("\n[2/3] Applying feature filtering...")

# Remove NaN/inf
cols_with_nan = df_descriptors.columns[df_descriptors.isna().any()].tolist()
if cols_with_nan:
    df_descriptors = df_descriptors.drop(columns=cols_with_nan)
df_descriptors = df_descriptors.replace([np.inf, -np.inf], np.nan)
cols_with_nan = df_descriptors.columns[df_descriptors.isna().any()].tolist()
if cols_with_nan:
    df_descriptors = df_descriptors.drop(columns=cols_with_nan)

# Variance threshold
if VAR_THRESH > 0:
    selector = VarianceThreshold(threshold=VAR_THRESH)
    X_filt = selector.fit_transform(df_descriptors.values)
    selected_features = df_descriptors.columns[selector.get_support()].tolist()
    df_descriptors = pd.DataFrame(X_filt, index=df_descriptors.index, columns=selected_features)

print(f"  After variance threshold: {df_descriptors.shape[1]} features")

# Correlation threshold
if CORR_THRESH < 1.0 and len(df_descriptors.columns) > 1:
    corr_matrix = df_descriptors.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESH)]
    df_descriptors = df_descriptors.drop(columns=to_drop)

print(f"  After correlation threshold: {df_descriptors.shape[1]} features")

# Create expanded dataset
print("\n[3/3] Running LOO CV and generating predictions...")

ph_values = np.array([4, 5, 6, 7, 8])
ph_columns = ['pH4', 'pH5', 'pH6', 'pH7', 'pH8']

X = df_descriptors.values
protein_names = df_descriptors.index.tolist()

X_expanded = []
y_expanded = []
protein_ph_pairs = []

for prot_idx, prot in enumerate(protein_names):
    for ph_idx, (ph, ph_col) in enumerate(zip(ph_values, ph_columns)):
        y_val = df_nacl.loc[prot, ph_col]
        if not np.isnan(y_val):
            x_with_ph = np.concatenate([X[prot_idx], [ph]])
            X_expanded.append(x_with_ph)
            y_expanded.append(y_val)
            protein_ph_pairs.append((prot, ph))

X_expanded = np.array(X_expanded)
y_expanded = np.array(y_expanded)

print(f"  Total samples: {len(y_expanded)}")

# LOO CV
loo = LeaveOneOut()
y_pred_loo = np.zeros(len(y_expanded))

for train_idx, test_idx in loo.split(X_expanded):
    X_train, X_test = X_expanded[train_idx], X_expanded[test_idx]
    y_train, y_test = y_expanded[train_idx], y_expanded[test_idx]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Ridge
    ridge = Ridge(alpha=ALPHA, random_state=SEED)
    ridge.fit(X_train_pca, y_train)
    y_pred_loo[test_idx] = ridge.predict(X_test_pca)

# Create results dataframe
results_df = pd.DataFrame({
    'protein': [p[0] for p in protein_ph_pairs],
    'pH': [p[1] for p in protein_ph_pairs],
    'actual_NaCl_M': y_expanded,
    'predicted_NaCl_M': y_pred_loo,
    'residual_M': y_expanded - y_pred_loo,
    'abs_error_M': np.abs(y_expanded - y_pred_loo),
    'percent_error': 100 * np.abs(y_expanded - y_pred_loo) / y_expanded
})

# Sort by protein and pH
results_df = results_df.sort_values(['protein', 'pH'])

# Save
output_file = 'ph_model_predictions_vs_actual.csv'
results_df.to_csv(output_file, index=False)

print(f"\n✓ Saved predictions to: {output_file}")

# Summary statistics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_expanded, y_pred_loo)
rmse = np.sqrt(mean_squared_error(y_expanded, y_pred_loo))
mae = mean_absolute_error(y_expanded, y_pred_loo)

print(f"\nModel Performance:")
print(f"  R² = {r2:.4f}")
print(f"  RMSE = {rmse:.4f} M")
print(f"  MAE = {mae:.4f} M")
print(f"  Mean % Error = {results_df['percent_error'].mean():.2f}%")

print(f"\nSample of predictions:")
print(results_df.head(10).to_string(index=False))

print("\n" + "="*120)
print("COMPLETE - Ready for plotting!")
print("="*120)

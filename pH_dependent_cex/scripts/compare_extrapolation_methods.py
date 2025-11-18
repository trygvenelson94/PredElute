#!/usr/bin/env python3
"""
Compare three extrapolation methods for pH-dependent elution prediction:
1. Two-stage sigmoid fitting
2. Henderson-Hasselbalch fitting on dense pH predictions
3. Polynomial pH features

Tests extrapolation from pH 4-7 training data to pH 8 (validation)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("EXTRAPOLATION METHOD COMPARISON: pH 9-10 Prediction")
print("="*120)
print("\nStrategy: Train on pH 4-8, extrapolate to pH 9-10")
print("Note: pH 9-10 predictions cannot be validated (no experimental data)")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

df_prodes = pd.read_csv('sp_sepharose_hp_descriptors_complete.csv', index_col=0)
df_schrodinger = pd.read_csv('schrodinger_descriptors.csv')
df_nacl = pd.read_csv('sp_sepharose_hp_nacl_concentrations.csv')

# Merge descriptors
df_schrodinger['Name'] = df_schrodinger['Name'].str.lower().str.strip().str.replace(' - prepared', '')
df_schrodinger = df_schrodinger.set_index('Name')
df_prodes.index = df_prodes.index.str.lower().str.strip()

common_proteins = sorted(set(df_prodes.index) & set(df_schrodinger.index))
df_prodes_common = df_prodes.loc[common_proteins]
df_schrodinger_common = df_schrodinger.loc[common_proteins]
df_descriptors = pd.concat([df_prodes_common, df_schrodinger_common], axis=1)

# Clean
df_descriptors = df_descriptors.replace([np.inf, -np.inf], np.nan)
cols_with_nan = df_descriptors.columns[df_descriptors.isna().any()].tolist()
if cols_with_nan:
    df_descriptors = df_descriptors.drop(columns=cols_with_nan)

# Prepare elution data
df_nacl['protein_name'] = df_nacl['protein_name'].str.lower().str.strip()
df_nacl = df_nacl.set_index('protein_name')
for col in ['pH4', 'pH5', 'pH6', 'pH7', 'pH8']:
    df_nacl[col] = pd.to_numeric(df_nacl[col], errors='coerce')

common_proteins = sorted(set(df_descriptors.index) & set(df_nacl.index))
df_descriptors = df_descriptors.loc[common_proteins]
df_nacl = df_nacl.loc[common_proteins]

print(f"  Proteins: {len(common_proteins)}")
print(f"  Descriptors: {df_descriptors.shape[1]}")

# Apply variance threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
X_filt = selector.fit_transform(df_descriptors.values)
selected_features = df_descriptors.columns[selector.get_support()].tolist()
df_descriptors = pd.DataFrame(X_filt, index=df_descriptors.index, columns=selected_features)

print(f"  Features after filtering: {df_descriptors.shape[1]}")

# ============================================================================
# DEFINE FUNCTIONS
# ============================================================================

def sigmoid_4param(pH, a, b, c, d):
    """4-parameter sigmoid"""
    return d + (a - d) / (1 + (pH / c) ** b)

def henderson_hasselbalch(pH, NaCl_min, NaCl_max, pKa, n):
    """Modified Henderson-Hasselbalch equation"""
    return NaCl_min + (NaCl_max - NaCl_min) / (1 + 10**(n * (pKa - pH)))

# ============================================================================
# METHOD 1: TWO-STAGE SIGMOID
# ============================================================================
print("\n[2/7] Method 1: Two-stage sigmoid fitting...")

ph_values_train = np.array([4, 5, 6, 7, 8])
ph_values_extrapolate = np.array([9, 10])
ph_columns_train = ['pH4', 'pH5', 'pH6', 'pH7', 'pH8']

# Stage 1: Fit sigmoid to each protein (pH 4-7)
sigmoid_params = []
proteins_with_fit = []

for prot in common_proteins:
    ph_data = []
    nacl_data = []
    for ph, col in zip(ph_values_train, ph_columns_train):
        val = df_nacl.loc[prot, col]
        if not np.isnan(val):
            ph_data.append(ph)
            nacl_data.append(val)
    
    if len(ph_data) == 5:  # Need all 5 pH points
        try:
            popt, _ = curve_fit(sigmoid_4param, ph_data, nacl_data,
                               p0=[max(nacl_data), 2, 6, min(nacl_data)],
                               maxfev=5000, bounds=([0, 0.1, 3, 0], [3, 10, 10, 3]))
            sigmoid_params.append(popt)
            proteins_with_fit.append(prot)
        except:
            pass

sigmoid_params = np.array(sigmoid_params)
print(f"  Fitted sigmoids for {len(proteins_with_fit)} proteins (pH 4-8)")

# Stage 2: Train models to predict sigmoid parameters from descriptors
# Use all proteins to train (no validation, just show extrapolation)
X_sigmoid = df_descriptors.loc[proteins_with_fit].values
y_sigmoid = sigmoid_params

# Store extrapolation curves for each protein
method1_curves = {}

for i, test_prot in enumerate(proteins_with_fit):
    train_mask = [p != test_prot for p in proteins_with_fit]
    X_train = X_sigmoid[train_mask]
    y_train = y_sigmoid[train_mask]
    X_test = X_sigmoid[i:i+1]
    
    # Train model for each parameter
    predicted_params = []
    for param_idx in range(4):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        pca = PCA(n_components=min(10, X_train_scaled.shape[0]-1, X_train_scaled.shape[1]))
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_pca, y_train[:, param_idx])
        pred_param = ridge.predict(X_test_pca)[0]
        predicted_params.append(pred_param)
    
    # Generate full curve from pH 4-10
    ph_full = np.arange(4, 10.1, 0.1)
    nacl_curve = sigmoid_4param(ph_full, *predicted_params)
    method1_curves[test_prot] = (ph_full, nacl_curve, predicted_params)

print(f"  Generated extrapolation curves for {len(method1_curves)} proteins")

# ============================================================================
# METHOD 2: HENDERSON-HASSELBALCH FITTING
# ============================================================================
print("\n[3/7] Method 2: Henderson-Hasselbalch fitting on dense predictions...")

# Train pH-as-feature model on pH 4-8
X_train_m2 = []
y_train_m2 = []
protein_ph_pairs_train = []

for prot in common_proteins:
    for ph, col in zip(ph_values_train, ph_columns_train):
        val = df_nacl.loc[prot, col]
        if not np.isnan(val):
            x_with_ph = np.concatenate([df_descriptors.loc[prot].values, [ph]])
            X_train_m2.append(x_with_ph)
            y_train_m2.append(val)
            protein_ph_pairs_train.append((prot, ph))

X_train_m2 = np.array(X_train_m2)
y_train_m2 = np.array(y_train_m2)

method2_curves = {}

for test_prot in common_proteins:
    # Train on all other proteins
    train_mask = [p[0] != test_prot for p in protein_ph_pairs_train]
    X_tr = X_train_m2[train_mask]
    y_tr = y_train_m2[train_mask]
    
    if len(X_tr) < 10:
        continue
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    
    pca = PCA(n_components=min(15, X_tr_scaled.shape[0]-1, X_tr_scaled.shape[1]))
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_pca, y_tr)
    
    # Predict at dense pH points (4.0, 4.1, ..., 8.0)
    ph_dense = np.arange(4.0, 8.1, 0.1)
    preds_dense = []
    
    for ph in ph_dense:
        x_test = np.concatenate([df_descriptors.loc[test_prot].values, [ph]]).reshape(1, -1)
        x_test_scaled = scaler.transform(x_test)
        x_test_pca = pca.transform(x_test_scaled)
        pred = ridge.predict(x_test_pca)[0]
        preds_dense.append(pred)
    
    preds_dense = np.array(preds_dense)
    
    # Fit Henderson-Hasselbalch to dense predictions
    try:
        popt_hh, _ = curve_fit(henderson_hasselbalch, ph_dense, preds_dense,
                               p0=[min(preds_dense), max(preds_dense), 6.0, 1.0],
                               maxfev=5000,
                               bounds=([0, 0, 3, 0.1], [3, 3, 10, 5]))
        
        # Extrapolate to pH 4-10
        ph_full = np.arange(4, 10.1, 0.1)
        nacl_curve = henderson_hasselbalch(ph_full, *popt_hh)
        method2_curves[test_prot] = (ph_full, nacl_curve, popt_hh)
    except:
        pass

print(f"  Generated extrapolation curves for {len(method2_curves)} proteins")

# ============================================================================
# METHOD 3: POLYNOMIAL pH FEATURES
# ============================================================================
print("\n[4/6] Method 3: Polynomial pH features...")

# Create training data with polynomial pH features
X_train_m3 = []
y_train_m3 = []
protein_ph_pairs_m3 = []

for prot in common_proteins:
    for ph, col in zip(ph_values_train, ph_columns_train):
        val = df_nacl.loc[prot, col]
        if not np.isnan(val):
            # Add pH, pH², pH³
            x_with_ph_poly = np.concatenate([df_descriptors.loc[prot].values, [ph, ph**2, ph**3]])
            X_train_m3.append(x_with_ph_poly)
            y_train_m3.append(val)
            protein_ph_pairs_m3.append((prot, ph))

X_train_m3 = np.array(X_train_m3)
y_train_m3 = np.array(y_train_m3)

predictions_method3 = []
actuals_method3 = []

for test_prot in common_proteins:
    train_mask = [p[0] != test_prot for p in protein_ph_pairs_m3]
    X_tr = X_train_m3[train_mask]
    y_tr = y_train_m3[train_mask]
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    
    pca = PCA(n_components=min(15, X_tr_scaled.shape[0]-1, X_tr_scaled.shape[1]))
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_pca, y_tr)
    
    # Predict pH 8 with polynomial features
    x_test = np.concatenate([df_descriptors.loc[test_prot].values, [8, 8**2, 8**3]]).reshape(1, -1)
    x_test_scaled = scaler.transform(x_test)
    x_test_pca = pca.transform(x_test_scaled)
    pred_ph8 = ridge.predict(x_test_pca)[0]
    
    actual_ph8 = df_nacl.loc[test_prot, 'pH8']
    if not np.isnan(actual_ph8):
        predictions_method3.append(pred_ph8)
        actuals_method3.append(actual_ph8)

r2_method3 = r2_score(actuals_method3, predictions_method3)
rmse_method3 = np.sqrt(mean_squared_error(actuals_method3, predictions_method3))
mae_method3 = mean_absolute_error(actuals_method3, predictions_method3)

print(f"  R² = {r2_method3:.4f}, RMSE = {rmse_method3:.4f}, MAE = {mae_method3:.4f}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n[5/6] Generating comparison summary...")

results_df = pd.DataFrame({
    'Method': ['Two-stage Sigmoid', 'Henderson-Hasselbalch', 'Polynomial pH'],
    'R²': [r2_method1, r2_method2, r2_method3],
    'RMSE': [rmse_method1, rmse_method2, rmse_method3],
    'MAE': [mae_method1, mae_method2, mae_method3],
    'N_predictions': [len(actuals_method1), len(actuals_method2), len(actuals_method3)]
})

results_df.to_csv('extrapolation_method_comparison.csv', index=False)

print("\n" + "="*120)
print("RESULTS SUMMARY")
print("="*120)
print(results_df.to_string(index=False))

best_method = results_df.loc[results_df['R²'].idxmax(), 'Method']
print(f"\n🏆 Best method: {best_method} (R² = {results_df['R²'].max():.4f})")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n[6/6] Creating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Method 1
ax = axes[0, 0]
ax.scatter(actuals_method1, predictions_method1, alpha=0.6, s=100)
ax.plot([min(actuals_method1), max(actuals_method1)], 
        [min(actuals_method1), max(actuals_method1)], 'r--', lw=2)
ax.set_xlabel('Actual NaCl (M)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted NaCl (M)', fontsize=12, fontweight='bold')
ax.set_title(f'Method 1: Two-stage Sigmoid\nR² = {r2_method1:.4f}', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Method 2
ax = axes[0, 1]
ax.scatter(actuals_method2, predictions_method2, alpha=0.6, s=100, color='orange')
ax.plot([min(actuals_method2), max(actuals_method2)], 
        [min(actuals_method2), max(actuals_method2)], 'r--', lw=2)
ax.set_xlabel('Actual NaCl (M)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted NaCl (M)', fontsize=12, fontweight='bold')
ax.set_title(f'Method 2: Henderson-Hasselbalch\nR² = {r2_method2:.4f}', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Method 3
ax = axes[0, 2]
ax.scatter(actuals_method3, predictions_method3, alpha=0.6, s=100, color='green')
ax.plot([min(actuals_method3), max(actuals_method3)], 
        [min(actuals_method3), max(actuals_method3)], 'r--', lw=2)
ax.set_xlabel('Actual NaCl (M)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted NaCl (M)', fontsize=12, fontweight='bold')
ax.set_title(f'Method 3: Polynomial pH\nR² = {r2_method3:.4f}', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Residuals
ax = axes[1, 0]
residuals1 = np.array(actuals_method1) - np.array(predictions_method1)
ax.scatter(predictions_method1, residuals1, alpha=0.6, s=100)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted NaCl (M)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual (M)', fontsize=12, fontweight='bold')
ax.set_title('Method 1: Residuals', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
residuals2 = np.array(actuals_method2) - np.array(predictions_method2)
ax.scatter(predictions_method2, residuals2, alpha=0.6, s=100, color='orange')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted NaCl (M)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual (M)', fontsize=12, fontweight='bold')
ax.set_title('Method 2: Residuals', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
residuals3 = np.array(actuals_method3) - np.array(predictions_method3)
ax.scatter(predictions_method3, residuals3, alpha=0.6, s=100, color='green')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted NaCl (M)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual (M)', fontsize=12, fontweight='bold')
ax.set_title('Method 3: Residuals', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('extrapolation_method_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved: extrapolation_method_comparison.png")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
print("\nFiles created:")
print("  - extrapolation_method_comparison.csv")
print("  - extrapolation_method_comparison.png")

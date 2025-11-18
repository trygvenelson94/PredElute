#!/usr/bin/env python3
"""
Multi-Experiment QSPR - CORRECTED VERSION

Key fixes:
1. First column of each experiment is protein subset (not elution data)
2. Each condition gets a separate model (not concentration as predictor)
3. Option to combine pH 5/6 experiments
4. Experiment 5: Treats glycols as separate conditions (not series)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import warnings

# Suppress pandas performance warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Configuration
CONFIG = {
    'n_features': 50,
    'pca_variance': 0.95,
    'use_pca': True,
    'ridge_alpha': 1.0,
    'var_thresh': 0.01,
    'corr_thresh': 0.95,
    'seed': 42,
    'combine_ph_experiments': True,  # Combine experiments 3 and 4 with pH as feature
}


def parse_experiments_corrected(csv_path):
    """
    Parse experiments with corrected structure:
    - First column in each group = protein subset
    - Next 3 columns = elution data
    """
    print("\n" + "="*80)
    print("PARSING EXPERIMENTS (CORRECTED)")
    print("="*80)
    
    # Read metadata
    df_meta = pd.read_csv(csv_path, nrows=5, header=None)
    
    # Define experiments manually based on user specification
    experiments = [
        {
            'id': 1,
            'name': 'Arginine on CEX',
            'protein_col': 1,  # Column B
            'elution_cols': [2, 3, 4],  # Columns C, D, E
            'conditions': ['0M', '0.025M', '0.1M'],
            'ph': 6,
            'modifier': 'Arginine',
            'resin': 'CEX',
            'type': 'concentration_series'
        },
        {
            'id': 2,
            'name': 'Arginine on Capto MMC',
            'protein_col': 5,  # Column F
            'elution_cols': [6, 7, 8],  # Columns G, H, I
            'conditions': ['0M', '0.025M', '0.1M'],
            'ph': 6,
            'modifier': 'Arginine',
            'resin': 'Capto MMC',
            'type': 'concentration_series'
        },
        {
            'id': 3,
            'name': 'Guanidine on CEX',
            'protein_col': 9,  # Column J
            'elution_cols': [10, 11, 12],  # Columns K, L, M
            'conditions': ['0M', '0.025M', '0.1M'],
            'ph': 6,
            'modifier': 'Guanidine',
            'resin': 'CEX',
            'type': 'concentration_series'
        },
        {
            'id': 4,
            'name': 'Guanidine on Capto MMC',
            'protein_col': 13,  # Column N
            'elution_cols': [14, 15, 16],  # Columns O, P, Q
            'conditions': ['0M', '0.025M', '0.1M'],
            'ph': 6,
            'modifier': 'Guanidine',
            'resin': 'Capto MMC',
            'type': 'concentration_series'
        },
        {
            'id': 5,
            'name': 'pH5 Multi-resin',
            'protein_col': 17,  # Column R
            'elution_cols': [18, 19, 20],  # Columns S, T, U
            'conditions': ['Capto MMC', 'CM Sepharose FF', 'SP Sepharose'],
            'ph': 5,
            'modifier': 'None',
            'resin': 'Multi',
            'type': 'multi_resin'
        },
        {
            'id': 6,
            'name': 'pH6 Multi-resin',
            'protein_col': 21,  # Column V
            'elution_cols': [22, 23, 24],  # Columns W, X, Y
            'conditions': ['Capto MMC', 'CM Sepharose FF', 'SP Sepharose'],
            'ph': 6,
            'modifier': 'None',
            'resin': 'Multi',
            'type': 'multi_resin'
        },
        {
            'id': 7,
            'name': 'Glycols on Capto MMC',
            'protein_col': 25,  # Column Z
            'elution_cols': [26, 27, 28],  # Columns AA, AB, AC
            'conditions': ['No modifier', '20% Ethylene glycol', '20% Propylene glycol'],
            'ph': 6,
            'modifier': 'Glycols',
            'resin': 'Capto MMC',
            'type': 'multi_condition'
        },
        {
            'id': 8,
            'name': 'Sodium Caprylate on CM Seph FF',
            'protein_col': 29,  # Column AD
            'elution_cols': [30, 31, 32],  # Columns AE, AF, AG
            'conditions': ['0M', '0.01M', '0.025M'],
            'ph': 6,
            'modifier': 'Sodium Caprylate',
            'resin': 'CM Sepharose FF',
            'type': 'concentration_series'
        },
        {
            'id': 9,
            'name': 'Sodium Caprylate on Capto MMC',
            'protein_col': 33,  # Column AH
            'elution_cols': [34, 35, 36],  # Columns AI, AJ, AK
            'conditions': ['0M', '0.01M', '0.025M'],
            'ph': 6,
            'modifier': 'Sodium Caprylate',
            'resin': 'Capto MMC',
            'type': 'concentration_series'
        },
    ]
    
    for exp in experiments:
        print(f"\nExperiment {exp['id']}: {exp['name']}")
        print(f"  pH: {exp['ph']}, Modifier: {exp['modifier']}, Resin: {exp['resin']}")
        print(f"  Conditions: {', '.join(exp['conditions'])}")
    
    return experiments


def load_experiment_data_corrected(csv_path, experiment):
    """
    Load data for a single experiment (corrected version)
    Now handles pH-specific descriptors!
    """
    # Read full data (skip header rows for elution data)
    df_data = pd.read_csv(csv_path, skiprows=4)
    
    # Get protein subset for this experiment
    protein_col_idx = experiment['protein_col']
    proteins_subset = df_data.iloc[:, protein_col_idx].dropna().tolist()
    
    # Get elution data
    elution_data = []
    for elution_col_idx in experiment['elution_cols']:
        elution_col = df_data.iloc[:, elution_col_idx]
        elution_data.append(elution_col.values)
    
    elution_matrix = np.array(elution_data).T  # Shape: (n_proteins, n_conditions)
    
    # Determine which descriptor block to use based on pH
    # Read full CSV without skipping rows
    df_full = pd.read_csv(csv_path, header=None)
    
    if experiment['ph'] == 5:
        # pH 5 descriptors: rows 31-54 (indices 30-53)
        descriptor_row_start = 30
        descriptor_row_end = 54
        print(f"  [INFO] Using pH 5 descriptors (rows 31-54)")
    else:
        # pH 6 descriptors: rows 6-29 (indices 5-28)
        descriptor_row_start = 5
        descriptor_row_end = 29
        print(f"  [INFO] Using pH 6 descriptors (rows 6-29)")
    
    # Column AL = index 37 (protein names), AM onwards = index 38+ (descriptors)
    descriptor_start_col = 38
    
    # Extract descriptor block
    descriptor_proteins = df_full.iloc[descriptor_row_start:descriptor_row_end, 37].tolist()
    descriptor_values = df_full.iloc[descriptor_row_start:descriptor_row_end, descriptor_start_col:]
    
    # Create descriptor dataframe and convert to numeric
    descriptor_cols = [f'Desc_{i}' for i in range(descriptor_values.shape[1])]
    descriptors_all = pd.DataFrame(descriptor_values.values, columns=descriptor_cols, index=descriptor_proteins)
    
    # Convert all columns to numeric, replacing non-numeric with NaN
    for col in descriptors_all.columns:
        descriptors_all[col] = pd.to_numeric(descriptors_all[col], errors='coerce')
    
    # Fill NaN with 0
    descriptors_all = descriptors_all.fillna(0)
    
    # Match proteins from experiment to descriptor block
    protein_indices = []
    valid_proteins = []
    for prot in proteins_subset:
        if prot in descriptor_proteins:
            protein_indices.append(descriptor_proteins.index(prot))
            valid_proteins.append(prot)
        else:
            print(f"  [WARNING] Protein '{prot}' not found in pH {experiment['ph']} descriptor block")
    
    # Extract descriptors for this subset
    descriptors_subset = descriptors_all.iloc[protein_indices].reset_index(drop=True)
    
    # Match elution data (need to find where these proteins are in elution data)
    elution_master_proteins = df_data.iloc[:, 0].tolist()
    elution_indices = []
    final_valid_proteins = []
    final_descriptor_indices = []
    
    for i, prot in enumerate(valid_proteins):
        try:
            idx = elution_master_proteins.index(prot)
            elution_indices.append(idx)
            final_valid_proteins.append(prot)
            final_descriptor_indices.append(i)
        except ValueError:
            print(f"  [WARNING] Protein '{prot}' not found in elution master list")
    
    elution_subset = elution_matrix[elution_indices, :]
    descriptors_subset_matched = descriptors_subset.iloc[final_descriptor_indices].reset_index(drop=True)
    
    # Filter out proteins with all NaN elution values
    valid_mask = ~np.all(np.isnan(elution_subset), axis=1)
    
    return {
        'proteins': [final_valid_proteins[i] for i in range(len(final_valid_proteins)) if valid_mask[i]],
        'descriptors': descriptors_subset_matched[valid_mask].reset_index(drop=True),
        'elution': elution_subset[valid_mask],
        'conditions': experiment['conditions']
    }


def engineer_features(descriptors, quiet=True):
    """Feature engineering (optimized to avoid fragmentation warnings)"""
    if not quiet:
        print(f"  [Feature Engineering] Creating derived features...")
    
    X_orig = descriptors.copy()
    orig_cols = X_orig.columns.tolist()
    
    # Build all new columns in a dict, then concat once
    new_cols = {}
    
    # Log transforms
    for col in orig_cols[:20]:  # Top 20 descriptors
        new_cols[f'{col}_log'] = np.log(np.abs(X_orig[col]) + 1e-6)
    
    # Square root
    for col in orig_cols[:15]:
        new_cols[f'{col}_sqrt'] = np.sqrt(np.abs(X_orig[col]))
    
    # Squared
    for col in orig_cols[:15]:
        new_cols[f'{col}_sq'] = X_orig[col] ** 2
    
    # Ratios (top descriptors)
    top_cols = X_orig[orig_cols].var().nlargest(8).index.tolist()
    for i, col1 in enumerate(top_cols):
        for col2 in top_cols[i+1:]:
            new_cols[f'{col1}/{col2}'] = X_orig[col1] / (X_orig[col2] + 1e-6)
    
    # Products
    for i, col1 in enumerate(top_cols[:5]):
        for col2 in top_cols[i+1:6]:
            new_cols[f'{col1}*{col2}'] = X_orig[col1] * X_orig[col2]
    
    # Absolute differences
    for i, col1 in enumerate(top_cols[:5]):
        for col2 in top_cols[i+1:6]:
            new_cols[f'|{col1}-{col2}|'] = np.abs(X_orig[col1] - X_orig[col2])
    
    # Concatenate all at once (avoids fragmentation)
    X_new = pd.DataFrame(new_cols, index=X_orig.index)
    X = pd.concat([X_orig, X_new], axis=1)
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if not quiet:
        print(f"  [Feature Engineering] {X.shape[1] - len(orig_cols)} new features (total: {X.shape[1]})")
    
    return X


def clean_features(X, var_thresh=0.01, corr_thresh=0.95):
    """Remove low-variance and highly correlated features"""
    X = X.loc[:, X.var() > var_thresh]
    
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    X_clean = X.drop(columns=to_drop)
    
    print(f"  [Feature Cleaning] Removed {len(to_drop)} correlated features ({X_clean.shape[1]} remain)")
    
    return X_clean


def build_model_per_condition(data, experiment, config):
    """
    Build SEPARATE model for EACH condition
    
    Key change: No concentration as predictor!
    Each condition (0M, 0.025M, 0.1M) gets its own model.
    """
    exp_id = experiment['id']
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {exp_id}: {experiment['name']}")
    print(f"{'='*80}")
    
    proteins = data['proteins']
    elution = data['elution']
    descriptors = data['descriptors']
    conditions = data['conditions']
    
    print(f"[INFO] Proteins: {len(proteins)}")
    print(f"[INFO] Conditions: {len(conditions)}")
    print(f"[INFO] Strategy: Separate model per condition")
    
    # Feature engineering
    desc_eng = engineer_features(descriptors, quiet=False)
    desc_clean = clean_features(desc_eng, 
                                var_thresh=config['var_thresh'],
                                corr_thresh=config['corr_thresh'])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(desc_clean),
        columns=desc_clean.columns,
        index=desc_clean.index
    )
    
    results_per_condition = []
    
    # Build model for EACH condition separately
    for cond_idx, cond_name in enumerate(conditions):
        print(f"\n  --- Condition: {cond_name} ---")
        
        # Get elution values for this condition
        y_cond = elution[:, cond_idx]
        
        # Remove NaN values
        valid_mask = ~np.isnan(y_cond)
        X_cond = X_scaled[valid_mask]
        y_cond = y_cond[valid_mask]
        proteins_cond = [proteins[i] for i in range(len(proteins)) if valid_mask[i]]
        
        if len(y_cond) < 5:
            print(f"    [SKIP] Insufficient data ({len(y_cond)} samples)")
            continue
        
        print(f"    Samples: {len(y_cond)}, Features: {X_cond.shape[1]}")
        
        # PCA or RFE
        if config['use_pca']:
            pca = PCA(n_components=min(config['pca_variance'], len(y_cond)-1), 
                     random_state=config['seed'])
            X_pca = pca.fit_transform(X_cond)
            n_comp = X_pca.shape[1]
            X_sel = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_comp)])
            print(f"    [PCA] Reduced to {n_comp} components")
        else:
            n_feat = min(config['n_features'], X_cond.shape[1], len(y_cond) - 2)
            model_pre = Ridge(alpha=config['ridge_alpha'], random_state=config['seed'])
            rfe = RFE(estimator=model_pre, n_features_to_select=n_feat)
            X_sel = pd.DataFrame(
                rfe.fit_transform(X_cond, y_cond),
                columns=[X_cond.columns[i] for i in range(X_cond.shape[1]) if rfe.support_[i]]
            )
            print(f"    [RFE] Selected {X_sel.shape[1]} features")
        
        # Train model
        model = Ridge(alpha=config['ridge_alpha'], random_state=config['seed'])
        model.fit(X_sel, y_cond)
        y_pred = model.predict(X_sel)
        
        # Metrics
        r2 = r2_score(y_cond, y_pred)
        rmse = np.sqrt(mean_squared_error(y_cond, y_pred))
        mae = mean_absolute_error(y_cond, y_pred)
        
        print(f"    [RESULTS] R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        results_per_condition.append({
            'condition': cond_name,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_cond),
            'y_true': y_cond,
            'y_pred': y_pred,
            'proteins': proteins_cond
        })
    
    # Overall summary
    avg_r2 = np.mean([r['r2'] for r in results_per_condition])
    avg_rmse = np.mean([r['rmse'] for r in results_per_condition])
    avg_mae = np.mean([r['mae'] for r in results_per_condition])
    
    print(f"\n[EXPERIMENT SUMMARY]")
    print(f"  Average R²:   {avg_r2:.4f}")
    print(f"  Average RMSE: {avg_rmse:.4f}")
    print(f"  Average MAE:  {avg_mae:.4f}")
    
    # Create plot
    plot_experiment_results(experiment, results_per_condition)
    
    return {
        'exp_id': exp_id,
        'name': experiment['name'],
        'ph': experiment['ph'],
        'modifier': experiment['modifier'],
        'resin': experiment['resin'],
        'avg_r2': avg_r2,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'per_condition': results_per_condition
    }


def plot_experiment_results(experiment, results_per_condition):
    """Create plot for experiment with subplots per condition"""
    n_conditions = len(results_per_condition)
    fig, axes = plt.subplots(1, n_conditions, figsize=(5*n_conditions, 4))
    
    if n_conditions == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results_per_condition):
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='black', linewidth=1)
        
        # Diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Actual Elution', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Elution', fontsize=11, fontweight='bold')
        ax.set_title(f"{result['condition']}\nR²={result['r2']:.3f}", 
                    fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle(f"Exp {experiment['id']}: {experiment['name']}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    results_dir = r"C:\Users\tryg.nelson\predelute\results_corrected"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"exp{experiment['id']}_by_condition.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [PLOT] Saved to {output_path}")
    plt.close()


def main():
    """Run corrected analysis"""
    csv_path = r"C:\Users\tryg.nelson\predelute\multi_qspr.csv"
    
    # Parse experiments
    experiments = parse_experiments_corrected(csv_path)
    
    all_results = []
    
    # Process each experiment
    for exp in experiments:
        try:
            data = load_experiment_data_corrected(csv_path, exp)
            result = build_model_per_condition(data, exp, CONFIG)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp['id']} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    summary_df = pd.DataFrame([
        {
            'Exp_ID': r['exp_id'],
            'Name': r['name'],
            'pH': r['ph'],
            'Modifier': r['modifier'],
            'Resin': r['resin'],
            'Avg_R2': r['avg_r2'],
            'Avg_RMSE': r['avg_rmse'],
            'Avg_MAE': r['avg_mae'],
            'N_Conditions': len(r['per_condition'])
        }
        for r in all_results
    ])
    
    print(summary_df.to_string(index=False))
    
    # Save summary
    results_dir = r"C:\Users\tryg.nelson\predelute\results_corrected"
    summary_path = os.path.join(results_dir, 'experiment_summary_corrected.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] Summary to {summary_path}")
    
    print("\n" + "="*80)
    print(f"COMPLETED: {len(all_results)}/{len(experiments)} experiments successful")
    print("="*80)


if __name__ == '__main__':
    main()

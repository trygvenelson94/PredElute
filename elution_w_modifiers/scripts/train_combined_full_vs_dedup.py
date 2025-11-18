#!/usr/bin/env python3
"""
Grid search comparing Full vs Deduplicated protein pools
Combined Schrodinger + ProDes descriptors
Outputs best models with CLI parameters
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def parse_list(s):
    """Parse comma-separated list"""
    return [float(x) if '.' in x else int(x) for x in s.split(',')]

def define_protein_families():
    """Group proteins by family to avoid family leakage"""
    families = {
        'Albumin': ['bovine serum albumin', 'human serum albumin', 'porcine albumin', 'sheep albumin'],
        'Chymotrypsin': ['alpha-chymotrypsin', 'beta-chymotrypsin', 'alpha-chymotrypsinogen a'],
        'Trypsin': ['bovine trypsin', 'porcine trypsin', 'trypsinogen'],
        'Ribonuclease': ['ribonuclease a', 'ribonuclease b'],
        'Cytochrome c': ['horse cytochrome c', 'bovine cytochrome c'],
        'Aprotinin': ['aprotinin'],
        'Avidin': ['avidin'],
        'Carbonic anhydrase': ['carbonic anhydrase'],
        'Conalbumin': ['conalbumin'],
        'Glutamic dehydrogenase': ['l-glutamic dehydrogenase'],
        'Lipoxygenase': ['lipoxidase'],
        'Lysozyme': ['lysozyme'],
        'Papain': ['papain'],
        'Peanut lectin': ['peanut lectin'],
        'Ubiquitin': ['ubiquitin'],
        'Beta-lactoglobulin': ['beta-lactoglobulin b'],
        'Phospholipase': ['bee phospholipase a2'],
        'Lactoferrin': ['lactoferrin'],
        'Pyruvate kinase': ['pyruvate kinase'],
        'Elastase': ['elastase']
    }
    return families

def create_deduplicated_dataset(families, available_proteins):
    """Select one representative protein per family"""
    deduplicated = []
    family_mapping = {}
    
    for family_name, members in families.items():
        for member in members:
            if member in available_proteins:
                deduplicated.append(member)
                family_mapping[member] = family_name
                break
    
    return deduplicated, family_mapping

# Parse arguments
parser = argparse.ArgumentParser(description='Grid search: Full vs Dedup with combined descriptors')
parser.add_argument('--alpha', type=str, default='0.01,0.1,1.0,10.0,100.0')
parser.add_argument('--n_components', type=str, default='3,5,7,10,15,19')
parser.add_argument('--var_thresh', type=float, default=0.0)
parser.add_argument('--corr_thresh', type=str, default='1.0')
parser.add_argument('--experiment', type=str, default='all')
parser.add_argument('--condition', type=str, default='all')
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--n_seeds', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_dir', type=str, default='trained_models_combined_best')

args = parser.parse_args()

# Parse parameters
alpha_values = parse_list(args.alpha)
n_components_values = parse_list(args.n_components)
corr_thresh_values = parse_list(args.corr_thresh)

print("="*120)
print("GRID SEARCH: FULL VS DEDUPLICATED PROTEIN POOLS")
print("Combined Schrodinger + ProDes Descriptors")
print("="*120)
print(f"\nParameters:")
print(f"  Alpha: {alpha_values}")
print(f"  PCA components: {n_components_values}")
print(f"  Variance threshold: {args.var_thresh}")
print(f"  Correlation thresholds: {corr_thresh_values}")
print(f"  Seeds: {args.n_seeds}")

# Load data
print("\n[1/7] Loading data...")
df_prodes = pd.read_csv('prodes_descriptors_training_only.csv', index_col=0)
df_schrodinger = pd.read_csv('schrodinger_descriptors.csv')
df_multi = pd.read_csv('multi_qspr.csv', header=[0,1,2,3,4])

protein_names = df_multi.iloc[:, 0].values
df_elution = df_multi.iloc[:, 1:37]
df_elution.index = protein_names
df_elution.columns = [f'col_{i}' for i in range(len(df_elution.columns))]
df_elution = df_elution.apply(pd.to_numeric, errors='coerce')

# Standardize names
df_schrodinger['Name'] = df_schrodinger['Name'].str.lower().str.strip().str.replace(' - prepared', '')
df_schrodinger = df_schrodinger.set_index('Name')
df_prodes.index = df_prodes.index.str.lower().str.strip()
df_elution.index = df_elution.index.str.lower().str.strip()

# Find common proteins
common_proteins = sorted(set(df_prodes.index) & set(df_schrodinger.index) & set(df_elution.index))

print(f"  Common proteins: {len(common_proteins)}")

# Create deduplicated dataset
print("\n[2/7] Creating deduplicated dataset...")
families = define_protein_families()
deduplicated_proteins, family_mapping = create_deduplicated_dataset(families, common_proteins)

print(f"  Full dataset: {len(common_proteins)} proteins")
print(f"  Deduplicated: {len(deduplicated_proteins)} proteins ({len(families)} families)")

# Merge descriptors
print("\n[3/7] Merging descriptors...")
df_prodes_common = df_prodes.loc[common_proteins]
df_schrodinger_common = df_schrodinger.loc[common_proteins]
df_elution_common = df_elution.loc[common_proteins]

df_combined = pd.concat([df_prodes_common, df_schrodinger_common], axis=1)
df_combined = df_combined.replace([np.inf, -np.inf], np.nan)
cols_with_nan = df_combined.columns[df_combined.isna().any()].tolist()
if cols_with_nan:
    df_combined = df_combined.drop(columns=cols_with_nan)

print(f"  Features: {len(df_combined.columns)}")

# Experiment mapping
EXPERIMENT_MAP = {
    1: [1, 2, 3], 2: [5, 6, 7], 3: [9, 10, 11], 4: [13, 14, 15],
    5: [17, 18, 19], 6: [21, 22, 23], 7: [25, 26, 27],
    8: [29, 30, 31], 9: [33, 34, 35]
}

# Determine columns
print("\n[4/7] Selecting columns...")
if args.columns is not None:
    col_indices = parse_list(args.columns)
    columns_to_process = [df_elution_common.columns[i] for i in col_indices]
else:
    experiments = list(range(1, 10)) if args.experiment == 'all' else parse_list(args.experiment)
    conditions = [0, 1, 2] if args.condition == 'all' else parse_list(args.condition)
    
    col_indices = []
    for exp in experiments:
        if exp in EXPERIMENT_MAP:
            for cond in conditions:
                if cond < len(EXPERIMENT_MAP[exp]):
                    col_indices.append(EXPERIMENT_MAP[exp][cond])
    
    columns_to_process = [df_elution_common.columns[i] for i in sorted(set(col_indices))]

print(f"  Processing {len(columns_to_process)} columns")

# Grid search
print("\n[5/7] Running grid search (Full vs Dedup)...")
print("="*120)

os.makedirs(args.output_dir, exist_ok=True)
all_results = []
loo = LeaveOneOut()

for col_idx, column in enumerate(columns_to_process, 1):
    print(f"\n[{col_idx}/{len(columns_to_process)}] {column}")
    print("-"*120)
    
    y_full = df_elution_common[column].values
    y_dedup = df_elution_common.loc[deduplicated_proteins, column].values
    
    valid_mask_full = ~np.isnan(y_full)
    valid_mask_dedup = ~np.isnan(y_dedup)
    
    if valid_mask_full.sum() < 5:
        print(f"  [SKIP] Only {valid_mask_full.sum()} valid samples")
        continue
    
    X_full = df_combined.values[valid_mask_full]
    y_full_valid = y_full[valid_mask_full]
    proteins_full = [common_proteins[i] for i in range(len(valid_mask_full)) if valid_mask_full[i]]
    
    X_dedup = df_combined.loc[deduplicated_proteins].values[valid_mask_dedup]
    y_dedup_valid = y_dedup[valid_mask_dedup]
    proteins_dedup = [deduplicated_proteins[i] for i in range(len(valid_mask_dedup)) if valid_mask_dedup[i]]
    
    print(f"  Full: {len(y_full_valid)} samples, Dedup: {len(y_dedup_valid)} samples")
    
    # Test each correlation threshold
    for corr_thresh in corr_thresh_values:
        # Apply correlation filtering
        def apply_corr_filter(X, thresh):
            if thresh >= 1.0:
                return X, list(range(X.shape[1]))
            corr_matrix = np.corrcoef(X.T)
            np.fill_diagonal(corr_matrix, 0)
            to_remove = set()
            for i in range(len(corr_matrix)):
                if i in to_remove:
                    continue
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix[i, j]) > thresh:
                        to_remove.add(j)
            keep_indices = [i for i in range(X.shape[1]) if i not in to_remove]
            return X[:, keep_indices], keep_indices
        
        X_full_filt, keep_full = apply_corr_filter(X_full, corr_thresh)
        X_dedup_filt, keep_dedup = apply_corr_filter(X_dedup, corr_thresh)
        
        # Grid search
        best_full = {'r2': -np.inf}
        best_dedup = {'r2': -np.inf}
        
        for n_comp in n_components_values:
            max_comp_full = min(len(X_full_filt) - 1, X_full_filt.shape[1])
            max_comp_dedup = min(len(X_dedup_filt) - 1, X_dedup_filt.shape[1])
            
            if n_comp > max_comp_full and n_comp > max_comp_dedup:
                continue
            
            for alpha in alpha_values:
                # Full dataset
                if n_comp <= max_comp_full:
                    seed_scores_full = []
                    for seed_offset in range(args.n_seeds):
                        np.random.seed(args.seed + seed_offset)
                        y_pred = np.zeros(len(y_full_valid))
                        for train_idx, test_idx in loo.split(X_full_filt):
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_full_filt[train_idx])
                            X_test_scaled = scaler.transform(X_full_filt[test_idx])
                            pca = PCA(n_components=n_comp, random_state=args.seed+seed_offset)
                            X_train_pca = pca.fit_transform(X_train_scaled)
                            X_test_pca = pca.transform(X_test_scaled)
                            ridge = Ridge(alpha=alpha, random_state=args.seed+seed_offset)
                            ridge.fit(X_train_pca, y_full_valid[train_idx])
                            y_pred[test_idx] = ridge.predict(X_test_pca)
                        seed_scores_full.append(r2_score(y_full_valid, y_pred))
                    
                    r2_full = np.mean(seed_scores_full)
                    if r2_full > best_full['r2']:
                        best_full = {'r2': r2_full, 'n_comp': n_comp, 'alpha': alpha, 
                                    'corr_thresh': corr_thresh, 'n_features': X_full_filt.shape[1],
                                    'y_pred': y_pred}
                
                # Dedup dataset
                if n_comp <= max_comp_dedup:
                    seed_scores_dedup = []
                    for seed_offset in range(args.n_seeds):
                        np.random.seed(args.seed + seed_offset)
                        y_pred = np.zeros(len(y_dedup_valid))
                        for train_idx, test_idx in loo.split(X_dedup_filt):
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_dedup_filt[train_idx])
                            X_test_scaled = scaler.transform(X_dedup_filt[test_idx])
                            pca = PCA(n_components=n_comp, random_state=args.seed+seed_offset)
                            X_train_pca = pca.fit_transform(X_train_scaled)
                            X_test_pca = pca.transform(X_test_scaled)
                            ridge = Ridge(alpha=alpha, random_state=args.seed+seed_offset)
                            ridge.fit(X_train_pca, y_dedup_valid[train_idx])
                            y_pred[test_idx] = ridge.predict(X_test_pca)
                        seed_scores_dedup.append(r2_score(y_dedup_valid, y_pred))
                    
                    r2_dedup = np.mean(seed_scores_dedup)
                    if r2_dedup > best_dedup['r2']:
                        best_dedup = {'r2': r2_dedup, 'n_comp': n_comp, 'alpha': alpha,
                                     'corr_thresh': corr_thresh, 'n_features': X_dedup_filt.shape[1],
                                     'y_pred': y_pred}
        
        if best_full['r2'] > -np.inf or best_dedup['r2'] > -np.inf:
            delta_r2 = best_full['r2'] - best_dedup['r2']
            print(f"  corr={corr_thresh:.2f}: Full R²={best_full['r2']:.4f} (α={best_full['alpha']}, PC={best_full['n_comp']}) | " +
                  f"Dedup R²={best_dedup['r2']:.4f} (α={best_dedup['alpha']}, PC={best_dedup['n_comp']}) | dR²={delta_r2:+.4f}")
            
            all_results.append({
                'column': column,
                'corr_thresh': corr_thresh,
                'r2_full': best_full['r2'],
                'alpha_full': best_full['alpha'],
                'n_comp_full': best_full['n_comp'],
                'n_features_full': best_full['n_features'],
                'r2_dedup': best_dedup['r2'],
                'alpha_dedup': best_dedup['alpha'],
                'n_comp_dedup': best_dedup['n_comp'],
                'n_features_dedup': best_dedup['n_features'],
                'delta_r2': delta_r2,
                'n_samples_full': len(y_full_valid),
                'n_samples_dedup': len(y_dedup_valid)
            })

# Save results
print("\n" + "="*120)
print("[6/7] RESULTS SUMMARY")
print("="*120)

df_results = pd.DataFrame(all_results)
df_results.to_csv('full_vs_dedup_results.csv', index=False)

print(f"\nProcessed {len(df_results)} column-threshold combinations")
print(f"\nOverall statistics:")
print(f"  Full dataset - Mean R²: {df_results['r2_full'].mean():.4f} ± {df_results['r2_full'].std():.4f}")
print(f"  Dedup dataset - Mean R²: {df_results['r2_dedup'].mean():.4f} ± {df_results['r2_dedup'].std():.4f}")
print(f"  Average dR² (family bias): {df_results['delta_r2'].mean():.4f}")

# Create optimal models CSV
print("\n[7/7] CREATING OPTIMAL MODELS FILE")
print("="*120)

optimal_models = []
for column in df_results['column'].unique():
    col_data = df_results[df_results['column'] == column]
    
    # Find best for full and dedup
    best_full_row = col_data.loc[col_data['r2_full'].idxmax()]
    best_dedup_row = col_data.loc[col_data['r2_dedup'].idxmax()]
    
    # Choose better one
    if best_dedup_row['r2_dedup'] > best_full_row['r2_full']:
        use_dedup = True
        best_r2 = best_dedup_row['r2_dedup']
        best_alpha = best_dedup_row['alpha_dedup']
        best_n_comp = best_dedup_row['n_comp_dedup']
        best_corr = best_dedup_row['corr_thresh']
        n_features = best_dedup_row['n_features_dedup']
    else:
        use_dedup = False
        best_r2 = best_full_row['r2_full']
        best_alpha = best_full_row['alpha_full']
        best_n_comp = best_full_row['n_comp_full']
        best_corr = best_full_row['corr_thresh']
        n_features = best_full_row['n_features_full']
    
    optimal_models.append({
        'column': column,
        'dataset': 'dedup' if use_dedup else 'full',
        'best_r2': best_r2,
        'best_alpha': best_alpha,
        'best_n_components': int(best_n_comp),
        'best_corr_thresh': best_corr,
        'n_features': int(n_features),
        'cli_command': f"--alpha {best_alpha} --n_components {int(best_n_comp)} --corr_thresh {best_corr}"
    })

df_optimal = pd.DataFrame(optimal_models)
df_optimal = df_optimal.sort_values('best_r2', ascending=False)
df_optimal.to_csv('optimal_models_combined.csv', index=False)

print(f"\nTop 10 models:")
print(f"{'Rank':<6} {'Column':<12} {'Dataset':<8} {'R²':<8} {'Alpha':<8} {'PC':<5} {'Corr':<6} {'Features':<10}")
print("-"*120)
for rank, (_, row) in enumerate(df_optimal.head(10).iterrows(), 1):
    print(f"{rank:<6} {row['column']:<12} {row['dataset']:<8} {row['best_r2']:<8.4f} {row['best_alpha']:<8} " +
          f"{row['best_n_components']:<5} {row['best_corr_thresh']:<6.2f} {row['n_features']:<10}")

print(f"\n[OK] Saved optimal models to: optimal_models_combined.csv")
print(f"[OK] Saved full results to: full_vs_dedup_results.csv")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)

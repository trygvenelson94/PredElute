#!/usr/bin/env python3
"""
Analyze feature importance using PCA loadings and correlation analysis
Suggests domain-relevant feature engineering based on important descriptors
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import sys
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))
from multi_experiment_qspr_corrected import (
    parse_experiments_corrected, 
    load_experiment_data_corrected,
    engineer_features,
    clean_features
)


def analyze_pca_loadings(X_clean, n_components=5):
    """Analyze PCA loadings to find most important features"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # Get loadings (components)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=X_clean.columns
    )
    
    # Calculate absolute contribution to each PC
    abs_loadings = np.abs(loadings)
    
    # Get top features per PC
    top_features_per_pc = {}
    for pc in loadings.columns:
        top_10 = abs_loadings[pc].nlargest(10)
        top_features_per_pc[pc] = top_10
    
    # Overall importance: weighted by explained variance
    explained_var = pca.explained_variance_ratio_
    weighted_importance = np.zeros(len(X_clean.columns))
    
    for i, var in enumerate(explained_var):
        weighted_importance += var * abs_loadings[f'PC{i+1}'].values
    
    overall_importance = pd.Series(weighted_importance, index=X_clean.columns)
    overall_importance = overall_importance.sort_values(ascending=False)
    
    return {
        'loadings': loadings,
        'abs_loadings': abs_loadings,
        'top_features_per_pc': top_features_per_pc,
        'overall_importance': overall_importance,
        'explained_variance': explained_var,
        'pca': pca
    }


def analyze_correlation_with_target(X_clean, y_delta):
    """Analyze correlation between features and target (delta)"""
    correlations = []
    
    for col in X_clean.columns:
        corr = np.corrcoef(X_clean[col], y_delta)[0, 1]
        correlations.append({
            'feature': col,
            'correlation': corr,
            'abs_correlation': np.abs(corr)
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    return corr_df


def suggest_ratio_features(important_descriptors):
    """Suggest domain-relevant ratio features based on important descriptors"""
    suggestions = []
    
    # Common chromatography-relevant ratios
    ratio_pairs = [
        ('SASA_hydrophobic', 'SASA_hydrophilic', 'Hydrophobicity ratio'),
        ('SASA_positive', 'SASA_negative', 'Charge balance'),
        ('num_positive', 'num_negative', 'Net charge ratio'),
        ('molecular_weight', 'SASA_total', 'Density proxy'),
        ('num_aromatic', 'molecular_weight', 'Aromaticity density'),
        ('dipole_moment', 'molecular_weight', 'Polarity per mass'),
        ('num_hbond_donors', 'num_hbond_acceptors', 'H-bond donor/acceptor'),
        ('SASA_polar', 'SASA_total', 'Polar fraction'),
        ('num_rotatable_bonds', 'molecular_weight', 'Flexibility per mass'),
    ]
    
    for feat1, feat2, description in ratio_pairs:
        # Check if similar features exist (may have different names)
        feat1_matches = [f for f in important_descriptors if feat1.lower() in f.lower()]
        feat2_matches = [f for f in important_descriptors if feat2.lower() in f.lower()]
        
        if feat1_matches and feat2_matches:
            for f1 in feat1_matches[:2]:  # Top 2 matches
                for f2 in feat2_matches[:2]:
                    suggestions.append({
                        'feature_1': f1,
                        'feature_2': f2,
                        'ratio_name': f'{f1}/{f2}',
                        'description': description
                    })
    
    return pd.DataFrame(suggestions)


def analyze_experiment(data, exp_name, condition_idx, config):
    """Analyze feature importance for one experiment/condition"""
    proteins = data['proteins']
    elution = data['elution']
    descriptors = data['descriptors']
    conditions = data['conditions']
    
    # Calculate deltas
    baseline = elution[:, 0]
    target = elution[:, condition_idx]
    delta = target - baseline
    
    # Remove NaN
    valid_mask = ~(np.isnan(delta) | np.isnan(baseline))
    delta_valid = delta[valid_mask]
    descriptors_valid = descriptors[valid_mask]
    
    if len(delta_valid) < 5:
        return None
    
    # Feature engineering
    X_eng = engineer_features(descriptors_valid, quiet=True)
    X_clean = clean_features(X_eng, var_thresh=config['var_thresh'], 
                            corr_thresh=config['corr_thresh'])
    X_clean.columns = X_clean.columns.astype(str)
    
    print(f"\n{'='*100}")
    print(f"{exp_name} - {conditions[condition_idx]}")
    print(f"{'='*100}")
    print(f"Samples: {len(delta_valid)}")
    print(f"Features after cleaning: {X_clean.shape[1]}")
    
    # PCA loadings analysis
    print(f"\n{'='*50}")
    print("PCA LOADINGS ANALYSIS")
    print(f"{'='*50}")
    
    pca_results = analyze_pca_loadings(X_clean, n_components=config['n_components'])
    
    print(f"\nExplained variance by PC:")
    for i, var in enumerate(pca_results['explained_variance']):
        print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    print(f"  Total: {sum(pca_results['explained_variance']):.3f} ({sum(pca_results['explained_variance'])*100:.1f}%)")
    
    print(f"\nTop 10 most important features (weighted by explained variance):")
    for i, (feat, importance) in enumerate(pca_results['overall_importance'].head(10).items()):
        print(f"  {i+1:2d}. {feat:<50} {importance:.4f}")
    
    # Correlation analysis
    print(f"\n{'='*50}")
    print("CORRELATION WITH TARGET (DELTA)")
    print(f"{'='*50}")
    
    corr_df = analyze_correlation_with_target(X_clean, delta_valid)
    
    print(f"\nTop 10 features by correlation with delta:")
    for i, row in corr_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:<50} r={row['correlation']:+.4f}")
    
    # Feature engineering suggestions
    print(f"\n{'='*50}")
    print("SUGGESTED RATIO FEATURES")
    print(f"{'='*50}")
    
    # Combine PCA and correlation important features
    top_pca = set(pca_results['overall_importance'].head(20).index)
    top_corr = set(corr_df.head(20)['feature'].values)
    important_features = list(top_pca | top_corr)
    
    ratio_suggestions = suggest_ratio_features(important_features)
    
    if len(ratio_suggestions) > 0:
        print(f"\nFound {len(ratio_suggestions)} potential ratio features:")
        for i, row in ratio_suggestions.head(15).iterrows():
            print(f"  {row['ratio_name']:<60} ({row['description']})")
    else:
        print("\nNo obvious ratio pairs found. Consider manual feature engineering.")
    
    return {
        'exp_name': exp_name,
        'condition': conditions[condition_idx],
        'pca_results': pca_results,
        'correlations': corr_df,
        'ratio_suggestions': ratio_suggestions,
        'important_features': important_features
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance and suggest engineered features')
    
    parser.add_argument('--n_components', type=int, default=5, help='PCA components (default: 5)')
    parser.add_argument('--var_thresh', type=float, default=0.01, help='Variance threshold')
    parser.add_argument('--corr_thresh', type=float, default=0.95, help='Correlation threshold')
    parser.add_argument('--experiments', type=str, default='1,3',
                       help='Experiments to analyze')
    parser.add_argument('--output', type=str, default='feature_importance',
                       help='Output filename prefix')
    
    args = parser.parse_args()
    
    config = {
        'n_components': args.n_components,
        'var_thresh': args.var_thresh,
        'corr_thresh': args.corr_thresh
    }
    
    exp_ids = [int(x.strip()) for x in args.experiments.split(',')]
    
    print("="*100)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*100)
    print(f"Strategy:")
    print(f"  1. Analyze PCA loadings to find features contributing most to variance")
    print(f"  2. Analyze correlation with target (delta)")
    print(f"  3. Suggest domain-relevant ratio features")
    print(f"\nParameters:")
    print(f"  PCA components:   {config['n_components']}")
    print(f"  Experiments:      {exp_ids}")
    print("="*100)
    
    csv_path = r"C:\Users\tryg.nelson\predelute\multi_qspr.csv"
    experiments = parse_experiments_corrected(csv_path)
    
    all_results = []
    
    for exp_id in exp_ids:
        exp = experiments[exp_id - 1]
        
        try:
            data = load_experiment_data_corrected(csv_path, exp)
            
            # Analyze each non-baseline condition
            for condition_idx in range(1, len(data['conditions'])):
                result = analyze_experiment(data, exp['name'], condition_idx, config)
                if result:
                    all_results.append(result)
        
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary across all experiments
    if len(all_results) > 0:
        print(f"\n{'='*100}")
        print("SUMMARY ACROSS ALL EXPERIMENTS")
        print(f"{'='*100}")
        
        # Most frequently important features
        all_important = []
        for result in all_results:
            all_important.extend(result['important_features'])
        
        importance_counts = pd.Series(all_important).value_counts()
        
        print(f"\nMost consistently important features:")
        for i, (feat, count) in enumerate(importance_counts.head(15).items()):
            print(f"  {i+1:2d}. {feat:<50} (appeared in {count}/{len(all_results)} analyses)")
        
        # Save results
        output_dir = r"C:\Users\tryg.nelson\predelute\results_corrected"
        
        # Save importance counts
        importance_counts.to_csv(
            os.path.join(output_dir, f'{args.output}_summary.csv'),
            header=['count']
        )
        print(f"\n[SAVED] Summary to {args.output}_summary.csv")
        
        # Save detailed results for each experiment
        for result in all_results:
            exp_name = result['exp_name'].replace(' ', '_').replace('/', '_')
            cond_name = result['condition'].replace(' ', '_').replace('/', '_')
            
            # Save top features
            filename = f"{args.output}_{exp_name}_{cond_name}.csv"
            result['pca_results']['overall_importance'].to_csv(
                os.path.join(output_dir, filename),
                header=['importance']
            )
        
        print(f"[SAVED] Detailed results for {len(all_results)} analyses")
        
        print(f"\n{'='*100}")
        print("RECOMMENDATIONS:")
        print(f"{'='*100}")
        print("1. Focus on top 15 consistently important features")
        print("2. Create ratio features from suggested pairs")
        print("3. Test engineered features with train_delta_loo.py")
        print("4. Monitor if R² improves with new features")
        print(f"{'='*100}")


if __name__ == '__main__':
    main()

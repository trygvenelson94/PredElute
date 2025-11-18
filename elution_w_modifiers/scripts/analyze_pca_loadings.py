#!/usr/bin/env python3
"""
Analyze PCA loadings to identify which features drive each model
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import argparse


def analyze_pca_loadings(model_path, feature_names=None, top_n=10):
    """
    Analyze PCA loadings from a trained model
    
    Parameters:
    -----------
    model_path : str
        Path to pickled model file
    feature_names : list, optional
        List of feature names. If None, will try to extract from model
    top_n : int
        Number of top features to show per component
    """
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract PCA object
    if 'pca' in model_data:
        pca = model_data['pca']
    elif hasattr(model_data, 'named_steps') and 'pca' in model_data.named_steps:
        pca = model_data.named_steps['pca']
    else:
        print(f"Could not find PCA in model: {model_path}")
        return None
    
    # Get feature names
    if feature_names is None:
        if 'feature_names' in model_data:
            feature_names = model_data['feature_names']
        elif 'features' in model_data:
            feature_names = model_data['features']
        else:
            feature_names = [f"Feature_{i}" for i in range(pca.n_features_in_)]
    
    # Get number of components
    n_components = pca.n_components_
    
    print(f"\n{'='*100}")
    print(f"Model: {Path(model_path).name}")
    print(f"{'='*100}")
    print(f"\nTotal features: {len(feature_names)}")
    print(f"PCA components: {n_components}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Analyze each component
    results = []
    
    for pc_idx in range(n_components):
        loadings = pca.components_[pc_idx]
        explained_var = pca.explained_variance_ratio_[pc_idx]
        
        # Get top features by absolute loading
        abs_loadings = np.abs(loadings)
        top_indices = np.argsort(abs_loadings)[-top_n:][::-1]
        
        print(f"\n{'-'*100}")
        print(f"PC{pc_idx + 1} (explains {explained_var:.2%} of variance)")
        print(f"{'-'*100}")
        print(f"{'Rank':<6} {'Feature':<40} {'Loading':<12} {'Abs Loading':<12}")
        print(f"{'-'*100}")
        
        for rank, idx in enumerate(top_indices, 1):
            feat_name = feature_names[idx]
            loading = loadings[idx]
            abs_loading = abs_loadings[idx]
            
            print(f"{rank:<6} {feat_name:<40} {loading:>11.4f} {abs_loading:>11.4f}")
            
            results.append({
                'PC': f'PC{pc_idx + 1}',
                'PC_variance': explained_var,
                'Rank': rank,
                'Feature': feat_name,
                'Loading': loading,
                'Abs_Loading': abs_loading
            })
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results)
    
    # Overall top features (across all PCs)
    print(f"\n{'='*100}")
    print("OVERALL TOP FEATURES (weighted by PC variance)")
    print(f"{'='*100}")
    
    # Weight loadings by explained variance
    weighted_importance = np.zeros(len(feature_names))
    for pc_idx in range(n_components):
        loadings = np.abs(pca.components_[pc_idx])
        variance = pca.explained_variance_ratio_[pc_idx]
        weighted_importance += loadings * variance
    
    top_overall = np.argsort(weighted_importance)[-20:][::-1]
    
    print(f"\n{'Rank':<6} {'Feature':<40} {'Weighted Importance':<20}")
    print(f"{'-'*100}")
    for rank, idx in enumerate(top_overall, 1):
        print(f"{rank:<6} {feature_names[idx]:<40} {weighted_importance[idx]:>19.4f}")
    
    return df_results, weighted_importance, feature_names


def analyze_multiple_models(model_dir, pattern="*.pkl", top_n=10):
    """Analyze all models in a directory"""
    
    model_dir = Path(model_dir)
    model_files = list(model_dir.glob(pattern))
    
    if len(model_files) == 0:
        print(f"No models found in {model_dir} matching pattern {pattern}")
        return
    
    print(f"\nFound {len(model_files)} models in {model_dir}")
    
    all_results = {}
    
    for model_file in model_files:
        try:
            df_results, weighted_importance, feature_names = analyze_pca_loadings(
                str(model_file), top_n=top_n
            )
            all_results[model_file.stem] = {
                'loadings': df_results,
                'importance': weighted_importance,
                'features': feature_names
            }
        except Exception as e:
            print(f"\nError analyzing {model_file.name}: {e}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Analyze PCA loadings from trained models')
    parser.add_argument('--model', type=str, help='Path to single model file')
    parser.add_argument('--model_dir', type=str, help='Directory containing multiple models')
    parser.add_argument('--pattern', type=str, default='*.pkl', help='File pattern for models')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top features per PC')
    parser.add_argument('--output', type=str, help='Output CSV file for results')
    
    args = parser.parse_args()
    
    if args.model:
        # Analyze single model
        df_results, weighted_importance, feature_names = analyze_pca_loadings(
            args.model, top_n=args.top_n
        )
        
        if args.output:
            df_results.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
    
    elif args.model_dir:
        # Analyze multiple models
        all_results = analyze_multiple_models(
            args.model_dir, pattern=args.pattern, top_n=args.top_n
        )
        
        if args.output and all_results:
            # Combine all results
            combined_df = pd.concat([
                df['loadings'].assign(Model=name) 
                for name, df in all_results.items()
            ])
            combined_df.to_csv(args.output, index=False)
            print(f"\nCombined results saved to: {args.output}")
    
    else:
        print("Please specify either --model or --model_dir")
        parser.print_help()


if __name__ == '__main__':
    main()

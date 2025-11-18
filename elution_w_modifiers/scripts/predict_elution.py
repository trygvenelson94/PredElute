"""
Predict protein elution concentration using trained models
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os


def load_model(model_path):
    """Load a trained model from pickle file"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_elution(protein_name, model_data, df_prodes):
    """
    Predict elution for a protein using a trained model
    
    Args:
        protein_name: Name of protein (must be in df_prodes index)
        model_data: Loaded model dictionary
        df_prodes: DataFrame with ProDes descriptors
    
    Returns:
        Predicted elution concentration (M)
    """
    if protein_name not in df_prodes.index:
        raise ValueError(f"Protein '{protein_name}' not found in ProDes descriptors")
    
    # Get descriptors
    X = df_prodes.loc[[protein_name]].values
    
    # Apply same preprocessing as training
    scaler = model_data['scaler']
    pca = model_data['pca']
    model = model_data['model']
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # PCA
    X_pca = pca.transform(X_scaled)
    
    # Predict
    prediction = model.predict(X_pca)[0]
    
    return prediction


def main():
    parser = argparse.ArgumentParser(description='Predict protein elution concentration')
    parser.add_argument('--protein', type=str, required=True, help='Protein name')
    parser.add_argument('--experiment', type=int, help='Specific experiment number (1-9)')
    parser.add_argument('--condition', type=str, help='Specific condition name')
    parser.add_argument('--model', type=str, help='Specific model file (e.g., exp8_0M.pkl)')
    parser.add_argument('--show_all', action='store_true', help='Show predictions for all available models')
    args = parser.parse_args()
    
    print("="*100)
    print(f"PROTEIN ELUTION PREDICTION: {args.protein}")
    print("="*100)
    
    # Load ProDes descriptors
    prodes_csv = r'C:\Users\tryg.nelson\predelute\prodes_descriptors_complete.csv'
    df_prodes = pd.read_csv(prodes_csv, index_col=0)
    
    # Normalize to lowercase
    df_prodes.index = [str(idx).strip().lower() for idx in df_prodes.index]
    protein_name = args.protein.strip().lower()
    
    # Check protein exists
    if protein_name not in df_prodes.index:
        print(f"\n[ERROR] Protein '{args.protein}' not found in ProDes descriptors")
        print(f"\nAvailable proteins:")
        for prot in sorted(df_prodes.index):
            print(f"  - {prot}")
        return
    
    # Load model inventory
    inventory_path = 'trained_models_final/model_inventory.csv'
    if not os.path.exists(inventory_path):
        print(f"\n[ERROR] Model inventory not found: {inventory_path}")
        print("Run compare_full_vs_deduplicated.py first to train models")
        return
    
    inventory = pd.read_csv(inventory_path)
    
    # Filter models if specified
    if args.model:
        selected_models = inventory[inventory['model_file'] == args.model]
    elif args.experiment:
        selected_models = inventory[inventory['experiment'] == args.experiment]
        if args.condition:
            selected_models = selected_models[selected_models['condition'] == args.condition]
    else:
        selected_models = inventory
    
    if len(selected_models) == 0:
        print("\n[ERROR] No matching models found")
        return
    
    print(f"\nPredicting elution for: {args.protein}")
    print(f"Models available: {len(selected_models)}")
    print("-" * 100)
    
    results = []
    
    for _, model_info in selected_models.iterrows():
        model_path = os.path.join('trained_models_final', model_info['model_file'])
        
        if not os.path.exists(model_path):
            print(f"  [SKIP] Model file not found: {model_info['model_file']}")
            continue
        
        try:
            # Load model
            model_data = load_model(model_path)
            
            # Predict
            prediction = predict_elution(protein_name, model_data, df_prodes)
            
            # Get dataset info if available
            dataset_used = model_info.get('dataset_used', 'full')
            r2_used = model_info.get('r2_used', model_info.get('r2_full', np.nan))
            mae_used = model_info.get('mae_used', model_info.get('mae_full', np.nan))
            
            results.append({
                'experiment': model_info['experiment'],
                'experiment_name': model_info['experiment_name'],
                'condition': model_info['condition'],
                'predicted_elution_M': prediction,
                'model_r2': r2_used,
                'model_mae': mae_used,
                'dataset_used': dataset_used
            })
            
            dataset_indicator = f"[{dataset_used}]" if dataset_used == "dedup" else ""
            print(f"  Exp {model_info['experiment']:2d} - {model_info['condition']:25s}: "
                  f"{prediction:.3f} M  (R²={r2_used:.3f}, MAE=±{mae_used:.3f}) {dataset_indicator}")
            
        except Exception as e:
            print(f"  [ERROR] {model_info['model_file']}: {e}")
    
    print("-" * 100)
    
    # Save results
    if results:
        df_results = pd.DataFrame(results)
        output_file = f"predictions_{args.protein.replace(' ', '_')}.csv"
        df_results.to_csv(output_file, index=False)
        
        print(f"\n✓ Predictions saved to: {output_file}")
        print(f"  Total predictions: {len(results)}")
        print(f"  Elution range: {df_results['predicted_elution_M'].min():.3f} - {df_results['predicted_elution_M'].max():.3f} M")
        
        # Highlight best/worst models for this protein
        best_model = df_results.loc[df_results['model_r2'].idxmax()]
        print(f"\n  Best model (highest R²): Exp {best_model['experiment']} - {best_model['condition']}")
        print(f"    Predicted: {best_model['predicted_elution_M']:.3f} M (R²={best_model['model_r2']:.3f})")
    else:
        print("\n[WARNING] No successful predictions made")
    
    print("="*100)


if __name__ == '__main__':
    main()

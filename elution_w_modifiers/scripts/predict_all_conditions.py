#!/usr/bin/env python3
"""
Unified prediction script for:
1. pH-dependent elution (pH 4-10 curves)
2. Elution with modifiers (specific experimental conditions)

Predicts for all proteins in a descriptor CSV file
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

def sigmoid_4param(pH, a, b, c, d):
    """4-parameter sigmoid function"""
    return d + (a - d) / (1 + (pH / c) ** b)

def load_model(model_path):
    """Load a trained model from pickle file"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_ph_curve(protein_name, df_descriptors, model_data, ph_range=None):
    """
    Predict pH-dependent elution curve
    
    Args:
        protein_name: Protein name
        df_descriptors: DataFrame with descriptors (combined ProDes + Schrodinger)
        model_data: Loaded pH model
        ph_range: Array of pH values (default: 4.0 to 10.0, step 0.1)
    
    Returns:
        DataFrame with pH and predicted NaCl
    """
    if ph_range is None:
        ph_range = np.arange(4.0, 10.1, 0.1)
    
    if protein_name not in df_descriptors.index:
        return None
    
    # Get base descriptors
    base_desc = df_descriptors.loc[protein_name].values
    
    # Check if model has engineered features
    if 'top_features' in model_data:
        # Engineered model
        feature_names = model_data['feature_names']
        top_features = model_data['top_features']
        
        predictions = []
        for ph in ph_range:
            # Build engineered features
            features = list(base_desc) + [ph]
            
            # Descriptor × pH interactions
            for feat_name in top_features[:10]:
                if feat_name in feature_names:
                    feat_idx = feature_names.index(feat_name)
                    features.append(base_desc[feat_idx] * ph)
            
            # pH polynomials
            features.extend([ph**2, ph**3])
            
            # Descriptor × descriptor interactions
            for i, feat1_name in enumerate(top_features[:5]):
                if feat1_name not in feature_names:
                    continue
                feat1_idx = feature_names.index(feat1_name)
                for j, feat2_name in enumerate(top_features[:5]):
                    if j <= i or feat2_name not in feature_names:
                        continue
                    feat2_idx = feature_names.index(feat2_name)
                    features.append(base_desc[feat1_idx] * base_desc[feat2_idx])
            
            X = np.array(features).reshape(1, -1)
            X_scaled = model_data['scaler'].transform(X)
            X_pca = model_data['pca'].transform(X_scaled)
            pred = model_data['model'].predict(X_pca)[0]
            predictions.append(pred)
    else:
        # Baseline model
        predictions = []
        for ph in ph_range:
            X = np.concatenate([base_desc, [ph]]).reshape(1, -1)
            X_scaled = model_data['scaler'].transform(X)
            X_pca = model_data['pca'].transform(X_scaled)
            pred = model_data['model'].predict(X_pca)[0]
            predictions.append(pred)
    
    return pd.DataFrame({
        'pH': ph_range,
        'predicted_NaCl_M': predictions
    })

def predict_modifier_condition(protein_name, df_prodes, model_data):
    """
    Predict elution for a specific modifier condition
    
    Args:
        protein_name: Protein name
        df_prodes: DataFrame with ProDes descriptors only
        model_data: Loaded modifier model
    
    Returns:
        Predicted NaCl concentration (M)
    """
    if protein_name not in df_prodes.index:
        return None
    
    # Filter to features used in this specific model
    if 'feature_names' in model_data:
        training_features = model_data['feature_names']
        available_features = [f for f in training_features if f in df_prodes.columns]
        X = df_prodes.loc[[protein_name], available_features].values
    else:
        X = df_prodes.loc[[protein_name]].values
    
    X_scaled = model_data['scaler'].transform(X)
    X_pca = model_data['pca'].transform(X_scaled)
    prediction = model_data['model'].predict(X_pca)[0]
    
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Predict elution for all proteins in descriptor CSV')
    parser.add_argument('--desc_csv', type=str, required=True,
                        help='CSV file with combined ProDes + Schrodinger descriptors (rows=proteins)')
    parser.add_argument('--prodes_csv', type=str, 
                        default='prodes_descriptors_complete.csv',
                        help='CSV file with ProDes descriptors only (for modifier models)')
    parser.add_argument('--ph_model', type=str,
                        default='final_ph_model_engineered/ph_model_engineered.pkl',
                        help='Path to pH-dependent model')
    parser.add_argument('--modifier_dir', type=str,
                        default='trained_models_final',
                        help='Directory containing modifier models')
    parser.add_argument('--output_dir', type=str,
                        default='predictions_all',
                        help='Output directory for results')
    parser.add_argument('--ph_min', type=float, default=4.0)
    parser.add_argument('--ph_max', type=float, default=10.0)
    parser.add_argument('--ph_step', type=float, default=0.1)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*120)
    print("UNIFIED ELUTION PREDICTION")
    print("="*120)
    print(f"\nDescriptor CSV: {args.desc_csv}")
    print(f"pH model: {args.ph_model}")
    print(f"Modifier models: {args.modifier_dir}")
    print(f"Output: {args.output_dir}")
    
    # ========================================================================
    # LOAD DESCRIPTORS
    # ========================================================================
    print("\n[1/4] Loading descriptors...")
    
    # Combined descriptors for pH model
    df_combined = pd.read_csv(args.desc_csv, index_col=0)
    df_combined.index = df_combined.index.str.lower().str.strip()
    
    # ProDes only for modifier models
    df_prodes = pd.read_csv(args.prodes_csv, index_col=0)
    df_prodes.index = df_prodes.index.str.lower().str.strip()
    
    proteins = df_combined.index.tolist()
    print(f"  Proteins to predict: {len(proteins)}")
    for prot in proteins:
        print(f"    - {prot}")
    
    print(f"  Initial features: {df_combined.shape[1]}")
    
    # ========================================================================
    # LOAD MODELS
    # ========================================================================
    print("\n[2/4] Loading models...")
    
    # pH model
    if os.path.exists(args.ph_model):
        ph_model = load_model(args.ph_model)
        print(f"  ✓ pH model loaded (R² = {ph_model.get('r2_loo', 'N/A')})")
        
        # Apply feature filtering to match training
        if 'feature_names' in ph_model:
            # Filter to only the features used in training
            training_features = ph_model['feature_names']
            missing_features = [f for f in training_features if f not in df_combined.columns]
            
            if missing_features:
                print(f"  ⚠️  Warning: {len(missing_features)} training features not found in input data")
                print(f"      This may affect prediction quality")
            
            # Keep only features that exist in both
            available_features = [f for f in training_features if f in df_combined.columns]
            df_combined = df_combined[available_features]
            print(f"  Filtered to {len(available_features)} features (matching training)")
        
        has_ph_model = True
    else:
        print(f"  ⚠️  pH model not found: {args.ph_model}")
        has_ph_model = False
    
    # Modifier models
    modifier_models = {}
    inventory_path = os.path.join(args.modifier_dir, 'model_inventory.csv')
    
    if os.path.exists(inventory_path):
        inventory = pd.read_csv(inventory_path)
        print(f"  ✓ Found {len(inventory)} modifier models")
        
        for _, row in inventory.iterrows():
            model_path = os.path.join(args.modifier_dir, row['model_file'])
            if os.path.exists(model_path):
                model_data = load_model(model_path)
                key = f"Exp{row['experiment']}_{row['condition']}"
                modifier_models[key] = {
                    'model': model_data,
                    'experiment': row['experiment'],
                    'experiment_name': row['experiment_name'],
                    'condition': row['condition'],
                    'r2': row.get('r2_used', 'N/A'),
                    'mae': row.get('mae_used', 0.05)
                }
    else:
        print(f"  ⚠️  Modifier model inventory not found: {inventory_path}")
    
    # ========================================================================
    # PREDICT pH CURVES
    # ========================================================================
    if has_ph_model:
        print("\n[3/4] Predicting pH-dependent elution curves...")
        
        ph_range = np.arange(args.ph_min, args.ph_max + args.ph_step/2, args.ph_step)
        
        all_ph_predictions = {}
        
        for prot in proteins:
            pred_df = predict_ph_curve(prot, df_combined, ph_model, ph_range)
            if pred_df is not None:
                all_ph_predictions[prot] = pred_df
                
                # Save individual CSV
                output_file = os.path.join(args.output_dir, f"{prot}_ph_curve.csv")
                pred_df.to_csv(output_file, index=False)
                print(f"  ✓ {prot}: pH {args.ph_min}-{args.ph_max} -> {output_file}")
            else:
                print(f"  ✗ {prot}: Not found in descriptors")
        
        # Create combined plot with all proteins on same axes
        if all_ph_predictions:
            n_proteins = len(all_ph_predictions)
            
            # Create single plot with all proteins
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Color map for proteins
            colors = plt.cm.tab10(np.linspace(0, 1, n_proteins))
            
            # Store pH 9 and 10 predictions for text box
            predictions_text = []
            
            for idx, (prot, pred_df) in enumerate(all_ph_predictions.items()):
                color = colors[idx]
                
                # Plot raw predictions as scatter points
                ax.scatter(pred_df['pH'], pred_df['predicted_NaCl_M'], 
                          s=40, alpha=0.5, color=color, zorder=3)
                
                # Fit sigmoid to predictions (pH 4-8 only for fitting)
                ph_fit = pred_df[pred_df['pH'] <= 8.0]['pH'].values
                nacl_fit = pred_df[pred_df['pH'] <= 8.0]['predicted_NaCl_M'].values
                
                try:
                    # Fit sigmoid
                    popt, _ = curve_fit(sigmoid_4param, ph_fit, nacl_fit,
                                       p0=[max(nacl_fit), 2, 6, min(nacl_fit)],
                                       maxfev=5000,
                                       bounds=([0, 0.1, 3, 0], [3, 10, 10, 3]))
                    
                    # Generate smooth curve from pH 4-10
                    ph_smooth = np.linspace(4, args.ph_max, 200)
                    nacl_smooth = sigmoid_4param(ph_smooth, *popt)
                    
                    ax.plot(ph_smooth, nacl_smooth, '-', linewidth=2.5, 
                           label=prot.title(), alpha=0.8, color=color, zorder=2)
                    
                    # Store predictions
                    pred_ph9 = sigmoid_4param(9, *popt)
                    pred_ph10 = sigmoid_4param(10, *popt)
                    predictions_text.append(f"{prot.title()}: pH9={pred_ph9:.3f}M, pH10={pred_ph10:.3f}M")
                except:
                    # If sigmoid fitting fails, just plot the raw predictions
                    ax.plot(pred_df['pH'], pred_df['predicted_NaCl_M'], 
                           '-', linewidth=2.5, alpha=0.8, color=color, 
                           label=prot.title(), zorder=2)
            
            # Shade extrapolation region
            ax.axvspan(8, args.ph_max, alpha=0.15, color='red', label='Extrapolation region', zorder=1)
            ax.axvline(x=8, color='red', linestyle='--', alpha=0.5, linewidth=2)
            
            # Add prediction text box
            if predictions_text:
                textstr = '\n'.join(predictions_text)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
            
            ax.set_xlabel('pH', fontsize=14, fontweight='bold')
            ax.set_ylabel('NaCl Concentration (M)', fontsize=14, fontweight='bold')
            ax.set_title('pH-Dependent Elution Predictions with Extrapolation', 
                        fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='best', framealpha=0.9)
            ax.set_xlim(3.5, args.ph_max + 0.5)
            
            # Set y-axis minimum to 0
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(0, y_max)
            
            plt.tight_layout()
            plot_file = os.path.join(args.output_dir, 'ph_curves_all.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"\n  ✓ Saved combined plot: {plot_file}")
    
    # ========================================================================
    # PREDICT MODIFIER CONDITIONS
    # ========================================================================
    if modifier_models:
        print("\n[4/4] Predicting elution with modifiers...")
        
        modifier_results = []
        
        for prot in proteins:
            if prot not in df_prodes.index:
                print(f"  ✗ {prot}: Not found in ProDes descriptors")
                continue
            
            prot_results = {'protein': prot}
            
            for key, model_info in modifier_models.items():
                pred = predict_modifier_condition(prot, df_prodes, model_info['model'])
                if pred is not None:
                    col_name = f"{model_info['experiment_name']}_{model_info['condition']}"
                    prot_results[col_name] = pred
            
            modifier_results.append(prot_results)
            print(f"  ✓ {prot}: {len(prot_results)-1} conditions predicted")
        
        # Save to CSV
        if modifier_results:
            modifier_df = pd.DataFrame(modifier_results)
            output_file = os.path.join(args.output_dir, 'modifier_predictions.csv')
            modifier_df.to_csv(output_file, index=False)
            print(f"\n  ✓ Saved modifier predictions: {output_file}")
            
            # Create scatter plot with error bars (R² > 0.6 only)
            print("\n  Creating modifier predictions scatter plot...")
            
            # Filter models by R² > 0.6
            good_models = {k: v for k, v in modifier_models.items() 
                          if isinstance(v['r2'], (int, float)) and v['r2'] > 0.6}
            
            if good_models:
                # Prepare data for plotting
                plot_data = []
                for key, model_info in good_models.items():
                    col_name = f"{model_info['experiment_name']}_{model_info['condition']}"
                    mae = model_info.get('mae', 0.05)  # Get MAE from inventory
                    
                    for prot in proteins:
                        if prot in df_prodes.index and col_name in modifier_df.columns:
                            pred_val = modifier_df[modifier_df['protein'] == prot][col_name].values
                            if len(pred_val) > 0 and not pd.isna(pred_val[0]):
                                plot_data.append({
                                    'protein': prot,
                                    'condition': col_name,
                                    'prediction': pred_val[0],
                                    'mae': mae,
                                    'r2': model_info['r2']
                                })
                
                if plot_data:
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Get unique conditions and proteins
                    conditions = plot_df['condition'].unique()
                    n_conditions = len(conditions)
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(max(12, n_conditions * 0.8), 8))
                    
                    # Color map for proteins
                    colors = plt.cm.tab10(np.linspace(0, 1, len(proteins)))
                    protein_colors = {prot: colors[i] for i, prot in enumerate(proteins)}
                    
                    # Plot each protein
                    x_positions = np.arange(n_conditions)
                    offset_step = 0.15
                    n_proteins = len(proteins)
                    offsets = np.linspace(-offset_step * (n_proteins-1)/2, 
                                         offset_step * (n_proteins-1)/2, 
                                         n_proteins)
                    protein_offsets = {prot: offsets[i] for i, prot in enumerate(proteins)}
                    
                    for prot in proteins:
                        prot_data = plot_df[plot_df['protein'] == prot]
                        if len(prot_data) == 0:
                            continue
                        
                        x_vals = []
                        y_vals = []
                        yerr_vals = []
                        
                        for cond in conditions:
                            cond_data = prot_data[prot_data['condition'] == cond]
                            if len(cond_data) > 0:
                                x_idx = list(conditions).index(cond)
                                x_vals.append(x_idx + protein_offsets[prot])
                                y_vals.append(cond_data['prediction'].values[0])
                                yerr_vals.append(cond_data['mae'].values[0])
                        
                        if x_vals:
                            ax.errorbar(x_vals, y_vals, yerr=yerr_vals,
                                       fmt='o', markersize=8, capsize=5, capthick=2,
                                       label=prot.title(), color=protein_colors[prot],
                                       alpha=0.8, linewidth=2)
                    
                    # Formatting
                    # Add vertical lines between conditions
                    for i in range(1, n_conditions):
                        ax.axvline(x=i - 0.5, color='gray', linestyle='-', linewidth=1, alpha=0.3)
                    
                    # Set tick marks between conditions (on the dividing lines)
                    tick_positions = [-0.5] + [i - 0.5 for i in range(1, n_conditions)] + [n_conditions - 0.5]
                    ax.set_xticks(tick_positions, minor=False)
                    ax.set_xticklabels([''] * len(tick_positions))  # No labels on dividing lines
                    
                    # Add condition labels centered in each section
                    for idx, cond in enumerate(conditions):
                        ax.text(idx, -0.02, cond, ha='center', va='top', fontsize=10, 
                               rotation=45, transform=ax.get_xaxis_transform())
                    
                    ax.set_xlabel('Experiment + Condition', fontsize=13, fontweight='bold')
                    ax.set_ylabel('Predicted NaCl Concentration (M)', fontsize=13, fontweight='bold')
                    ax.set_title('Elution Predictions with Modifiers (R² > 0.6)\nError bars = Model MAE', 
                                fontsize=14, fontweight='bold')
                    ax.legend(fontsize=10, loc='best', ncol=min(3, len(proteins)))
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Set y-axis minimum to 0
                    y_min, y_max = ax.get_ylim()
                    ax.set_ylim(0, y_max)
                    
                    # Add R² labels below each condition
                    for idx, cond in enumerate(conditions):
                        # Get R² for this condition
                        cond_data = plot_df[plot_df['condition'] == cond]
                        if len(cond_data) > 0:
                            r2_val = cond_data['r2'].values[0]
                            mae_val = cond_data['mae'].values[0]
                            label_text = f"R²={r2_val:.3f}\nMAE={mae_val:.3f}M"
                            ax.text(idx, -0.08 * y_max, label_text, 
                                   ha='center', va='top', fontsize=8,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
                    
                    plt.tight_layout()
                    plot_file = os.path.join(args.output_dir, 'modifier_predictions_scatter.png')
                    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                    print(f"  ✓ Saved scatter plot: {plot_file}")
                    print(f"    Models plotted: {len(good_models)} (R² > 0.6)")
                    print(f"    Total predictions: {len(plot_data)}")
                else:
                    print(f"  ⚠️  No valid predictions to plot")
            else:
                print(f"  ⚠️  No models with R² > 0.6 found")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*120)
    print("PREDICTION COMPLETE")
    print("="*120)
    print(f"\nResults saved to: {args.output_dir}/")
    if has_ph_model:
        print(f"  - Individual pH curves: {len(all_ph_predictions)} CSV files")
        print(f"  - pH curves plot: ph_curves_all.png (with sigmoid fit + extrapolation)")
    if modifier_models:
        print(f"  - Modifier predictions: modifier_predictions.csv")
        print(f"  - Modifier scatter plot: modifier_predictions_scatter.png (R² > 0.6 only)")
    print("="*120)

if __name__ == '__main__':
    main()

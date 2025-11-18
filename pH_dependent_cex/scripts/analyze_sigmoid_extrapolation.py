#!/usr/bin/env python3
"""
Analyze which proteins are safe for sigmoid extrapolation
Evaluates fit quality and data coverage
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

def sigmoid(pH, a, b, c, d):
    """Sigmoidal function"""
    return a / (1 + np.exp(-b * (pH - c))) + d

print("="*120)
print("SIGMOID EXTRAPOLATION SAFETY ANALYSIS")
print("="*120)

# Load data
df_nacl = pd.read_csv('sp_sepharose_hp_nacl_concentrations.csv')
df_nacl['protein_name'] = df_nacl['protein_name'].str.lower().str.strip()
df_nacl = df_nacl.set_index('protein_name')

# Convert to numeric
ph_columns = ['pH4', 'pH5', 'pH6', 'pH7', 'pH8']
for col in ph_columns:
    if col in df_nacl.columns:
        df_nacl[col] = pd.to_numeric(df_nacl[col], errors='coerce')

ph_values = np.array([4, 5, 6, 7, 8])

print("\n[1/2] Fitting sigmoid curves and evaluating quality...")
print("="*120)

results = []

for protein in df_nacl.index:
    # Get elution data
    y_data = df_nacl.loc[protein, ph_columns].values
    y_data = pd.to_numeric(pd.Series(y_data), errors='coerce').values
    valid_mask = ~np.isnan(y_data)
    
    n_points = valid_mask.sum()
    missing_ph = ph_values[~valid_mask].tolist()
    measured_ph = ph_values[valid_mask].tolist()
    
    # Determine pH range
    if n_points > 0:
        ph_min = ph_values[valid_mask].min()
        ph_max = ph_values[valid_mask].max()
        ph_range = ph_max - ph_min
    else:
        ph_min = ph_max = ph_range = np.nan
    
    # Try sigmoid fit
    fit_quality = "NO_FIT"
    r2 = np.nan
    rmse = np.nan
    params = [np.nan] * 4
    extrapolation_safety = "UNSAFE"
    
    if n_points >= 4:
        ph_valid = ph_values[valid_mask]
        y_valid = y_data[valid_mask]
        
        # Initial guess
        y_range = y_valid.max() - y_valid.min()
        y_min = y_valid.min()
        mid_ph = ph_valid[len(ph_valid)//2]
        p0 = [y_range, 1.0, mid_ph, y_min]
        
        try:
            params, pcov = curve_fit(sigmoid, ph_valid, y_valid, p0=p0, maxfev=10000,
                                     bounds=([0, -10, 3, -1], [2, 10, 9, 2]))
            
            # Calculate fit quality
            y_pred = sigmoid(ph_valid, *params)
            ss_res = np.sum((y_valid - y_pred)**2)
            ss_tot = np.sum((y_valid - y_valid.mean())**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((y_valid - y_pred)**2))
            
            # Categorize fit quality
            if r2 >= 0.95:
                fit_quality = "EXCELLENT"
            elif r2 >= 0.90:
                fit_quality = "GOOD"
            elif r2 >= 0.80:
                fit_quality = "FAIR"
            else:
                fit_quality = "POOR"
            
            # Assess extrapolation safety
            # Safe if: good fit + wide pH range + not extrapolating too far
            if r2 >= 0.90 and ph_range >= 3:
                extrapolation_safety = "SAFE"
            elif r2 >= 0.85 and ph_range >= 2:
                extrapolation_safety = "MODERATE"
            else:
                extrapolation_safety = "RISKY"
                
        except Exception as e:
            fit_quality = f"FAILED: {str(e)[:20]}"
    
    elif n_points == 3:
        fit_quality = "INSUFFICIENT (3 pts)"
    elif n_points == 2:
        fit_quality = "INSUFFICIENT (2 pts)"
    elif n_points == 1:
        fit_quality = "INSUFFICIENT (1 pt)"
    else:
        fit_quality = "NO DATA"
    
    results.append({
        'protein': protein,
        'n_measured': n_points,
        'n_missing': 5 - n_points,
        'measured_pH': measured_ph,
        'missing_pH': missing_ph,
        'pH_range': ph_range,
        'fit_quality': fit_quality,
        'r2': r2,
        'rmse': rmse,
        'extrapolation_safety': extrapolation_safety,
        'a': params[0],
        'b': params[1],
        'c': params[2],
        'd': params[3]
    })

df_results = pd.DataFrame(results)

# Sort by extrapolation safety and R²
safety_order = {'SAFE': 0, 'MODERATE': 1, 'RISKY': 2, 'UNSAFE': 3}
df_results['safety_rank'] = df_results['extrapolation_safety'].map(safety_order)
df_results = df_results.sort_values(['safety_rank', 'r2'], ascending=[True, False])

print("\n[2/2] EXTRAPOLATION SAFETY REPORT")
print("="*120)

# SAFE proteins
safe = df_results[df_results['extrapolation_safety'] == 'SAFE']
if len(safe) > 0:
    print(f"\n✓ SAFE FOR EXTRAPOLATION ({len(safe)} proteins)")
    print(f"  Criteria: R² ≥ 0.90, pH range ≥ 3")
    print("-"*120)
    print(f"{'Protein':<30} {'Measured':<12} {'Missing':<12} {'R²':<8} {'RMSE':<8} {'Fit Quality'}")
    print("-"*120)
    for _, row in safe.iterrows():
        measured_str = ','.join(map(str, row['measured_pH']))
        missing_str = ','.join(map(str, row['missing_pH'])) if row['missing_pH'] else 'None'
        print(f"{row['protein']:<30} {measured_str:<12} {missing_str:<12} {row['r2']:<8.4f} {row['rmse']:<8.4f} {row['fit_quality']}")

# MODERATE proteins
moderate = df_results[df_results['extrapolation_safety'] == 'MODERATE']
if len(moderate) > 0:
    print(f"\n⚠ MODERATE RISK ({len(moderate)} proteins)")
    print(f"  Criteria: R² ≥ 0.85, pH range ≥ 2")
    print("-"*120)
    print(f"{'Protein':<30} {'Measured':<12} {'Missing':<12} {'R²':<8} {'RMSE':<8} {'Fit Quality'}")
    print("-"*120)
    for _, row in moderate.iterrows():
        measured_str = ','.join(map(str, row['measured_pH']))
        missing_str = ','.join(map(str, row['missing_pH'])) if row['missing_pH'] else 'None'
        print(f"{row['protein']:<30} {measured_str:<12} {missing_str:<12} {row['r2']:<8.4f} {row['rmse']:<8.4f} {row['fit_quality']}")

# RISKY proteins
risky = df_results[df_results['extrapolation_safety'] == 'RISKY']
if len(risky) > 0:
    print(f"\n✗ RISKY ({len(risky)} proteins)")
    print(f"  Criteria: Poor fit (R² < 0.85) or narrow pH range")
    print("-"*120)
    print(f"{'Protein':<30} {'Measured':<12} {'Missing':<12} {'R²':<8} {'RMSE':<8} {'Fit Quality'}")
    print("-"*120)
    for _, row in risky.iterrows():
        measured_str = ','.join(map(str, row['measured_pH']))
        missing_str = ','.join(map(str, row['missing_pH'])) if row['missing_pH'] else 'None'
        r2_str = f"{row['r2']:.4f}" if not np.isnan(row['r2']) else "N/A"
        rmse_str = f"{row['rmse']:.4f}" if not np.isnan(row['rmse']) else "N/A"
        print(f"{row['protein']:<30} {measured_str:<12} {missing_str:<12} {r2_str:<8} {rmse_str:<8} {row['fit_quality']}")

# UNSAFE proteins
unsafe = df_results[df_results['extrapolation_safety'] == 'UNSAFE']
if len(unsafe) > 0:
    print(f"\n✗ UNSAFE - CANNOT EXTRAPOLATE ({len(unsafe)} proteins)")
    print(f"  Reason: Insufficient data points (< 4) or failed fit")
    print("-"*120)
    print(f"{'Protein':<30} {'Measured':<12} {'Missing':<12} {'Reason'}")
    print("-"*120)
    for _, row in unsafe.iterrows():
        measured_str = ','.join(map(str, row['measured_pH']))
        missing_str = ','.join(map(str, row['missing_pH'])) if row['missing_pH'] else 'None'
        print(f"{row['protein']:<30} {measured_str:<12} {missing_str:<12} {row['fit_quality']}")

# Summary statistics
print("\n" + "="*120)
print("SUMMARY")
print("="*120)
print(f"Total proteins: {len(df_results)}")
print(f"  ✓ Safe for extrapolation: {len(safe)} ({100*len(safe)/len(df_results):.1f}%)")
print(f"  ⚠ Moderate risk: {len(moderate)} ({100*len(moderate)/len(df_results):.1f}%)")
print(f"  ✗ Risky: {len(risky)} ({100*len(risky)/len(df_results):.1f}%)")
print(f"  ✗ Unsafe: {len(unsafe)} ({100*len(unsafe)/len(df_results):.1f}%)")

# Potential data gain
safe_and_moderate = df_results[df_results['extrapolation_safety'].isin(['SAFE', 'MODERATE'])]
total_missing = safe_and_moderate['n_missing'].sum()
total_current = df_results['n_measured'].sum()
potential_total = total_current + total_missing

print(f"\nPotential data gain from extrapolation (SAFE + MODERATE):")
print(f"  Current data points: {total_current}")
print(f"  Missing points that could be filled: {total_missing}")
print(f"  Potential total: {potential_total} (+{100*total_missing/total_current:.1f}%)")

# Save results
df_results.to_csv('sigmoid_extrapolation_analysis.csv', index=False)
print(f"\nDetailed results saved to: sigmoid_extrapolation_analysis.csv")

print("\n" + "="*120)
print("RECOMMENDATIONS")
print("="*120)
print("1. Use SAFE proteins for extrapolation without hesitation")
print("2. Use MODERATE proteins with caution - validate if possible")
print("3. Do NOT extrapolate RISKY proteins - poor fit quality")
print("4. Do NOT extrapolate UNSAFE proteins - insufficient data")
print("="*120)

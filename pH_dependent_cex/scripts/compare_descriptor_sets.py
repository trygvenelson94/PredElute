#!/usr/bin/env python3
"""
Compare performance of ProDes-only vs Combined (ProDes + Schrodinger) descriptors
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*100)
print("COMPARING PRODES-ONLY VS COMBINED DESCRIPTORS")
print("="*100)

# Load results
df_prodes = pd.read_csv('model_performance_with_interactions.csv')
df_combined = pd.read_csv('model_performance_combined_descriptors.csv')

print(f"\nProDes-only models: {len(df_prodes)}")
print(f"Combined models: {len(df_combined)}")

# ProDes models have actual column names, combined has col_0, col_1, etc.
# We need to match by position
print("\n" + "="*100)
print("OVERALL COMPARISON")
print("="*100)

print(f"\nProDes-only (116 features: 105 ProDes + 11 interactions):")
print(f"  Mean R²: {df_prodes['r2_loo'].mean():.4f} ± {df_prodes['r2_loo'].std():.4f}")
print(f"  Mean RMSE: {df_prodes['rmse_loo'].mean():.4f} ± {df_prodes['rmse_loo'].std():.4f}")

print(f"\nCombined (268 features: 105 ProDes + 166 Schrodinger - 3 with NaN):")
print(f"  Mean R²: {df_combined['r2_loo'].mean():.4f} ± {df_combined['r2_loo'].std():.4f}")
print(f"  Mean RMSE: {df_combined['rmse_loo'].mean():.4f} ± {df_combined['rmse_loo'].std():.4f}")

# Calculate improvement
r2_improvement = df_combined['r2_loo'].mean() - df_prodes['r2_loo'].mean()
rmse_improvement = df_prodes['rmse_loo'].mean() - df_combined['rmse_loo'].mean()

print(f"\nImprovement:")
print(f"  R² change: {r2_improvement:+.4f}")
print(f"  RMSE change: {rmse_improvement:+.4f}")

# Count how many models improved
if len(df_prodes) == len(df_combined):
    df_comparison = pd.DataFrame({
        'column': df_prodes['column'],
        'r2_prodes': df_prodes['r2_loo'].values,
        'r2_combined': df_combined['r2_loo'].values,
        'rmse_prodes': df_prodes['rmse_loo'].values,
        'rmse_combined': df_combined['rmse_loo'].values
    })
    
    df_comparison['r2_improvement'] = df_comparison['r2_combined'] - df_comparison['r2_prodes']
    df_comparison['rmse_improvement'] = df_comparison['rmse_prodes'] - df_comparison['rmse_combined']
    
    improved_r2 = (df_comparison['r2_improvement'] > 0).sum()
    improved_rmse = (df_comparison['rmse_improvement'] > 0).sum()
    
    print(f"\nModels improved (R²): {improved_r2}/{len(df_comparison)} ({100*improved_r2/len(df_comparison):.1f}%)")
    print(f"Models improved (RMSE): {improved_rmse}/{len(df_comparison)} ({100*improved_rmse/len(df_comparison):.1f}%)")
    
    print(f"\nTop 5 improvements (R²):")
    for idx, row in df_comparison.nlargest(5, 'r2_improvement').iterrows():
        print(f"  {row['column']:40s} {row['r2_improvement']:+.4f}  (ProDes: {row['r2_prodes']:.4f} → Combined: {row['r2_combined']:.4f})")
    
    print(f"\nTop 5 declines (R²):")
    for idx, row in df_comparison.nsmallest(5, 'r2_improvement').iterrows():
        print(f"  {row['column']:40s} {row['r2_improvement']:+.4f}  (ProDes: {row['r2_prodes']:.4f} → Combined: {row['r2_combined']:.4f})")
    
    df_comparison.to_csv('descriptor_comparison.csv', index=False)
    print(f"\nSaved detailed comparison to: descriptor_comparison.csv")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² comparison
    axes[0].scatter(df_comparison['r2_prodes'], df_comparison['r2_combined'], alpha=0.6, s=100)
    axes[0].plot([df_comparison['r2_prodes'].min(), df_comparison['r2_prodes'].max()],
                 [df_comparison['r2_prodes'].min(), df_comparison['r2_prodes'].max()],
                 'k--', alpha=0.3, label='y=x')
    axes[0].set_xlabel('ProDes-only R²', fontsize=12)
    axes[0].set_ylabel('Combined R²', fontsize=12)
    axes[0].set_title('R² Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # RMSE comparison
    axes[1].scatter(df_comparison['rmse_prodes'], df_comparison['rmse_combined'], alpha=0.6, s=100)
    axes[1].plot([df_comparison['rmse_prodes'].min(), df_comparison['rmse_prodes'].max()],
                 [df_comparison['rmse_prodes'].min(), df_comparison['rmse_prodes'].max()],
                 'k--', alpha=0.3, label='y=x')
    axes[1].set_xlabel('ProDes-only RMSE', fontsize=12)
    axes[1].set_ylabel('Combined RMSE', fontsize=12)
    axes[1].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('descriptor_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: descriptor_comparison.png")

print("\n" + "="*100)
print("CONCLUSION")
print("="*100)

if r2_improvement > 0:
    print(f"✓ Combined descriptors show {abs(r2_improvement):.4f} improvement in mean R²")
else:
    print(f"✗ ProDes-only descriptors perform {abs(r2_improvement):.4f} better in mean R²")

if rmse_improvement > 0:
    print(f"✓ Combined descriptors show {abs(rmse_improvement):.4f} improvement in mean RMSE")
else:
    print(f"✗ ProDes-only descriptors perform {abs(rmse_improvement):.4f} better in mean RMSE")

print("="*100)

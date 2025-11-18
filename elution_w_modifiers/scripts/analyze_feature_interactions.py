#!/usr/bin/env python3
"""
Analyze feature importance patterns to identify potential feature interactions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

# Load feature importance matrix
df_importance = pd.read_csv('feature_importance_matrix.csv', index_col=0)

print("="*100)
print("FEATURE INTERACTION ANALYSIS")
print("="*100)

# 1. CORRELATION OF FEATURE IMPORTANCE ACROSS MODELS
print("\n1. FINDING FEATURES WITH CORRELATED IMPORTANCE PATTERNS")
print("-"*100)
print("Features with similar importance patterns across models likely interact or")
print("represent related physicochemical properties.\n")

# Calculate correlation between features based on their importance patterns
corr_matrix = df_importance.T.corr(method='spearman')

# Find highly correlated pairs (excluding self-correlation)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.7:  # Strong correlation threshold
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr,
                'type': 'positive' if corr > 0 else 'negative'
            })

df_corr_pairs = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)

print(f"Found {len(df_corr_pairs)} feature pairs with |correlation| > 0.7:")
print(df_corr_pairs.head(20).to_string(index=False))

# 2. IDENTIFY COMPLEMENTARY FEATURES
print("\n\n2. COMPLEMENTARY FEATURE PATTERNS")
print("-"*100)
print("Features that are important in different models may capture complementary information.\n")

# Calculate variance of importance across models
importance_variance = df_importance.var(axis=1).sort_values(ascending=False)

print("Features with HIGHEST variance (model-specific importance):")
print(importance_variance.head(15))

print("\nFeatures with LOWEST variance (consistently important):")
print(importance_variance.tail(15))

# 3. DOMAIN-SPECIFIC INTERACTION CANDIDATES
print("\n\n3. DOMAIN-SPECIFIC INTERACTION CANDIDATES")
print("-"*100)

# Group features by type
surface_fracs = [f for f in df_importance.index if 'SurfFrac' in f]
charge_features = [f for f in df_importance.index if 'Ep' in f or 'charge' in f.lower()]
hydrophobic_features = [f for f in df_importance.index if 'Mhp' in f]
shape_features = [f for f in df_importance.index if 'Shape' in f or 'Area' in f]

print("\nFeature categories:")
print(f"  Surface fractions: {len(surface_fracs)} features")
print(f"  Charge-related: {len(charge_features)} features")
print(f"  Hydrophobicity: {len(hydrophobic_features)} features")
print(f"  Shape/geometry: {len(shape_features)} features")

# Propose interaction terms
print("\n\nPROPOSED INTERACTION TERMS:")
print("-"*100)

interactions = []

# 1. Charged amino acid ratios (balance)
charged_positive = ['ARGSurfFrac', 'LYSSurfFrac', 'HISSurfFrac']
charged_negative = ['ASPSurfFrac', 'GLUSurfFrac']
print("\n1. CHARGE BALANCE INTERACTIONS:")
print("   - (ARG + LYS + HIS) / (ASP + GLU) ratio [cation/anion balance]")
print("   - ARG/LYS ratio [arginine vs lysine dominance]")
interactions.append(('charge_balance', 'sum(ARG,LYS,HIS)/sum(ASP,GLU)'))
interactions.append(('arg_lys_ratio', 'ARGSurfFrac/LYSSurfFrac'))

# 2. Aromatic cluster
aromatic = ['TRPSurfFrac', 'TYRSurfFrac', 'PHESurfFrac']
print("\n2. AROMATIC INTERACTIONS:")
print("   - (TRP + TYR + PHE) sum [aromatic patch size]")
print("   - TRP * TYR [tryptophan-tyrosine synergy]")
interactions.append(('aromatic_sum', 'sum(TRP,TYR,PHE)'))
interactions.append(('trp_tyr_interaction', 'TRPSurfFrac*TYRSurfFrac'))

# 3. Hydrophobic + aromatic
print("\n3. HYDROPHOBIC-AROMATIC PATCHES:")
print("   - (TRP + TYR) * SurfMhpMean [aromatic in hydrophobic regions]")
print("   - ILE * LEU * VAL [aliphatic hydrophobic cluster]")
interactions.append(('aromatic_hydrophobic', '(TRPSurfFrac+TYRSurfFrac)*SurfMhpMean'))
interactions.append(('aliphatic_cluster', 'ILESurfFrac*LEUSurfFrac*VALSurfFrac'))

# 4. Polar + charged
polar = ['SERSurfFrac', 'THRSurfFrac', 'GLNSurfFrac', 'ASNSurfFrac']
print("\n4. POLAR-CHARGE INTERACTIONS:")
print("   - (SER + THR) * (ARG + LYS) [polar near positive charges]")
print("   - GLN * charge features [glutamine-mediated charge interactions]")
interactions.append(('polar_positive', '(SERSurfFrac+THRSurfFrac)*(ARGSurfFrac+LYSSurfFrac)'))
interactions.append(('gln_charge', 'GLNSurfFrac*SurfEpPosSumAverage'))

# 5. Shape-charge coupling
print("\n5. SHAPE-CHARGE COUPLING:")
print("   - Shape_max * ARGSurfFrac [arginine on convex surfaces]")
print("   - Shape_min * charge density [charges in concave pockets]")
interactions.append(('shape_arg', 'Shape max*ARGSurfFrac'))
interactions.append(('pocket_charge', 'Shape min*SurfEpPosSumAverage'))

# 6. Isoelectric point interactions
print("\n6. pI-DEPENDENT INTERACTIONS:")
print("   - Isoelectric_point * pH-dependent features")
print("   - (pI - pH) * charge features [distance from pI]")
interactions.append(('pi_charge', 'Isoelectric point*SurfEpPosSumAverage'))

# 4. STATISTICAL EVIDENCE FOR INTERACTIONS
print("\n\n4. DETECTING INTERACTIONS FROM MODEL PATTERNS")
print("-"*100)

# Look for models where multiple features are simultaneously important
top_n = 5
model_top_features = {}

for col in df_importance.columns:
    top_features = df_importance[col].nlargest(top_n).index.tolist()
    model_top_features[col] = top_features

# Find co-occurring pairs
cooccurrence = {}
for model, features in model_top_features.items():
    for i, f1 in enumerate(features):
        for f2 in features[i+1:]:
            pair = tuple(sorted([f1, f2]))
            cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

# Sort by frequency
sorted_cooccur = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)

print(f"\nFeature pairs that co-occur in top {top_n} (multiple models):")
print(f"{'Feature 1':<30} {'Feature 2':<30} {'Co-occur in N models'}")
print("-"*100)
for (f1, f2), count in sorted_cooccur[:25]:
    if count >= 3:  # Appears together in at least 3 models
        print(f"{f1:<30} {f2:<30} {count}/27")

# 5. SAVE INTERACTION PROPOSALS
print("\n\n5. SAVING INTERACTION PROPOSALS")
print("-"*100)

df_interactions = pd.DataFrame(interactions, columns=['interaction_name', 'formula'])
df_interactions.to_csv('proposed_feature_interactions.csv', index=False)

print(f"[OK] Saved {len(interactions)} proposed interactions to: proposed_feature_interactions.csv")

# Save correlation pairs
df_corr_pairs.to_csv('feature_correlation_pairs.csv', index=False)
print(f"[OK] Saved {len(df_corr_pairs)} correlated pairs to: feature_correlation_pairs.csv")

print("\n" + "="*100)
print("NEXT STEPS:")
print("="*100)
print("1. Implement these interaction terms as new features")
print("2. Test model performance with interaction features added")
print("3. Use feature selection to identify which interactions are most valuable")
print("4. Consider non-linear models (Random Forest, XGBoost) to auto-discover interactions")
print("="*100)

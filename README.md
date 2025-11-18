# PredElute

Predictive models for protein elution behavior in cation exchange chromatography (CEX)

---

## Overview

PredElute provides machine learning models to predict NaCl elution concentrations for proteins under various chromatographic conditions, reducing experimental trial-and-error in resin screening and process development.

This repository contains two complementary predictive models:

1. **Elution with Modifiers** - Predicts how additives (Arginine, Guanidine, etc.) affect protein elution across multiple resin types
2. **pH-Dependent Elution** - Predicts protein elution as a function of pH on SP Sepharose HP

Both models use protein structural descriptors (ProDes, Schrödinger) combined with experimental conditions to predict NaCl elution concentrations.

---

## Model 1: Elution with Modifiers

### Overview
Predicts protein elution behavior under different modifier conditions across 9 experimental setups:
- **Modifiers**: Arginine, Guanidine, Sodium Caprylate, Polyols (Ethylene/Propylene Glycol)
- **Resins**: CM Sepharose FF, SP Sepharose HP, Capto MMC
- **Conditions**: Various modifier concentrations (0M, 0.01M, 0.025M, 0.1M, 20%)

### Model Architecture
- **Algorithm**: Ridge Regression on PCA-transformed features
- **Feature Engineering**: Variance and correlation-based feature selection (thresholds vary per condition)
- **Training**: Leave-One-Out Cross-Validation (LOO CV)
- **Performance**: R² ranges from 0.44 to 0.85 across 27 condition-specific models

### Directory Structure
```
elution_w_modifiers/
├── data/                           # Raw experimental data
├── trained_models_final/           # 27 trained models (.pkl files)
│   └── model_inventory_master.csv  # Model metadata and performance metrics
├── scripts/                        # Training and analysis scripts
├── images/                         # Model performance plots
└── prodes_descriptors_training_only.csv  # ProDes descriptors for training proteins
```

### Key Files
- **Models**: `trained_models_final/*.pkl` - Pickled scikit-learn models with scaler, PCA, and Ridge components
- **Inventory**: `model_inventory_master.csv` - Contains R², MAE, hyperparameters, and protein lists for each model
- **Data**: `multi_qspr (version 1).xlsb.xlsx` - Experimental elution data across all conditions

### Usage Example
```python
import pickle
import pandas as pd
import numpy as np

# Load a model
with open('elution_w_modifiers/trained_models_final/exp1_0M.pkl', 'rb') as f:
    model_data = pickle.load(f)

scaler = model_data['scaler']
pca = model_data['pca']
ridge = model_data['ridge']  # or model_data['model']
feature_names = model_data['feature_names']
protein_names = model_data['protein_names']

# Load your protein descriptors
df_descriptors = pd.read_csv('elution_w_modifiers/prodes_descriptors_training_only.csv', index_col=0)

# Prepare features for a protein
protein = 'lysozyme'
X = df_descriptors.loc[protein][feature_names].values.reshape(1, -1)

# Predict
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)
prediction = ridge.predict(X_pca)[0]

print(f"Predicted elution for {protein}: {prediction:.3f} M NaCl")
```

### Model Performance
| Experiment | Modifier | Resin | R² Range | MAE Range (M) |
|------------|----------|-------|----------|---------------|
| 1 | Arginine | CEX | 0.74-0.77 | 0.03-0.04 |
| 2 | Arginine | Capto MMC | 0.44-0.61 | 0.15-0.22 |
| 3 | Guanidine | CEX | 0.78-0.82 | 0.03 |
| 4 | Guanidine | Capto MMC | 0.45-0.57 | 0.17-0.22 |
| 5-6 | Various | Multi-resin | 0.49-0.79 | 0.03-0.22 |
| 7 | Polyols | Capto MMC | 0.45-0.84 | 0.07-0.21 |
| 8-9 | Sodium Caprylate | CM/Capto | 0.50-0.85 | 0.02-0.20 |

---

## Model 2: pH-Dependent Elution

### Overview
Predicts protein elution concentration as a function of pH (4-8) on SP Sepharose HP resin.

### Model Architecture
- **Algorithm**: Ridge Regression on PCA-transformed engineered features
- **Feature Engineering**:
  - Descriptor × pH interactions
  - pH polynomial terms (pH², pH³)
  - Descriptor × Descriptor interactions
  - Top feature selection based on importance
- **Training**: Leave-One-Out Cross-Validation
- **Performance**: R² = 0.937, MAE = 0.050 M NaCl

### Directory Structure
```
pH_dependent_cex/
├── data/                              # Training data
│   ├── sp_sepharose_hp_descriptors_complete.csv
│   ├── sp_sepharose_hp_nacl_concentrations.csv
│   └── schrodinger_descriptors.csv
├── final_ph_model_engineered/         # Trained model
│   └── ph_model_engineered.pkl
├── scripts/                           # Training and analysis scripts
└── images/                            # Performance plots
```

### Key Features
- **Input**: ProDes + Schrödinger descriptors + pH value
- **Output**: Predicted NaCl elution concentration (M)
- **Training Set**: 14 proteins, 62 protein-pH pairs
- **pH Range**: 4.0 - 8.0

### Usage Example
```python
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load model
with open('pH_dependent_cex/final_ph_model_engineered/ph_model_engineered.pkl', 'rb') as f:
    model_data = pickle.load(f)

scaler = model_data['scaler']
pca = model_data['pca']
ridge = model_data['model']
feature_names = model_data['feature_names']

print(f"Model R²: {model_data['r2_loo']:.3f}")
print(f"Model MAE: {model_data['mae_loo']:.3f} M")

# Note: This model requires engineered features (descriptor×pH interactions, etc.)
# See scripts/engineer_ph_features.py for feature engineering details
```

### Model Performance
- **R² (LOO CV)**: 0.937
- **MAE (LOO CV)**: 0.050 M NaCl
- **RMSE (LOO CV)**: 0.065 M NaCl
- **Improvement**: +3% R² over base model through feature engineering

---

## Visualization

Both model directories contain `images/` folders with:
- **Actual vs Predicted plots** with error bars showing LOO CV performance
- **Feature importance analyses**
- **Applicability domain assessments** (PCA Hotelling's T², Q-residuals)

Example plots show:
- Scatter plots with perfect prediction line (red dashed)
- Error bars representing absolute prediction errors
- R² and MAE metrics for each condition

---

## Requirements

### Python Version
- Python 3.8+

### Core Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Optional (for GUI in legacy script)
```bash
# Linux only
sudo apt-get install python3-tk
```

---

## Data Files

### Elution with Modifiers
- **Descriptors**: ProDes descriptors (105 features) for 24-25 proteins
- **Elution Data**: Experimental NaCl concentrations across 9 experiments × 3 conditions each
- **Format**: CSV files with protein names as index

### pH-Dependent Elution
- **Descriptors**: ProDes (105 features) + Schrödinger (165 features) = 270 total
- **Elution Data**: NaCl concentrations at pH 4, 5, 6, 7, 8 for 14-17 proteins
- **Format**: CSV files with protein names and pH columns

---

## Model Training Details

### Feature Selection
Both models use:
1. **Variance Thresholding**: Remove low-variance features (threshold varies: 0.0-0.05)
2. **Correlation Filtering**: Remove highly correlated features (threshold varies: 0.85-0.99)
3. **PCA Dimensionality Reduction**: Retain 10-20 components

### Hyperparameter Optimization
- **Ridge Alpha**: Optimized per model (typical range: 0.01-10.0)
- **PCA Components**: Optimized per model (typical range: 10-20)
- **Feature Thresholds**: Optimized per condition

### Cross-Validation
- **Method**: Leave-One-Out (LOO) for small datasets
- **Rationale**: Maximizes training data while providing unbiased performance estimates
- **Protein Family Handling**: Some models use deduplicated datasets to avoid family leakage

---

## Citation

If you use PredElute in your research, please cite:

```
[Citation information to be added]
```

---

## License

MIT License - See LICENSE file for details

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description of changes

---

## Contact

For questions, issues, or collaboration inquiries:
- Open an issue on GitHub
- Contact: [Your contact information]

---

## Related Resources

- **ProDes**: Protein descriptor calculation tool
- **Schrödinger Suite**: Molecular modeling software for descriptor generation
- **scikit-learn**: Machine learning library used for model development

---

## Notes

- Models are trained on specific protein sets and may not generalize to highly dissimilar proteins
- Predictions are most reliable within the training data's applicability domain
- Feature engineering significantly improved pH model performance (R² 0.907 → 0.937)
- Each modifier/condition combination has its own optimized model due to different physicochemical interactions

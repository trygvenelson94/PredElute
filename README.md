# Arginine_Guanidine_QSPR_pred.py

Standalone GUI (Tkinter) for predicting elution behavior using embedded linear QSPR models under six arginine/guanidine scenarios. You provide a CSV of raw descriptors (per sequence or molecule), and the app outputs a predictions CSV.

## What this does
- Uses six embedded linear models (RFE-trained; no external files needed):
  - `Gua_NoMod`
  - `Gua_0.025M`
  - `Gua_0.1M`
  - `Arg_NoMod`
  - `Arg_0.025M`
  - `Arg_0.1M`
- For each input row, computes a linear score per scenario:
  - score = intercept + Σ coef(feature) × zscore(feature)
  - zscore(feature) uses per-scenario means and std devs that are embedded in the script.
- Robust column matching: case-insensitive, whitespace/underscore/dash/slash-insensitive, with common aliases supported.
- Outputs a CSV with one column per scenario (plus an optional ID column if you supply one).

## Requirements
- Python 3.8+
- Packages: `pandas` (GUI uses the built-in `tkinter` on Windows/macOS)

Install dependencies:
```bash
pip install pandas
```

On some Linux distros you may need Tkinter separately:
```bash
# Debian/Ubuntu
sudo apt-get install python3-tk
```

## Running the GUI
From the directory containing the script:
```bash
python Arginine_Guanidine_QSPR_pred.py
```

## Using the app
1. __Raw Descriptors CSV__: Click “Browse” and select your input CSV.
2. __Optional ID column__: If your CSV has an identifier column (e.g., `SequenceID`), enter that column name to pass it through to the output.
3. __Run Predictions__: Click the button, then choose where to save the output CSV.

## Input CSV schema
Your CSV must contain these descriptor columns (names are case-insensitive and allow flexible spacing/underscore/dash/slash):
- `AGGRESCAN_a3v_value`
- `All_Greasy_SASA`
- `All_HB_Acceptor_SASA`
- `All_HB_Donor_SASA`
- `All_Zyggregator_profile_smoothed_pos`
- `Avg_Score_Hyd_Patches`
- `Avg_Score_Pos_Patches`
- `Avg_Size_Hyd_Patches`
- `Avg_Size_Neg_Patches`
- `Connectivity`
- `Dipole_Y_direction`
- `Dipole_Z_direction`
- `Disorder_Propensity_DisProt`
- `Exposed_agg_surf_area`
- `Hplc_Hfba_Retention`
- `Hydrophobic_Y_direction`
- `Hydrophobicity_Kyte_Doolittle`
- `Max_Score_Hyd_Patches`
- `Max_Score_Pos_Patches`
- `Max_Size_Pos_Patches`
- `Net_Charge_model_based`
- `Net_Charge_propka_based`
- `Nr_Pos_Patches_gt250`
- `Nr_rotatable_bonds`
- `Total_positive_SASA`

### Column matching and aliases
The app automatically:
- Normalizes names by removing spaces, tabs, underscores, dashes, and slashes, and lowercasing.
- Recognizes common synonyms, for example:
  - `All Greasy SASA` → `All_Greasy_SASA`
  - `Hydrophobicity Kyte Doolittle` → `Hydrophobicity_Kyte_Doolittle`
  - `All HB Donor SASA` → `All_HB_Donor_SASA`

If any required column is missing after matching/aliasing, the app shows a schema error listing the missing fields.

### Data quality
- All descriptor columns must be numeric. The app coerces to numeric and will stop with an error if any row contains non-numeric/missing values.

## Output CSV
The saved predictions file contains:
- Optional ID column (if you provided one that exists in the input CSV)
- Scenario scores:
  - `Gua_NoMod`
  - `Gua_0.025M`
  - `Gua_0.1M`
  - `Arg_NoMod`
  - `Arg_0.025M`
  - `Arg_0.1M`

Scores are unitless linear model outputs based on standardized descriptors. They are not guaranteed to be bounded to [0,1]. Interpretation is relative (higher/lower within a scenario).

## Troubleshooting
- __Schema Error: Missing required columns__
  - Check that your CSV headers match the required names (or common variants) and that there are no typos.
- __Non-numeric or missing values detected__
  - Ensure all descriptor columns are fully numeric. Remove or impute invalid values before loading.
- __Failed to read CSV__
  - Confirm the file is a comma-separated text file with a header row. If exported from Excel, save as CSV (UTF-8).
- __Tkinter is not available__
  - Install `python3-tk` (Linux), or use a standard Python distribution that includes Tkinter (Windows/macOS).

## Programmatic use (optional)
While the GUI is the intended interface, you can also call the predictor from Python:
```python
import pandas as pd
from Arginine_Guanidine_QSPR_pred import _predict_row, REQUIRED

# Build a feature dict mapping every REQUIRED name to a float value
sample = {name: 0.0 for name in REQUIRED}  # fill with your descriptor values
preds = _predict_row(sample)
print(preds)  # dict with keys: Gua_NoMod, Gua_0.025M, Gua_0.1M, Arg_NoMod, Arg_0.025M, Arg_0.1M
```
Note: leading underscores indicate internal helpers; API stability is not guaranteed.

## Notes
- All six models and their per-scenario normalization stats are embedded in the script (`MODELS` and `STATS`).
- No internet connection or external model files are required.

---
If you need a command-line (non-GUI) mode or batch processing enhancements, open an issue or request and we can extend the script accordingly.

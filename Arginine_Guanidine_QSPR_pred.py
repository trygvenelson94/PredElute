"""
elution_gui_embedded_rfe.py
Truly standalone GUI: select only the raw descriptors CSV.
The six RFE models and their per-scenario training means/stds are embedded.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd

# -------------------- Embedded models & stats (auto-generated) --------------------
MODELS = {
  "Gua_NoMod": {
    "intercept": 0.5361111111111106,
    "coef": {
      "AGGRESCAN_a3v_value": -0.1890222303839721,
      "All_Greasy_SASA": -0.0073049534014091,
      "All_HB_Acceptor_SASA": 0.111911050069331,
      "Avg_Score_Hyd_Patches": 0.1859517673825526,
      "Connectivity": -0.1387601441872159,
      "Disorder_Propensity_DisProt": -0.3702044875986551,
      "Exposed_agg_surf_area": 0.1250158968437469,
      "Hplc_Hfba_Retention": 0.1940922266604454,
      "Max_Score_Pos_Patches": 0.1117083202929567,
      "Net_Charge_model_based": -0.1351414481416847
    }
  },
  "Gua_0.025M": {
    "intercept": 0.5155555555555564,
    "coef": {
      "AGGRESCAN_a3v_value": -0.1810957826525484,
      "All_Greasy_SASA": 0.2627287154245484,
      "All_Zyggregator_profile_smoothed_pos": 0.2437519129330159,
      "Avg_Score_Hyd_Patches": 0.1404375746579429,
      "Avg_Score_Pos_Patches": -0.1147878948471982,
      "Dipole_Z_direction": -0.1749399912770238,
      "Exposed_agg_surf_area": -0.1474224572215625,
      "Hydrophobic_Y_direction": -0.1563589523244315,
      "Max_Score_Pos_Patches": 0.2232516851604068,
      "Max_Size_Pos_Patches": -0.0655265439015108
    }
  },
  "Gua_0.1M": {
    "intercept": 0.5383333333333332,
    "coef": {
      "AGGRESCAN_a3v_value": -0.3479662541825427,
      "All_Greasy_SASA": 0.2335088733195452,
      "All_Zyggregator_profile_smoothed_pos": 0.3477389746489749,
      "Avg_Size_Hyd_Patches": -0.2210408816959004,
      "Dipole_Z_direction": -0.1935649753338007,
      "Disorder_Propensity_DisProt": -0.5881529370592662,
      "Hydrophobicity_Kyte_Doolittle": -0.1128480357584805,
      "Max_Score_Hyd_Patches": 0.31453412963218,
      "Nr_rotatable_bonds": 0.3766013509218194,
      "Total_positive_SASA": -0.2738624303479691
    }
  },
  "Arg_NoMod": {
    "intercept": 0.5355555555555547,
    "coef": {
      "AGGRESCAN_a3v_value": 0.1063975529844521,
      "All_Greasy_SASA": -0.003635398491126,
      "All_HB_Acceptor_SASA": 0.1991882266154975,
      "All_Zyggregator_profile_smoothed_pos": -0.2148529683360459,
      "Avg_Score_Hyd_Patches": 0.3530102316553121,
      "Connectivity": -0.1856895113258684,
      "Disorder_Propensity_DisProt": 0.0396540564438479,
      "Max_Size_Pos_Patches": 0.2375152209404209,
      "Net_Charge_propka_based": -0.1525777807006661,
      "Nr_Pos_Patches_gt250": 0.029412954525408
    }
  },
  "Arg_0.025M": {
    "intercept": 0.4749999999999998,
    "coef": {
      "AGGRESCAN_a3v_value": -0.3180127908615859,
      "All_Greasy_SASA": 0.1758484878008783,
      "All_Zyggregator_profile_smoothed_pos": 0.242827978963259,
      "Avg_Score_Hyd_Patches": 0.0755163433271576,
      "Avg_Size_Hyd_Patches": -0.0568484895426069,
      "Dipole_Z_direction": -0.0977076392433154,
      "Disorder_Propensity_DisProt": -0.3835111390867722,
      "Hplc_Hfba_Retention": 0.22354583358999,
      "Max_Size_Pos_Patches": 0.0811422298881129,
      "Net_Charge_propka_based": -0.0706523410429384
    }
  },
  "Arg_0.1M": {
    "intercept": 0.2616666666666663,
    "coef": {
      "All_HB_Acceptor_SASA": 0.0619927775623249,
      "All_HB_Donor_SASA": 0.0507381264741229,
      "Avg_Score_Hyd_Patches": 0.1382073179064222,
      "Avg_Size_Neg_Patches": -0.0829927958499676,
      "Connectivity": -0.0612927419181858,
      "Dipole_Y_direction": 0.0582737461187261,
      "Hydrophobic_Y_direction": -0.0687922669161416,
      "Max_Size_Pos_Patches": 0.0934551497630222,
      "Net_Charge_propka_based": -0.0598997217530402,
      "Nr_Pos_Patches_gt250": 0.0602036013420094
    }
  }
}
STATS = {
  "Gua_NoMod": {
    "AGGRESCAN_a3v_value": {
      "mean": -0.043747909168642,
      "std": 0.0800006639289493
    },
    "All_Greasy_SASA": {
      "mean": 0.0044167177951927,
      "std": 0.0036794383665248
    },
    "All_HB_Acceptor_SASA": {
      "mean": 0.0209142212991308,
      "std": 0.0061482598459173
    },
    "Avg_Score_Hyd_Patches": {
      "mean": 0.5602690997995726,
      "std": 0.0405694179223382
    },
    "Connectivity": {
      "mean": 0.9470262823409354,
      "std": 0.2184162332899703
    },
    "Disorder_Propensity_DisProt": {
      "mean": 3.605277777777786,
      "std": 10.514647734393924
    },
    "Exposed_agg_surf_area": {
      "mean": 57.15766666666666,
      "std": 66.4170828000347
    },
    "Hplc_Hfba_Retention": {
      "mean": 606.1611111111116,
      "std": 566.9635654949665
    },
    "Max_Score_Pos_Patches": {
      "mean": 1194.9021384838848,
      "std": 634.932448687507
    },
    "Net_Charge_model_based": {
      "mean": 6.425671249622781,
      "std": 3.9746100675769287
    }
  },
  "Gua_0.025M": {
    "AGGRESCAN_a3v_value": {
      "mean": -0.043747909168642,
      "std": 0.0800006639289493
    },
    "All_Greasy_SASA": {
      "mean": 0.0044167177951927,
      "std": 0.0036794383665248
    },
    "All_Zyggregator_profile_smoothed_pos": {
      "mean": 22.12118731881509,
      "std": 9.212985678833078
    },
    "Avg_Score_Hyd_Patches": {
      "mean": 0.5602690997995726,
      "std": 0.0405694179223382
    },
    "Avg_Score_Pos_Patches": {
      "mean": 1.1016288489392023,
      "std": 0.047526303449007
    },
    "Dipole_Z_direction": {
      "mean": 0.532596058918805,
      "std": 0.3914194841277589
    },
    "Exposed_agg_surf_area": {
      "mean": 57.15766666666666,
      "std": 66.4170828000347
    },
    "Hydrophobic_Y_direction": {
      "mean": 0.5499357774765774,
      "std": 0.4033715772850324
    },
    "Max_Score_Pos_Patches": {
      "mean": 1194.9021384838848,
      "std": 634.932448687507
    },
    "Max_Size_Pos_Patches": {
      "mean": 1045.9123361139905,
      "std": 573.4743776388865
    }
  },
  "Gua_0.1M": {
    "AGGRESCAN_a3v_value": {
      "mean": -0.043747909168642,
      "std": 0.0800006639289493
    },
    "All_Greasy_SASA": {
      "mean": 0.0044167177951927,
      "std": 0.0036794383665248
    },
    "All_Zyggregator_profile_smoothed_pos": {
      "mean": 22.12118731881509,
      "std": 9.212985678833078
    },
    "Avg_Size_Hyd_Patches": {
      "mean": 158.93713768126295,
      "std": 34.248983924964314
    },
    "Dipole_Z_direction": {
      "mean": 0.532596058918805,
      "std": 0.3914194841277589
    },
    "Disorder_Propensity_DisProt": {
      "mean": 3.605277777777786,
      "std": 10.514647734393924
    },
    "Hydrophobicity_Kyte_Doolittle": {
      "mean": -102.73333333333332,
      "std": 104.8296183761483
    },
    "Max_Score_Hyd_Patches": {
      "mean": 292.2388314885971,
      "std": 173.020639286335
    },
    "Nr_rotatable_bonds": {
      "mean": 1146.0,
      "std": 858.7617054029986
    },
    "Total_positive_SASA": {
      "mean": 1093.4794444444442,
      "std": 712.7796093913854
    }
  },
  "Arg_NoMod": {
    "AGGRESCAN_a3v_value": {
      "mean": -0.043747909168642,
      "std": 0.0800006639289493
    },
    "All_Greasy_SASA": {
      "mean": 0.0044167177951927,
      "std": 0.0036794383665248
    },
    "All_HB_Acceptor_SASA": {
      "mean": 0.0209142212991308,
      "std": 0.0061482598459173
    },
    "All_Zyggregator_profile_smoothed_pos": {
      "mean": 22.12118731881509,
      "std": 9.212985678833078
    },
    "Avg_Score_Hyd_Patches": {
      "mean": 0.5602690997995726,
      "std": 0.0405694179223382
    },
    "Connectivity": {
      "mean": 0.9470262823409354,
      "std": 0.2184162332899703
    },
    "Disorder_Propensity_DisProt": {
      "mean": 3.605277777777786,
      "std": 10.514647734393924
    },
    "Max_Size_Pos_Patches": {
      "mean": 1045.9123361139905,
      "std": 573.4743776388865
    },
    "Net_Charge_propka_based": {
      "mean": 6.007482786924675,
      "std": 4.469049378075739
    },
    "Nr_Pos_Patches_gt250": {
      "mean": 4.111111111111111,
      "std": 2.6434171674156266
    }
  },
  "Arg_0.025M": {
    "AGGRESCAN_a3v_value": {
      "mean": -0.043747909168642,
      "std": 0.0800006639289493
    },
    "All_Greasy_SASA": {
      "mean": 0.0044167177951927,
      "std": 0.0036794383665248
    },
    "All_Zyggregator_profile_smoothed_pos": {
      "mean": 22.12118731881509,
      "std": 9.212985678833078
    },
    "Avg_Score_Hyd_Patches": {
      "mean": 0.5602690997995726,
      "std": 0.0405694179223382
    },
    "Avg_Size_Hyd_Patches": {
      "mean": 158.93713768126295,
      "std": 34.248983924964314
    },
    "Dipole_Z_direction": {
      "mean": 0.532596058918805,
      "std": 0.3914194841277589
    },
    "Disorder_Propensity_DisProt": {
      "mean": 3.605277777777786,
      "std": 10.514647734393924
    },
    "Hplc_Hfba_Retention": {
      "mean": 606.1611111111116,
      "std": 566.9635654949665
    },
    "Max_Size_Pos_Patches": {
      "mean": 1045.9123361139905,
      "std": 573.4743776388865
    },
    "Net_Charge_propka_based": {
      "mean": 6.007482786924675,
      "std": 4.469049378075739
    }
  },
  "Arg_0.1M": {
    "All_HB_Acceptor_SASA": {
      "mean": 0.0209142212991308,
      "std": 0.0061482598459173
    },
    "All_HB_Donor_SASA": {
      "mean": 0.034063912110038,
      "std": 0.0095988526345553
    },
    "Avg_Score_Hyd_Patches": {
      "mean": 0.5602690997995726,
      "std": 0.0405694179223382
    },
    "Avg_Size_Neg_Patches": {
      "mean": 87.05099449301194,
      "std": 30.002140318866036
    },
    "Connectivity": {
      "mean": 0.9470262823409354,
      "std": 0.2184162332899703
    },
    "Dipole_Y_direction": {
      "mean": 0.4268066369141027,
      "std": 0.3543294738990772
    },
    "Hydrophobic_Y_direction": {
      "mean": 0.5499357774765774,
      "std": 0.4033715772850324
    },
    "Max_Size_Pos_Patches": {
      "mean": 1045.9123361139905,
      "std": 573.4743776388865
    },
    "Net_Charge_propka_based": {
      "mean": 6.007482786924675,
      "std": 4.469049378075739
    },
    "Nr_Pos_Patches_gt250": {
      "mean": 4.111111111111111,
      "std": 2.6434171674156266
    }
  }
}
REQUIRED = [
  "AGGRESCAN_a3v_value",
  "All_Greasy_SASA",
  "All_HB_Acceptor_SASA",
  "All_HB_Donor_SASA",
  "All_Zyggregator_profile_smoothed_pos",
  "Avg_Score_Hyd_Patches",
  "Avg_Score_Pos_Patches",
  "Avg_Size_Hyd_Patches",
  "Avg_Size_Neg_Patches",
  "Connectivity",
  "Dipole_Y_direction",
  "Dipole_Z_direction",
  "Disorder_Propensity_DisProt",
  "Exposed_agg_surf_area",
  "Hplc_Hfba_Retention",
  "Hydrophobic_Y_direction",
  "Hydrophobicity_Kyte_Doolittle",
  "Max_Score_Hyd_Patches",
  "Max_Score_Pos_Patches",
  "Max_Size_Pos_Patches",
  "Net_Charge_model_based",
  "Net_Charge_propka_based",
  "Nr_Pos_Patches_gt250",
  "Nr_rotatable_bonds",
  "Total_positive_SASA"
]
_ALIAS = {
  "aggrescana3vvalue": "AGGRESCAN_a3v_value",
  "allgreasysasa": "All_Greasy_SASA",
  "allgreasysurfacearea": "All_Greasy_SASA",
  "allhbacceptorsasa": "All_HB_Acceptor_SASA",
  "allhbacceptorsurfacearea": "All_HB_Acceptor_SASA",
  "allhbdonorsasa": "All_HB_Donor_SASA",
  "allhbdonorsurfacearea": "All_HB_Donor_SASA",
  "allzyggregatorprofilesmoothedpos": "All_Zyggregator_profile_smoothed_pos",
  "avgscorehydpatches": "Avg_Score_Hyd_Patches",
  "avgscoreshydpatches": "Avg_Score_Hyd_Patches",
  "avgscorepospatches": "Avg_Score_Pos_Patches",
  "avgscorespospatches": "Avg_Score_Pos_Patches",
  "avgsizehydpatches": "Avg_Size_Hyd_Patches",
  "avgsizeshydpatches": "Avg_Size_Hyd_Patches",
  "avgsizenegpatches": "Avg_Size_Neg_Patches",
  "avgsizesnegpatches": "Avg_Size_Neg_Patches",
  "connectivity": "Connectivity",
  "dipoleydirection": "Dipole_Y_direction",
  "dipolezdirection": "Dipole_Z_direction",
  "disorderpropensitydisprot": "Disorder_Propensity_DisProt",
  "exposedaggsurfarea": "Exposed_agg_surf_area",
  "hplchfbaretention": "Hplc_Hfba_Retention",
  "hydrophobicydirection": "Hydrophobic_Y_direction",
  "hydrophobicitykytedoolittle": "Hydrophobicity_Kyte_Doolittle",
  "maxscorehydpatches": "Max_Score_Hyd_Patches",
  "maxscoreshydpatches": "Max_Score_Hyd_Patches",
  "maxscorepospatches": "Max_Score_Pos_Patches",
  "maxscorespospatches": "Max_Score_Pos_Patches",
  "maxsizepospatches": "Max_Size_Pos_Patches",
  "maxsizespospatches": "Max_Size_Pos_Patches",
  "netchargemodelbased": "Net_Charge_model_based",
  "netchargepropkabased": "Net_Charge_propka_based",
  "nrpospatchesgt250": "Nr_Pos_Patches_gt250",
  "nrrotatablebonds": "Nr_rotatable_bonds",
  "totalpositivesasa": "Total_positive_SASA",
  "totalpositivesurfacearea": "Total_positive_SASA"
}

def _norm_key(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("\t", "").replace("_", "").replace("-", "").replace("/", "")

def _build_column_map(df_cols):
    # map canonical required feature -> actual column name in the CSV
    mapping = {req: None for req in REQUIRED}
    colset = set(df_cols)
    # 1) exact match
    for req in REQUIRED:
        if req in colset:
            mapping[req] = req
    if all(v is not None for v in mapping.values()):
        return mapping
    # 2) normalized match
    normalized = {_norm_key(c): c for c in df_cols}
    for req in REQUIRED:
        if mapping[req] is not None: continue
        key = _norm_key(req)
        if key in normalized:
            mapping[req] = normalized[key]
    if all(v is not None for v in mapping.values()):
        return mapping
    # 3) alias match
    for req in REQUIRED:
        if mapping[req] is not None: continue
        for nkey, orig in normalized.items():
            if nkey in _ALIAS and _ALIAS[nkey] == req:
                mapping[req] = orig
                break
    missing = [r for r,c in mapping.items() if c is None]
    if missing:
        raise KeyError("Missing required columns in input CSV: " + ", ".join(missing))
    return mapping

def _z(x, mu, sd):
    if pd.isna(sd) or sd == 0:
        return 0.0
    return (x - mu) / sd

def _predict_row(sample_dict):
    out = {}
    for scen, spec in MODELS.items():
        total = spec["intercept"]
        for feat, coef in spec["coef"].items():
            mu = STATS[scen][feat]["mean"]
            sd = STATS[scen][feat]["std"]
            total += coef * _z(float(sample_dict[feat]), mu, sd)
        out[scen] = total
    return out

# -------------------- GUI --------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Resin Elution Predictor (Embedded RFE Models)")
        self.input_csv = tk.StringVar()
        self.id_col = tk.StringVar(value="")  # optional

        tk.Label(root, text="Raw Descriptors CSV:").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        tk.Entry(root, textvariable=self.input_csv, width=60).grid(row=0, column=1, padx=6, pady=6)
        tk.Button(root, text="Browse", command=self._browse_csv).grid(row=0, column=2, padx=6)

        tk.Label(root, text="Optional ID column:").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        tk.Entry(root, textvariable=self.id_col, width=28).grid(row=1, column=1, sticky="w", padx=6, pady=6)

        tk.Button(root, text="Run Predictions", command=self._run).grid(row=2, column=1, pady=12)

        self.status = tk.StringVar(value="Ready.")
        tk.Label(root, textvariable=self.status, fg="blue").grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=6)

    def _browse_csv(self):
        path = filedialog.askopenfilename(title="Select raw descriptors CSV", filetypes=[("CSV files","*.csv")])
        if path:
            self.input_csv.set(path)

    def _run(self):
        path = self.input_csv.get().strip()
        idc = (self.id_col.get().strip() or None)
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid CSV file.")
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
            return

        try:
            col_map = _build_column_map(df.columns.tolist())
        except Exception as e:
            messagebox.showerror("Schema Error", str(e))
            return

        keep = [col_map[r] for r in REQUIRED]
        if idc and idc in df.columns:
            keep = [idc] + keep
        data = df[keep].copy()

        # numeric coercion
        for c in keep:
            if idc and c == idc: 
                continue
            data[c] = pd.to_numeric(data[c], errors="coerce")
        data_cols = [c for c in keep if not (idc and c == idc)]
        if data[data_cols].isna().any().any():
            bad = data.index[data[data_cols].isna().any(axis=1)].tolist()
            messagebox.showerror("Error", f"Non-numeric or missing values detected. Example rows: {bad[:10]}")
            return

        out_path = filedialog.asksaveasfilename(title="Save predictions CSV as...", defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not out_path:
            return

        rows = []
        for _, r in data.iterrows():
            sample = {req: float(r[col_map[req]]) for req in REQUIRED}
            preds = _predict_row(sample)
            out_row = {
                "Gua_NoMod": preds["Gua_NoMod"],
                "Gua_0.025M": preds["Gua_0.025M"],
                "Gua_0.1M": preds["Gua_0.1M"],
                "Arg_NoMod": preds["Arg_NoMod"],
                "Arg_0.025M": preds["Arg_0.025M"],
                "Arg_0.1M": preds["Arg_0.1M"],
            }
            if idc and idc in data.columns:
                out_row[idc] = r[idc]
            rows.append(out_row)

        cols = ([idc] if (idc and idc in data.columns) else []) + ["Gua_NoMod","Gua_0.025M","Gua_0.1M","Arg_NoMod","Arg_0.025M","Arg_0.1M"]
        pd.DataFrame(rows)[cols].to_csv(out_path, index=False)
        self.status.set(f"Success! Saved predictions to: {out_path}")
        messagebox.showinfo("Done", f"Predictions saved:\n{out_path}")

def main():
    root = tk.Tk()
    App(root)
    root.resizable(False, False)
    root.mainloop()

if __name__ == "__main__":
    main()

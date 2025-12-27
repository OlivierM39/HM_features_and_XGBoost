# Hydrometeorological features and XGBoost modelling

This repository provides two complementary workflows used for landslide hydro‑meteorological analyses:

1. **Hydrometeorological feature engineering** (rainfall / effective rainfall / groundwater level):  
   rolling sums and maxima, saturation indices, anomaly indices, and event‑based strength indices.

2. **XGBoost modelling workflow**:  
   merge a target time series with pre‑computed features, split the record into **N equal temporal blocks**
   (leave‑one‑block‑out testing), train an `XGBRegressor`, compute metrics (**R², RMSE, MAE**), and export
   feature importance + SHAP diagnostics.

> **Open‑access note:** this repo is meant for **open code**. Raw `.xlsx` data are usually not tracked and should be placed under `data/`
> (see `.gitignore`). If you want to share data, publish a small anonymised sample and document it in this README.

---

## Repository structure

```
.
├─ notebooks/
│  ├─ Hydrometeorological_features_Calculations.ipynb
│  └─ XGBoost_model.ipynb
├─ src/
│  ├─ hm_features/        # feature-engineering functions (R/ER + GWL/WL)
│  └─ xgb_pipeline/       # XGBoost workflow functions (merge, split, train, plots, save)
├─ data/                  # put raw inputs here (ignored by git by default)
├─ outputs/               # results (Excel/PNG/models; usually ignored by git)
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## Quickstart

### 1) Install dependencies

```bash
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Place your input files

Put your site files under `data/<SITE>/...` using the recommended layout:

```
data/
└─ <SITE>/
   ├─ 0_Input_dataset/
   │  ├─ Hydro_Meteo_traitées.xlsx
   │  └─ vel_traité_all_synthese.xlsx
   └─ 1_Features_ER_R_GWL/
      └─ Features_HM.xlsx
```

### 3) Run notebooks

- `notebooks/Hydrometeorological_features_Calculations.ipynb`
- `notebooks/XGBoost_model.ipynb`

---

## Input data requirements (important)

### A) Hydro‑meteorological input (HM)

Used by the **feature engineering** notebook/pipeline.

**Minimal requirement**
- A column named `Date`
- At least rainfall and/or effective rainfall and/or groundwater level columns

The code reads an Excel file and can optionally keep selected columns via `input_usecols`.
It then standardises column names (renaming) to match the expected conventions below.

#### A.1 Viella / Villerville / generic single‑GWL site

The pipeline expects (after renaming):
- `Date` (datetime)
- `R`  : rainfall (typically mm/day)
- `ER` : effective rainfall (typically mm/day)
- `GWL`: groundwater level (typically m)

Example (your raw file can have different names):
```python
input_usecols = ["Rain", "EffRain", "PZ3"]
# the code renames these into ["R", "ER", "GWL"] in that order
```

#### A.2 Séchilienne (two water‑level columns)

The pipeline expects (after renaming):
- `Date` (datetime)
- `R`
- `ER`
- `WLI` : first water level series (site-specific meaning)
- `WLM` : second water level series (site-specific meaning)

Example:
```python
input_usecols = ["R", "ER", "WLI_raw", "WLM_raw"]
# the code renames these into ["R", "ER", "WLI", "WLM"] in that order
```

#### HM file format notes

- Excel format is expected (`.xlsx`)
- `Date` is parsed with `pd.to_datetime(..., errors="coerce")`
- Rows are sorted by `Date`
- Recommended time step: **daily** (rolling windows are expressed as number of rows/days)
- **Column order matters**: `input_usecols` is renamed in the same order as provided (via `zip(...)`)

---

### B) Target / movement time series (prisme / velocity)

Used by the **XGBoost** workflow.

**Minimal requirement**
- A column named `Date`
- One target column (your displacement/velocity) referenced by `col_prisme`

Example target column naming convention used in this workflow:
- `BAV-01_vel_processed`
- `E-A13_vel_processed_decomposed`

The code automatically builds a short target name:
```python
prisme_name = col_prisme.split("_")[0]
# e.g. "BAV-01_vel_processed" -> "BAV-01"
```

**Important**
- The merge is performed on `Date` (**inner join**)
- The merged dataset is then `dropna()` (rows must have non‑NaN in target + all selected features)

---

### C) Feature table input for XGBoost

The XGBoost workflow expects a **feature table** with:
- `Date`
- engineered feature columns (rolling sums/maxima, saturation, anomalies, event strength, GWL lags, etc.)

Optionally, it can also include external raw series (e.g. `R`, `ER`, `GWL` / `WLI` / `WLM`) if you want them in `base_df`.

Typical filename produced by the feature engineering pipeline:
- `Features_HM.xlsx` (saved under a site folder)

---

## Expected columns summary

### Feature engineering (HM → features)

At minimum:
- `Date`
- plus one or more of:
  - `R` and/or `ER` (for rainfall features)
  - `GWL` (for groundwater features) **or** `WLI`, `WLM` (Séchilienne)

### XGBoost modelling (target + features)

At minimum:
- Target table: `Date` + one target column `col_prisme`
- Feature table: `Date` + engineered feature columns

---

## Outputs

### Feature engineering outputs

Saved by default under:

- `outputs/<SITE>/1_Features_ER_R_GWL/`
  - `Features_HM.xlsx` (merged features)
  - `Features_METEO.xlsx` (R/ER-based features)
  - `Features_HYDRO.xlsx` (GWL/WL-based features)

### XGBoost outputs

Saved by default under:

- `outputs/<SITE>/2_XGBoost_results/<TARGET>/Bloc_<k>/`
  - time series results: `Results_time_series.xlsx`
  - hyperparameters: `Results_hyperparameters.xlsx`
  - metrics: `Results_metrics.xlsx`
  - feature importance: `Results_features_importance.xlsx`, `Results_top_features_importance.xlsx`
  - figures: `.png`
  - trained model: `XGB_trained_model.joblib`

---

## Minimal example usage (optional)

### Build features (Viella/Villerville-like site)

```python
from hm_features.features import features_pipeline_totale

SITE = "Villerville"
input_path = f"data/{SITE}/0_Input_dataset"
input_file = "Hydro_Meteo_traitées.xlsx"

input_usecols = ["R", "ER", "PZ3"]      # <- your raw column names (order matters)
externe_col_list = ["R", "ER", "GWL"]   # <- columns to keep as externals in outputs
output_folder_path = f"outputs/{SITE}"

features_hm_df, R_ER_features_df, GWL_features_df = features_pipeline_totale(
    input_path, input_file, input_usecols, externe_col_list, output_folder_path
)
```

### Run XGBoost (block cross-validation)

Open `notebooks/XGBoost_model.ipynb` and edit the “Application” cell, or call your pipeline function(s)
directly if you expose them from `src/xgb_pipeline/`.

---

## Data policy

This repository is open access for **code and documentation**.  
The input datasets used in the associated study are **not included** in this repository.

To run the workflows, place your own input files under `data/` (see the recommended layout above).  
If you plan to share data with this code, please provide:
- units (e.g. mm/day, m),
- time step (daily recommended),
- date range,
- preprocessing steps (gap filling, resampling, filtering, etc.).
---

## License

This project is distributed under the **MIT License** (see `LICENSE`).
---

## Citation

If you use this code, please cite **(1) this repository** and, when relevant, **(2) the related manuscript**.

### (1) Software / repository
Béjean-Maillard, O. (2025). *Hydrometeorological features and XGBoost modelling* (v0.1.0). GitHub repository.  

### (2) Related manuscript (in preparation)
Béjean-Maillard, O., Bertrand, C., Malet, J.-P., Dubois, L., Batailles, C., Lespine, L., & Ducasse, J. (in preparation).  
*A Generic and Explainable AI-based Workflow to Simulate Landslide Dynamics Using Hydro-Meteorological Observations*.

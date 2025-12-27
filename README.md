# Hydrometeorological features and XGBoost modelling

This repository contains two complementary workflows used in landslide hydro-meteorological analyses:

1. **Hydrometeorological feature engineering** (rainfall / effective rainfall / groundwater level):
   rolling sums and maxima, saturation indices, anomaly indices, and event-based strength indices.

2. **XGBoost modelling workflow**:
   merge a target time series with pre-computed features, split the record into **N equal temporal blocks**
   (leave-one-block-out testing), train an `XGBRegressor`, compute metrics (**R², RMSE, MAE**), and export
   feature importance + SHAP diagnostics.

## Repository structure

- `src/hm_features/` : feature-engineering functions
- `src/xgb_pipeline/` : XGBoost workflow functions
- `notebooks/` : runnable notebooks
- `data/` : **not versioned** (place your Excel input data here)
- `outputs/` : generated outputs (results/figures)

Expected layout for each site:

- `data/<SITE>/0_Input_dataset/` : movement/target time series (Excel)
- `data/<SITE>/1_Features_ER_R_GWL/` : pre-computed features (Excel)
- `outputs/<SITE>/` : written automatically

## Quick start

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the notebooks:

- `notebooks/Hydrometeorological_features_Calculations.ipynb`
- `notebooks/XGBoost_model.ipynb`

## Notes on data

The repository is intended for **open access code**. Raw datasets (e.g., `.xlsx`) are typically not tracked
and should be placed under `data/` (see `.gitignore`). If you want to provide an example dataset, use a small
anonymised sample and document it in the README.

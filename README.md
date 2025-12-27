# Hydrometeorological feature calculations

Utilities to compute hydro-meteorological feature sets from daily time series:
- **Rainfall (R)** and **effective rainfall (ER)**: rolling sums/maxima, saturation indices, anomaly indices, and event-strength indices.
- **Groundwater / water level (GWL / WL)**: lagged series, differences, rolling means, rolling extrema, and deviation from the global mean.

The code is extracted from a Jupyter notebook and packaged under `src/hm_features/` for easier reuse and versioning.

## Expected input format

An Excel file with at least a **Date** column (daily timestamps) and one or more of the following columns:
- `R`: rainfall
- `ER`: effective rainfall
- `GWL` (or `PZ*` columns that will be renamed to `GWL` via `input_usecols`)
- `WLI`, `WLM` (Séchilienne)

## Install

```bash
pip install -r requirements.txt
# optional: install as a package (editable)
# pip install -e .
```

## Quick start (Python)

```python
from hm_features import features_pipeline_totale

features_hm_df, r_er_df, gwl_df = features_pipeline_totale(
    input_path="data/Villerville/0_Input_dataset",
    input_file="Hydro_Meteo_traitées.xlsx",
    input_usecols=["R", "ER", "PZ3"],      # will be renamed to R/ER/GWL
    externe_col_list=["R", "ER", "GWL"],
    output_folder_path="outputs/Villerville",
)
```

## Notebook

See [`notebooks/Hydrometeorological_features_Calculations.ipynb`](notebooks/Hydrometeorological_features_Calculations.ipynb) for an end-to-end example.

## Folder conventions

By default the notebook expects:
- `data/<SITE>/0_Input_dataset/<your_excel>.xlsx`
- outputs written to `outputs/<SITE>/1_Features_ER_R_GWL/`

If you prefer a different structure, just change the paths in the example cells.

## License

MIT — see `LICENSE`.

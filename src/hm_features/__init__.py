"""Hydrometeorological feature engineering utilities.

This package provides utilities to compute rolling, saturation, anomaly and event-based
features from rainfall / effective rainfall (R/ER) and groundwater-level (GWL/WL) time series.
"""

from .features import (
    read_excel_file,
    data_importation,
    data_importation_Sech,
    rainfall_and_effective_rainfall_features_calcul,
    gwl_features_calcul,
    compute_features,
    compute_features_Sech,
    save_features_df,
    features_pipeline_totale,
    features_pipeline_totale_Sech,
)

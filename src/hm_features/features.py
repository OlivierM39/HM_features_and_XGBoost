from __future__ import annotations

import os
from functools import reduce
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd


PathLike = Union[str, os.PathLike, Path]


# =============================================================================
# I/O utilities
# =============================================================================
def read_excel_file(input_path: PathLike, input_file: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read an Excel file and return a clean, date-sorted DataFrame.

    Parameters
    ----------
    input_path : str | PathLike
        Folder containing the Excel file.
    input_file : str
        Excel filename.
    usecols : list[str] | None
        Optional list of column names to keep (in addition to 'Date').
        If provided, only existing columns are kept.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing at least a 'Date' column (if present in the source file).
    """
    df = pd.read_excel(os.path.join(os.fspath(input_path), input_file))

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    if usecols:
        keep = ["Date"] + [c for c in usecols if c in df.columns]
        df = df[keep]

    return df


def data_importation(input_path: PathLike, input_file: str, input_usecols: Sequence[str]) -> pd.DataFrame:
    """
    Import hydro-meteorological time series (Viella / Villerville version).

    Reads the Excel file and renames the provided columns to standard names:
    ['R', 'ER', 'GWL'] (truncated to the number of columns provided).

    Returns
    -------
    pd.DataFrame
        Output DataFrame with columns: Date + standardised external series.
    """
    hm_df = read_excel_file(input_path, input_file, list(input_usecols))
    rename_map = {old: new for old, new in zip(input_usecols, ["R", "ER", "GWL"][: len(input_usecols)])}
    return hm_df.rename(columns=rename_map)


def data_importation_Sech(input_path: PathLike, input_file: str, input_usecols: Sequence[str]) -> pd.DataFrame:
    """
    Import hydro-meteorological time series (Séchilienne version).

    Reads the Excel file and renames the provided columns to standard names:
    ['R', 'ER', 'WLI', 'WLM'] (truncated to the number of columns provided).

    Returns
    -------
    pd.DataFrame
        Output DataFrame with columns: Date + standardised external series.
    """
    hm_df = read_excel_file(input_path, input_file, list(input_usecols))
    rename_map = {old: new for old, new in zip(input_usecols, ["R", "ER", "WLI", "WLM"][: len(input_usecols)])}
    return hm_df.rename(columns=rename_map)


# =============================================================================
# R / ER feature blocks
# =============================================================================
def _rolling_features(df: pd.DataFrame, cols: Sequence[str], T_periods: Sequence[int]) -> Dict[str, pd.Series]:
    """Rolling sum and rolling maximum for multiple window sizes."""
    out: Dict[str, pd.Series] = {}
    for col in cols:
        s = df[col]
        for T in T_periods:
            out[f"{col}_{T}"] = s.rolling(window=T).sum()
            out[f"{col}_max{T}"] = s.rolling(window=T).max()
    return out


def _saturation_indices(
    df: pd.DataFrame,
    cols: Sequence[str],
    short_terms: Sequence[int] = (3, 5, 10, 20, 30, 40, 60),
    long_terms: Sequence[int] = (30, 60, 90),
    rain_threshold: float = 3.0,
    eps: float = 1e-6,
) -> Dict[str, pd.Series]:
    """Local saturation indices based on a ratio of recent vs antecedent filtered rainfall."""
    out: Dict[str, pd.Series] = {}
    for col in cols:
        filtered = df[col].where(df[col] >= rain_threshold, 0.0)

        for sT in short_terms:
            short_sum = filtered.rolling(window=sT, min_periods=1).sum()
            for lT in long_terms:
                long_sum = filtered.shift(sT).rolling(window=lT, min_periods=1).sum()
                ratio = short_sum / long_sum.mask(long_sum.abs() < eps, 1.0)
                out[f"{col}_sat{sT}/{lT}"] = ratio
    return out


# Backward-compatibility alias (keeps your original function name)
_staturation_indices = _saturation_indices


def _anomaly_indices(df: pd.DataFrame, cols: Sequence[str], T_periods: Sequence[int]) -> Dict[str, pd.Series]:
    """Anomaly indices: rolling mean / climatological mean computed on rainy days."""
    out: Dict[str, pd.Series] = {}
    for col in cols:
        clim_mean = df.loc[df[col] > 0, col].mean()
        for T in T_periods:
            out[f"{col}_anomaly{T}"] = df[col].rolling(window=T).mean() / clim_mean
    return out


def _compute_event_ids(series: pd.Series, dry_tolerance: int = 2) -> np.ndarray:
    """
    Event IDs with a tolerance of 'dry_tolerance' internal dry days.

    Returns
    -------
    np.ndarray
        Array of event IDs (0 = no event).
    """
    ids = np.zeros(len(series), dtype=int)
    ev_id = 0
    dry = 0

    for i, v in enumerate(series.values):
        if v > 0:
            if ev_id == 0:
                ev_id = (ids.max() + 1) if ids.max() > 0 else 1
            dry = 0
            ids[i] = ev_id
        else:
            if ev_id == 0:
                ids[i] = 0
            else:
                dry += 1
                if dry <= dry_tolerance:
                    ids[i] = ev_id
                else:
                    ev_id = 0
                    dry = 0
                    ids[i] = 0
    return ids


def _event_strength_for_col(df: pd.DataFrame, col: str, dry_tolerance: int = 2) -> pd.Series:
    """
    Event-strength index for one column:
    (event cumulative sum) / (mean cumulative sum of events with the same duration).
    """
    s = df[col]
    ev_ids = _compute_event_ids(s, dry_tolerance=dry_tolerance)

    ev_df = pd.DataFrame({"id": ev_ids, "val": s.values})
    ev_df = ev_df[ev_df["id"] != 0]
    if ev_df.empty:
        return pd.Series(np.zeros(len(df)), index=df.index)

    grp = ev_df.groupby("id")["val"]
    cumuls = grp.sum()
    durations = ev_df.groupby("id").size()

    tmp = pd.DataFrame({"Duration": durations, "Cumul": cumuls})
    mean_cumul_by_duration = tmp.groupby("Duration")["Cumul"].mean().to_dict()

    # Map back to the original index
    ev_map_cumul = pd.Series(0.0, index=np.unique(ev_ids[ev_ids != 0]))
    ev_map_dur = pd.Series(0, index=np.unique(ev_ids[ev_ids != 0]))
    ev_map_cumul.loc[cumuls.index] = cumuls.values
    ev_map_dur.loc[durations.index] = durations.values

    idx_vals: List[float] = []
    for eid in ev_ids:
        if eid == 0:
            idx_vals.append(np.nan)
            continue

        dur = int(ev_map_dur.loc[eid])
        cumul = float(ev_map_cumul.loc[eid])
        denom = float(mean_cumul_by_duration.get(dur, np.nan))

        if (not np.isfinite(denom)) or denom == 0.0:
            idx_vals.append(np.nan)
        else:
            idx_vals.append(cumul / denom)

    return pd.Series(idx_vals, index=df.index)


def _event_strength_pipeline(df: pd.DataFrame, cols: Sequence[str], dry_tolerance: int = 2) -> Dict[str, pd.Series]:
    """Compute event-strength indices for a list of columns."""
    out: Dict[str, pd.Series] = {}
    for col in cols:
        out[f"{col}_event_strength"] = _event_strength_for_col(df, col, dry_tolerance=dry_tolerance)
    return out


def rainfall_and_effective_rainfall_features_calcul(
    df: pd.DataFrame,
    meteo_col_list: Sequence[str],
    T_periods: Sequence[int] = (1, 2, 5, 10, 20, 30, 60, 90),
) -> pd.DataFrame:
    """
    Compute rainfall (R) and effective rainfall (ER) feature sets.

    Notes
    -----
    This corresponds to the "rainfall_effrainfall_features_calcul_V3" function used for Viella,
    with minor readability improvements.
    """
    base_df = df[["Date"] + list(meteo_col_list)].copy()

    roll_dict = _rolling_features(df, meteo_col_list, T_periods)
    sat_dict = _saturation_indices(df, meteo_col_list, short_terms=(3, 5, 10, 20, 30, 40, 60), long_terms=(30, 60, 90), rain_threshold=3.0)
    anom_dict = _anomaly_indices(df, meteo_col_list, T_periods)
    evt_dict = _event_strength_pipeline(df, meteo_col_list, dry_tolerance=2)

    feats = pd.concat(
        [base_df, pd.DataFrame(roll_dict), pd.DataFrame(sat_dict), pd.DataFrame(anom_dict), pd.DataFrame(evt_dict)],
        axis=1,
    )

    # Drop unnecessary columns
    drop_cols = [c for c in feats.columns if ("rainy_days" in c or "dry_days" in c)]
    if drop_cols:
        feats = feats.drop(columns=drop_cols)

    # NaNs: saturation => 0 (neutral), event_strength => 0 (neutral)
    sat_cols = [c for c in feats.columns if "sat" in c]
    if sat_cols:
        feats[sat_cols] = feats[sat_cols].fillna(0)

    evt_cols = [c for c in feats.columns if "event_strength" in c]
    if evt_cols:
        feats[evt_cols] = feats[evt_cols].fillna(0)

    # Remove raw meteorological columns
    feats = feats.drop(columns=list(meteo_col_list))

    return feats


# =============================================================================
# GWL/WL feature blocks
# =============================================================================
def _lag_blocks(s: pd.Series, shift: int) -> pd.Series:
    """Lagged series (shifted by 'shift' steps)."""
    return s.shift(shift)


def _diff_mean_pipeline(s: pd.Series, lag_name: str, periods: Sequence[int]) -> Dict[str, pd.Series]:
    """Differences and rolling means over multiple periods."""
    out: Dict[str, pd.Series] = {}
    for P in periods:
        out[f"{lag_name}_diff{P}"] = s.diff(P)
        out[f"{lag_name}_mean{P}"] = s.rolling(window=P).mean()
    return out


def _extrema_pipeline(s: pd.Series, lag_name: str, windows: Sequence[int]) -> Dict[str, pd.Series]:
    """Rolling max/min over multiple window sizes."""
    out: Dict[str, pd.Series] = {}
    for w in windows:
        out[f"{lag_name}_max{w}"] = s.rolling(window=w, min_periods=1).max()
        out[f"{lag_name}_min{w}"] = s.rolling(window=w, min_periods=1).min()
    return out


def _global_mean_diff(s: pd.Series, lag_name: str, global_mean: float) -> Dict[str, pd.Series]:
    """Deviation from the (constant) global mean."""
    return {f"{lag_name}_diff_Gmean": s - global_mean}


def gwl_features_calcul(
    df: pd.DataFrame,
    hydro_col_list: Sequence[str],
    shift_list: Sequence[int] = (0, 1, 5, 8, 20, 30),
    P_periods: Sequence[int] = (2, 5, 10, 20, 30, 60),
    W_periods: Sequence[int] = (5, 10, 20, 30, 60, 90),
) -> pd.DataFrame:
    """
    Compute groundwater-level features (GWL/WL).

    Notes
    -----
    This corresponds to the "piezo_features_calcul_V6" function used for Viella,
    with minor readability improvements.
    """
    parts = [df[["Date"]].copy()]

    for gwl_col in hydro_col_list:
        s = df[gwl_col]
        gmean = float(s.mean())

        feat_dict: Dict[str, pd.Series] = {}
        for sh in shift_list:
            lag_s = _lag_blocks(s, sh)
            lag_name = f"{gwl_col}_lag{sh}"

            feat_dict[lag_name] = lag_s
            feat_dict.update(_diff_mean_pipeline(lag_s, lag_name, P_periods))
            feat_dict.update(_extrema_pipeline(lag_s, lag_name, W_periods))
            feat_dict.update(_global_mean_diff(lag_s, lag_name, gmean))

        parts.append(pd.DataFrame(feat_dict, index=df.index))

    return pd.concat(parts, axis=1)


# =============================================================================
# Full-series feature computation
# =============================================================================
def compute_features(
    hm_df: pd.DataFrame,
    externe_col_list: Sequence[str],
    meteo_col_list: Tuple[str, ...] = ("R", "ER"),
    hydro_col_list: Tuple[str, ...] = ("GWL",),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute features over the entire time series (including a future/forecast period).

    The goal is to compute the feature matrix once, then reuse it elsewhere (e.g. ML workflows).
    """
    if hm_df.empty:
        empty = pd.DataFrame({"Date": []})
        return empty, empty.copy(), empty.copy()

    r_er_df = rainfall_and_effective_rainfall_features_calcul(hm_df, list(meteo_col_list))
    gwl_df = gwl_features_calcul(hm_df, list(hydro_col_list))

    features_hm_df = reduce(lambda L, R: pd.merge(L, R, on="Date", how="outer"), [r_er_df, gwl_df])

    if externe_col_list:
        extern_part = hm_df[["Date"] + [c for c in externe_col_list if c in hm_df.columns]].copy()
        r_er_df = r_er_df.merge(extern_part, on="Date", how="left")
        gwl_df = gwl_df.merge(extern_part, on="Date", how="left")
        features_hm_df = features_hm_df.merge(extern_part, on="Date", how="left")

    return features_hm_df, r_er_df, gwl_df


def compute_features_Sech(
    hm_df: pd.DataFrame,
    externe_col_list: Sequence[str],
    meteo_col_list: Tuple[str, ...] = ("R", "ER"),
    hydro_col_list: Tuple[str, ...] = ("WLI", "WLM"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same as compute_features, but adapted to Séchilienne and its two WL columns."""
    return compute_features(hm_df, externe_col_list, meteo_col_list=meteo_col_list, hydro_col_list=hydro_col_list)


# =============================================================================
# Saving
# =============================================================================
def save_features_df(
    features_hm_df: pd.DataFrame,
    R_ER_features_df: pd.DataFrame,
    GWL_features_df: pd.DataFrame,
    output_folder_path: PathLike,
) -> None:
    """Save computed feature tables to Excel files."""
    out_dir = os.path.join(os.fspath(output_folder_path), "1_Features_ER_R_GWL")
    os.makedirs(out_dir, exist_ok=True)

    features_hm_df.to_excel(os.path.join(out_dir, "Features_HM.xlsx"), index=False)
    R_ER_features_df.to_excel(os.path.join(out_dir, "Features_METEO.xlsx"), index=False)
    GWL_features_df.to_excel(os.path.join(out_dir, "Features_HYDRO.xlsx"), index=False)

    print(f"Files successfully saved in {out_dir}")


# =============================================================================
# End-to-end pipelines
# =============================================================================
def features_pipeline_totale(
    input_path: PathLike,
    input_file: str,
    input_usecols: Sequence[str],
    externe_col_list: Sequence[str],
    output_folder_path: PathLike,
):
    """Viella and Villerville version."""
    hm_df = data_importation(input_path, input_file, input_usecols)
    features_hm_df, r_er_df, gwl_df = compute_features(
        hm_df,
        externe_col_list,
        meteo_col_list=("R", "ER"),
        hydro_col_list=("GWL",),
    )
    save_features_df(features_hm_df, r_er_df, gwl_df, output_folder_path)
    return features_hm_df, r_er_df, gwl_df


def features_pipeline_totale_Sech(
    input_path: PathLike,
    input_file: str,
    input_usecols: Sequence[str],
    externe_col_list: Sequence[str],
    output_folder_path: PathLike,
):
    """Séchilienne version."""
    hm_df = data_importation_Sech(input_path, input_file, input_usecols)
    features_hm_df, r_er_df, gwl_df = compute_features_Sech(
        hm_df,
        externe_col_list,
        meteo_col_list=("R", "ER"),
        hydro_col_list=("WLI", "WLM"),
    )
    save_features_df(features_hm_df, r_er_df, gwl_df, output_folder_path)
    return features_hm_df, r_er_df, gwl_df

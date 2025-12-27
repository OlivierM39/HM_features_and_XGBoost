from __future__ import annotations

import os
from functools import reduce
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ===== 1. Data importation ===== #
def data_importation(prisme_path, prisme_file, features_path, features_file, col_prisme, externe_col_list):
    
    """
    Imports the file containing the target variables and the file containing the features, and merges them into a single DataFrame.

    Requires that the name of the column containing the target values starts with the target identifier
    and that any suffixes are separated by an underscore "_" (e.g., "P500_vel", in which case the target name is "P500").
    """

    # === i. Read === #
    prisme_data  = pd.read_excel(os.path.join(prisme_path, prisme_file)) # Movement data
    features_data = pd.read_excel(os.path.join(features_path, features_file)) # Features_data

    # === ii. Formatting === #
    prisme_name = col_prisme.split('_')[0]
    prisme_data = prisme_data[['Date', col_prisme]].copy()
    prisme_data.rename(columns={col_prisme: prisme_name}, inplace=True)

    # === iii. Parsing & sorting === #
    prisme_data['Date']  = pd.to_datetime(prisme_data['Date'], errors='coerce')
    features_data['Date'] = pd.to_datetime(features_data['Date'], errors='coerce')
    prisme_data  = prisme_data.sort_values('Date').reset_index(drop=True)
    features_data = features_data.sort_values('Date').reset_index(drop=True)
    
    # === iv. Build merged DataFrame === #
    dfs = [prisme_data, features_data]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='inner'), dfs)
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)

    # Info before cleaning
    print(f"[Avant dropna] n={len(merged_df)} | Date range: {merged_df['Date'].min()} -> {merged_df['Date'].max()}")

    # === v. Observable base (Date + target + external variables) === #
    externes_ok = [c for c in externe_col_list if c in merged_df.columns]
    base_df = merged_df[['Date', prisme_name] + externes_ok].copy()
    
    # === vi. Drop incomplete rows (target + features) === #
    merged_df = merged_df.dropna().reset_index(drop=True)

    # Info after cleaning
    if len(merged_df):
        print(f"[Après  dropna] n={len(merged_df)} | Date range: {merged_df['Date'].min()} -> {merged_df['Date'].max()}")
    else:
        print("[Après  dropna] n=0")
    
    return base_df, merged_df, prisme_name

# ===== 1Bis. Data importation of Séchilienne landslide ===== #
def data_importation_Sech(prisme_data, features_data, col_prisme, externe_col_list):
    
    """
    Note: An adapted version of "data_importation" function, specific to Séchilienne data application.
    """
    # === ii. Formatting === #
    prisme_name = col_prisme.split('_')[0]
    prisme_data = prisme_data[['Date', col_prisme]].copy()
    prisme_data.rename(columns={col_prisme: prisme_name}, inplace=True)

    # === iii. Parsing & sorting === #
    prisme_data['Date']  = pd.to_datetime(prisme_data['Date'], errors='coerce')
    features_data['Date'] = pd.to_datetime(features_data['Date'], errors='coerce')
    prisme_data  = prisme_data.sort_values('Date').reset_index(drop=True)
    features_data = features_data.sort_values('Date').reset_index(drop=True)
    
    # === iv. Build merged DataFrame === #
    dfs = [prisme_data, features_data]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='inner'), dfs)
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)

    # Info before cleaning
    print(f"[Avant dropna] n={len(merged_df)} | Date range: {merged_df['Date'].min()} -> {merged_df['Date'].max()}")

    # === v. Observable base (Date + target + external variables) === #
    externes_ok = [c for c in externe_col_list if c in merged_df.columns]
    base_df = merged_df[['Date', prisme_name] + externes_ok].copy()
    
    # === vi. Drop incomplete rows (target + features) === #
    merged_df = merged_df.dropna().reset_index(drop=True)

    # Info after cleaning
    if len(merged_df):
        print(f"[Après  dropna] n={len(merged_df)} | Date range: {merged_df['Date'].min()} -> {merged_df['Date'].max()}")
    else:
        print("[Après  dropna] n=0")
    
    return base_df, merged_df, prisme_name

# ===== 2. Division de la série en 5 parts égales ===== #
def data_splitting_preparation(merged_df, bloc_numbers):

    """
    Splits the full time series into N equal parts.
    """

    df = merged_df.copy()

    bloc_size = len(df) // bloc_numbers
    bloc_list = list(range(1, bloc_numbers + 1))

    sub_dfs = {}
    for i in range(bloc_numbers):
        start_index = i*bloc_size
        # Include the remaining rows in the last block
        end_index = start_index + bloc_size if i != bloc_numbers - 1 else len(df)
        sub_dfs[f'subdf_{i+1}'] = df.iloc[start_index : end_index]

    # Check the length of each sub_df
    for bloc in bloc_list:
        subdf_name = f"subdf_{bloc}"
        length = len(sub_dfs[subdf_name]) # (kept for debugging / verification)

    # Build the list of sub_dfs
    subdf_list = [sub_dfs[f'subdf_{i+1}'] for i in range(bloc_numbers)]

    return subdf_list

# ===== 3. Build train_df and test_df ===== #
def train_test_splitting(subdf_list, test_df_index):

    """
    Defines the test_df and the resulting train_df based on the index of the selected test block.
    """

    test_df = subdf_list[test_df_index]

    train_df_list = [subdf for i, subdf in enumerate(subdf_list) if i != test_df_index]
    train_df = pd.concat(train_df_list, ignore_index=True)
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    train_df = train_df.sort_values(by='Date').reset_index(drop=True)

    return train_df, test_df

# ===== 4. Define targets (y) and feature matrices (X) for train_df and test_df ===== #
def X_and_y_definition(train_df, test_df, prisme_name, externe_col_list): 

    """
    Splits X and y for both the train_df and the test_df.
    """

    drop_cols = ['Date', prisme_name] + externe_col_list

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[prisme_name]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[prisme_name]

    return X_train, y_train, X_test, y_test

# ===== 5. Pipeline step ===== #
def pipeline_data_preparation(prisme_path, prisme_file, features_path, features_file, col_prisme, externe_col_list,
                             bloc_numbers, test_df_index):

    # == 1. Import == #
    base_df, merged_df, prisme_name = data_importation(prisme_path, prisme_file, features_path, features_file, 
                                                       col_prisme, externe_col_list)

    # == 2. Split the time series == #
    subdf_list = data_splitting_preparation(merged_df, bloc_numbers)

    # == 3. Define train_df and test_df == #
    train_df, test_df = train_test_splitting(subdf_list, test_df_index)

    # == 4. Define X and y for training and testing sets == #
    X_train, y_train, X_test, y_test = X_and_y_definition(train_df, test_df, prisme_name, externe_col_list)

    return base_df, train_df, test_df, X_train, y_train, X_test, y_test, subdf_list, prisme_name

# ===== 5Bis. Pipeline step (Séchilienne case) ===== #
def pipeline_data_preparation_Sech(prisme_data, features_data, col_prisme, externe_col_list, bloc_numbers, test_df_index):

    # == 1. Import == #
    base_df, merged_df, prisme_name = data_importation_Sech(prisme_data, features_data, col_prisme, externe_col_list)

    # == 2. Split the time series == #
    subdf_list = data_splitting_preparation(merged_df, bloc_numbers)

    # == 3. Define train_df and test_df == #
    train_df, test_df = train_test_splitting(subdf_list, test_df_index)

    # == 4. Define X and y for training and testing sets == #
    X_train, y_train, X_test, y_test = X_and_y_definition(train_df, test_df, prisme_name, externe_col_list)

    return base_df, train_df, test_df, X_train, y_train, X_test, y_test, subdf_list, prisme_name

# ===== 1. Metric computation ===== #
def metrics_evaluation(y_obs, y_pred):

    R2 = float(r2_score(y_obs, y_pred))
    RMSE = float(np.sqrt(mean_squared_error(y_obs, y_pred)))
    MAE = float(mean_absolute_error(y_obs, y_pred))

    return {"R2": R2, "RMSE": RMSE, "MAE": MAE, "n": int(len(y_obs))}

# ===== 2. XGBoost model ===== #
def XGB_model(X_train, y_train, X_test, y_test):

    # == 1. Model training == #
    model = XGBRegressor(n_estimators=1000)
    model.fit(X_train, y_train)

    # == 2. Retrieve hyperparameter values == #
    hyperparameter_values = model.get_params()
    params_df = pd.DataFrame(hyperparameter_values.items(), columns=["Hyperparameter", "Value"])

    # == 3. Predictions on train and test sets == #
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # == 4. Compute metrics == #
    train_metrics = metrics_evaluation(y_train, y_pred_train)
    test_metrics = metrics_evaluation(y_test, y_pred_test)

    metrics_df = pd.DataFrame.from_dict({"train": train_metrics, "test": test_metrics}, 
                                        orient="index").loc[["train", "test"]].round({"R2": 3, "RMSE": 3, "MAE": 3})

    # == 5. Feature importance == #
    # Feature importance based on "gain"
    importances = model.feature_importances_
    array_importance = np.array([X_test.columns, importances])
    
    # Build a DataFrame of features ranked by importance
    FI_df = pd.DataFrame(np.transpose(array_importance), columns=['Features', 'Importances'])
    FI_df = FI_df.sort_values(by='Importances', ascending=False)
    FI_top = FI_df.head(20)

    # == 6. SHAP values == #
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    return y_pred_train, y_pred_test, params_df, metrics_df, FI_df, FI_top, shap_values, model

# ===== 3. Create the output DataFrame for predicted values ===== #
def output_df_creation(base_df, train_df, test_df, y_test, y_pred_train, y_pred_test):

    # Dates and time span
    date_start = pd.to_datetime(base_df["Date"]).min()
    date_end   = pd.to_datetime(base_df["Date"]).max()
    full_dates = pd.DataFrame({"Date": pd.date_range(start=date_start, end=date_end, freq="D")})

    # Train and test dates
    train_dates = train_df["Date"].values 
    test_dates = test_df["Date"].values

    pred_train_df = pd.DataFrame({
        'Date': train_dates, 
        'vel_train_pred': pd.Series(y_pred_train).values
    })

    pred_test_df = pd.DataFrame({
        'Date': test_dates, 
        'vel_test': pd.Series(y_test).values, 
        'vel_test_pred': pd.Series(y_pred_test).values
    })

    dfs = [base_df, pred_train_df, pred_test_df]
    merged = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), dfs)
    out_df = full_dates.merge(merged, on="Date", how="left").sort_values("Date").reset_index(drop=True)

    return out_df

# ===== 4. Pipeline step ===== #
def pipeline_XGBoost_application(base_df, train_df, test_df, X_train, y_train, X_test, y_test):

    # == 1. XGBoost == #
    y_pred_train, y_pred_test, params_df, metrics_df, FI_df, FI_top, shap_values, model = XGB_model(X_train, y_train, X_test, y_test)

    # == 2. Output DataFrame == #
    out_df = output_df_creation(base_df, train_df, test_df, y_test, y_pred_train, y_pred_test)

    return out_df, params_df, metrics_df, FI_df, FI_top, shap_values, model
 

# ===== 0. Extract metric values for display ===== #
def metrics_box_text(metrics_df, split="test", keys=("RMSE","MAE","R2"), fmt=("{:.3f}","{:.3f}","{:.3f}")):
    row = metrics_df.loc[split]
    lines = []
    for k, f in zip(keys, fmt):
        if k in metrics_df.columns:
            val = row[k]
            lines.append(f"{k}: {f.format(val)}")
    return "\n".join(lines)

# ===== 1. Time-series plot ===== #
def graphique_1(out_df, prisme_name, test_df_index, metrics_df, show_figure):

    """
    Displays:
      - Top subplot   : prisme_name (observed, full time series) + vel_pred (on train and test)
      - Bottom subplot: External time series (ER + GWL)
    """
    
    df = out_df.copy()
    
    # === Figure === #
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,7))

    # Figure 1 
    ax1.plot(df['Date'], df[prisme_name], label=f'{prisme_name} observed', color='grey', alpha=0.8, lw=1.3)
    ax1.plot(df['Date'], df['vel_train_pred'], label=f'{prisme_name} simulated on train', color='maroon', lw=1)
    ax1.plot(df['Date'], df['vel_test_pred'], label=f'{prisme_name} simulated on test', color='blue', lw=1)

    ax1.set_ylabel("Velocity (cm/day)")
    ax1.set_title(f"Observed vs simulated velocity with test_set n°{test_df_index+1}")
    ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False), ax1.spines['right'].set_visible(False)
    ax1.legend(loc="best")

    txt = metrics_box_text(metrics_df, split="test", keys=("RMSE","MAE","R2"), fmt=("{:.3f}","{:.3f}","{:.3f}"))
    ax1.text(0.01, 0.98, txt,
        transform=ax1.transAxes, va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Figure 2
    ax2.plot(df['Date'], df['GWL'], label='Groundwater level', color='navy')
    ax2.set_ylabel("Groundwater level (m)")
    ax2.spines['top'].set_visible(False)

    ax2b = ax2.twinx()
    ax2b.bar(df['Date'], df['ER'], label='Effective rainfall', color='deepskyblue', width=0.7, alpha=0.7)
    ax2b.invert_yaxis()
    ax2b.set_ylabel("Effective rainfall (mm)")
    ax2b.spines['top'].set_visible(False)

    ax2.set_title("External time series (ER + GWL)")
    ax2.grid(True, alpha=0.2)

    fig1.tight_layout()

    if show_figure:
        plt.show()
    else:
        plt.close(fig1)

    return fig1

# ===== 1BIS.Time-series plot (Séchilienne case) ===== #
def graphique_1bis(out_df, prisme_name, test_df_index, metrics_df, show_figure):

    """
    Displays:
      - Top subplot   : prisme_name (observed, full time series) + vel_pred (on train and test)
      - Bottom subplot: External time series (ER + GWL)
    """
    
    df = out_df.copy()
    
    # === Figure === #
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,7))

    # Figure 1 
    ax1.plot(df['Date'], df[prisme_name], label=f'{prisme_name} observed', color='grey', alpha=0.8, lw=1.3)
    ax1.plot(df['Date'], df['vel_train_pred'], label=f'{prisme_name} simulated on train', color='maroon', lw=1)
    ax1.plot(df['Date'], df['vel_test_pred'], label=f'{prisme_name} simulated on test', color='blue', lw=1)

    ax1.set_ylabel("Velocity (cm/day)")
    ax1.set_title(f"Observed vs simulated velocity with test_set n°{test_df_index+1}")
    ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False), ax1.spines['right'].set_visible(False)
    ax1.legend(loc="best")

    txt = metrics_box_text(metrics_df, split="test", keys=("RMSE","MAE","R2"), fmt=("{:.3f}","{:.3f}","{:.3f}"))
    ax1.text(0.01, 0.98, txt,
        transform=ax1.transAxes, va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Figure 2
    ax2.plot(df['Date'], df['WLI'], label='Groundwater level', color='dodgerblue')
    ax2.plot(df['Date'], df['WLM'], label='Groundwater level', color='navy')
    ax2.set_ylabel("Groundwater level (m)")
    ax2.spines['top'].set_visible(False)

    ax2b = ax2.twinx()
    ax2b.bar(df['Date'], df['ER'], label='Effective rainfall', color='deepskyblue', width=0.7, alpha=0.7)
    ax2b.invert_yaxis()
    ax2b.set_ylabel("Effective rainfall (mm)")
    ax2b.spines['top'].set_visible(False)

    ax2.set_title("External time series (ER + GWL)")
    ax2.grid(True, alpha=0.2)

    fig1.tight_layout()

    if show_figure:
        plt.show()
    else:
        plt.close(fig1)

    return fig1

# ===== 2. Features importance ===== #
def graphique_2(FI_top, prisme_name, test_df_index, show_figure):

    """
    Feature importance ranking (displays only a limited number).
    """

    # === Prepare colour assignment for bars === #
    prefix_color_map = {"GWL": "rebeccapurple",
                        "WLI": "darkorchid",
                        "WLM": "rebeccapurple",
                        "R":   "steelblue",
                        "ER":  "deepskyblue",
                       }

    # Colour assignment function
    def assign_color(feature):
        for prefix, color in prefix_color_map.items():
            if feature.startswith(prefix):
                return color
        return "gray"  # couleur par défaut

    # Preparation
    df = FI_top.copy()
    colors = df["Features"].map(assign_color)
    
    # === Figure === #
    fig2, ax = plt.subplots(figsize=(12,4))
    bars = ax.bar(df["Features"], df["Importances"], color=colors, edgecolor="black")
    ax.set_ylabel("Importance")
    ax.set_xticks(range(len(df["Features"])))
    ax.set_xticklabels(df["Features"], rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    ax.set_title(f"Top Feature Importance for {prisme_name} | Test bloc n°{test_df_index+1}")

    fig2.tight_layout()
    
    if show_figure:
        plt.show()
    else:
        plt.close(fig2)

    return fig2

# ===== 3. SHAP values ===== #
def graphique_3(shap_values, X_test, prisme_name, test_df_index, show_figure):

    """
    SHAP values.
    """
    plt.figure(figsize=(12,4))

    shap.summary_plot(shap_values, X_test, show=False)
    fig3 = plt.gcf()
    fig3.suptitle(f"SHAP summary for {prisme_name} | Test bloc n°{test_df_index+1}")
    fig3.tight_layout()

    if show_figure:
        plt.show()
    else:
        plt.close(fig3)    

    return fig3

# ===== 4. Pipeline step ===== #
def pipeline_graphique(out_df, FI_top, shap_values, X_test, prisme_name, test_df_index, metrics_df, show_figure):
    
    fig1 = graphique_1(out_df, prisme_name, test_df_index, metrics_df, show_figure)
    fig2 = graphique_2(FI_top, prisme_name, test_df_index, show_figure)
    fig3 = graphique_3(shap_values, X_test, prisme_name, test_df_index, show_figure)

    return fig1, fig2, fig3

# ===== 4BIS. Pipeline step for Séchilienne ===== #
def pipeline_graphique_Sech(out_df, FI_top, shap_values, X_test, prisme_name, test_df_index, metrics_df, show_figure):
    
    fig1 = graphique_1bis(out_df, prisme_name, test_df_index, metrics_df, show_figure)
    fig2 = graphique_2(FI_top, prisme_name, test_df_index, show_figure)
    fig3 = graphique_3(shap_values, X_test, prisme_name, test_df_index, show_figure)

    return fig1, fig2, fig3

def Save(out_df, params_df, metrics_df, FI_df, FI_top, fig1, fig2, fig3, model, prisme_name, test_df_index, output_path):

    # == 1. Create the output directory == #
    out_dir = os.path.join(output_path, fr'2_XGBoost_results\{prisme_name}\Bloc_{test_df_index+1}')
    os.makedirs(out_dir, exist_ok=True)

    # == 2. Save outputs == #
    # Save Excel files
    out_df.to_excel(os.path.join(out_dir, 'Results_time_series.xlsx'), index=False)
    params_df.to_excel(os.path.join(out_dir, 'Results_hyperparameters.xlsx'), index=False)
    metrics_df.to_excel(os.path.join(out_dir, 'Results_metrics.xlsx'), index=False)
    FI_df.to_excel(os.path.join(out_dir, 'Results_features_importance.xlsx'), index=False)
    FI_top.to_excel(os.path.join(out_dir, 'Results_top_features_importance.xlsx'), index=False) 
    print(f".XLSX files saved in {out_dir}")

    # Save figures
    fig1.savefig(os.path.join(out_dir, "Results_time_series.png"), dpi=300, bbox_inches="tight")
    fig2.savefig(os.path.join(out_dir, "Results_features_importance.png"), dpi=300, bbox_inches="tight")
    fig3.savefig(os.path.join(out_dir, "Results_shap_values.png"), dpi=300, bbox_inches="tight")
    print(f".PNG files saved in {out_dir}")

    # Save the trained model
    joblib.dump(model, os.path.join(out_dir, "XGB_trained_model.joblib"))
    print(f".JOBLIB file saved in {out_dir}")

    return

def pipeline_totale(prisme_path, prisme_file, features_path, features_file, col_prisme, externe_col_list,
                    bloc_numbers, test_df_index, output_path, show_figure):

    # ===== 1. Data reading and preparation ===== #
    base_df, train_df, test_df, X_train, y_train, X_test, y_test, subdf_list, prisme_name = pipeline_data_preparation(prisme_path, prisme_file, 
                                                                                                             features_path, features_file,
                                                                                                             col_prisme, externe_col_list,
                                                                                                             bloc_numbers, test_df_index)

    # ===== 2. XGBoost model ===== #
    out_df, params_df, metrics_df, FI_df, FI_top, shap_values, model = pipeline_XGBoost_application(base_df, train_df, test_df, 
                                                                                                    X_train, y_train, X_test, y_test)

    
    # ===== 3. Figures ===== #
    fig1, fig2, fig3 = pipeline_graphique(out_df, FI_top, shap_values, X_test, prisme_name, test_df_index, metrics_df, show_figure)

    # ===== 4. Saving ===== #
    Save(out_df, params_df, metrics_df, FI_df, FI_top, fig1, fig2, fig3, model, prisme_name, test_df_index, output_path)

    return out_df, metrics_df, FI_df

def pipeline_totale_Sech(prisme_data, features_data, col_prisme, externe_col_list,
                    bloc_numbers, test_df_index, output_path, show_figure):

    # ===== 1. Data reading and preparation ===== #
    base_df, train_df, test_df, X_train, y_train, X_test, y_test, subdf_list, prisme_name = pipeline_data_preparation_Sech(prisme_data, 
                                                                                                       features_data, col_prisme, 
                                                                                                       externe_col_list, bloc_numbers, 
                                                                                                       test_df_index)

    # ===== 2. XGBoost model ===== #
    out_df, params_df, metrics_df, FI_df, FI_top, shap_values, model = pipeline_XGBoost_application(base_df, train_df, test_df, X_train, 
                                                                                                    y_train, X_test, y_test)

    
    # ===== 3. Figures ===== #
    fig1, fig2, fig3 = pipeline_graphique_Sech(out_df, FI_top, shap_values, X_test, prisme_name, test_df_index, metrics_df, show_figure)

    # ===== 4. Saving ===== #
    Save(out_df, params_df, metrics_df, FI_df, FI_top, fig1, fig2, fig3, model, prisme_name, test_df_index, output_path)

    return out_df, metrics_df, FI_df

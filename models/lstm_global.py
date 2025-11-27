import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse, mae
from darts.models import BlockRNNModel
import torch.nn as nn
from matplotlib import pyplot as plt
from numpy.f2py.crackfortran import verbose
from sklearn.preprocessing import MinMaxScaler

MODEL_NAME = 'BlockRNN_LSTM'

TARGET_VAR = 'prec'
target_scaler = Scaler()
cov_scaler = Scaler()


def load_data_into_df(filepath):
    df = pd.read_csv(
        filepath,
        sep=',',
    )

    df['Date'] = pd.to_datetime(df[['YYYY', 'MM', 'DD']].rename(columns={
        'YYYY': 'year',
        'MM': 'month',
        'DD': 'day'
    }))

    # Set the index to the new Date column
    df = df.set_index('Date')

    # drop data columns
    df.drop(columns=['YYYY', 'MM', 'DD', 'DOY'], inplace=True)
    return df


def separate_target_variable(df):
    # 1. Create the Target Series (What we want to predict)
    target_series = TimeSeries.from_series(df[TARGET_VAR], freq='D')

    # 2. Create the Covariates Series (All OTHER features)
    # We drop the target column to create a dataframe of just features
    covariates_df = df.drop(columns=[TARGET_VAR])

    # Convert features to Darts TimeSeries
    covariates_series = TimeSeries.from_dataframe(covariates_df, freq='D')

    return covariates_series, target_series


def scale(covariates_series, target_series):
    # Fit and transform
    target_scaled = target_scaler.fit_transform(target_series)
    cov_scaled = cov_scaler.fit_transform(covariates_series)
    return target_scaled, cov_scaled


if __name__ == '__main__':

    ATTR_PATH = '../data_for_prediction/1_attributes/Catchment_attributes.csv'

    attr_df = pd.read_csv(ATTR_PATH, sep=';', index_col='ID')
    # Select useful features. DO NOT use all columns (too much noise).
    # Good candidates: Area, Mean Elevation, Mean Slope, Forest Cover.
    selected_attrs = ['area_calc', 'elev_mean', 'slope_mean', 'forest_fra']
    # selected_attrs = [
    #     # --- Topography (How fast water moves) ---
    #     'area_calc',  # Catchment size (Critical scaling factor)
    #     'elev_mean',  # Mean elevation (Determines temperature/snow)
    #     'slope_mean',  # Steepness (Steep = Fast runoff)
    #
    #     # --- Climate Context (Long-term averages) ---
    #     'p_mean',  # Mean annual precip (Wet vs Dry basin)
    #     'frac_snow',  # Fraction of precip as snow (CRITICAL for Austria)
    #     'p_season',  # Seasonality (Does it rain in Summer or Winter?)
    #
    #     # --- Land Cover (The "Sponge" effect) ---
    #     'forest_fra',  # Forest cover (Trees delay water)
    #     'glac_fra',  # Glacier fraction (Critical for summer melt in Alps)
    #     'urban_fra',  # Urban area (Concrete = Flash floods)
    #
    #     # --- Subsurface (Storage capacity) ---
    #     'soil_condu',  # Soil hydraulic conductivity (How fast water sinks in)
    #     'soil_tawc',  # Total Available Water Content (How much water soil holds)
    #     'geol_perme'  # Deep geological permeability (Groundwater loss)
    # ]
    attr_subset = attr_df[selected_attrs].copy()

    # SCALE THE ATTRIBUTES (Crucial!)
    # We use sklearn because this is a static dataframe, not a time series.
    scaler_static = MinMaxScaler()
    attr_scaled_df = pd.DataFrame(
        scaler_static.fit_transform(attr_subset),
        columns=attr_subset.columns,
        index=attr_subset.index
    )

    print(f"Loaded attributes for {len(attr_scaled_df)} basins.")

    data_dir = "../data_for_prediction/"
    file_pattern = str(Path(data_dir) / "*.csv")
    files = glob.glob(file_pattern)
    print(f"Found {len(files)} files")

    all_cov_ts = []
    all_target_ts = []
    for csv_path in files[:5]:
        filename = os.path.basename(csv_path)
        basin_id = int(os.path.splitext(filename)[0].removeprefix('ID_'))
        try:
            static_row = attr_scaled_df.loc[basin_id]  # Look up ID in attribute table

            # Convert to a DataFrame with 1 row, columns = feature names
            static_cov_df = pd.DataFrame([static_row.values], columns=static_row.index)
        except KeyError:
            print(f"Warning: ID {basin_id} not found in attributes file. Skipping.")
            continue
        covariates_ts, target_ts = separate_target_variable(load_data_into_df(csv_path))
        all_cov_ts.append(covariates_ts)
        # MAGIC: makes model take static attributes take into account per basin
        target_ts.with_static_covariates(static_cov_df)
        all_target_ts.append(target_ts)

    all_cov_ts_scaled, all_target_ts_scaled = scale(all_cov_ts, all_target_ts)

    # splits based on time so equal on any number of calls
    train_targets = [t.split_before(0.8)[0] for t in all_target_ts_scaled]
    val_targets = [t.split_before(0.8)[1] for t in all_target_ts_scaled]

    train_covs = [c.split_before(0.8)[0] for c in all_cov_ts_scaled]
    val_covs = [c.split_before(0.8)[1] for c in all_cov_ts_scaled]

    model = BlockRNNModel(
        input_chunk_length=30,  # Look back 30 days
        output_chunk_length=7,  # Predict 7 days
        model='LSTM',
        hidden_dim=20,
        n_rnn_layers=1,
        n_epochs=15,  # Increased epochs slightly for more complex data
        dropout=0.1,
        batch_size=16,
        random_state=42,
        loss_fn=nn.L1Loss()  # better for precipitation because it is less sensitive to extreme outliers
        # and discourages the model from "blurring" predictions (predicting constant drizzle to minimize squared error)
    )

    print(f"Fitting multivariate model for all files...")
    model.fit(
        series=train_targets,
        past_covariates=train_covs,
        val_series=val_targets,
        val_past_covariates=val_covs,
        verbose=True
    )
    pred_list_scaled = model.predict(
        n=7,
        series=train_targets,
        past_covariates=train_covs
    )

    # Inverse transform to get real units (mm)
    pred_list = target_scaler.inverse_transform(pred_list_scaled)
    val_list = target_scaler.inverse_transform(val_targets)

    # List comprehension to slice the Day 1 predictions for every basin
    preds_day_1 = [pred[0] for pred in pred_list]

    # List comprehension for Day 3 (Index 2)
    preds_day_3 = [pred[2] for pred in pred_list]

    # List comprehension for Day 7 (Index 6)
    preds_day_7 = [pred[6] for pred in pred_list]

    # ==========================================
    # Compute Metrics
    # ==========================================
    # Darts is smart: When you compare the 'preds_day_1' list against the full 'val_list',
    # it automatically finds the matching timestamp in the validation data for each point.

    # --- Day 1 ---
    rmse_d1 = rmse(val_list, preds_day_1, series_reduction=np.mean)
    mae_d1 = mae(val_list, preds_day_1, series_reduction=np.mean)

    # --- Day 3 ---
    rmse_d3 = rmse(val_list, preds_day_3, series_reduction=np.mean)
    mae_d3 = mae(val_list, preds_day_3, series_reduction=np.mean)

    # --- Day 7 ---
    rmse_d7 = rmse(val_list, preds_day_7, series_reduction=np.mean)
    mae_d7 = mae(val_list, preds_day_7, series_reduction=np.mean)

    print(f"--- Global Metrics (Average over {len(val_list)} basins) ---")
    print(f"Day 1: RMSE = {rmse_d1:.4f} mm | MAE = {mae_d1:.4f} mm")
    print(f"Day 3: RMSE = {rmse_d3:.4f} mm | MAE = {mae_d3:.4f} mm")
    print(f"Day 7: RMSE = {rmse_d7:.4f} mm | MAE = {mae_d7:.4f} mm")

    results_dir = Path("../prediction_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_results_df = pd.DataFrame([
        {
            'target': 'prec_1day_ahead',
            'model': MODEL_NAME,
            'RMSE': rmse_d1,
            'MAE': mae_d1,
        },
        {
            'target': 'prec_3day_ahead',
            'model': MODEL_NAME,
            'RMSE': rmse_d3,
            'MAE': mae_d3,
        },
        {
            'target': 'prec_7day_ahead',
            'model': MODEL_NAME,
            'RMSE': rmse_d7,
            'MAE': mae_d7,
        },
    ])
    output_filename = f'nn_lstm_global_metrics_per_day.csv'
    metrics_results_df.to_csv(results_dir / output_filename, index=False)
    print(f"Metrics saved to {results_dir / output_filename}")

    # Create lists to store individual scores
    basin_rmses = []

    # Loop through the results we generated above
    for i in range(len(pred_list)):
        # Calculate RMSE for just this one basin
        # Note: Darts compares the specific dates in pred_list[i]
        # with the matching dates in val_list[i]
        err = rmse(val_list[i], pred_list[i])
        basin_rmses.append(err)

    # Convert to numpy for easy stats
    basin_rmses = np.array(basin_rmses)

    print("\n--- Per-Basin Statistics ---")
    print(f"Best Basin RMSE:  {np.min(basin_rmses):.4f}")
    print(f"Worst Basin RMSE: {np.max(basin_rmses):.4f}")
    print(f"Mean Basin RMSE:  {np.mean(basin_rmses):.4f}")

    metrics_results_df = pd.DataFrame([
        {
            'model': MODEL_NAME,
            'best basin RMSE': np.min(basin_rmses),
            'worst basin RMSE': np.max(basin_rmses),
            'mean basin RMSE': np.mean(basin_rmses),
        }
    ])
    output_filename = f'nn_lstm_global_metrics_per_basin.csv'
    metrics_results_df.to_csv(results_dir / output_filename, index=False)

    # Visualizing the distribution of errors
    plt.figure(figsize=(10, 5))
    plt.hist(basin_rmses, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of RMSE across Basins')
    plt.xlabel('RMSE (mm)')
    plt.ylabel('Count of Basins')
    plt.show()


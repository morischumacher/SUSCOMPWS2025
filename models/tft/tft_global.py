import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse, mae
from darts.models import TFTModel  # <--- CHANGED
from darts.utils.likelihood_models import QuantileRegression  # <--- NEW for TFT
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODEL_NAME = 'TFT_Model'
ATTR_PATH = '../../data_for_prediction/1_attributes/Catchment_attributes.csv'
TARGET_VAR = 'prec'
selected_attrs = ['area_calc', 'elev_mean', 'slope_mean', 'forest_fra']
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

    df = df.set_index('Date')
    df.drop(columns=['YYYY', 'MM', 'DD', 'DOY', 'prec_1d_ahead', 'prec_3d_ahead', 'prec_7d_ahead'], inplace=True)
    return df


def separate_target_variable(df):
    target_series = TimeSeries.from_series(df[TARGET_VAR], freq='D')
    covariates_df = df.drop(columns=[TARGET_VAR])
    covariates_series = TimeSeries.from_dataframe(covariates_df, freq='D')
    return covariates_series, target_series


def scale(covariates_series, target_series):
    # Fit and transform
    target_scaled = target_scaler.fit_transform(target_series)
    cov_scaled = cov_scaler.fit_transform(covariates_series)
    return target_scaled, cov_scaled

def calculate_metrics(day, data_points):
    y_true = np.array(data_points[day]['true'])
    y_pred = np.array(data_points[day]['pred'])

    if len(y_true) == 0:
        return np.nan, np.nan  # Safety for empty data

    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    return rmse_val, mae_val


def setup_preds_and_vals_for_metrics(preds, vals):
    # ==========================================
    # 1. Setup Containers for Pooled Data
    # ==========================================
    # We will collect matched pairs (Actual, Predicted) for each horizon
    # from all basins into these lists.
    data_points = {
        1: {'true': [], 'pred': []},
        3: {'true': [], 'pred': []},
        7: {'true': [], 'pred': []}
    }

    # The indices in the 7-day forecast vector corresponding to Day 1, 3, and 7
    horizon_indices = {1: 0, 3: 2, 7: 6}

    # ==========================================
    # 2. Loop & Align (The Robust "Time-Aware" Extraction)
    # ==========================================
    # Iterate through each basin
    for i in range(len(preds)):
        ts_pred = preds[i]  # The 7-day forecast for Basin i
        ts_val = vals[i]  # The actual validation data for Basin i

        # For each horizon of interest (1, 3, 7 days)
        for day, idx in horizon_indices.items():
            # Get the prediction value and its specific timestamp
            # ts_pred[idx] returns a TimeSeries of length 1.
            # .time_index[0] gives us the Pandas Timestamp.
            pred_value = ts_pred[idx].values()[0][0]
            pred_time = ts_pred[idx].time_index[0]

            # Find the matching 'Actual' value in the validation series
            # We try to slice the validation series at exactly that time.
            try:
                # ts_val[pred_time] will raise an error if the date is missing
                actual_value = ts_val[pred_time].values()[0][0]

                # If found, add to our global pool
                data_points[day]['true'].append(actual_value)
                data_points[day]['pred'].append(pred_value)

            except (KeyError, ValueError):
                # This handles cases where validation data might have gaps (missing days)
                # We skip this point to avoid corrupting the metrics.
                continue

    return data_points

if __name__ == '__main__':
    attr_df = pd.read_csv(ATTR_PATH, sep=';', index_col='ID')
    attr_subset = attr_df[selected_attrs].copy()

    scaler_static = MinMaxScaler()
    attr_scaled_df = pd.DataFrame(
        scaler_static.fit_transform(attr_subset),
        columns=attr_subset.columns,
        index=attr_subset.index
    )

    print(f"Loaded attributes for {len(attr_scaled_df)} basins.")

    data_dir = "../../data_for_prediction/"
    file_pattern = str(Path(data_dir) / "*.csv")
    files = glob.glob(file_pattern)
    print(f"Found {len(files)} files")

    all_cov_ts = []
    all_target_ts = []

    for csv_path in files:
        filename = os.path.basename(csv_path)
        basin_id = int(os.path.splitext(filename)[0].removeprefix('ID_'))

        try:
            static_row = attr_scaled_df.loc[basin_id]
            static_cov_df = pd.DataFrame([static_row.values], columns=static_row.index)
        except KeyError:
            print(f"Warning: ID {basin_id} not found in attributes file. Skipping.")
            continue

        covariates_ts, target_ts = separate_target_variable(load_data_into_df(csv_path))

        all_cov_ts.append(covariates_ts)

        # Attach static covariates to the target series
        target_ts = target_ts.with_static_covariates(static_cov_df)
        all_target_ts.append(target_ts)

    train_targets_raw = [t.split_before(0.8)[0] for t in all_target_ts]
    val_targets_raw = [t.split_before(0.8)[1] for t in all_target_ts]

    train_covs_raw = [c.split_before(0.8)[0] for c in all_cov_ts]
    val_covs_raw = [c.split_before(0.8)[1] for c in all_cov_ts]

    train_targets = target_scaler.fit_transform(train_targets_raw)
    train_covs = cov_scaler.fit_transform(train_covs_raw)

    # Transform validation list using the trained scaler (DO NOT FIT)
    val_targets = target_scaler.transform(val_targets_raw)
    val_covs = cov_scaler.transform(val_covs_raw)

    model = TFTModel(
        input_chunk_length=30,
        output_chunk_length=7,
        hidden_size=32,  # TFT specific: size of internal layers
        lstm_layers=1,  # TFT specific
        num_attention_heads=4,  # TFT specific
        dropout=0.1,
        batch_size=64,
        n_epochs=5,
        add_relative_index=True,  # Helps model track "steps since start"

        # TFT Feature: Automatically create cyclical date encodings (Month, Day)
        # This acts as "Future Covariates" known in advance.
        # it extracts that information directly from the Date Index (the timestamp) that is set in the loading function
        add_encoders={
            'cyclic': {'future': ['month', 'dayofyear']},
        },

        # TFT is probabilistic. We use QuantileRegression.
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        random_state=42
    )

    model.fit(
        series=train_targets,
        past_covariates=train_covs,
        val_series=val_targets,
        val_past_covariates=val_covs,
        verbose=True
    )

    # num_samples=1 gives a stochastic sample.
    # With QuantileRegression, we usually want the median (0.5) for metrics.
    pred_list_scaled = model.predict(
        n=7,
        series=train_targets,
        past_covariates=train_covs,
        num_samples=1  # Generates distribution, 1 sample for median pred (0.5 Quantile)
    )

    val_list = target_scaler.inverse_transform(val_targets)
    pred_list = target_scaler.inverse_transform(pred_list_scaled)

    result_data_points = setup_preds_and_vals_for_metrics(pred_list, val_list)

    rmse_d1, mae_d1 = calculate_metrics(1, result_data_points)
    rmse_d3, mae_d3 = calculate_metrics(3, result_data_points)
    rmse_d7, mae_d7 = calculate_metrics(7, result_data_points)

    print(f"--- Global Metrics (Average over {len(val_list)} basins) ---")
    print(f"Day 1: RMSE = {rmse_d1:.4f} mm | MAE = {mae_d1:.4f} mm")
    print(f"Day 3: RMSE = {rmse_d3:.4f} mm | MAE = {mae_d3:.4f} mm")
    print(f"Day 7: RMSE = {rmse_d7:.4f} mm | MAE = {mae_d7:.4f} mm")

    # Save Results
    results_dir = Path("../prediction_results/tft/")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_results_df = pd.DataFrame([
        {'target': 'prec_1day_ahead', 'model': MODEL_NAME, 'RMSE': rmse_d1, 'MAE': mae_d1},
        {'target': 'prec_3day_ahead', 'model': MODEL_NAME, 'RMSE': rmse_d3, 'MAE': mae_d3},
        {'target': 'prec_7day_ahead', 'model': MODEL_NAME, 'RMSE': rmse_d7, 'MAE': mae_d7},
    ])
    output_filename = f'tft_global_metrics_per_day.csv'
    metrics_results_df.to_csv(results_dir / output_filename, index=False)
    print(f"Metrics saved to {results_dir / output_filename}")

    # Per-Basin Statistics
    basin_rmses = []
    for j in range(len(pred_list)):
        err = rmse(val_list[j], pred_list[j])
        basin_rmses.append(err)

    basin_rmses = np.array(basin_rmses)

    print("\n--- Per-Basin Statistics ---")
    print(f"Best Basin RMSE:  {np.min(basin_rmses):.4f}")
    print(f"Worst Basin RMSE: {np.max(basin_rmses):.4f}")
    print(f"Mean Basin RMSE:  {np.mean(basin_rmses):.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.hist(basin_rmses, bins=20, color='lightgreen', edgecolor='black')
    plt.title(f'Distribution of RMSE across Basins ({MODEL_NAME})')
    plt.xlabel('RMSE (mm)')
    plt.ylabel('Count of Basins')
    plt.savefig(results_dir / 'tft_plot_per_basin_rmse.png')
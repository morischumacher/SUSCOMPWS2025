import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse
from darts.models import BlockRNNModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

ATTR_PATH = '../data_for_prediction/1_attributes/Catchment_attributes.csv'
MODEL_NAME = 'BlockRNN_LSTM'
TARGET_VAR = 'prec'
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


def calculate_permutation_importance(model, input_series, input_past_covariates, val_targets, verbose=True):
    """
    Calculates permutation importance for a Darts model.

    Args:
        model: Trained Darts model.
        input_series: List of Series used as input for prediction (e.g., train_targets).
        input_past_covariates: List of Series used as covariates (e.g., train_covs).
        val_targets: List of Series acting as Ground Truth (scaled).
    """

    # 1. Calculate Baseline Error (RMSE)
    if verbose: print("Calculating baseline performance...")
    baseline_preds = model.predict(n=7, series=input_series, past_covariates=input_past_covariates)

    # Compute global baseline RMSE (average over all basins)
    baseline_errors = [rmse(val, pred) for val, pred in zip(val_targets, baseline_preds)]
    baseline_score = np.mean(baseline_errors)

    importances = {}

    # ==========================================
    # 2. Permute DYNAMIC Covariates (Time Dependent)
    # ==========================================
    # Get list of dynamic feature names from the first series
    dynamic_features = input_past_covariates[0].components

    for feature in dynamic_features:
        if verbose: print(f"Testing importance of dynamic feature: {feature}")

        # Create a deep copy of covariates to modify
        shuffled_covs = [ts.copy() for ts in input_past_covariates]

        # Shuffle this feature within each time series (breaks temporal structure)
        for ts in shuffled_covs:
            # Extract the array for this feature
            feature_values = ts[feature].values()
            # Shuffle in place
            np.random.shuffle(feature_values)
            # Update the TimeSeries with shuffled values
            # (We use pandas mapping to handle Darts immutability easily)
            df = ts.pd_dataframe()
            df[feature] = feature_values
            ts_new = TimeSeries.from_dataframe(df)
            # Darts objects are tricky to mutate, simpler to overwrite list item if using index
            # ideally we reconstruct, but for this snippet we assume standard TimeSeries
            ts._series = df  # Hacky way to update, better to reconstruct:

        # Reconstruct properly to be safe
        shuffled_covs_safe = []
        for ts in input_past_covariates:
            df = ts.pd_dataframe().copy()
            df[feature] = np.random.permutation(df[feature].values)
            # Preserve static covariates if they exist on covariates (usually on target for BlockRNN)
            new_ts = TimeSeries.from_dataframe(df)
            shuffled_covs_safe.append(new_ts)

        # Predict with shuffled feature
        pred = model.predict(n=7, series=input_series, past_covariates=shuffled_covs_safe, verbose=False)

        # Calculate score
        errors = [rmse(val, p) for val, p in zip(val_targets, pred)]
        new_score = np.mean(errors)

        importances[feature] = new_score - baseline_score

    # ==========================================
    # 3. Permute STATIC Covariates (Attributes)
    # ==========================================
    # In Darts BlockRNN, static covariates are attached to the 'series' (target) input
    if input_series[0].static_covariates is not None:
        static_features = input_series[0].static_covariates.columns

        for feature in static_features:
            if verbose: print(f"Testing importance of static feature: {feature}")

            # For static features, we shuffle ACROSS basins (swap basin A's area with Basin B's area)
            # 1. Collect all values for this feature across the list
            all_values = [ts.static_covariates[feature].item() for ts in input_series]
            np.random.shuffle(all_values)

            # 2. Create new series list with shuffled static attributes
            shuffled_input_series = []
            for i, ts in enumerate(input_series):
                # Copy existing static covs
                new_static_df = ts.static_covariates.copy()
                # Assign the shuffled value
                new_static_df[feature] = all_values[i]

                # Create new TS with updated static covs
                new_ts = ts.with_static_covariates(new_static_df)
                shuffled_input_series.append(new_ts)

            # Predict
            pred = model.predict(n=7, series=shuffled_input_series, past_covariates=input_past_covariates,
                                 verbose=False)

            # Calculate score
            errors = [rmse(val, p) for val, p in zip(val_targets, pred)]
            new_score = np.mean(errors)

            importances[feature] = new_score - baseline_score

    return importances


def plot_importance(importances):
    # Sort importances
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_feats]
    values = [x[1] for x in sorted_feats]

    plt.figure(figsize=(10, 6))
    plt.barh(names, values, color='skyblue')
    plt.xlabel('Increase in RMSE (Scaled Units)')
    plt.title('Feature Importance (Permutation Method)')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.show()


if __name__ == '__main__':
    attr_df = pd.read_csv(ATTR_PATH, sep=';', index_col='ID')
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
    for csv_path in files:
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
        hidden_dim=43,
        n_rnn_layers=1,
        n_epochs=15,  # Increased epochs slightly for more complex data
        dropout=0.0019931860519986967,
        batch_size=64,
        random_state=42,
        loss_fn=nn.L1Loss(),  # better for precipitation because it is less sensitive to extreme outliers
        # and discourages the model from "blurring" predictions (predicting constant drizzle to minimize squared error)
        optimizer_kwargs={'lr': 0.0004290498767711267}
    )

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

    # ==========================================
    # Compute Metrics
    # ==========================================
    # since we are computing metrics for single days rmse function of darts equals to mae function
    # When you run rmse(val_list, preds_day_1, series_reduction=np.mean), Darts does this:
    # - Calculates RMSE for Basin A (Length 1) -> Returns |Error_A|
    # - Calculates RMSE for Basin B (Length 1) -> Returns |Error_B|
    # - ...
    # - Then it takes the average of these results.
    # so we manually align datapoints by timestamp and then use np arrays for sklearn functions
    result_data_points = setup_preds_and_vals_for_metrics(pred_list, val_list)

    rmse_d1, mae_d1 = calculate_metrics(1, result_data_points)
    rmse_d3, mae_d3 = calculate_metrics(3, result_data_points)
    rmse_d7, mae_d7 = calculate_metrics(7, result_data_points)

    print(f"--- Global Metrics (Average over {len(val_list)} basins) ---")
    print(f"Day 1: RMSE = {rmse_d1:.4f} mm | MAE = {mae_d1:.4f} mm")
    print(f"Day 3: RMSE = {rmse_d3:.4f} mm | MAE = {mae_d3:.4f} mm")
    print(f"Day 7: RMSE = {rmse_d7:.4f} mm | MAE = {mae_d7:.4f} mm")

    results_dir = Path("../prediction_results/lstm/")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_results_df = pd.DataFrame([
        {'target': 'prec_1day_ahead', 'model': MODEL_NAME, 'RMSE': rmse_d1, 'MAE': mae_d1},
        {'target': 'prec_3day_ahead', 'model': MODEL_NAME, 'RMSE': rmse_d3, 'MAE': mae_d3},
        {'target': 'prec_7day_ahead', 'model': MODEL_NAME, 'RMSE': rmse_d7, 'MAE': mae_d7},
    ])
    output_filename = f'nn_lstm_global_best_metrics_per_day.csv'
    metrics_results_df.to_csv(results_dir / output_filename, index=False)
    print(f"Metrics saved to {results_dir / output_filename}")

    basin_rmses = []

    # Loop through the results we generated above
    for j in range(len(pred_list)):
        # Calculate RMSE for just this one basin
        # Darts is smart: When pred_list[i] is compared against the val_list[i],
        # it automatically finds the matching timestamp in the validation data for each point.
        err = rmse(val_list[j], pred_list[j])
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
    output_filename = f'nn_lstm_global_best_metrics_per_basin.csv'
    metrics_results_df.to_csv(results_dir / output_filename, index=False)

    # Visualizing the distribution of errors
    plt.figure(figsize=(10, 5))
    plt.hist(basin_rmses, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of RMSE across Basins')
    plt.xlabel('RMSE (mm)')
    plt.ylabel('Count of Basins')
    plt.savefig(results_dir / 'nn_lstm_best_plot_per_basin_rmse.png')
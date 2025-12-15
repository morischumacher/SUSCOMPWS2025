import glob
import logging
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch.nn as nn
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse
from darts.models import BlockRNNModel
from sklearn.preprocessing import MinMaxScaler

# Suppress heavy logging from Darts/PyTorch during optimization
logging.getLogger("darts").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

ATTR_PATH = '../data_for_prediction/1_attributes/Catchment_attributes.csv'
DATA_DIR = '../data_for_prediction/'
TARGET_VAR = 'prec'

# Configuration
N_TRIALS = 10  # How many hyperparameter combinations to try
TIMEOUT = 3600*15  # Maximum time in seconds, 15h

selected_attrs = ['area_calc', 'elev_mean', 'slope_mean', 'forest_fra']


def load_data_into_df(filepath):
    df = pd.read_csv(filepath, sep=',')
    df['Date'] = pd.to_datetime(df[['YYYY', 'MM', 'DD']].rename(columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}))
    df = df.set_index('Date')
    df.drop(columns=['YYYY', 'MM', 'DD', 'DOY'], inplace=True)
    return df


def separate_target_variable(df):
    target_series = TimeSeries.from_series(df[TARGET_VAR], freq='D')
    covariates_df = df.drop(columns=[TARGET_VAR])
    covariates_series = TimeSeries.from_dataframe(covariates_df, freq='D')
    return covariates_series, target_series


def load_and_preprocess_data():
    """
    Loads all data once to be reused across optimization trials.
    """
    print("Loading and preprocessing data...")
    attr_df = pd.read_csv(ATTR_PATH, sep=';', index_col='ID')
    attr_subset = attr_df[selected_attrs].copy()

    # Scale static attributes
    scaler_static = MinMaxScaler()
    attr_scaled_df = pd.DataFrame(
        scaler_static.fit_transform(attr_subset),
        columns=attr_subset.columns,
        index=attr_subset.index
    )

    file_pattern = str(Path(DATA_DIR) / "*.csv")
    files = glob.glob(file_pattern)

    all_cov_ts = []
    all_target_ts = []

    valid_files_count = 0
    for csv_path in files:
        filename = os.path.basename(csv_path)
        basin_id = int(os.path.splitext(filename)[0].removeprefix('ID_'))

        try:
            static_row = attr_scaled_df.loc[basin_id]
            static_cov_df = pd.DataFrame([static_row.values], columns=static_row.index)
        except KeyError:
            continue

        cov_ts, target_ts = separate_target_variable(load_data_into_df(csv_path))

        # Add static covariates
        target_ts = target_ts.with_static_covariates(static_cov_df)

        all_cov_ts.append(cov_ts)
        all_target_ts.append(target_ts)
        valid_files_count += 1

    print(f"Loaded {valid_files_count} basins.")

    # Scale time series data
    target_scaler = Scaler()
    cov_scaler = Scaler()

    # Fit scaler on all data, transform all data
    all_target_scaled = target_scaler.fit_transform(all_target_ts)
    all_cov_scaled = cov_scaler.fit_transform(all_cov_ts)

    # Split into Train/Val
    # We pre-split here so we don't do it inside every trial
    train_targets = [t.split_before(0.8)[0] for t in all_target_scaled]
    val_targets = [t.split_before(0.8)[1] for t in all_target_scaled]

    train_covs = [c.split_before(0.8)[0] for c in all_cov_scaled]
    val_covs = [c.split_before(0.8)[1] for c in all_cov_scaled]

    return train_targets, val_targets, train_covs, val_covs, target_scaler


def objective(trial, train_targets, val_targets, train_covs, val_covs):
    # Show progress
    print(f"\n--- Starting Trial {trial.number + 1}/{N_TRIALS} ---")

    # 1. Define Hyperparameter Search Space
    hidden_dim = trial.suggest_int("hidden_dim", 16, 64)
    n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    print(f"hidden layers: {hidden_dim}, n_rnn_layers: {n_rnn_layers}, dropout: {dropout}, lr: {lr}")

    # 2. Setup Model with explicit pathing to avoid "Expected a parent" errors
    # We use a unique name for each trial so logs don't collide
    model_name = f"trial_{trial.number}"
    work_dir = "darts_logs"

    # Instantiate Model
    model = BlockRNNModel(
        input_chunk_length=30,
        output_chunk_length=7,
        model='LSTM',
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        batch_size=64,
        n_epochs=10,
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={
            "enable_progress_bar": False,  # Keep output clean
            "enable_model_summary": False
        },
        loss_fn=nn.L1Loss(),
        random_state=42,
        save_checkpoints=False,  # Crucial: Disable auto-saving to prevent path errors
        work_dir=work_dir,
        model_name=model_name,
        force_reset=True
    )

    # 3. Train
    try:
        model.fit(
            series=train_targets,
            past_covariates=train_covs,
            val_series=val_targets,
            val_past_covariates=val_covs,
        )
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return infinity so Optuna learns this set of params is bad
        return float('inf')

    # 4. Cleanup
    # Delete the trial folder to save disk space
    try:
        shutil.rmtree(Path(work_dir) / model_name, ignore_errors=True)
    except:
        pass

    # 5. Evaluate
    preds = model.predict(n=7, series=train_targets, past_covariates=train_covs)
    error = rmse(val_targets, preds, series_reduction=np.mean)
    
    print(f"--- Trial {trial.number + 1}/{N_TRIALS} finished with score (RMSE): {error:.4f} ---")

    return error


if __name__ == '__main__':
    # 1. Load Data Once
    train_y, val_y, train_x, val_x, scaler = load_and_preprocess_data()

    # 2. Setup Optuna Study
    print("Starting Hyperparameter Optimization...")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

    # We use a lambda to pass our data into the objective function
    study.optimize(
        lambda trial: objective(trial, train_y, val_y, train_x, val_x),
        n_trials=N_TRIALS,
        timeout=TIMEOUT
    )

    # 3. Print and save Results
    results_dir = Path("../prediction_results/lstm")
    results_file = "lstm_optimization_results.txt"
    with open(results_dir / results_file, "w") as f:
        try:
            f.write(f"Best Trial Score (RMSE): {study.best_value:.4f}\n")
            f.write("Best Parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
        except ValueError:
            f.write("No successful trials completed.\n")

    print(f"Optimization results saved to {results_file}")
    # 4. (Optional) Retrain Best Model on full epochs
    # You can now take study.best_params and feed them into your original script logic
    # with higher epochs (e.g., 20 or 50) for the final production model.
import glob
import os
from pathlib import Path

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse, mae
from darts.models import BlockRNNModel
import torch.nn as nn
from numpy.f2py.crackfortran import verbose

MODEL_NAME = 'BlockRNN_LSTM'

TARGET_VAR = 'prec'

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

def scale_and_split(covariates_series, target_series):
    target_scaler = Scaler()
    cov_scaler = Scaler()
    # Fit and transform
    target_scaled = target_scaler.fit_transform(target_series)
    cov_scaled = cov_scaler.fit_transform(covariates_series)
    # Split into train/validation (80/20 split)
    train_target, val_target = target_scaled.split_before(0.8)
    train_cov, val_cov = cov_scaled.split_before(0.8)
    return train_cov, val_cov, train_target, val_target, target_scaler

def append_predictions_to_results(prec_1day_ahead, prec_3day_ahead, prec_7day_ahead, location_id):
    prediction_results.append(
        {
            'location': location_id,
            'target': 'prec_1day_ahead',
            'model': MODEL_NAME,
            'prediction': prec_1day_ahead
        })
    prediction_results.append(
        {
            'location': location_id,
            'target': 'prec_3day_ahead',
            'model': MODEL_NAME,
            'prediction': prec_3day_ahead
        })
    prediction_results.append(
        {
            'location': location_id,
            'target': 'prec_7day_ahead',
            'model': MODEL_NAME,
            'prediction': prec_7day_ahead
        })


def append_metrics_to_results(rmse_day1, mae_day1, rmse_day3, mae_day3, rmse_day7, mae_day7, location_id):
    metric_results.append(
        {
            'location': location_id,
            'target': 'prec_1day_ahead',
            'model': MODEL_NAME,
            'RMSE': rmse_day1,
            'MAE': mae_day1,
        }
    )
    metric_results.append(
        {
            'location': location_id,
            'target': 'prec_3day_ahead',
            'model': MODEL_NAME,
            'RMSE': rmse_day3,
            'MAE': mae_day3,
        }
    )
    metric_results.append(
        {
            'location': location_id,
            'target': 'prec_7day_ahead',
            'model': MODEL_NAME,
            'RMSE': rmse_day7,
            'MAE': mae_day7,
        }
    )


if __name__ == '__main__':
    data_dir = "../data_for_prediction/"
    file_pattern = str(Path(data_dir) / "*.csv")
    files = glob.glob(file_pattern)
    print(f"Found {len(files)} files")

    prediction_results = []
    metric_results = []
    for csv_path in files[:5]:
        covariates_timeseries, target_timeseries = separate_target_variable(load_data_into_df(csv_path))
        train_ts_cov, val_ts_cov, train_ts_target, val_ts_target, tar_scaler = scale_and_split(covariates_timeseries, target_timeseries)
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

        filename = os.path.basename(os.path.normpath(csv_path))
        location = os.path.splitext(filename)[0]

        print(f"Fitting multivariate model for file {location}...")
        model.fit(
            series=train_ts_target,
            past_covariates=train_ts_cov,
            val_series=val_ts_target,
            val_past_covariates=val_ts_cov,
            verbose=False
        )
        pred_scaled = model.predict(
            n=7,
            series=train_ts_target,
            past_covariates=train_ts_cov
        )

        # Inverse transform to get real units (mm)
        pred = tar_scaler.inverse_transform(pred_scaled)

        day_1 = pred[0].values()[0][0]
        day_3 = pred[2].values()[0][0]
        day_7 = pred[6].values()[0][0]

        append_predictions_to_results(day_1, day_3, day_7, location)

        # 1. Day 1 Metrics
        rmse_d1 = rmse(target_timeseries, pred[0])
        mae_d1  = mae(target_timeseries, pred[0])

        # 2. Day 3 Metrics
        rmse_d3 = rmse(target_timeseries, pred[2])
        mae_d3  = mae(target_timeseries, pred[2])

        # 3. Day 7 Metrics
        rmse_d7 = rmse(target_timeseries, pred[6])
        mae_d7  = mae(target_timeseries, pred[6])

        append_metrics_to_results(rmse_d1, mae_d1, rmse_d3, mae_d3, rmse_d7, mae_d7, location)

    results_dir = Path("../prediction_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # save predictions to csv
    results_df = pd.DataFrame(prediction_results)
    output_filename = f'nn_lstm_local_predictions.csv'
    results_df.to_csv(results_dir / output_filename, index=False)
    print(f"Predictions saved to {results_dir / output_filename}")

    # save metrics to csv
    metrics_df = pd.DataFrame(metric_results)
    output_filename = f'nn_lstm_local_metrics.csv'
    metrics_df.to_csv(results_dir / output_filename, index=False)
    print(f"Metrics saved to {results_dir / output_filename}")



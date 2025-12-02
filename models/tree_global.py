import glob
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed


# ============================================================
# 1. CONFIG
# ============================================================

DATA_DIR = "../data_for_prediction/"
RESULTS_DIR = Path("../prediction_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["prec_1d_ahead", "prec_3d_ahead", "prec_7d_ahead"]
LAGS = [1, 2, 3, 5, 7, 14, 30]

MODEL_NAME = "tree_global_parallel"

files = sorted(glob.glob(str(Path(DATA_DIR) / "*.csv")))
print(f"Found {len(files)} station files.")


# ============================================================
# Helper: create lag features
# ============================================================

def add_lag_features(df, lag_list):
    df = df.copy()
    for lag in lag_list:
        df[f"prec_lag_{lag}"] = df["prec"].shift(lag)
    return df


# ============================================================
# 2. Function to process ONE station (parallelised)
# ============================================================

def process_station(path):
    """Load station, train 3 models (1d, 3d, 7d), return predictions & metrics."""
    location_id = Path(path).stem

    # Load and prepare
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df[["YYYY", "MM", "DD"]].rename(columns={
            "YYYY": "year",
            "MM": "month",
            "DD": "day"
        })
    )

    df = df.drop(columns=["YYYY", "MM", "DD", "DOY"])
    df = add_lag_features(df, LAGS)
    df = df.dropna().reset_index(drop=True)

    # Train/val split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    # Feature columns
    lag_cols = [c for c in df.columns if c.startswith("prec_lag_")]
    other_features = [c for c in df.columns
                      if c not in TARGETS + ["prec", "date"] + lag_cols]

    feature_cols = lag_cols + other_features

    station_predictions = []
    station_metrics = []

    # Train model for each target
    for target_col in TARGETS:

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        # Fit tree
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # Predictions (per-row)
        pred_df = pd.DataFrame({
            "location": location_id,
            "date": val_df["date"],
            "target": target_col,
            "model": MODEL_NAME,
            "y_true": y_val.values,
            "y_pred": y_pred
        })
        station_predictions.append(pred_df)

        # Metrics (one row)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)

        station_metrics.append(pd.DataFrame({
            "location": [location_id],
            "target": [target_col],
            "model": [MODEL_NAME],
            "RMSE": [rmse],
            "MAE": [mae],
            "n": [len(y_val)]
        }))

    # Return combined results for this station
    return (
        pd.concat(station_predictions, ignore_index=True),
        pd.concat(station_metrics, ignore_index=True)
    )


# ============================================================
# 3. RUN ALL STATIONS IN PARALLEL
# ============================================================

print("Running stations in parallel...")

results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
    delayed(process_station)(path) for path in files
)

print("Parallel processing finished.")


# ============================================================
# 4. MERGE ALL RESULTS + SAVE
# ============================================================

all_predictions = pd.concat([r[0] for r in results], ignore_index=True)
all_metrics = pd.concat([r[1] for r in results], ignore_index=True)

pred_file = RESULTS_DIR / f"predictions_{MODEL_NAME}.csv"
metrics_file = RESULTS_DIR / f"metrics_{MODEL_NAME}.csv"

all_predictions.to_csv(pred_file, index=False)
all_metrics.to_csv(metrics_file, index=False)

print("------------------------------------------------")
print("Saved predictions to:", pred_file)
print("Saved metrics to:", metrics_file)
print("Prediction rows:", len(all_predictions))
print("Metric rows:", len(all_metrics))

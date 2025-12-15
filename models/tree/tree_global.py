import glob
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "../data_for_prediction/"
ATTR_PATH = "../data_for_prediction/1_attributes/Catchment_attributes.csv"

RESULTS_DIR = Path("prediction_results/tree_global/")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Forecast horizons
TARGETS = ["prec_1d_ahead", "prec_3d_ahead", "prec_7d_ahead"]
LAGS = [1, 2, 3, 5, 7, 14, 30]

selected_attrs = ['area_calc', 'elev_mean', 'slope_mean', 'forest_fra']

# ============================================================
# HELPERS
# ============================================================

def add_lag_features(df, lag_list):
    df = df.copy()
    for lag in lag_list:
        df[f"prec_lag_{lag}"] = df["prec"].shift(lag)
    return df


def load_and_prepare_station(path, static_attributes):
    """
    Loads one CSV, creates lags, attaches static attributes,
    and returns a dataframe with all features.
    """
    location_id = int(Path(path).stem.replace("ID_", ""))

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df[["YYYY", "MM", "DD"]].rename(columns={
            "YYYY": "year",
            "MM": "month",
            "DD": "day"
        })
    )

    df = df.drop(columns=["YYYY", "MM", "DD", "DOY"])

    # Add lag features
    df = add_lag_features(df, LAGS)

    # Drop NaN (lag creation)
    df = df.dropna().reset_index(drop=True)

    # Add static attributes to each row
    for col in selected_attrs:
        df[col] = static_attributes.loc[location_id, col]

    # Return with location included
    df["location"] = location_id
    return df


# ============================================================
# MAIN GLOBAL PROCESSING
# ============================================================

if __name__ == "__main__":

    # Load & scale static attributes
    attr_df = pd.read_csv(ATTR_PATH, sep=";", index_col="ID")[selected_attrs]

    static_scaler = MinMaxScaler()
    attr_scaled = pd.DataFrame(
        static_scaler.fit_transform(attr_df),
        columns=attr_df.columns,
        index=attr_df.index
    )

    # Load all station files
    files = sorted(glob.glob(str(Path(DATA_DIR) / "*.csv")))
    print(f"Found {len(files)} basin files")

    # ============================================================
    # Build one big GLOBAL DATAFRAME
    # ============================================================

    all_data = []

    for path in files:
        try:
            df_station = load_and_prepare_station(path, static_attributes=attr_scaled)
            all_data.append(df_station)
        except KeyError:
            print(f"Skipping: {path} (missing static attributes)")
            continue

    global_df = pd.concat(all_data, ignore_index=True)
    print("Global dataset size:", global_df.shape)

    # Train/val split on GLOBAL time axis
    split_idx = int(len(global_df) * 0.8)
    train_df = global_df.iloc[:split_idx]
    val_df = global_df.iloc[split_idx:]

    # Feature columns
    lag_cols = [c for c in global_df.columns if c.startswith("prec_lag_")]
    dynamic_features = [
        c for c in global_df.columns
        if c not in TARGETS + ["prec", "date", "location"] + lag_cols + selected_attrs
    ]
    feature_cols = lag_cols + dynamic_features + selected_attrs

    # To store results
    all_predictions = []
    all_metrics = []

    # ============================================================
    # TRAIN A GLOBAL MODEL FOR EACH HORIZON
    # ============================================================

    for target in TARGETS:
        print(f"\nTraining global model for: {target}")

        X_train = train_df[feature_cols]
        y_train = train_df[target]

        X_val = val_df[feature_cols]
        y_val = val_df[target]

        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # Save predictions
        pred_df = pd.DataFrame({
            "date": val_df["date"],
            "location": val_df["location"],
            "target": target,
            "model": "tree_global",
            "y_true": y_val.values,
            "y_pred": y_pred
        })
        all_predictions.append(pred_df)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)

        print(f"{target}: RMSE={rmse:.4f}, MAE={mae:.4f}")

        all_metrics.append(pd.DataFrame({
            "target": [target],
            "model": ["tree_global"],
            "RMSE": [rmse],
            "MAE": [mae],
            "n": [len(y_val)]
        }))

    # ============================================================
    # SAVE OUTPUT
    # ============================================================

    all_predictions = pd.concat(all_predictions, ignore_index=True)
    all_metrics = pd.concat(all_metrics, ignore_index=True)

    all_predictions.to_csv(RESULTS_DIR / "predictions_tree_global.csv", index=False)
    all_metrics.to_csv(RESULTS_DIR / "metrics_tree_global.csv", index=False)

    print("\nSaved global predictions and metrics.")
    print("Prediction rows:", len(all_predictions))
    print("Metric rows:", len(all_metrics))

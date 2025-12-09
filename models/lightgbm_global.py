import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data_for_prediction/"
ATTR_PATH = "data_for_prediction/1_attributes/Catchment_attributes.csv"

RESULTS_DIR = Path("prediction_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Forecast horizons
TARGETS = ["prec_1d_ahead", "prec_3d_ahead", "prec_7d_ahead"]
LAGS = [1, 2, 3, 5, 7, 14, 30]

# Required static attributes (as agreed in the team)
SELECTED_ATTRS = ["area_calc", "elev_mean", "slope_mean", "forest_fra"]


# ============================================================
# HELPERS
# ============================================================

def add_lag_features(df: pd.DataFrame, lag_list):
    """
    Add precipitation lag features (prec_lag_k for each k in lag_list).
    """
    df = df.copy()
    for lag in lag_list:
        df[f"prec_lag_{lag}"] = df["prec"].shift(lag)
    return df


def load_and_prepare_station(path: str, static_attributes: pd.DataFrame) -> pd.DataFrame:
    """
    Load one basin file, create lags, attach static attributes, and return
    a prepared dataframe including location ID.
    """
    # Extract numeric ID from filename like "ID_123.csv"
    location_id = int(Path(path).stem.replace("ID_", ""))

    df = pd.read_csv(path)

    # Build proper datetime
    df["date"] = pd.to_datetime(
        df[["YYYY", "MM", "DD"]].rename(columns={
            "YYYY": "year",
            "MM": "month",
            "DD": "day"
        })
    )

    # Drop redundant time columns
    df = df.drop(columns=["YYYY", "MM", "DD", "DOY"])

    # Add lag features based on 'prec'
    df = add_lag_features(df, LAGS)

    # Remove rows with NaNs (created by lagging)
    df = df.dropna().reset_index(drop=True)

    # Attach scaled static attributes to each row
    for col in SELECTED_ATTRS:
        df[col] = static_attributes.loc[location_id, col]

    # Keep location ID as a column (for analysis, not used as feature)
    df["location"] = location_id

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # 1. Load & scale static attributes
    # --------------------------------------------------------
    attr_df = pd.read_csv(ATTR_PATH, sep=";", index_col="ID")[SELECTED_ATTRS]

    static_scaler = MinMaxScaler()
    attr_scaled = pd.DataFrame(
        static_scaler.fit_transform(attr_df),
        columns=attr_df.columns,
        index=attr_df.index,
    )

    # --------------------------------------------------------
    # 2. Load all stations & build one global dataframe
    # --------------------------------------------------------
    files = sorted(glob.glob(str(Path(DATA_DIR) / "ID_*.csv")))
    print(f"Found {len(files)} basin files")

    all_data = []

    for path in files:
        try:
            station_df = load_and_prepare_station(path, static_attributes=attr_scaled)
            all_data.append(station_df)
        except KeyError:
            # Missing static attributes for this ID
            print(f"Skipping {path}: missing static attributes for this basin.")
            continue

    if not all_data:
        raise RuntimeError("No valid basins loaded. Check attribute IDs and file names.")

    global_df = pd.concat(all_data, ignore_index=True)
    print("Global dataset size:", global_df.shape)

    # --------------------------------------------------------
    # 3. Train/validation split (same style as tree_global)
    # --------------------------------------------------------
    split_idx = int(len(global_df) * 0.8)
    train_df = global_df.iloc[:split_idx]
    val_df = global_df.iloc[split_idx:]

    # --------------------------------------------------------
    # 4. Define features
    # --------------------------------------------------------
    lag_cols = [c for c in global_df.columns if c.startswith("prec_lag_")]

    dynamic_features = [
        c for c in global_df.columns
        if c not in TARGETS
        and c not in ["prec", "date", "location"]
        and c not in lag_cols
        and c not in SELECTED_ATTRS
    ]

    feature_cols = lag_cols + dynamic_features + SELECTED_ATTRS
    print("Number of features:", len(feature_cols))

    all_predictions = []
    all_metrics = []

    # --------------------------------------------------------
    # 5. Train one global LightGBM model per horizon
    # --------------------------------------------------------
    for target in TARGETS:
        print(f"\nTraining LightGBM global model for: {target}")

        X_train = train_df[feature_cols]
        y_train = train_df[target]

        X_val = val_df[feature_cols]
        y_val = val_df[target]

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # --- Save predictions ---
        pred_df = pd.DataFrame({
            "date": val_df["date"],
            "location": val_df["location"],
            "target": target,
            "model": "lightgbm_global",
            "y_true": y_val.values,
            "y_pred": y_pred,
        })
        all_predictions.append(pred_df)

        # --- Compute metrics ---
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)

        print(f"{target}: RMSE={rmse:.4f}, MAE={mae:.4f}")

        metric_df = pd.DataFrame({
            "target": [target],
            "model": ["lightgbm_global"],
            "RMSE": [rmse],
            "MAE": [mae],
            "n": [len(y_val)],
        })
        all_metrics.append(metric_df)

    # --------------------------------------------------------
    # 6. Save outputs
    # --------------------------------------------------------
    all_predictions = pd.concat(all_predictions, ignore_index=True)
    all_metrics = pd.concat(all_metrics, ignore_index=True)

    preds_path = RESULTS_DIR / "predictions_lightgbm_global.csv"
    metrics_path = RESULTS_DIR / "metrics_lightgbm_global.csv"

    all_predictions.to_csv(preds_path, index=False)
    all_metrics.to_csv(metrics_path, index=False)

    print("\nSaved LightGBM global predictions and metrics.")
    print("Predictions file:", preds_path)
    print("Metrics file:", metrics_path)
    print("Prediction rows:", len(all_predictions))
    print("Metric rows:", len(all_metrics))

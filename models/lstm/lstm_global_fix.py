import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
import logging
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import BlockRNNModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# =========================
# Config
# =========================
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*no accelerator is found.*"
)

# Silence pin_memory warning
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*no accelerator is found.*"
)

# Silence Lightning device info spam
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators").setLevel(logging.ERROR)

# Optional: silence Lightning progress bars & info
os.environ["PL_DISABLE_FORK"] = "1"
os.environ["PL_LOGGING"] = "0"

BASE_DIR = Path(__file__).resolve().parent
ATTR_PATH = BASE_DIR / ".." / ".." / "data_for_prediction" / "1_attributes" / "Catchment_attributes.csv"
DATA_DIR = (BASE_DIR / ".." / ".." / "data_for_prediction").resolve()
RESULTS_DIR=(BASE_DIR / ".." / ".." / "prediction_results").resolve()

MODEL_NAME = "BlockRNN_LSTM"
TARGET_VAR = "prec"
HORIZON = 7
INPUT_LEN = 30

# Static attributes
SELECTED_ATTRS = ["area_calc", "elev_mean", "slope_mean", "forest_fra"]

# Columns we never want the model to see
LEAK_COLS = [
    "prec_1d_ahead",
    "prec_3d_ahead",
    "prec_7d_ahead",
]

# =========================
# Helpers
# =========================
def load_basin_df(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=",")
    df["Date"] = pd.to_datetime(
        df[["YYYY", "MM", "DD"]].rename(columns={"YYYY": "year", "MM": "month", "DD": "day"})
    )
    df = df.set_index("Date")
    df = df.drop(columns=["YYYY", "MM", "DD", "DOY"], errors="ignore")
    return df


def make_series(df: pd.DataFrame):
    """
    Target = prec.
    Past covariates = all non-leaky, non-target dynamic columns.
    """
    # drop explicit future-label columns if present
    df = df.drop(columns=[c for c in LEAK_COLS if c in df.columns], errors="ignore")

    # target
    target = TimeSeries.from_series(df[TARGET_VAR], freq="D")

    # covariates: everything except target (and except leak cols already dropped)
    cov_df = df.drop(columns=[TARGET_VAR], errors="ignore")
    covs = TimeSeries.from_dataframe(cov_df, freq="D") if len(cov_df.columns) else None

    return covs, target


def attach_static(target: TimeSeries, static_row: pd.Series) -> TimeSeries:
    static_df = pd.DataFrame([static_row.values], columns=static_row.index)
    return target.with_static_covariates(static_df)


def split_train_val(ts: TimeSeries, frac=0.8):
    train, val = ts.split_before(frac)
    return train, val


def rmse_mm(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae_mm(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def evaluate_one_forecast(pred_ts: TimeSeries, true_ts: TimeSeries):
    """
    pred_ts and true_ts must both be exactly length HORIZON, in chronological order.
    No time-index intersection; no slicing; no surprises.
    """
    pred_vals = pred_ts.values().flatten()
    true_vals = true_ts.values().flatten()

    assert len(pred_vals) == HORIZON, f"pred length {len(pred_vals)} != {HORIZON}"
    assert len(true_vals) == HORIZON, f"true length {len(true_vals)} != {HORIZON}"

    idx_map = {1: 0, 3: 2, 7: 6}
    out = {}
    for day, i in idx_map.items():
        out[day] = {
            "rmse": rmse_mm([true_vals[i]], [pred_vals[i]]),
            "mae": mae_mm([true_vals[i]], [pred_vals[i]]),
            "true": float(true_vals[i]),
            "pred": float(pred_vals[i]),
        }

    out["full_rmse"] = rmse_mm(true_vals, pred_vals)
    return out



def pooled_metrics(day_points):
    y_true = np.array([p["true"] for p in day_points if np.isfinite(p["true"])])
    y_pred = np.array([p["pred"] for p in day_points if np.isfinite(p["pred"])])
    if len(y_true) == 0:
        return np.nan, np.nan
    return rmse_mm(y_true, y_pred), mae_mm(y_true, y_pred)


def pooled_metrics_conditional(day_points, threshold_mm=1.0):
    """
    Metrics only on cases where true precip >= threshold_mm.
    Helps avoid the 'day7 looks great because most days are dry' illusion.
    """
    filt = [p for p in day_points if np.isfinite(p["true"]) and p["true"] >= threshold_mm]
    if not filt:
        return np.nan, np.nan, 0
    y_true = np.array([p["true"] for p in filt])
    y_pred = np.array([p["pred"] for p in filt])
    return rmse_mm(y_true, y_pred), mae_mm(y_true, y_pred), len(filt)


def build_baselines(train_target_unscaled: TimeSeries, horizon=7):
    """
    Two simple baselines:
    - persistence: repeat last observed value
    - climatology: repeat train mean
    """
    last_val = float(train_target_unscaled.values()[-1][0])
    mean_val = float(np.mean(train_target_unscaled.values()))

    # Build series at the correct timestamps later by reindexing; we return value vectors
    persistence = np.array([last_val] * horizon)
    climatology = np.array([mean_val] * horizon)
    return persistence, climatology


# =========================
# Permutation Importance (repeatable, in mm)
# =========================
def permutation_importance_mm(
    model: BlockRNNModel,
    train_targets_scaled,
    full_covs_scaled,          # must cover train + HORIZON
    true_horizon_unscaled,     # list of TimeSeries (unscaled) length HORIZON each basin
    target_scaler: Scaler,
    n_repeats=20,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # baseline predictions (scaled -> unscaled)
    baseline_scaled = model.predict(
        n=HORIZON,
        series=train_targets_scaled,
        past_covariates=full_covs_scaled,
        verbose=False,
    )
    baseline_unscaled = target_scaler.inverse_transform(baseline_scaled)

    baseline_errors = [
        rmse_mm(t.values().flatten(), p.values().flatten())
        for t, p in zip(true_horizon_unscaled, baseline_unscaled)
    ]
    baseline_score = float(np.mean(baseline_errors))

    importances = {}

    if full_covs_scaled is None:
        return importances, baseline_score

    feat_names = list(full_covs_scaled[0].components)

    for feat in feat_names:
        rep_scores = []
        for _ in range(n_repeats):
            shuffled_covs = []
            for ts in full_covs_scaled:
                df = ts.to_dataframe().copy()
                vals = df[feat].to_numpy()
                rng.shuffle(vals)  # in-place shuffle
                df[feat] = vals
                shuffled_covs.append(TimeSeries.from_dataframe(df, freq=ts.freq_str))

            pred_scaled = model.predict(
                n=HORIZON,
                series=train_targets_scaled,
                past_covariates=shuffled_covs,
                verbose=False,
            )
            pred_unscaled = target_scaler.inverse_transform(pred_scaled)

            errors = [
                rmse_mm(t.values().flatten(), p.values().flatten())
                for t, p in zip(true_horizon_unscaled, pred_unscaled)
            ]
            rep_scores.append(float(np.mean(errors)))

        importances[feat] = float(np.mean(rep_scores) - baseline_score)

    return importances, baseline_score


def plot_importance(importances, title="Permutation Importance (ΔRMSE in mm)"):
    if not importances:
        print("No dynamic covariates => no permutation importance to plot.")
        return
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    plt.figure(figsize=(10, 6))
    plt.barh(names, vals)
    plt.xlabel("Increase in RMSE (mm)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# =========================
# Main
# =========================
if __name__ == "__main__":
    # ---- load + scale static attributes (only for basins used)
    attr_df = pd.read_csv(ATTR_PATH, sep=";", index_col="ID")
    attr_subset = attr_df[SELECTED_ATTRS].copy()

    # Find basin files
    files = glob.glob(str(DATA_DIR / "*.csv"))
    print(f"Found {len(files)} files in {DATA_DIR}")

    # Get basin IDs present in this run
    basin_ids = []
    for p in files:
        fn = os.path.basename(p)
        try:
            basin_id = int(os.path.splitext(fn)[0].removeprefix("ID_"))
            basin_ids.append(basin_id)
        except Exception:
            continue

    basin_ids = sorted(set(basin_ids))
    attr_subset = attr_subset.loc[attr_subset.index.intersection(basin_ids)]
    print(f"Loaded attributes for {len(attr_subset)} basins (intersection with files).")

    # Fit scaler on static only for these basins (no point fitting on all 859 if not used)
    scaler_static = MinMaxScaler()
    attr_scaled_df = pd.DataFrame(
        scaler_static.fit_transform(attr_subset),
        columns=attr_subset.columns,
        index=attr_subset.index,
    )

    # ---- build series lists
    cov_list_raw = []
    target_list_raw = []

    used_files = 0
    for csv_path in files:
        fn = os.path.basename(csv_path)
        try:
            basin_id = int(os.path.splitext(fn)[0].removeprefix("ID_"))
        except Exception:
            continue

        if basin_id not in attr_scaled_df.index:
            continue

        df = load_basin_df(csv_path)
        cov_ts, target_ts = make_series(df)

        # attach static to target
        target_ts = attach_static(target_ts, attr_scaled_df.loc[basin_id])

        cov_list_raw.append(cov_ts)
        target_list_raw.append(target_ts)
        used_files += 1

    print(f"Using {used_files} basins.")

    # ---- split train/val BEFORE scaling (to prevent leakage)
    train_targets_raw, val_targets_raw = [], []
    train_covs_raw, val_covs_raw = [], []

    for cov_ts, target_ts in zip(cov_list_raw, target_list_raw):
        tr_t, va_t = split_train_val(target_ts, frac=0.8)
        train_targets_raw.append(tr_t)
        val_targets_raw.append(va_t)

        if cov_ts is None:
            train_covs_raw.append(None)
            val_covs_raw.append(None)
        else:
            tr_c, va_c = split_train_val(cov_ts, frac=0.8)
            train_covs_raw.append(tr_c)
            val_covs_raw.append(va_c)

    # ---- fit scalers on TRAIN ONLY
    target_scaler = Scaler()
    target_scaler.fit(train_targets_raw)
    train_targets = target_scaler.transform(train_targets_raw)
    val_targets = target_scaler.transform(val_targets_raw)

    # keep static covariates after scaling (darts can drop them depending on version)
    train_targets = [t.with_static_covariates(orig.static_covariates) for t, orig in zip(train_targets, train_targets_raw)]
    val_targets = [t.with_static_covariates(orig.static_covariates) for t, orig in zip(val_targets, val_targets_raw)]

    # covariate scaling (if covariates exist)
    if train_covs_raw[0] is not None:
        cov_scaler = Scaler()
        cov_scaler.fit(train_covs_raw)
        train_covs = cov_scaler.transform(train_covs_raw)
        val_covs = cov_scaler.transform(val_covs_raw)
    else:
        cov_scaler = None
        train_covs = None
        val_covs = None

    # ---- build "full covariates" covering train end + horizon
    # For past_covariates, Darts needs covariates to extend through prediction horizon.
    # ---- build "full covariates" covering ENTIRE val (so any random window can be evaluated)
    if train_covs is not None:
        full_covs = []
        for tr_c, va_c in zip(train_covs, val_covs):
            # full covs = train covs + all validation covs
            full_covs.append(tr_c.append(va_c))
    else:
        full_covs = None


    # ---- model (fix dropout by using >=2 layers)
    model = BlockRNNModel(
        input_chunk_length=INPUT_LEN,
        output_chunk_length=HORIZON,
        model="LSTM",
        hidden_dim=32,
        n_rnn_layers=2,     # enables dropout properly
        dropout=0.1,
        n_epochs=0,
        batch_size=64,
        random_state=42,
        loss_fn=nn.L1Loss(),
    )

    model.fit(
        series=train_targets,
        past_covariates=train_covs,
        val_series=val_targets,
        val_past_covariates=val_covs,
        verbose=True,
    )

    # ---- RANDOM 7-DAY EVALUATION WINDOW PER BASIN (alleviates dry dominance)
    rng = np.random.default_rng(42) # 42 the answer to everything lol

    eval_train_raw = []
    eval_train_scaled = []
    eval_full_covs = []
    true_horizon = []

    for tr_raw, va_raw, tr_scaled, va_scaled, fc in zip(
        train_targets_raw,
        val_targets_raw,
        train_targets,
        val_targets,
        full_covs
    ):
        max_start = len(va_raw) - HORIZON
        if max_start <= 0:
            continue

        start = rng.integers(0, max_start + 1)

        # history up to forecast origin
        hist_raw = tr_raw.append(va_raw[:start])
        hist_scaled = tr_scaled.append(va_scaled[:start])

        # true future (unscaled)
        truth = va_raw[start:start + HORIZON]

        # covariates must end at or after end of horizon
        horizon_end = truth.end_time()
        cov_hist = fc.slice(fc.start_time(), horizon_end)

        eval_train_raw.append(hist_raw)
        eval_train_scaled.append(hist_scaled)
        eval_full_covs.append(cov_hist)
        true_horizon.append(truth)


    print("Example series end:", eval_train_scaled[0].end_time())
    print("Example cov end:   ", eval_full_covs[0].end_time())
    print("Example truth end: ", true_horizon[0].end_time())
    # predict from random origins
    preds_scaled = model.predict(
        n=HORIZON,
        series=eval_train_scaled,
        past_covariates=eval_full_covs,
        verbose=False
    )
    preds = target_scaler.inverse_transform(preds_scaled)


    # ---- compute pooled per-horizon metrics
    day_points = {1: [], 3: [], 7: []}
    basin_full_rmses = []

    for p_ts, t_ts, tr_raw in zip(preds, true_horizon, train_targets_raw):
        res = evaluate_one_forecast(p_ts, t_ts)
        for day in (1, 3, 7):
            day_points[day].append(res[day])
        basin_full_rmses.append(res["full_rmse"])

    rmse_d1, mae_d1 = pooled_metrics(day_points[1])
    rmse_d3, mae_d3 = pooled_metrics(day_points[3])
    rmse_d7, mae_d7 = pooled_metrics(day_points[7])

    print(f"\n--- Global Metrics (Average over {len(true_horizon)} basins) ---")
    print(f"Day 1: RMSE = {rmse_d1:.4f} mm | MAE = {mae_d1:.4f} mm")
    print(f"Day 3: RMSE = {rmse_d3:.4f} mm | MAE = {mae_d3:.4f} mm")
    print(f"Day 7: RMSE = {rmse_d7:.4f} mm | MAE = {mae_d7:.4f} mm")

    print("\n--- Per-Basin (7-day vector) RMSE stats ---")
    basin_full_rmses = np.asarray(basin_full_rmses)
    print(f"Best Basin RMSE:  {np.min(basin_full_rmses):.4f} mm")
    print(f"Worst Basin RMSE: {np.max(basin_full_rmses):.4f} mm")
    print(f"Mean Basin RMSE:  {np.mean(basin_full_rmses):.4f} mm")

    # ---- conditional metrics (prec >= 1 mm)
    for day in (1, 3, 7):
        r_c, m_c, n_c = pooled_metrics_conditional(day_points[day], threshold_mm=1.0)
        print(f"Day {day} (true ≥ 1 mm): RMSE={r_c:.4f} | MAE={m_c:.4f} | n={n_c}")

    def dry_fraction(points, eps=0.1):
        y = np.array([p["true"] for p in points if np.isfinite(p["true"])])
        return float(np.mean(y <= eps)), len(y)

    print("\n--- Dry-day dominance (true ≤ 0.1 mm) ---")
    for day in (1, 3, 7):
        frac, n = dry_fraction(day_points[day], eps=0.1)
        print(f"Day {day}: fraction={frac:.3f} (n={n})")

    # ---- baselines (persistence + climatology)
    # Build and evaluate pooled baselines at day1/3/7
    base_points_pers = {1: [], 3: [], 7: []}
    base_points_clim = {1: [], 3: [], 7: []}

    for tr_raw, true7 in zip(train_targets_raw, true_horizon):
        pers_vec, clim_vec = build_baselines(tr_raw, horizon=HORIZON)
        true_vals = true7.values().flatten()

        for day, idx in {1: 0, 3: 2, 7: 6}.items():
            if idx < len(true_vals):
                base_points_pers[day].append({"true": true_vals[idx], "pred": pers_vec[idx]})
                base_points_clim[day].append({"true": true_vals[idx], "pred": clim_vec[idx]})

    def pooled_simple(points):
        y_true = np.array([p["true"] for p in points])
        y_pred = np.array([p["pred"] for p in points])
        return rmse_mm(y_true, y_pred), mae_mm(y_true, y_pred)

    print("\n--- Baselines (pooled across basins) ---")
    for day in (1, 3, 7):
        r, m = pooled_simple(base_points_pers[day])
        print(f"Persistence Day {day}: RMSE={r:.4f} | MAE={m:.4f}")
    for day in (1, 3, 7):
        r, m = pooled_simple(base_points_clim[day])
        print(f"Climatology Day {day}: RMSE={r:.4f} | MAE={m:.4f}")

    # ---- permutation importance on dynamic covariates (in mm)
    print("\n--- Permutation Importance (dynamic covariates only) ---")
    if full_covs is not None:
        imps, base = permutation_importance_mm(
            model=model,
            train_targets_scaled=train_targets,
            full_covs_scaled=full_covs,
            true_horizon_unscaled=true_horizon,
            target_scaler=target_scaler,
            n_repeats=20,
            seed=42,
        )
        print(f"Baseline (mean 7-day RMSE across basins): {base:.4f} mm")
        imp_df = pd.DataFrame(sorted(imps.items(), key=lambda x: x[1], reverse=True),
                              columns=["Feature", "Delta_RMSE_mm"])
        print(imp_df.head(20).to_string(index=False))
        imp_df.to_csv(RESULTS_DIR / "nn_lstm_global_feature_importance.csv", index=False)
        plot_importance(imps)
    else:
        print("No dynamic covariates found, skipping permutation importance.")

    # ---- save outputs
    RESULTS_DIR = (BASE_DIR / ".." / ".." / "prediction_results/lstm").resolve()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([
        {"target": "prec_1d", "model": MODEL_NAME, "RMSE": rmse_d1, "MAE": mae_d1},
        {"target": "prec_3d", "model": MODEL_NAME, "RMSE": rmse_d3, "MAE": mae_d3},
        {"target": "prec_7d", "model": MODEL_NAME, "RMSE": rmse_d7, "MAE": mae_d7},
    ])
    metrics_df.to_csv(RESULTS_DIR / "nn_lstm_global_metrics_per_day.csv", index=False)

    basin_df = pd.DataFrame([{
        "model": MODEL_NAME,
        "best_basin_RMSE_7day": float(np.min(basin_full_rmses)),
        "worst_basin_RMSE_7day": float(np.max(basin_full_rmses)),
        "mean_basin_RMSE_7day": float(np.mean(basin_full_rmses)),
    }])
    basin_df.to_csv(RESULTS_DIR / "nn_lstm_global_metrics_per_basin.csv", index=False)

    print(f"\nSaved metrics to: {RESULTS_DIR}")

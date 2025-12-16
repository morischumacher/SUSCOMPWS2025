import optuna
import numpy as np
import torch.nn as nn
from darts.models import BlockRNNModel

from lstm_global_fix import build_pipeline, evaluate_one_forecast, HORIZON

# ---------------------------------------------------------
# Build pipeline ONCE
# ---------------------------------------------------------
PIPE = build_pipeline()

# ---------------------------------------------------------
# Optuna objective (ONE scalar only!)
# ---------------------------------------------------------
def objective(trial):

    pipe = PIPE

    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, step=16)
    n_layers   = trial.suggest_int("n_rnn_layers", 1, 3)
    dropout    = trial.suggest_float("dropout", 0.0, 0.3)
    input_len  = trial.suggest_int("input_chunk_length", 14, 56, step=7)
    lr         = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = BlockRNNModel(
        model="LSTM",
        input_chunk_length=input_len,
        output_chunk_length=HORIZON,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_layers,
        dropout=dropout if n_layers > 1 else 0.0,
        batch_size=batch_size,
        optimizer_kwargs={"lr": lr},
        n_epochs=5,
        random_state=42,
        loss_fn=nn.L1Loss(),
    )

    model.fit(
        series=pipe["train_targets"],
        past_covariates=pipe["train_covs"],
        val_series=pipe["val_targets"],
        val_past_covariates=pipe["val_covs"],
        verbose=False,
    )

    preds = pipe["target_scaler"].inverse_transform(
        model.predict(
            n=HORIZON,
            series=pipe["eval_train_scaled"],
            past_covariates=pipe["eval_full_covs"],
            verbose=False,
        )
    )

    basin_full_rmses = []
    for p_ts, t_ts in zip(preds, pipe["true_horizon"]):
        res = evaluate_one_forecast(p_ts, t_ts)
        basin_full_rmses.append(res["full_rmse"])

    return float(np.mean(basin_full_rmses))



# ---------------------------------------------------------
# Run study + FINAL evaluation
# ---------------------------------------------------------
if __name__ == "__main__":

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=12, show_progress_bar=True)

    print("\n--- Optuna Results (LSTM) ---")
    print(study.best_trial.params)
    print(f"Best mean 7-day RMSE: {study.best_value:.4f} mm")

    # ---------- FINAL REPORTING ----------
    best = study.best_trial.params
    pipe = PIPE

    model = BlockRNNModel(
        model="LSTM",
        input_chunk_length=best["input_chunk_length"],
        output_chunk_length=HORIZON,
        hidden_dim=best["hidden_dim"],
        n_rnn_layers=best["n_rnn_layers"],
        dropout=best["dropout"] if best["n_rnn_layers"] > 1 else 0.0,
        batch_size=best["batch_size"],
        optimizer_kwargs={"lr": best["learning_rate"]},
        n_epochs=15,
        random_state=42,
        loss_fn=nn.L1Loss(),
    )

    model.fit(
        series=pipe["train_targets"],
        past_covariates=pipe["train_covs"],
        val_series=pipe["val_targets"],
        val_past_covariates=pipe["val_covs"],
        verbose=False,
    )

    preds = pipe["target_scaler"].inverse_transform(
        model.predict(
            n=HORIZON,
            series=pipe["eval_train_scaled"],
            past_covariates=pipe["eval_full_covs"],
            verbose=False,
        )
    )

    rmse_1d, rmse_3d, rmse_7d, rmse_full = [], [], [], []

    for p_ts, t_ts in zip(preds, pipe["true_horizon"]):
        res = evaluate_one_forecast(p_ts, t_ts)
        rmse_1d.append(res[1]["rmse"])
        rmse_3d.append(res[3]["rmse"])
        rmse_7d.append(res[7]["rmse"])
        rmse_full.append(res["full_rmse"])

    print("\n--- Global Metrics ---")
    print(f"Day 1 RMSE: {np.mean(rmse_1d):.4f} mm")
    print(f"Day 3 RMSE: {np.mean(rmse_3d):.4f} mm")
    print(f"Day 7 RMSE: {np.mean(rmse_7d):.4f} mm")
    print(f"Full 7-day RMSE: {np.mean(rmse_full):.4f} mm")

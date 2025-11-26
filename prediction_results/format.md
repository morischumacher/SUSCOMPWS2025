### Prediction and metrics format for all models

Each location is stored in a separate CSV file with daily weather data.  
Each file has columns:

- `YYYY, MM, DD, DOY` – date fields  
- Current-day variables, including `prec` (precipitation on that day)  
- Target columns (already aligned as supervised labels):  
  - `prec_1d_ahead` – precipitation 1 day ahead  
  - `prec_3d_ahead` – precipitation 3 days ahead  
  - `prec_7d_ahead` – precipitation 7 days ahead  

On top of this data we fit different models (baselines, regression models, (deep) neural networks, …).  
**All models use the same output format:**

---

#### Result CSV per model

For each model we write a *results* file to `../prediction_results/`, e.g.:

- `baseline_predictions_ma-3.csv`  
- `baseline_predictions_persistence.csv`  

(And later e.g. `nn_predictions.csv` …)

Each *results* CSV is format, one row per forecast:

- `location` – station / location ID  
- `date` – forecast origin date (day *t*)  
- `target` – which horizon is predicted  
  (`prec_1d_ahead`, `prec_3d_ahead`, `prec_7d_ahead`)  
- `model` – model name (e.g. `ma-3`, `persistence`)  
- `y_true` – true future precipitation (from the corresponding target column)  
- `y_pred` – model’s prediction for that value  

Example:

```text
location,date,target,model,y_true,y_pred
ID_177,1981-01-03,prec_1d_ahead,ma-3,5.55,10.96
ID_177,1981-01-04,prec_1d_ahead,ma-3,5.36,11.01
```

---

#### Metrics CSV per model

In addition to the per-forecast results, we compute **summary metrics** for
each combination of `location`, `target`, and `model`.

We create one *metrics* CSV per model in `../prediction_results/`, e.g.:

- `baseline_metrics_ma-3.csv`  
- `baseline_metrics_persistence.csv`  

(And later e.g. `nn_metrics.csv` …)

Each *metrics* CSV contains **one row per `(location, target, model)`** with:

- `location` – station / location ID  
- `target` – forecast horizon (`prec_1d_ahead`, `prec_3d_ahead`, `prec_7d_ahead`)  
- `model` – model name  
- `RMSE` – root mean squared error over all forecast cases for that triple  
- `MAE` – mean absolute error over all forecast cases for that triple  
- `n` – number of forecast cases used to compute `RMSE` and `MAE`  

Example (excerpt from `baseline_metrics_ma-3.csv`):

```text
location,target,model,RMSE,MAE,n
ID_10,prec_1d_ahead,ma-3,8.270982421743303,5.359646117118386,14242.0
ID_10,prec_3d_ahead,ma-3,9.02165781859148,5.991323784112722,14242.0
ID_10,prec_7d_ahead,ma-3,9.080617727619387,6.105340542058702,14242.0
ID_104,prec_1d_ahead,ma-3,7.5520532374470335,4.99521228291906,14242.0
ID_104,prec_3d_ahead,ma-3,8.117878500213244,5.505462247811632,14242.0
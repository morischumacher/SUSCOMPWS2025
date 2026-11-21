# Precipitation Forecasting for Austria

A comparison of models that forecast daily precipitation 1, 3 and 7 days ahead at 100 locations around the centre of Austria. Team project in Introduction to Computational Sustainability, TU Wien (winter term 2025/26).

## Approach

- Data cleaning and analysis first.
- Simple baselines (persistence, 3-day moving average) against gradient boosting, LightGBM, an LSTM and a Temporal Fusion Transformer.
- Every model writes its results in one shared format, so they can be compared directly ([results notebook](results.ipynb)).

From the same course: [Quantizing Qwen2-VL](https://github.com/morischumacher/Quantizing-Qwen2-VL).

## Tech stack

Python, pandas, scikit-learn, LightGBM, Darts (PyTorch), Optuna, Jupyter

### Python

Install virtualenv

```console
$ python3 -m venv venv
(can be python or py)
```
For VSCode select kernel from the venv folder

Setup virtualenv and activate it

```console
venv/Scripts/activate
pip install -r requirements.txt
```

Install strip package to not run into many merge conflict
```console
$ pip install nbstripout
nbstripout --install
```

For outputs we want to commit we can add this metadata to a cell:
```
Set the "keep_output": true metadata on the cell. To do this, select the "Edit Metadata" Cell Toolbar, and then use the "Edit Metadata" button on the desired cell to enter something like:

{
  "keep_output": true,
}
```
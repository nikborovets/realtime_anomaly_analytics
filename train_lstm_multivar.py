#!/usr/bin/env python
"""Batch training of several LSTMMultivariateForecast configs with logging & plots.

Запускайте на ночь: python train_lstm_multivar.py  (данные тянутся через fetch_frame)
"""
import datetime as dt
import io
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from ts_toolkit.metrics import global_metrics, daily_mae
from ts_toolkit.viz import plot_history_forecast

from src.data_loader import fetch_frame
from src.models import LSTMMultivariateForecast

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────
@contextmanager
def capture_to_logging(logger):
    """Redirect stdout/stderr into logger inside with-block."""
    original_stdout, original_stderr = sys.stdout, sys.stderr

    class _LoggingWriter(io.TextIOBase):
        def __init__(self, _logger, level):
            self.logger, self.level, self.buf = _logger, level, ""
        def write(self, txt):
            self.buf += txt
            lines = self.buf.split("\n")
            for line in lines[:-1]:
                if line.strip():
                    self.logger.log(self.level, line.strip())
            self.buf = lines[-1]
        def flush(self):
            if self.buf.strip():
                self.logger.log(self.level, self.buf.strip())
            self.buf = ""

    sys.stdout, sys.stderr = _LoggingWriter(logger, logging.INFO), _LoggingWriter(logger, logging.ERROR)
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        sys.stdout, sys.stderr = original_stdout, original_stderr


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ────────────────────────────────────────────────────────────────
#  Load data once
# ────────────────────────────────────────────────────────────────
print("Loading data…")
df: pd.DataFrame = fetch_frame(
    start_date="2024-11-25 18:00:00",
    end_date="2024-12-11 12:10:00",
    use_cache=True,
    cache_filename="multivar_9001_20241125_20241211.parquet",
)
# rename cols to concise names
renames = {
    'common_cad_avg1h_instance_consumer_from_ws_with_metrics:8000_job_consumer_from_ws_with_metrics_service_castle': 'common_cad_avg1h',
    'db_insert_cad_avg1h_instance_10.201.92.176:8010_job_consumer_ml_to_db_service_castle': 'db_insert_cad_avg1h',
    'kafka_network_cad_avg1h_instance_10.201.92.176:8010_job_consumer_ml_to_db_service_castle': 'kafka_network_cad_avg1h',
    'counter_events_total_instance_consumer_from_ws_with_metrics:8000_job_consumer_from_ws_with_metrics_service_castle': 'counter_events_total',
}
df.rename(columns=renames, inplace=True)

df.dropna(inplace=True)
print("Data shape after dropna:", df.shape)

HORIZON = 900  # 15-sec steps → ~3.75 h

# ────────────────────────────────────────────────────────────────
#  Parameter grid (add more dicts if needed)
# ────────────────────────────────────────────────────────────────
experiments: List[Dict] = [
    dict(hidden_size=64 , dropout=0.10, weight_decay=1e-4),
    dict(hidden_size=128, dropout=0.20, weight_decay=1e-4),
    dict(hidden_size=256, dropout=0.30, weight_decay=5e-4),
    dict(hidden_size=128, dropout=0.30, weight_decay=1e-4),
    dict(hidden_size=256, dropout=0.20, weight_decay=1e-4),
    dict(hidden_size=128, dropout=0.10, weight_decay=5e-4),
    dict(hidden_size=64 , dropout=0.20, weight_decay=5e-4),
    dict(hidden_size=256, dropout=0.10, weight_decay=1e-4),
    dict(hidden_size=64 , dropout=0.30, weight_decay=1e-3),
    dict(hidden_size=192, dropout=0.25, weight_decay=5e-4),
]

# ────────────────────────────────────────────────────────────────
#  TimeSeriesSplit (5 folds × test_size=900)
# ────────────────────────────────────────────────────────────────
cv = TimeSeriesSplit(n_splits=5, test_size=HORIZON)

# ────────────────────────────────────────────────────────────────
#  Run
# ────────────────────────────────────────────────────────────────
for exp_id, params in enumerate(experiments, 1):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lstm_multivar_hs{params['hidden_size']}_do{params['dropout']}_wd{params['weight_decay']}_{ts}"
    out_dir = f"plots/lstm_multivar_night/{model_name}"
    ensure_dir(out_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{out_dir}/results.log"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True,
    )
    log = logging.getLogger(model_name)

    log.info(f"========== Experiment {exp_id}/{len(experiments)} — {model_name} ==========")
    log.info(f"Parameter set: {params}")

    fold_metrics = []

    # redirect prints from CatBoost etc.
    with capture_to_logging(log):
        for fold, (train_idx, test_idx) in enumerate(cv.split(df)):
            log.info(f"--- Fold {fold+1}/{cv.n_splits} ---")
            df_train = df.iloc[train_idx]
            df_hold = df.iloc[test_idx]

            try:
                model = LSTMMultivariateForecast(
                    target_col="common_cad_avg1h",
                    horizon=HORIZON,
                    seq_len=5760,
                    n_epochs=30,
                    patience=6,
                    device="cuda:1",
                    **params,
                )
                model.fit(df_train)
                y_pred_series = model.predict(df_train)
            except Exception as err:
                log.exception(f"Fold {fold+1} crashed: {err}", exc_info=True)
                # cleanup GPU and continue
                import gc, torch; del model; gc.collect(); torch.cuda.empty_cache();
                continue

            # align to hold indices
            y_true = df_hold["common_cad_avg1h"].reindex(y_pred_series.index)
            y_pred = y_pred_series.reindex(y_true.index)
            # dropna safety
            mask = (~y_true.isna()) & (~y_pred.isna())
            y_true, y_pred = y_true[mask], y_pred[mask]
            if len(y_true) == 0:
                log.warning("No overlap between prediction and hold-out!")
                continue
            metrics = global_metrics(y_true, y_pred)
            fold_metrics.append(metrics)
            log.info(f"Fold {fold+1}: MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} MAPE={metrics['MAPE']:.2f}% on {len(y_true)} pts")

            # quick chart per fold
            hist_start = y_true.index[0] - pd.Timedelta(minutes=30)
            plot_history_forecast(
                history=df.loc[hist_start:y_true.index[0], "common_cad_avg1h"],
                forecast=y_pred,
                actual=y_true,
                title=f"Fold {fold+1}",
                filename=f"{out_dir}/cv_fold_{fold+1}_forecast.png",
            )

            # free GPU mem
            del model; import gc, torch; gc.collect(); torch.cuda.empty_cache()

    #  aggregate & save
    if fold_metrics:
        agg = pd.DataFrame(fold_metrics).mean().to_dict()
        log.info(f"===== mean over {len(fold_metrics)} folds: MAE={agg['MAE']:.2f} RMSE={agg['RMSE']:.2f} MAPE={agg['MAPE']:.2f}% =====")
        pd.DataFrame(fold_metrics).to_csv(f"{out_dir}/cv_metrics.csv", index=False)

print("Done.") 
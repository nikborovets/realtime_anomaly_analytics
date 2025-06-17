#!/usr/bin/env python
"""Nightly training for LSTMDecompMultivariateForecast on multiple param sets."""
import datetime as dt
import io
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from ts_toolkit.metrics import global_metrics
from ts_toolkit.viz import plot_history_forecast

from src.data_loader import fetch_frame
from src.models import LSTMDecompMultivariateForecast

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────── helpers ──────────────────────────
@contextmanager
def capture_to_logging(logger):
    original_stdout, original_stderr = sys.stdout, sys.stderr

    class _Writer(io.TextIOBase):
        def __init__(self, lg, lvl):
            self.logger, self.level, self.buf = lg, lvl, ""
        def write(self, txt):
            self.buf += txt
            parts = self.buf.split("\n")
            for line in parts[:-1]:
                if line.strip():
                    self.logger.log(self.level, line.strip())
            self.buf = parts[-1]
        def flush(self):
            if self.buf.strip():
                self.logger.log(self.level, self.buf.strip())
            self.buf = ""
    sys.stdout, sys.stderr = _Writer(logger, logging.INFO), _Writer(logger, logging.ERROR)
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        sys.stdout, sys.stderr = original_stdout, original_stderr

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ───────────────────────── data ────────────────────────────
print("Loading data…")
df: pd.DataFrame = fetch_frame(
    start_date="2024-11-25 18:00:00",
    end_date="2024-12-11 12:10:00",
    use_cache=True,
    cache_filename="multivar_9001_20241125_20241211.parquet",
)
renames = {
    'common_cad_avg1h_instance_consumer_from_ws_with_metrics:8000_job_consumer_from_ws_with_metrics_service_castle': 'common_cad_avg1h',
    'db_insert_cad_avg1h_instance_10.201.92.176:8010_job_consumer_ml_to_db_service_castle': 'db_insert_cad_avg1h',
    'kafka_network_cad_avg1h_instance_10.201.92.176:8010_job_consumer_ml_to_db_service_castle': 'kafka_network_cad_avg1h',
    'counter_events_total_instance_consumer_from_ws_with_metrics:8000_job_consumer_from_ws_with_metrics_service_castle': 'counter_events_total',
}
df.rename(columns=renames, inplace=True)

df.dropna(inplace=True)
print("Data shape:", df.shape)

HORIZON = 900
# ───────────────────────── experiments ─────────────────────
experiments: List[Dict] = [
    dict(hidden_size=64, dropout=0.1, weight_decay=1e-4),
    dict(hidden_size=128, dropout=0.2, weight_decay=5e-4),
]

cv = TimeSeriesSplit(n_splits=5, test_size=HORIZON)

for exp_id, params in enumerate(experiments, 1):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lstm_decomp_hs{params['hidden_size']}_do{params['dropout']}_wd{params['weight_decay']}_{ts}"
    out_dir = f"plots/lstm_decomp_night/{model_name}"
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
    logger = logging.getLogger(model_name)

    logger.info(f"==== Experiment {exp_id}/{len(experiments)} → {model_name} ====")
    logger.info(f"Params: {params}")

    fold_metrics = []
    with capture_to_logging(logger):
        for fold, (train_idx, test_idx) in enumerate(cv.split(df)):
            logger.info(f"-- Fold {fold+1}/{cv.n_splits}")
            df_train = df.iloc[train_idx]
            df_hold = df.iloc[test_idx]

            try:
                model = LSTMDecompMultivariateForecast(
                    target_col="common_cad_avg1h",
                    horizon=HORIZON,
                    seq_len=5760,
                    n_epochs=30,
                    patience=6,
                    device="cuda:6",
                    **params,
                )
                model.fit(df_train)
                y_pred_series = model.predict(df_train)
            except Exception as err:
                logger.exception(f"Fold {fold+1} crashed: {err}", exc_info=True)
                # cleanup GPU and continue
                import gc, torch; del model; gc.collect(); torch.cuda.empty_cache()
                continue

            y_true = df_hold["common_cad_avg1h"].reindex(y_pred_series.index)
            y_pred = y_pred_series.reindex(y_true.index)
            mask = (~y_true.isna()) & (~y_pred.isna())
            y_true, y_pred = y_true[mask], y_pred[mask]
            if len(y_true) == 0:
                logger.warning("No overlap between prediction and hold-out"); continue

            m = global_metrics(y_true, y_pred)
            fold_metrics.append(m)
            logger.info(f"Fold {fold+1}: MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} MAPE={m['MAPE']:.2f}% on {len(y_true)} pts")

            hist_start = y_true.index[0] - pd.Timedelta(minutes=30)
            plot_history_forecast(
                history=df.loc[hist_start:y_true.index[0], "common_cad_avg1h"],
                forecast=y_pred,
                actual=y_true,
                title=f"Fold {fold+1}",
                filename=f"{out_dir}/cv_fold_{fold+1}_forecast.png",
            )

            # cleanup
            del model; import gc, torch; gc.collect(); torch.cuda.empty_cache()

    if fold_metrics:
        agg = pd.DataFrame(fold_metrics).mean().to_dict()
        logger.info(f"=== mean over folds: MAE={agg['MAE']:.2f} RMSE={agg['RMSE']:.2f} MAPE={agg['MAPE']:.2f}% ===")
        pd.DataFrame(fold_metrics).to_csv(f"{out_dir}/cv_metrics.csv", index=False)

print("ALL DONE") 
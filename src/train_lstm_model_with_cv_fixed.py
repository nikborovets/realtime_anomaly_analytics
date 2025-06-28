"""LSTM‑прогноз с честной walk‑forward‑валидацией без утечек.
Версия от 2025‑06‑27.
"""

import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from src.data_loader import fetch_frame
from ts_toolkit.io import clean_timeseries
from ts_toolkit.metrics import global_metrics
from ts_toolkit.viz import plot_history_forecast

# --- logging -----------------------------------------------------------------
LOGS_ROOT = Path("logs/lstm_multivariate_feat_cv_recursive")
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

log_file = LOGS_ROOT / "training.log"
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in root_logger.handlers):
    fh_root = logging.FileHandler(log_file)
    fh_root.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(fh_root)

if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    sh_root = logging.StreamHandler()
    sh_root.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(sh_root)

# --- feature engineering ------------------------------------------------------

def _calculate_features(df_in: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Вспомогательная функция для расчёта фичей на основе истории."""
    df = df_in.copy()
    seconds_in_hour = 3600
    seconds_in_day = 86400

    ts_s = df.index.map(pd.Timestamp.timestamp)
    df["hour_sin"] = np.sin(ts_s * 2 * np.pi / seconds_in_hour)
    df["hour_cos"] = np.cos(ts_s * 2 * np.pi / seconds_in_hour)
    df["day_sin"] = np.sin(ts_s * 2 * np.pi / seconds_in_day)
    df["day_cos"] = np.cos(ts_s * 2 * np.pi / seconds_in_day)

    window_5m, window_1h = 20, 240
    shifted = df[target_col].shift(1)
    df[f"{target_col}_roll_mean_5m"] = shifted.rolling(window_5m, min_periods=1).mean()
    df[f"{target_col}_roll_mean_1h"] = shifted.rolling(window_1h, min_periods=1).mean()
    df[f"{target_col}_roll_std_1h"] = shifted.rolling(window_1h, min_periods=1).std()

    # Лаги
    for i in range(1, 4):
        df[f"{target_col}_lag_{i}"] = df[target_col].shift(i)

    return df


def add_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Добавляет циклические и rolling‑фичи. Использует shift(1), поэтому
    не смотрит в будущее на момент расчёта.
    """
    df_f = _calculate_features(df, target_col)
    return df_f.dropna()


def df_to_multivariate_X_y(df: pd.DataFrame, window: int):
    arr = df.to_numpy()
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i : i + window])
        y.append(arr[i + window][0])  # target в первой колонке
    return np.asarray(X), np.asarray(y)

# --- recursive (walk‑forward) inference --------------------------------------

def _calculate_single_step_features(
    history_df: pd.DataFrame,
    new_value: float,
    new_timestamp: pd.Timestamp,
    target_col: str,
) -> pd.DataFrame:
    """Эффективно рассчитывает фичи для одного нового шага, используя историю."""
    feat_cols = [c for c in history_df.columns if c != target_col]
    new_row = pd.DataFrame(
        [[new_value] + [np.nan] * len(feat_cols)],
        columns=[target_col] + feat_cols,
        index=[new_timestamp],
    )

    # Циклические
    seconds_in_hour, seconds_in_day = 3600, 86400
    ts_s = new_timestamp.timestamp()
    new_row["hour_sin"] = np.sin(ts_s * 2 * np.pi / seconds_in_hour)
    new_row["hour_cos"] = np.cos(ts_s * 2 * np.pi / seconds_in_hour)
    new_row["day_sin"] = np.sin(ts_s * 2 * np.pi / seconds_in_day)
    new_row["day_cos"] = np.cos(ts_s * 2 * np.pi / seconds_in_day)

    # Rolling
    window_5m, window_1h = 20, 240
    temp_target_series = pd.concat([history_df[target_col], new_row[target_col]])
    shifted = temp_target_series.shift(1)

    new_row[f"{target_col}_roll_mean_5m"] = (
        shifted.rolling(window_5m, min_periods=1).mean().iloc[-1]
    )
    new_row[f"{target_col}_roll_mean_1h"] = (
        shifted.rolling(window_1h, min_periods=1).mean().iloc[-1]
    )
    new_row[f"{target_col}_roll_std_1h"] = (
        shifted.rolling(window_1h, min_periods=1).std().iloc[-1]
    )

    # Лаги
    for i in range(1, 4):
        if len(history_df) >= i:
            new_row[f"{target_col}_lag_{i}"] = history_df[target_col].iloc[-i]
        else:
            new_row[f"{target_col}_lag_{i}"] = np.nan

    return new_row


def recursive_predict(model, history_df, n_steps, window, scaler, target_col):
    """Пошаговый прогноз без заглядывания в будущие истинные значения."""
    hist = history_df.copy()
    preds = []
    feat_cols = [c for c in hist.columns if c != target_col]

    for _ in range(n_steps):
        X_t = hist.iloc[-window:].to_numpy()[None, ...]
        y_hat = model.predict(X_t, verbose=0)[0, 0]
        preds.append(y_hat)

        # timestamp новой точки (шаг 15 сек)
        t_next = hist.index[-1] + pd.Timedelta(seconds=15)

        # Считаем фичи для новой точки
        new_row = _calculate_single_step_features(hist, y_hat, t_next, target_col)

        # масштабирование feature‑колонок
        new_row.loc[:, feat_cols] = scaler.transform(new_row[feat_cols])

        hist = pd.concat([hist, new_row])

    return np.asarray(preds)

# --- main --------------------------------------------------------------------

def main(cfg):
    logging.info("Загрузка данных…")
    df_raw = fetch_frame(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        use_cache=cfg.use_cache,
        cache_filename=cfg.cache_filename,
    )

    if any(c.startswith(cfg.target_col) for c in df_raw.columns):
        df_raw.rename(columns={df_raw.columns[df_raw.columns.str.startswith(cfg.target_col)][0]: cfg.target_col}, inplace=True)

    df_initial = df_raw[[cfg.target_col]].copy()

    ts_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"models/lstm_recursive_cv_{ts_stamp}"); run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(f"plots/lstm_recursive_cv_{ts_stamp}"); plots_dir.mkdir(parents=True, exist_ok=True)

    splitter = TimeSeriesSplit(n_splits=cfg.n_splits, test_size=cfg.test_size)
    all_metrics = []

    for fold, (train_val_idx, test_idx) in enumerate(splitter.split(df_initial), 1):
        logging.info(f"===== Фолд {fold}/{cfg.n_splits} =====")

        df_trv_raw = df_initial.iloc[train_val_idx]
        df_test_raw = df_initial.iloc[test_idx]

        # очистка
        df_trv_clean = clean_timeseries(df_trv_raw, cfg.target_col)
        fill_seed = df_trv_clean.iloc[-1:] if not df_trv_clean.empty else pd.DataFrame()
        df_test_clean = pd.concat([fill_seed, df_test_raw]).ffill().iloc[len(fill_seed):]

        # признаки для train+val
        df_trv_feat = add_features(df_trv_clean, cfg.target_col)

        # split train/val
        val_size = cfg.test_size
        gap = cfg.gap
        if len(df_trv_feat) <= val_size + gap:
            logging.warning("Мало точек; пропуск фолда.")
            continue
        val_start = len(df_trv_feat) - val_size
        train_end = val_start - gap
        df_train = df_trv_feat.iloc[:train_end]
        df_val = df_trv_feat.iloc[val_start:]

        feature_cols = [c for c in df_train.columns if c != cfg.target_col]
        scaler = StandardScaler()
        df_train.loc[:, feature_cols] = scaler.fit_transform(df_train[feature_cols])
        df_val.loc[:, feature_cols] = scaler.transform(df_val[feature_cols])

        # окна
        X_train, y_train = df_to_multivariate_X_y(df_train, cfg.window_size)
        X_val, y_val = df_to_multivariate_X_y(df_val, cfg.window_size)
        if not len(X_train) or not len(X_val):
            logging.warning("Нулевая выборка; пропуск фолда.")
            continue

        n_feat = X_train.shape[2]
        model = Sequential([
            InputLayer((cfg.window_size, n_feat)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        mdl_path = run_dir / f"fold_{fold}.h5"
        model.compile(loss=MeanSquaredError(), optimizer=Adam(1e-4), metrics=[RootMeanSquaredError()])

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
        mcp = ModelCheckpoint(mdl_path, save_best_only=True, monitor="val_loss")

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.epochs,
            callbacks=[early_stopping, mcp],
            verbose=1,
        )
        best = model  # EarlyStopping с restore_best_weights=True вернёт лучшие веса

        # === Recursion ===
        initial_hist = df_trv_feat.copy()
        initial_hist.loc[:, feature_cols] = scaler.transform(initial_hist[feature_cols])

        n_steps = len(df_test_clean)
        preds = recursive_predict(best, initial_hist, n_steps, cfg.window_size, scaler, cfg.target_col)

        idx_test = df_test_clean.index
        y_true = df_test_clean[cfg.target_col]
        y_pred = pd.Series(preds, index=idx_test)

        metrics = global_metrics(y_true, y_pred)
        all_metrics.append(metrics)
        logging.info(f"Метрики фолда {fold}: {metrics}")

        # график
        plot_history_forecast(
            history=df_initial.loc[idx_test[0] - pd.Timedelta(minutes=cfg.history_minutes) : idx_test[0], cfg.target_col],
            forecast=y_pred,
            actual=y_true,
            title=f"Recursive forecast vs actual — fold {fold}",
            filename=str(plots_dir / f"fold_{fold}_{ts_stamp}.png"),
        )

    if all_metrics:
        mean_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
        logging.info(f"==== Средние метрики: {mean_metrics}")
    else:
        logging.warning("Ни одного завершённого фолда.")

if __name__ == "__main__":
    p = argparse.ArgumentParser("LSTM multivariate TS forecasting (walk‑forward CV)")
    p.add_argument("--start-date", default="2024-11-25 18:00:00")
    p.add_argument("--end-date",   default="2024-12-11 12:10:00")
    p.add_argument("--use-cache", action="store_true", default=True)
    p.add_argument("--cache-filename", default="common_cad_avg1h_20241125_20241211.parquet")
    p.add_argument("--target-col", default="common_cad_avg1h")
    p.add_argument("--window-size", type=int, default=720) # 15 * 720 = 3 часа
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--history-minutes", type=int, default=30)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=int, default=900)
    p.add_argument("--gap", type=int, default=0)
    cfg = p.parse_args()
    main(cfg)

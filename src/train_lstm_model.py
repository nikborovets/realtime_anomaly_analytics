import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from src.data_loader import fetch_frame
from ts_toolkit.io import clean_timeseries
from ts_toolkit.split import three_way_split
from ts_toolkit.viz import plot_history_forecast

# --- Настройка логирования ---
LOGS_ROOT = Path("logs/lstm_multivariate_feat")
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

log_file = LOGS_ROOT / "training.log"
# Проверяем, что обработчик для файла еще не добавлен
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in root_logger.handlers):
    fh_root = logging.FileHandler(log_file)
    fh_root.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(fh_root)

# Проверяем, что обработчик для консоли еще не добавлен
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    sh_root = logging.StreamHandler()
    sh_root.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(sh_root)


def add_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Добавляет в датафрейм релевантные фичи для данных с высокой частотой."""
    df_featured = df.copy()

    # 1. Циклические признаки
    # Добавляем часовые и суточные циклы. Недельные убраны по запросу.
    seconds_in_hour = 60 * 60
    seconds_in_day = 24 * seconds_in_hour
    
    timestamp_s = df_featured.index.map(pd.Timestamp.timestamp)
    df_featured['hour_sin'] = np.sin(timestamp_s * (2 * np.pi / seconds_in_hour))
    df_featured['hour_cos'] = np.cos(timestamp_s * (2 * np.pi / seconds_in_hour))
    df_featured['day_sin'] = np.sin(timestamp_s * (2 * np.pi / seconds_in_day))
    df_featured['day_cos'] = np.cos(timestamp_s * (2 * np.pi / seconds_in_day))

    # 2. Признаки на основе скользящего среднего
    # Окна под 15-секундную частоту: 5 мин = 20, 1 час = 240
    window_5m = 20
    window_1h = 240
    
    df_featured[f'{target_col}_roll_mean_5m'] = df_featured[target_col].rolling(window=window_5m, min_periods=1).mean()
    df_featured[f'{target_col}_roll_mean_1h'] = df_featured[target_col].rolling(window=window_1h, min_periods=1).mean()
    df_featured[f'{target_col}_roll_std_1h'] = df_featured[target_col].rolling(window=window_1h, min_periods=1).std()
    
    # Удаляем строки с NaN, которые могли появиться, если min_periods > 1
    df_featured = df_featured.dropna()
    
    return df_featured


def df_to_multivariate_X_y(df: pd.DataFrame, window_size: int = 6):
    """Преобразует временной ряд с признаками в датасет с окнами для LSTM."""
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size][0]  # Целевая переменная - первая колонка
        y.append(label)
    return np.array(X), np.array(y)


def preprocess_inputs(X_train, X_val, X_hold):
    """Масштабирует первую фичу (целевую) на основе трейна."""
    train_mean = np.mean(X_train[:, :, 0])
    train_std = np.std(X_train[:, :, 0])

    def scale(X):
        X_scaled = X.copy()
        X_scaled[:, :, 0] = (X_scaled[:, :, 0] - train_mean) / train_std
        return X_scaled

    return scale(X_train), scale(X_val), scale(X_hold)


def main(args):
    """Основной скрипт для обучения и оценки модели."""
    # --- Загрузка и подготовка данных ---
    logging.info("Загрузка и очистка данных...")
    df_raw = fetch_frame(
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=args.use_cache,
        cache_filename=args.cache_filename,
    )

    long_cols = [c for c in df_raw.columns if c.startswith(args.target_col)]
    if long_cols:
        df_raw.rename(columns={long_cols[0]: args.target_col}, inplace=True)

    df_clean = clean_timeseries(df_raw, args.target_col)

    # --- Добавление признаков ---
    logging.info("Добавление кастомных признаков (циклических и rolling)...")
    df_featured = add_features(df_clean[[args.target_col]], args.target_col)

    # --- Разделение данных ---
    logging.info("Разделение данных...")
    df_train, df_val, df_hold = three_way_split(
        df_featured, train_ratio=0.8, val_ratio=0.19
    )

    # --- Создание окон ---
    X_train, y_train = df_to_multivariate_X_y(df_train, args.window_size)
    X_val, y_val = df_to_multivariate_X_y(df_val, args.window_size)
    X_hold, y_hold = df_to_multivariate_X_y(df_hold, args.window_size)

    # --- Масштабирование входов ---
    logging.info("Масштабирование входных данных...")
    X_train_scaled, X_val_scaled, X_hold_scaled = preprocess_inputs(X_train, X_val, X_hold)
    
    num_features = X_train_scaled.shape[2]
    logging.info(f"Количество признаков: {num_features}")

    logging.info(f"Размер обучающей выборки: {X_train_scaled.shape}, {y_train.shape}")
    logging.info(f"Размер валидационной выборки: {X_val_scaled.shape}, {y_val.shape}")
    logging.info(f"Размер отложенной выборки: {X_hold_scaled.shape}, {y_hold.shape}")

    # --- Определение модели ---
    logging.info("Построение модели...")
    model = Sequential()
    model.add(InputLayer((args.window_size, num_features)))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))
    model.summary(print_fn=logging.info)

    # --- Обучение модели ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"models/lstm_multivariate_feat_{ts}")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.h5"

    cp = ModelCheckpoint(str(model_path), save_best_only=True, monitor='val_loss', mode='min')
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    logging.info("Обучение модели...")
    model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=args.epochs, callbacks=[cp])

    # --- Оценка ---
    logging.info("Оценка на отложенной выборке...")
    best_model = load_model(model_path)
    
    hold_predictions_flat = best_model.predict(X_hold_scaled).flatten()
    
    hold_index = df_hold.index[args.window_size:]
    
    y_pred = pd.Series(hold_predictions_flat, index=hold_index)
    y_true = pd.Series(y_hold, index=hold_index)

    # --- Построение графиков ---
    plots_dir = Path(f"plots/lstm_multivariate_feat_{ts}")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if not y_pred.empty and not y_true.empty:
        logging.info("Сохранение графика прогноза на отложенной выборке...")
        history_end_point = y_pred.index[0]
        history_start_point = history_end_point - pd.Timedelta(minutes=args.history_minutes)

        plot_history_forecast(
            history=df_clean.loc[history_start_point:history_end_point, args.target_col],
            forecast=y_pred,
            actual=y_true,
            title="Blind forecast vs actual — hold-out",
            filename=str(plots_dir / f"hold_out_forecast_{ts}.png"),
        )
    else:
        logging.warning("Отложенная выборка: нет данных для построения графика.")

    logging.info(f"Скрипт завершен. Модель сохранена в {model_dir}, графики в {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Multivariate Time Series Forecasting with Cyclical Features")
    parser.add_argument("--start-date", type=str, default="2024-11-25 18:00:00")
    parser.add_argument("--end-date", type=str, default="2024-12-11 12:10:00")
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--cache-filename", type=str, default="common_cad_avg1h_20241125_20241211.parquet")
    parser.add_argument("--target-col", type=str, default="common_cad_avg1h")
    parser.add_argument("--window-size", type=int, default=40, help="Количество прошлых временных шагов для входа (10 минут).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--history-minutes", type=int, default=180, help="Количество минут истории для отображения на графике.")

    args = parser.parse_args()
    main(args) 
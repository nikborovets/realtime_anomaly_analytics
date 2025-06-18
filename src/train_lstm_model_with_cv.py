import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from src.data_loader import fetch_frame
from ts_toolkit.io import clean_timeseries
from ts_toolkit.metrics import global_metrics
from ts_toolkit.viz import plot_history_forecast

# --- Настройка логирования ---
LOGS_ROOT = Path("logs/lstm_multivariate_feat_cv")
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


def get_scaler_params(X_train):
    """Вычисляет параметры масштабирования (среднее и std) на обучающих данных."""
    train_mean = np.mean(X_train[:, :, 0])
    train_std = np.std(X_train[:, :, 0])
    return train_mean, train_std


def scale_data(X, mean, std):
    """Масштабирует данные, используя предоставленные среднее и std."""
    X_scaled = X.copy()
    # Масштабируем только первую фичу (целевую переменную)
    X_scaled[:, :, 0] = (X_scaled[:, :, 0] - mean) / std
    return X_scaled


def main(args):
    """Основной скрипт для обучения и оценки модели с кросс-валидацией."""
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

    # --- Кросс-валидация ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"models/lstm_multivariate_cv_{ts}")
    plots_dir = Path(f"plots/lstm_multivariate_cv_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    tscv = TimeSeriesSplit(n_splits=args.n_splits, test_size=args.test_size)
    metrics_list = []

    logging.info(f"Запускаю кросс-валидацию с {args.n_splits} фолдами...")

    for fold_idx, (train_val_idx, test_idx) in enumerate(tscv.split(df_featured)):
        fold_num = fold_idx + 1
        logging.info(f"==== ФОЛД {fold_num}/{args.n_splits} ====")
        
        # 1. Разделение на train+val и test для этого фолда
        df_train_val_fold = df_featured.iloc[train_val_idx]
        df_test_fold = df_featured.iloc[test_idx]

        # 2. Разделение train+val на train и val для .fit()
        val_size = args.test_size 
        if len(df_train_val_fold) <= val_size:
            logging.warning(f"Фолд {fold_num}: недостаточно данных для создания валидационного сета, пропускаю.")
            continue
        df_train_fold = df_train_val_fold.iloc[:-val_size]
        df_val_fold = df_train_val_fold.iloc[-val_size:]
        
        # 3. Создание оконных данных
        X_train, y_train = df_to_multivariate_X_y(df_train_fold, args.window_size)
        X_val, y_val = df_to_multivariate_X_y(df_val_fold, args.window_size)
        X_test, y_test = df_to_multivariate_X_y(df_test_fold, args.window_size)

        # 4. Масштабирование (скейлер обучается ТОЛЬКО на X_train)
        mean, std = get_scaler_params(X_train)
        X_train_scaled = scale_data(X_train, mean, std)
        X_val_scaled = scale_data(X_val, mean, std)
        X_test_scaled = scale_data(X_test, mean, std)
        
        num_features = X_train_scaled.shape[2]
        logging.info(f"Количество признаков: {num_features}")

        # 5. Определение и обучение модели
        logging.info("Построение и обучение модели...")
        model = Sequential([
            InputLayer((args.window_size, num_features)),
            LSTM(64),
            Dense(8, 'relu'),
            Dense(1, 'linear')
        ])
        
        model_path = run_dir / f"model_fold_{fold_num}.h5"
        cp = ModelCheckpoint(str(model_path), save_best_only=True, monitor='val_loss', mode='min')
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
        
        model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=args.epochs, callbacks=[cp], verbose=1)

        # 6. Оценка на тестовом фолде
        logging.info("Оценка на тестовом фолде...")
        best_model = load_model(model_path)
        test_predictions_flat = best_model.predict(X_test_scaled).flatten()
        
        test_index = df_test_fold.index[args.window_size:]
        y_pred = pd.Series(test_predictions_flat, index=test_index)
        y_true = pd.Series(y_test, index=test_index)

        if y_pred.empty or y_true.empty:
            logging.warning(f"Фолд {fold_num}: нет данных для оценки, пропускаю.")
            continue

        fold_metrics = global_metrics(y_true, y_pred)
        metrics_list.append(fold_metrics)
        logging.info(f"Метрики на фолде {fold_num}: {fold_metrics}")
        
        # 7. Построение графика для фолда
        logging.info("Сохранение графика прогноза для фолда...")
        history_end_point = y_pred.index[0]
        history_start_point = history_end_point - pd.Timedelta(minutes=args.history_minutes)

        plot_history_forecast(
            history=df_clean.loc[history_start_point:history_end_point, args.target_col],
            forecast=y_pred,
            actual=y_true,
            title=f"Blind forecast vs actual — CV Fold {fold_num}",
            filename=str(plots_dir / f"cv_fold_{fold_num}_forecast_{ts}.png"),
        )

    # --- Итоги по кросс-валидации ---
    if metrics_list:
        avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
        logging.info("==== ИТОГОВЫЕ МЕТРИКИ (среднее по фолдам) ====")
        logging.info(avg_metrics)
    else:
        logging.warning("Не удалось рассчитать метрики ни на одном фолде.")

    logging.info(f"Скрипт завершен. Модели сохранены в {run_dir}, графики в {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Multivariate Time Series Forecasting with Cross-Validation")
    parser.add_argument("--start-date", type=str, default="2024-11-25 18:00:00")
    parser.add_argument("--end-date", type=str, default="2024-12-11 12:10:00")
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--cache-filename", type=str, default="common_cad_avg1h_20241125_20241211.parquet")
    parser.add_argument("--target-col", type=str, default="common_cad_avg1h")
    parser.add_argument("--window-size", type=int, default=40, help="Количество прошлых временных шагов для входа (10 минут).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--history-minutes", type=int, default=30, help="Количество минут истории для отображения на графике.")
    # Аргументы для CV
    parser.add_argument("--n-splits", type=int, default=5, help="Количество фолдов в TimeSeriesSplit.")
    parser.add_argument("--test-size", type=int, default=900, help="Размер тестового набора в каждом фолде (в точках данных).")

    args = parser.parse_args()
    main(args) 
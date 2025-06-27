import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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
LOGS_ROOT = Path("logs/lstm_multivariate_feat_cv_fixed")
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
    
    target_shifted = df_featured[target_col].shift(1)
    df_featured[f'{target_col}_roll_mean_5m'] = target_shifted.rolling(window=window_5m, min_periods=1).mean()
    df_featured[f'{target_col}_roll_mean_1h'] = target_shifted.rolling(window=window_1h, min_periods=1).mean()
    df_featured[f'{target_col}_roll_std_1h'] = target_shifted.rolling(window=window_1h, min_periods=1).std()
    
    # Удаляем строки с NaN, которые могли появиться из-за shift() и rolling()
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


def recursive_predict(model, initial_history_df, n_steps, window_size, scaler, target_col):
    """
    Делает рекурсивный прогноз на n_steps вперед.
    На каждом шаге использует свой собственный предыдущий прогноз для генерации фичей.
    """
    history_df = initial_history_df.copy()
    predictions = []
    
    feature_cols = [c for c in history_df.columns if c != target_col]

    for _ in range(n_steps):
        # 1. Берем последние window_size точек для создания входа в модель
        last_window_df = history_df.iloc[-window_size:]
        X = last_window_df.to_numpy().reshape(1, window_size, -1)
        
        # 2. Делаем прогноз на один шаг
        next_pred = model.predict(X, verbose=0)[0][0]
        predictions.append(next_pred)

        # 3. Готовим новую строку для добавления в историю
        last_timestamp = history_df.index[-1]
        next_timestamp = last_timestamp + pd.Timedelta(seconds=15)

        new_row_df = pd.DataFrame([[next_pred] + [0]*(len(feature_cols))], columns=[target_col] + feature_cols, index=[next_timestamp])
        
        # 4. Обновляем фичи для новой строки
        # 4.1 Циклические фичи
        timestamp_s = next_timestamp.timestamp()
        seconds_in_hour = 3600
        seconds_in_day = 86400
        new_row_df['hour_sin'] = np.sin(timestamp_s * (2 * np.pi / seconds_in_hour))
        new_row_df['hour_cos'] = np.cos(timestamp_s * (2 * np.pi / seconds_in_hour))
        new_row_df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / seconds_in_day))
        new_row_df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / seconds_in_day))

        # 4.2 Rolling-фичи
        temp_history_for_rolling = pd.concat([history_df[[target_col]], new_row_df[[target_col]]])
        target_shifted = temp_history_for_rolling[target_col].shift(1)
        window_5m = 20
        window_1h = 240
        new_row_df[f'{target_col}_roll_mean_5m'] = target_shifted.rolling(window=window_5m, min_periods=1).mean().iloc[-1]
        new_row_df[f'{target_col}_roll_mean_1h'] = target_shifted.rolling(window=window_1h, min_periods=1).mean().iloc[-1]
        new_row_df[f'{target_col}_roll_std_1h'] = target_shifted.rolling(window=window_1h, min_periods=1).std().iloc[-1]
        
        # 5. Масштабируем фичи в новой строке
        new_row_df.loc[:, feature_cols] = scaler.transform(new_row_df[feature_cols])

        # 6. Добавляем полностью готовую новую строку в историю
        history_df = pd.concat([history_df, new_row_df])
        
    return np.array(predictions)


def main(args):
    """Основной скрипт для обучения и оценки модели с кросс-валидацией."""
    # --- Загрузка и подготовка данных ---
    logging.info("Загрузка данных...")
    df_raw = fetch_frame(
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=args.use_cache,
        cache_filename=args.cache_filename,
    )

    long_cols = [c for c in df_raw.columns if c.startswith(args.target_col)]
    if long_cols:
        df_raw.rename(columns={long_cols[0]: args.target_col}, inplace=True)

    # ВАЖНО: Очистка (особенно интерполяция) будет производиться внутри фолдов CV,
    # чтобы избежать утечки данных из будущего (теста) в прошлое (трейн).
    df_initial = df_raw[[args.target_col]].copy()

    # --- Кросс-валидация ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"models/lstm_multivariate_cv_fixed_{ts}")
    plots_dir = Path(f"plots/lstm_multivariate_cv_fixed_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    tscv = TimeSeriesSplit(n_splits=args.n_splits, test_size=args.test_size)
    metrics_list = []

    logging.info(f"Запускаю кросс-валидацию с {args.n_splits} фолдами...")

    for fold_idx, (train_val_idx, test_idx) in enumerate(tscv.split(df_initial)):
        fold_num = fold_idx + 1
        logging.info(f"==== ФОЛД {fold_num}/{args.n_splits} ====")
        
        # 1. Разделение на train+val и test для этого фолда
        df_train_val_fold_raw = df_initial.iloc[train_val_idx]
        df_test_fold_raw = df_initial.iloc[test_idx]

        # 2. Очистка данных строго внутри фолда
        # Для train+val можно использовать interpolate, т.к. он не видит тестовые данные
        df_train_val_fold_clean = clean_timeseries(df_train_val_fold_raw, args.target_col)

        # Для теста используем ffill, чтобы избежать подглядывания в будущее внутри самого теста.
        # Берем последний известный хороший поинт из трейна для заполнения возможных NaN в начале теста.
        if not df_train_val_fold_clean.empty:
            history_for_fill = df_train_val_fold_clean.iloc[-1:]
            df_test_fold_clean = pd.concat([history_for_fill, df_test_fold_raw]).ffill().iloc[1:]
        else:
            df_test_fold_clean = df_test_fold_raw.ffill().dropna()


        # 3. Добавление признаков ОТДЕЛЬНО для каждого набора
        logging.info("Добавление кастомных признаков (циклических и rolling)...")
        df_train_val_featured = add_features(df_train_val_fold_clean, args.target_col)
        
        # Для тестового сета нужно "заглянуть" в историю из трейна для корректного расчета rolling-фич
        history_len = 240 + args.window_size # Макс. окно rolling + окно для LSTM
        history_for_test_features = df_train_val_fold_clean.iloc[-history_len:]
        df_test_with_history = pd.concat([history_for_test_features, df_test_fold_clean])
        df_test_featured_full = add_features(df_test_with_history, args.target_col)
        df_test_featured = df_test_featured_full.loc[df_test_fold_clean.index]


        # 4. Разделение train+val на train и val с учетом gap
        val_size = args.test_size 
        gap_size = args.gap
        if len(df_train_val_featured) <= val_size + gap_size:
            logging.warning(f"Фолд {fold_num}: недостаточно данных для создания валидационного сета с gap, пропускаю.")
            continue
            
        val_start_idx = len(df_train_val_featured) - val_size
        train_end_idx = val_start_idx - gap_size
        df_train_featured = df_train_val_featured.iloc[:train_end_idx]
        df_val_featured = df_train_val_featured.iloc[val_start_idx:]

        if df_train_featured.empty or df_val_featured.empty or df_test_featured.empty:
            logging.warning(f"Фолд {fold_num}: один из датасетов пуст после всех приготовлений, пропускаю.")
            continue
            
        # 5. Масштабирование всех признаков с помощью StandardScaler
        feature_cols = [c for c in df_train_featured.columns if c != args.target_col]
        scaler = StandardScaler()

        df_train_featured.loc[:, feature_cols] = scaler.fit_transform(df_train_featured[feature_cols])
        df_val_featured.loc[:, feature_cols] = scaler.transform(df_val_featured[feature_cols])
        df_test_featured.loc[:, feature_cols] = scaler.transform(df_test_featured[feature_cols])
        
        # 6. Создание оконных данных
        X_train, y_train = df_to_multivariate_X_y(df_train_featured, args.window_size)
        X_val, y_val = df_to_multivariate_X_y(df_val_featured, args.window_size)
        X_test, y_test = df_to_multivariate_X_y(df_test_featured, args.window_size)

        if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
            logging.warning(f"Фолд {fold_num}: не удалось создать оконные данные (слишком мало точек), пропускаю.")
            continue
        
        num_features = X_train.shape[2]
        logging.info(f"Количество признаков: {num_features}")

        # 7. Определение и обучение модели
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
        
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, callbacks=[cp], verbose=1)

        # 8. Оценка на тестовом фолде (рекурсивный прогноз)
        logging.info("Оценка на тестовом фолде (рекурсивный прогноз)...")
        best_model = load_model(model_path)
        
        # Начальная история для рекурсивного прогноза - это полный train+val набор
        initial_history = df_train_val_featured.copy()
        
        # Прогноз на всю длину тестового сета
        n_test_steps = len(df_test_fold_raw)
        test_predictions_flat = recursive_predict(best_model, initial_history, n_test_steps, args.window_size, scaler, args.target_col)
        
        test_index = df_test_fold_raw.index
        y_pred = pd.Series(test_predictions_flat, index=test_index)
        y_true = df_test_fold_raw[args.target_col]

        if y_pred.empty or y_true.empty:
            logging.warning(f"Фолд {fold_num}: нет данных для оценки, пропускаю.")
            continue

        fold_metrics = global_metrics(y_true, y_pred)
        metrics_list.append(fold_metrics)
        logging.info(f"Метрики на фолде {fold_num}: {fold_metrics}")
        
        # 9. Построение графика для фолда
        logging.info("Сохранение графика прогноза для фолда...")
        history_end_point = y_pred.index[0]
        history_start_point = history_end_point - pd.Timedelta(minutes=args.history_minutes)

        plot_history_forecast(
            history=df_initial.loc[history_start_point:history_end_point, args.target_col], # Берем изначальные "чистые" данные для графика
            forecast=y_pred,
            actual=y_true,
            title=f"Recursive Blind forecast vs actual — CV Fold {fold_num}",
            filename=str(plots_dir / f"cv_fold_{fold_num}_forecast_recursive_{ts}.png"),
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
    parser.add_argument("--gap", type=int, default=0, help="Зазор между train и val в точках данных.")

    args = parser.parse_args()
    main(args) 
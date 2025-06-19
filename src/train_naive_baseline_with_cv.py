import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data_loader import fetch_frame
from ts_toolkit.io import clean_timeseries
from ts_toolkit.metrics import global_metrics
from ts_toolkit.viz import plot_history_forecast

# --- Настройка логирования ---
# Оставляем только базовый логгер для вывода в консоль
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Проверяем, что обработчик для консоли еще не добавлен
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    sh_root = logging.StreamHandler()
    sh_root.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(sh_root)


def main(args):
    """Основной скрипт для оценки сезонного наивного baseline-прогноза с кросс-валидацией."""
    # --- Генерация имени и папки для этого запуска ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"seasonal_naive_baseline_{ts}"
    plots_dir = Path(f"plots/{run_name}")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Настройка логгирования в файл для этого запуска ---
    log_file_path = plots_dir / "run.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(file_handler)

    # Логгирование параметров запуска
    logging.info("--- Параметры запуска ---")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("--------------------------")

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
    df_final = df_clean[[args.target_col]]

    # --- Кросс-валидация ---
    tscv = TimeSeriesSplit(n_splits=args.n_splits, test_size=args.test_size)
    metrics_list = []

    logging.info(f"Запускаю кросс-валидацию с {args.n_splits} фолдами...")

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_final)):
        fold_num = fold_idx + 1
        logging.info(f"==== ФОЛД {fold_num}/{args.n_splits} ====")
        
        df_train_fold = df_final.iloc[train_idx]
        df_test_fold = df_final.iloc[test_idx]
        
        logging.info(f"Размер обучающей выборки: {len(df_train_fold)}, тестовой: {len(df_test_fold)}")
        if df_train_fold.empty or df_test_fold.empty:
            logging.warning("Фолд пропущен из-за пустой выборки.")
            continue

        # --- Сезонный наивный прогноз ("как сутки назад") ---
        logging.info("Генерация сезонного наивного прогноза...")
        
        # 1. Определяем, откуда брать данные для прогноза (временные метки в прошлом)
        lookup_timestamps = df_test_fold.index - pd.Timedelta(days=1)

        # 2. Ищем соответствующие значения в ТРЕНИРОВОЧНОМ наборе.
        # reindex найдет совпадающие по времени точки и вернет NaN, если их нет.
        lookup_values = df_train_fold[args.target_col].reindex(lookup_timestamps)

        # 3. Cоздаем прогноз. Индекс берем от тестового набора, а значения - из прошлого.
        y_pred = pd.Series(lookup_values.values, index=df_test_fold.index)
        
        # 4. Обработка пропусков: если для точки в тесте не нашлось значения
        # ровно 24 часа назад, заполним последним известным значением из трейна.
        if y_pred.isnull().any():
            nan_count = y_pred.isnull().sum()
            last_known_value = df_train_fold[args.target_col].iloc[-1]
            y_pred.fillna(value=last_known_value, inplace=True)
            logging.warning(f"{nan_count} пропусков в прогнозе были заменены последним известным значением: {last_known_value:.4f}")

        y_true = df_test_fold[args.target_col]
        logging.info("Прогноз 'как сутки назад' сгенерирован.")
        
        fold_metrics = global_metrics(y_true, y_pred)
        metrics_list.append(fold_metrics)
        logging.info(f"Метрики на фолде {fold_num}: {fold_metrics}")
        
        # --- Построение графика для фолда ---
        logging.info("Сохранение графика прогноза для фолда...")
        history_end_point = y_pred.index[0]
        history_start_point = history_end_point - pd.Timedelta(minutes=args.history_minutes)

        plot_history_forecast(
            history=df_clean.loc[history_start_point:history_end_point, args.target_col],
            forecast=y_pred,
            actual=y_true,
            title=f"Seasonal Naive Forecast vs Actual — CV Fold {fold_num}",
            filename=str(plots_dir / f"cv_fold_{fold_num}_forecast_{ts}.png"),
        )

    # --- Итоги по кросс-валидации ---
    if metrics_list:
        avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
        logging.info("==== ИТОГОВЫЕ МЕТРИКИ (среднее по фолдам) ====")
        logging.info(avg_metrics)
    else:
        logging.warning("Не удалось рассчитать метрики ни на одном фолде.")

    logging.info(f"Скрипт завершен. Графики и лог сохранены в {plots_dir}")

    # Убираем файловый обработчик, чтобы избежать дублирования при повторных вызовах
    root_logger.removeHandler(file_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seasonal Naive Baseline Forecasting with Cross-Validation")
    parser.add_argument("--start-date", type=str, default="2024-11-25 18:00:00")
    parser.add_argument("--end-date", type=str, default="2024-12-11 12:10:00")
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--cache-filename", type=str, default="common_cad_avg1h_20241125_20241211.parquet")
    parser.add_argument("--target-col", type=str, default="common_cad_avg1h")
    parser.add_argument("--history-minutes", type=int, default=180, help="Количество минут истории для отображения на графике.")
    # Аргументы для CV
    parser.add_argument("--n-splits", type=int, default=5, help="Количество фолдов в TimeSeriesSplit.")
    parser.add_argument("--test-size", type=int, default=900, help="Размер тестового набора в каждом фолде (в точках данных).")
    
    args = parser.parse_args()
    main(args) 
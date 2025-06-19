import argparse
import ast
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
    """Основной скрипт для обучения и оценки модели SARIMA с кросс-валидацией."""
    # --- Генерация имени и папки для этого запуска ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    order_str = "_".join(map(str, args.order))
    seasonal_order_str = "_".join(map(str, args.seasonal_order))
    run_name = f"sarima_order_{order_str}_seasonal_{seasonal_order_str}_{ts}"
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

    # Явно задаем частоту, чтобы избежать ValueWarning от statsmodels
    if pd.infer_freq(df_final.index) == "15S":
        df_final = df_final.asfreq('15s')

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

        # --- Обучение модели SARIMA ---
        logging.info("Обучение модели SARIMA... Это может занять время.")
        try:
            model = SARIMAX(
                df_train_fold[args.target_col],
                order=args.order,
                seasonal_order=args.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit_results = model.fit(disp=False)
            logging.info("Модель обучена.")
        except Exception as e:
            logging.error(f"Ошибка при обучении модели на фолде {fold_num}: {e}")
            continue

        # --- Оценка на тестовом фолде ---
        logging.info("Получение прогноза...")
        y_pred = fit_results.get_forecast(steps=len(df_test_fold)).predicted_mean
        y_true = df_test_fold[args.target_col]

        if y_pred.empty or y_true.empty:
            logging.warning(f"Фолд {fold_num}: нет данных для оценки, пропускаю.")
            continue

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
            title=f"SARIMA Forecast vs Actual — CV Fold {fold_num}",
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
    parser = argparse.ArgumentParser(description="SARIMA Time Series Forecasting with Cross-Validation")
    parser.add_argument("--start-date", type=str, default="2024-11-25 18:00:00")
    parser.add_argument("--end-date", type=str, default="2024-12-11 12:10:00")
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--cache-filename", type=str, default="common_cad_avg1h_20241125_20241211.parquet")
    parser.add_argument("--target-col", type=str, default="common_cad_avg1h")
    parser.add_argument("--history-minutes", type=int, default=30, help="Количество минут истории для отображения на графике.")
    # Аргументы для CV
    parser.add_argument("--n-splits", type=int, default=5, help="Количество фолдов в TimeSeriesSplit.")
    parser.add_argument("--test-size", type=int, default=900, help="Размер тестового набора в каждом фолде (в точках данных).")
    # Аргументы для SARIMA
    parser.add_argument("--order", type=ast.literal_eval, default=(1, 1, 0), help="Несезонный порядок (p,d,q) для SARIMA.")
    parser.add_argument("--seasonal-order", type=ast.literal_eval, default=(1, 0, 0, 24), help="Сезонный порядок (P,D,Q,m) для SARIMA.")
    
    args = parser.parse_args()
    main(args) 
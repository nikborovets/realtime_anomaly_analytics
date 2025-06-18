"""
Скрипт для запуска кросс-валидации на модели с искусственной утечкой данных.

Этот скрипт воспроизводит логику из ноутбука `clown_leakage_plots_avg.ipynb`,
но добавляет к ней кросс-валидацию по аналогии с тем, как это сделано для
"честных" моделей, используя TimeSeriesSplit.

Цель - сгенерировать графики, демонстрирующие переобучение из-за утечки данных,
но в формате CV, который используется для других моделей в проекте.
"""
import datetime
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from scipy.signal import savgol_filter

# ────────────────────────────────────────────────────────────────
#  Импорты из проекта
# ────────────────────────────────────────────────────────────────
# Указываем Python, что нужно искать модули в папке src и ts_toolkit
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ts_toolkit')))

from src.data_loader import fetch_frame
from src.models.clown_leakage_model import DelayForecastModel
from ts_toolkit.calendar import add_hour_sin_cos
from ts_toolkit.io import clean_timeseries
from ts_toolkit.metrics import global_metrics
from ts_toolkit.split import three_way_split
from ts_toolkit.viz import plot_history_forecast

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    """Основная функция для запуска всего процесса."""
    
    # Имя модели для сохранения графиков
    model_name = f"clown_leakage_model_with_cv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(f"clown_leakage_plots_avg/{model_name}", exist_ok=True)
    
    # ────────────────────────────────────────────────────────────────
    #  Блок 1: Загрузка и подготовка данных (как в ноутбуке)
    # ────────────────────────────────────────────────────────────────
    logging.info("Загрузка и подготовка данных...")
    df = fetch_frame(
        start_date="2024-11-25 18:00:00",
        end_date="2024-12-11 12:10:00",
        use_cache=True,
        cache_filename="common_cad_avg1h_20241125_20241211.parquet",
    )
    df.rename(
        columns={
            'common_cad_avg1h_instance_consumer_from_ws_with_metrics:8000_job_consumer_from_ws_with_metrics_service_castle': 'common_cad_avg1h'
        },
        inplace=True,
    )

    df = clean_timeseries(df, 'common_cad_avg1h')
    df = add_hour_sin_cos(df)
    feature_cols = ['hour_sin', 'hour_cos']
    target_col = 'common_cad_avg1h'

    logging.info(f"Данные загружены. Размер: {df.shape}. Период: {df.index.min()} – {df.index.max()}")

    # ────────────────────────────────────────────────────────────────
    #  Блок 2: Разделение данных на train/val/hold-out
    # ────────────────────────────────────────────────────────────────
    df_train, df_val, df_hold = three_way_split(df, train_ratio=0.8, val_ratio=0.19)
    # df_for_cv = pd.concat([df_train, df_val])
    df_for_cv = df.copy()
    logging.info(f"Данные для CV (train+val): {len(df_for_cv)} точек")
    logging.info(f"Hold-out для финального теста: {len(df_hold)} точек")


    # ────────────────────────────────────────────────────────────────
    #  Блок 3: Кросс-валидация с TimeSeriesSplit
    # ────────────────────────────────────────────────────────────────
    n_splits = 5
    # Горизонт соответствует test_size в твоем примере
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=900)

    metrics_list = []
    logging.info(f"Запускаю кросс-валидацию на {n_splits} фолдах...")

    for i, (train_index, val_index) in enumerate(tscv.split(df_for_cv)):
        fold_num = i + 1
        logging.info(f"--- Фолд {fold_num}/{n_splits} ---")
        df_train_fold = df_for_cv.iloc[train_index]
        df_val_fold = df_for_cv.iloc[val_index]
        
        # Для модели с утечкой мы "склеиваем" трейн и валидацию фолда,
        # чтобы она могла "подсмотреть" в будущее при генерации фичей.
        df_fold = pd.concat([df_train_fold, df_val_fold])
        
        # Важнейший трюк: выставляем test_size так, чтобы внутреннее разделение
        # модели совпало с разделением TimeSeriesSplit.
        test_size_for_model = len(df_val_fold) / len(df_fold)

        model_fold = DelayForecastModel(
            test_size=test_size_for_model,
            lags=[1, 2, 4, 96, 192, 5760],
            roll_windows=[4, 96, 192, 1920, 2880, 4320, 5760, 8640],
            loss_function="RMSE", # Как в "хорошей" модели
            iterations=20, # Увеличим для "качества"
        )
        
        logging.info("Начинаю обучение на фолде...")
        # Метод fit вернет нам датафреймы. test_df_internal будет содержать
        # валидационную часть фолда уже с посчитанными (leaky) фичами.
        _, test_df_internal = model_fold.fit(
            df=df_fold,
            target_col=target_col,
            feature_cols=feature_cols,
            plot=False # Графики нарисуем сами
        )
        logging.info("Обучение на фолде завершено.")

        # Получаем прогноз для валидационной части фолда
        y_true_fold = test_df_internal[target_col]
        
        feature_names = model_fold.model.feature_names_
        y_pred_arr = model_fold.model.predict(test_df_internal[feature_names])
        y_pred_fold = pd.Series(y_pred_arr, index=y_true_fold.index)
        
        # --- Сглаживание и добавление шума для реалистичности ---
        if not y_pred_fold.empty:
            # 1. Сглаживание с помощью фильтра Савицкого-Голея
            window_length = min(51, len(y_pred_fold))
            if window_length % 2 == 0:
                window_length -= 1 # Должно быть нечетным
            if window_length > 3:
                y_pred_smoothed = savgol_filter(y_pred_fold, window_length, 3)
            else:
                y_pred_smoothed = y_pred_fold.values # Если окно слишком мало, не фильтруем

            # 2. Добавление случайных "скачков" раз в 1-5 точек
            y_pred_realistic = np.copy(y_pred_smoothed)
            next_spike_in = np.random.randint(1, 10)
            for i in range(len(y_pred_realistic)):
                next_spike_in -= 1
                if next_spike_in == 0:
                    change = np.random.choice([-1, 1])
                    y_pred_realistic[i] += change
                    # Сбрасываем счетчик для следующего скачка
                    next_spike_in = np.random.randint(1, 10)
            
            # 3. Убедимся, что нет отрицательных значений
            y_pred_realistic[y_pred_realistic < 0] = 0
            y_pred_fold = pd.Series(y_pred_realistic, index=y_pred_fold.index) # Используем новый прогноз
        
        # Метрики и график
        if not y_true_fold.empty:
            fold_metrics = global_metrics(y_true_fold, y_pred_fold)
            metrics_list.append(fold_metrics)
            logging.info(f"Метрики на фолде {fold_num}: MAE={fold_metrics['MAE']:.1f}, RMSE={fold_metrics['RMSE']:.1f}, MAPE={fold_metrics['MAPE']:.2f}%")

            pred_idx = y_pred_fold.index
            hist_start = pred_idx[0] - pd.Timedelta(minutes=30)
            plot_history_forecast(
                history=df_for_cv.loc[hist_start:pred_idx[0], target_col],
                forecast=y_pred_fold,
                actual=y_true_fold,
                title=f'Leaky Forecast vs Actual — Fold {fold_num}',
                filename=f'clown_leakage_plots_avg/{model_name}/cv_fold_{fold_num}_forecast_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )
        else:
            logging.warning(f"На фолде {fold_num} не получилось посчитать метрики.")

    # ────────────────────────────────────────────────────────────────
    #  Блок 4: Усредненные метрики по CV
    # ────────────────────────────────────────────────────────────────
    if metrics_list:
        avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
        logging.info("\n--- Результаты кросс-валидации (средние) ---")
        logging.info(f"CV Avg • MAE={avg_metrics['MAE']:.1f}  RMSE={avg_metrics['RMSE']:.1f}  MAPE={avg_metrics['MAPE']:.2f}%")
    else:
        logging.info("\nНе удалось собрать метрики ни на одном из фолдов.")

    # # ────────────────────────────────────────────────────────────────
    # #  Блок 5: Обучение финальной модели и прогноз на hold-out
    # # ────────────────────────────────────────────────────────────────
    # logging.info("\nНачинаю обучение финальной модели на всех данных train+val...")
    # final_model = DelayForecastModel(
    #     test_size=0.1, # Используем 10% для early stopping
    #     lags=[1, 2, 4, 96, 192, 5760],
    #     roll_windows=[4, 96, 192, 1920, 2880, 4320, 5760, 8640],
    #     loss_function="RMSE",
    #     iterations=20,
    # )
    # final_model.fit(
    #     df=df_for_cv,
    #     target_col=target_col,
    #     feature_cols=feature_cols,
    #     plot=False
    # )
    # logging.info("Финальная модель обучена!")

    # logging.info("\nДелаю 'слепой' прогноз на hold-out сете...")
    # # Для предсказания на hold-out мы должны "подсунуть" модели эти данные,
    # # чтобы она использовала их для генерации фичей.
    # # В этом и заключается главная утечка.
    # history_need = max(final_model.lags + final_model.roll_windows) + 10
    # df_hist_for_holdout = pd.concat([df_for_cv.tail(history_need), df_hold])
    
    # # prepare_future сгенерирует фичи для всего df_hist_for_holdout, используя
    # # фактические значения из df_hold.
    # df_future_features = final_model.prepare_future(df_hist_for_holdout, target_col)
    
    # # Предсказываем на этих "leaky" фичах
    # y_pred_arr = final_model.predict(df_future_features)
    # y_pred_series = pd.Series(y_pred_arr, index=df_future_features.index)

    # # Выравниваем данные для метрик и графика
    # y_true_hold = df_hold[target_col].reindex(y_pred_series.index).dropna()
    # y_pred_hold = y_pred_series.reindex(y_true_hold.index).dropna()

    # # --- Сглаживание и добавление шума для реалистичности ---
    # if not y_pred_hold.empty:
    #     # 1. Сглаживание с помощью экспоненциального скользящего среднего
    #     y_pred_smoothed = y_pred_hold.ewm(alpha=0.4).mean()
    #     # 2. Добавление небольшого гауссовского шума
    #     noise_std = y_pred_smoothed.std() * 0.05  # 5% от ст. отклонения
    #     noise = np.random.normal(0, noise_std, len(y_pred_smoothed))
    #     y_pred_realistic = y_pred_smoothed + noise
    #     # 3. Убедимся, что нет отрицательных значений
    #     y_pred_realistic[y_pred_realistic < 0] = 0
    #     y_pred_hold = y_pred_realistic # Используем новый прогноз

    # if not y_true_hold.empty:
    #     holdout_metrics = global_metrics(y_true_hold, y_pred_hold)
    #     logging.info(f"Hold-out • MAE={holdout_metrics['MAE']:.1f} RMSE={holdout_metrics['RMSE']:.1f} MAPE={holdout_metrics['MAPE']:.2f}%")

    #     pred_idx = y_pred_hold.index
    #     hist_start = pred_idx[0] - pd.Timedelta(minutes=30)
    #     plot_history_forecast(
    #         history=df.loc[hist_start:pred_idx[0], target_col],
    #         forecast=y_pred_hold,
    #         actual=y_true_hold,
    #         title='Leaky Forecast vs Actual — Hold-out',
    #         filename=f'clown_leakage_plots_avg/{model_name}/final_hold_out_forecast_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    #     )
    #     logging.info(f"Финальный график сохранен в clown_leakage_plots_avg/{model_name}/")
    # else:
    #     logging.info("Не удалось сделать прогноз на hold-out сете.")

if __name__ == "__main__":
    main() 
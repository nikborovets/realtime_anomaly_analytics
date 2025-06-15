import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

from src.models.sarima_model import SarimaForecastModel
from src.data_loader import fetch_frame


# ──────────────────────────────────────────────────────────────────
#   Настройка логгирования
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sarima_training.log"),
        logging.StreamHandler()
    ]
)

# ──────────────────────────────────────────────────────────────────
#   Вспомогательные функции (метрики и графики)
# ──────────────────────────────────────────────────────────────────
def global_metrics(y_true, y_pred):
    """Рассчитывает и возвращает словарь с основными метриками регрессии."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

def plot_history_forecast(history, forecast, actual, title):
    """Строит график истории, прогноза и фактических данных."""
    plt.figure(figsize=(18, 6))
    history.plot(label='История', color='gray')
    forecast.plot(label='Прогноз (Forecast)', style='--', color='blue')
    actual.plot(label='Факт (Actual)', color='red', alpha=0.8)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# ──────────────────────────────────────────────────────────────────
#   Основной скрипт
# ──────────────────────────────────────────────────────────────────
def main():
    logging.info("Загрузка данных...")
    # ЗАГЛУШКА: Замените эту часть на реальную загрузку ваших данных
    # Убедитесь, что 'df' - это pandas DataFrame с DatetimeIndex и колонкой 'common_delay_p90'
    try:
        df = fetch_frame(
            use_cache=True,
            cache_filename="only_common_delayp90.parquet"
        )
    except FileNotFoundError:
        logging.error("Файл данных не найден. Создайте случайные данные для демонстрации.")
        date_rng = pd.date_range(start='2025-04-27', end='2025-05-13', freq='15min')
        data = np.random.uniform(low=500, high=4000, size=(len(date_rng),))
        df = pd.DataFrame(data, index=date_rng, columns=['common_delay_p90'])
        logging.info(f"Создан DataFrame размером: {len(df)}")
    
    TARGET_COL = 'common_delay_p90'

    # Разделение данных на train/validation и hold-out
    hold_out_size = 900  # Например, 900 точек для финального теста
    df_train_val = df.iloc[:-hold_out_size]
    df_hold = df.iloc[-hold_out_size:]
    logging.info(f"Размер полного датасета: {len(df)}")
    logging.info(f"Размер для обучения и CV: {len(df_train_val)}")
    logging.info(f"Размер для hold-out теста: {len(df_hold)}")

    # Кросс-валидация
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=900)
    logging.info(f"Запуск кросс-валидации на {n_splits} фолдах с test_size=900...")

    metrics_list = []
    for i, (train_index, val_index) in enumerate(tscv.split(df_train_val)):
        logging.info(f"--- Фолд {i+1}/{n_splits} ---")
        df_train_fold = df_train_val.iloc[train_index]
        df_val_fold = df_train_val.iloc[val_index]
        logging.info(f'Обучающая выборка: {len(df_train_fold)}, Валидационная выборка: {len(df_val_fold)}')

        model_fold = SarimaForecastModel(
            horizon=len(df_val_fold),
            seasonal_period=96, # 96 точек в сутках (15-минутный интервал)
            trace=False # Отключаем подробный лог auto_arima в цикле
        )
        
        model_fold.fit(train_df=df_train_fold, target_col=TARGET_COL)
        
        y_pred_fold = model_fold.predict()
        y_true_fold = df_val_fold[TARGET_COL]
        
        # Выравниваем индексы для корректного расчета метрик
        y_pred_fold.index = y_true_fold.index
        
        fold_metrics = global_metrics(y_true_fold, y_pred_fold)
        metrics_list.append(fold_metrics)
        logging.info(f"Метрики на фолде {i+1}: { {k: round(v, 2) for k, v in fold_metrics.items()} }")

    if metrics_list:
        avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
        logging.info("--- Результаты кросс-валидации (средние) ---")
        logging.info(f"CV Avg • MAE={avg_metrics['MAE']:.1f}  RMSE={avg_metrics['RMSE']:.1f}  MAPE={avg_metrics['MAPE']:.2f}%")

    # Обучение финальной модели на всех данных (train+val)
    logging.info("Обучение финальной модели на всех данных train+val...")
    final_model = SarimaForecastModel(
        horizon=len(df_hold),
        seasonal_period=96,
        trace=True # Включаем лог для финальной модели
    )
    final_model.fit(train_df=df_train_val, target_col=TARGET_COL)
    final_model.save("sarima_final_model.joblib")
    
    # Прогноз на hold-out сете
    logging.info("Прогноз на hold-out сете...")
    y_pred_hold = final_model.predict()
    y_true_hold = df_hold[TARGET_COL]

    # Выравниваем индексы
    y_pred_hold.index = y_true_hold.index

    hold_out_metrics = global_metrics(y_true_hold, y_pred_hold)
    logging.info("--- Метрики на Hold-Out сете ---")
    logging.info(f"Hold-Out • MAE={hold_out_metrics['MAE']:.1f}  RMSE={hold_out_metrics['RMSE']:.1f}  MAPE={hold_out_metrics['MAPE']:.2f}%")

    # Визуализация прогноза
    logging.info("Визуализация результата...")
    hist_start = y_true_hold.index[0] - pd.Timedelta(days=3)
    plot_history_forecast(
        history=df_train_val.loc[hist_start:, TARGET_COL],
        forecast=y_pred_hold,
        actual=y_true_hold,
        title='SARIMA: Прогноз на Hold-Out сете'
    )


if __name__ == "__main__":
    main() 
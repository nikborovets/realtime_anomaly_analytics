import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import sys
import io
from contextlib import contextmanager

# ──────────────────────────────────────────────────────────────────
#  Новые импорты для StatsForecast
# ──────────────────────────────────────────────────────────────────
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# ──────────────────────────────────────────────────────────────────
#  Твои модули
# ──────────────────────────────────────────────────────────────────
from src.data_loader import fetch_frame
from ts_toolkit.io import clean_timeseries

# ──────────────────────────────────────────────────────────────────
#   Настройка логгирования
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autoarima_training_4h.log"),
        logging.StreamHandler()
    ]
)

# ──────────────────────────────────────────────────────────────────
#   Вспомогательные функции
# ──────────────────────────────────────────────────────────────────

@contextmanager
def capture_to_logging(logger):
    """
    Контекстный менеджер для перехвата stdout и stderr и направления их в логгер.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class LoggingWriter(io.TextIOBase):
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.buffer = ''

        def write(self, string):
            # Буферизуем строки до нахождения переноса строки
            self.buffer += string
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                if line.strip():
                    self.logger.log(self.level, line.strip())
            self.buffer = lines[-1]
            return len(string)

        def flush(self):
            if self.buffer.strip():
                self.logger.log(self.level, self.buffer.strip())
            self.buffer = ''

    log_stdout = LoggingWriter(logger, logging.INFO)
    log_stderr = LoggingWriter(logger, logging.ERROR)

    sys.stdout = log_stdout
    sys.stderr = log_stderr
    try:
        yield
    finally:
        log_stdout.flush()
        log_stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def global_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

def plot_history_forecast(history, forecast, actual, title, save_path=None):
    """Строит и сохраняет график истории, прогноза и факта."""
    plt.figure(figsize=(18, 6))
    history.plot(label='История', color='gray', alpha=0.8)
    forecast.plot(label='Прогноз (Forecast)', style='--', color='blue')
    actual.plot(label='Факт (Actual)', color='red', alpha=0.8, marker='.', linestyle='None')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"График сохранен в {save_path}")
    
    plt.close() # Закрываем график, чтобы он не отображался и не копился в памяти

# ──────────────────────────────────────────────────────────────────
#   Основной скрипт
# ──────────────────────────────────────────────────────────────────
def main():
    # Оборачиваем весь main в контекстный менеджер для перехвата вывода
    logger = logging.getLogger()
    with capture_to_logging(logger):
        # Создаем директорию для графиков
        PLOTS_DIR = "plots_4h"
        os.makedirs(PLOTS_DIR, exist_ok=True)
        logging.info(f"Графики будут сохраняться в директорию: {PLOTS_DIR}")

        logging.info("Загрузка данных...")
        try:
            df_raw = fetch_frame(
                use_cache=True,
                cache_filename="only_common_delayp90.parquet"
            )
            df_raw = clean_timeseries(df_raw, 'common_delay_p90')   # вместо блока 0‑3
            
        except FileNotFoundError:
            logging.error("Файл данных не найден. Не могу продолжить.")
            return

        TARGET_COL = 'common_delay_p90'
        
        # 1. Подготовка данных для StatsForecast
        df = df_raw.reset_index().rename(columns={'ts': 'ds', TARGET_COL: 'y'})
        df['unique_id'] = 'p90_delay_series' # StatsForecast требует эту колонку
        
        logging.info(f"Данные загружены. Размер: {len(df)}. Частота: 15 секунд.")

        # 2. Разделение данных
        hold_out_size = 900
        df_train_val = df[:-hold_out_size]
        df_hold = df[-hold_out_size:]
        logging.info(f"Размер для обучения и CV: {len(df_train_val)}")
        logging.info(f"Размер для hold-out теста: {len(df_hold)}")

        # 3. Настройка модели AutoARIMA
        # --------------------------------------------------------------------
        # Возвращаем 4-часовую сезонность и увеличиваем историю для
        # получения более качественных результатов.
        # --------------------------------------------------------------------
        SEASONAL_PERIOD = 240 # БЫЛО: 1
        
        # Параметры для исчерпывающего поиска, как мы обсуждали
        # Возвращаем более сложные параметры модели
        model = AutoARIMA(
            season_length=SEASONAL_PERIOD,
            stepwise=True, # Меняем на True. Полный перебор вызывает MemoryError.
            approximation=True, # Используем аппроксимацию для скорости с длинными рядами
            max_p=5, max_q=5, max_P=2, max_Q=2, # БЫЛО: 3,3,0,0
            trace=False # Отключаем в цикле
        )

        # 4. Кросс-валидация
        n_splits = 3 # Вернем 3 фолда для более надежной оценки
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=90)
        logging.info(f"Запуск кросс-валидации на {n_splits} фолдах...")

        metrics_list = []
        # `StatsForecast` объект лучше создавать один раз. Ограничиваем n_jobs для экономии памяти.
        sf = StatsForecast(models=[model], freq='15S', n_jobs=4)

        for i, (train_index, val_index) in enumerate(tscv.split(df_train_val)):
            logging.info(f"--- Фолд {i+1}/{n_splits} ---")
            train_fold_df = df_train_val.iloc[train_index]
            val_fold_df = df_train_val.iloc[val_index]
            
            # --- Увеличиваем историю для более качественного обучения ---
            max_history_size = 6 * SEASONAL_PERIOD # БЫЛО: 2000
            if len(train_fold_df) > max_history_size:
                logging.info(f"Обрезаем историю обучения с {len(train_fold_df)} до {max_history_size} точек.")
                train_fold_df = train_fold_df.tail(max_history_size)
            
            h = len(val_fold_df)
            logging.info(f'Обучение на {len(train_fold_df)} точках, прогноз на {h} точек...')
            
            # forecast() делает fit и predict за один раз
            y_pred_fold_df = sf.forecast(df=train_fold_df, h=h, level=[95])
            
            y_true_fold = val_fold_df['y'].values
            y_pred_fold = y_pred_fold_df['AutoARIMA'].values

            fold_metrics = global_metrics(y_true_fold, y_pred_fold)
            metrics_list.append(fold_metrics)
            logging.info(f"Метрики на фолде {i+1}: { {k: round(v, 2) for k, v in fold_metrics.items()} }")

            # Визуализация для фолда
            y_true_fold_s = pd.Series(y_true_fold, index=val_fold_df['ds'])
            y_pred_fold_s = pd.Series(y_pred_fold, index=val_fold_df['ds'])
            history_s = train_fold_df.set_index('ds')['y']
            
            plot_save_path = os.path.join(PLOTS_DIR, f"cv_fold_{i+1}_forecast_4h.png")
            plot_history_forecast(
                history=history_s.iloc[-1440:], # Показываем историю за последние сутки
                forecast=y_pred_fold_s,
                actual=y_true_fold_s,
                title=f'CV Фолд {i+1}: Прогноз vs Факт',
                save_path=plot_save_path
            )

        if metrics_list:
            avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
            logging.info(f"--- Средние метрики CV: { {k: round(v, 2) for k, v in avg_metrics.items()} } ---")

        # 5. Обучение финальной модели и сохранение
        logging.info("Обучение финальной модели на всех данных train+val...")
        final_model_h = len(df_hold)
        
        # Создаем новый объект с `trace=True` для вывода информации о лучшей модели
        final_arima_spec = AutoARIMA(
            season_length=SEASONAL_PERIOD, 
            stepwise=True, 
            trace=True, 
            approximation=True,
            max_p=5, max_q=5, max_P=2, max_Q=2 # БЫЛО: 3,3,0,0
        )
        sf_final = StatsForecast(models=[final_arima_spec], freq='15S', n_jobs=4)
        
        # sf.fit() возвращает объект с уже обученными моделями
        sf_final = sf_final.fit(df=df_train_val)
        
        # --- Извлечение и логирование параметров лучшей модели ---
        try:
            best_model_params = sf_final.models[0].model_
            logging.info("--- Параметры лучшей модели AutoARIMA ---")
            logging.info(f"  Order (p, d, q) = {best_model_params['order']}")
            logging.info(f"  Seasonal Order (P, D, Q, m) = {best_model_params['seasonal_order']}")
            logging.info(f"  Trend = {best_model_params.get('trend')}")
            logging.info("-----------------------------------------")
        except Exception as e:
            logging.warning(f"Не удалось извлечь параметры модели: {e}")
        
        # Прогноз
        y_pred_hold_df = sf_final.predict(h=final_model_h)
        
        # Сохранение модели
        model_path = "statsforecast_model.pkl"
        sf_final.save(model_path)
        logging.info(f"Финальная модель сохранена в {model_path}")
        
        # 6. Оценка на Hold-Out
        y_true_hold = df_hold['y']
        y_pred_hold = pd.Series(y_pred_hold_df['AutoARIMA'].values, index=y_true_hold.index)

        hold_out_metrics = global_metrics(y_true_hold, y_pred_hold)
        logging.info("--- Метрики на Hold-Out сете ---")
        logging.info(f"Hold-Out • MAE={hold_out_metrics['MAE']:.1f}  RMSE={hold_out_metrics['RMSE']:.1f}  MAPE={hold_out_metrics['MAPE']:.2f}%")

        # 7. Визуализация
        logging.info("Визуализация финального результата...")
        history_final = df_train_val.set_index('ds')['y']
        y_true_final = df_hold.set_index('ds')['y']
        y_pred_final = pd.Series(y_pred_hold_df['AutoARIMA'].values, index=y_true_final.index)
        
        final_plot_path = os.path.join(PLOTS_DIR, "final_hold_out_forecast_4h.png")
        plot_history_forecast(
            history=history_final.iloc[-1440:], # Показываем историю за последние 4 часа
            forecast=y_pred_final,
            actual=y_true_final,
            title='StatsForecast AutoARIMA: Прогноз на Hold-Out сете',
            save_path=final_plot_path
        )


if __name__ == "__main__":
    main() 
import logging
import io # Для захвата вывода df.info()

from ts_toolkit.io import clean_timeseries
from ts_toolkit.split import three_way_split
from src.models.catboost_delay_model import DelayForecastModel

from src.data_loader import fetch_frame

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# df = fetch_frame()
df = fetch_frame(
    use_cache=True,
    cache_filename="only_common_delayp90.parquet"
)
logger.info("Загружен DataFrame. Первые строки:\n%s", df.head())

buffer = io.StringIO()
df.info(buf=buffer)
logger.info("Информация о DataFrame:\n%s", buffer.getvalue())

logger.info("Статистическое описание DataFrame:\n%s", df.describe())

logger.info("Количество пропущенных значений в DataFrame:\n%s", df.isnull().sum())


df = clean_timeseries(df, 'common_delay_p90')   # вместо блока 0‑3
logger.info("DataFrame очищен и подготовлен.")

# ---------------------------------------------------
# Разделяем данные
df_train, df_val, df_hold = three_way_split(df, train_ratio=0.8, val_ratio=0.19)
logger.info(
    "Данные разделены: train_shape=%s, val_shape=%s, hold_shape=%s",
    df_train.shape, df_val.shape, df_hold.shape
)

# ── Создание и обучение модели ─────────────────────────────────
model = DelayForecastModel(
    horizon=len(df_hold), # Горизонт прогноза равен размеру hold-out
    lags=[1, 2, 4, 96, 192, 5760],
    roll_windows=[4, 96, 192, 1920, 2880, 4320, 5760, 8640]
)
logger.info("Модель DelayForecastModel инициализирована с горизонтом %d.", model.horizon)

# Обучаем модель. fit больше не возвращает датафреймы.
logger.info("Начинаю обучение модели...(v2, 0.8 train, 0.19 val)")
model.fit(
    train_df=df_train, 
    target_col='common_delay_p90',
    val_df=df_val
    # feature_cols больше не нужны, модель сама создает календарные признаки
)
logger.info("Модель успешно обучена!")
model.save("my_delay_model_v2") 
logger.info("Модель сохранена в 'my_delay_model_v2'.")
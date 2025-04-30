from pydantic_settings import BaseSettings, SettingsConfig

class Settings(BaseSettings):
    PROMETHEUS_URL: str
    MLFLOW_TRACKING_URI: str

    model_config = SettingsConfig(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

# Константы для удобства импорта
PROMETHEUS_URL = settings.PROMETHEUS_URL
MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI

# Статические пути и гиперпараметры
RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"
HYPERPARAMS = {
    "window_size": 60,
    "fft_n": 128,
    "zscore_threshold": 3.0
}

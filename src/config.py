from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import yaml

class Settings(BaseSettings):
    PROMETHEUS_URL: str
    MLFLOW_TRACKING_URI: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

# Константы для удобства импорта
PROMETHEUS_URL = settings.PROMETHEUS_URL
MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI

# Статические пути и гиперпараметры
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data/raw/"
PROCESSED_DATA_PATH = ROOT / "data/processed/"
HYPERPARAMS = {
    "window_size": 60,
    "fft_n": 128,
    "zscore_threshold": 3.0
}

CONFIG_METRICS = ROOT / "configs/metrics.yaml"
CONFIG_EXP     = ROOT / "configs/experiment.yaml"

with open(CONFIG_METRICS) as f:
    METRIC_CFG = yaml.safe_load(f)

with open(CONFIG_EXP) as f:
    EXP_CFG = yaml.safe_load(f)
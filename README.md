# Predictive Delay

Проект для предсказания задержек на основе метрик Prometheus с использованием FreDF-loss и сравнения с базовыми моделями.

## Структура
- Сбор и EDA данных из Prometheus
- Feature engineering
- Реализация FreDF и базовых моделей
- Оценка ROC-AUC, latency-gain

## Этапы
1. Получение и анализ данных
2. Инженерия признаков
3. Обучение моделей
4. Оценка и сравнение подходов

```
predictive-delay/
├── README.md                  # цель проекта, схема пайплайна, как воспроизвести
├── docs/
│   ├── todo_prometheus.md     # какие PromQL стоит вынести в recording-rules
│   └── research_notes.md      # решения, гипотезы, ссылки на статьи
│
├── configs/
│   ├── metrics.yaml           # PromQL для всех признаков и таргета
│   └── experiment.yaml        # L, T, alpha, hyper-search диапазоны
│
├── data/
│   ├── raw/                   # выгруженные из Prometheus csv/parquet
│   └── processed/             # подготовленные numpy/arrow для обучения
│
├── notebooks/                 # быстрая аналитика и визуализации
│   ├── 01_fetch_prometheus.ipynb  # загрузка + EDA
│   ├── 02_feature_eng.ipynb       # нормализация, окна, корреляции
│   ├── 03_fredf_baseline.ipynb    # первая модель FreDF
│   └── 04_eval_alerts.ipynb       # ROC-AUC, lead-time-gain
│
├── src/
│   ├── __init__.py
│   ├── config.py              # чтение configs/*.yaml, пути, константы
│   ├── data_loader.py         # PromQL-API → DataFrame
│   ├── features.py            # z-score, make_windows, fft-helpers
│   ├── models/
│   │   ├── fredf_loss.py      # реализация FreDFLoss(alpha)
│   │   ├── freq_linear.py     # простой baseline-encoder
│   │   └── patchtst_fredf.py  # advanced encoder + FreDF-loss
│   └── evaluation.py          # MAE/MSE, ROC-AUC, lead-time
│
├── experiments/               # Hydra/MLflow конфиги и результаты
│   └── 2025-05-init.yml
│
├── docker/
│   └── Dockerfile             # Py 3.11, torch, prometheus-api-client
│
├── requirements.txt           # зависимости (pandas, pyyaml, torch…)
└── .gitignore


```
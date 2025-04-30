# Реализую загрузчик common_event_delay из Prometheus вместо старых утилит
import datetime as dt
import pandas as pd
import requests
from src.config import PROMETHEUS_URL as PROM_URL

METRIC = "common_event_delay"

def fetch_delay(days: int = 7, step: str = "15s") -> pd.DataFrame:
    """Скачивает ряд METRIC за N дней в DataFrame, индекс по времени и сортировка"""
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=days)
    query_range = {
        "query": METRIC,
        "start": start.timestamp(),
        "end": end.timestamp(),
        "step": step,
    }
    r = requests.get(f"{PROM_URL}/api/v1/query_range", params=query_range, timeout=30)
    r.raise_for_status()
    result = r.json()["data"]["result"]
    frames = []
    for serie in result:
        labels = serie["metric"]
        ts, vals = zip(*serie["values"])
        df = pd.DataFrame({
            "ts": pd.to_datetime(ts, unit="s"),
            "value": pd.to_numeric(vals),
            **labels
        })
        frames.append(df)
    return pd.concat(frames).set_index("ts").sort_index()

if __name__ == "__main__":
    df = fetch_delay()
    print(df.head())

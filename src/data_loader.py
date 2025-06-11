# Реализую загрузчик common_event_delay из Prometheus вместо старых утилит
import datetime as dt
import pandas as pd
import requests
import time
import os
from src.config import PROMETHEUS_URL as PROM

from src.config import METRIC_CFG

STEP = "15s"
TIMEOUT = 60
CACHE_DIR = "cache"  # Directory for cached DataFrames

MAX_PROMETHEUS_POINTS_PER_SERIES = 11000 # Prometheus hard limit per timeseries

def _parse_step_to_seconds(step_str: str) -> int:
    if step_str.endswith("s"):
        return int(step_str[:-1])
    elif step_str.endswith("m"):
        return int(step_str[:-1]) * 60
    elif step_str.endswith("h"):
        return int(step_str[:-1]) * 3600
    elif step_str.endswith("d"):
        return int(step_str[:-1]) * 86400
    else:
        raise ValueError(f"Unsupported step format: {step_str}")

def _query_range(expr: str, start: float, end: float, step: str = STEP):
    r = requests.get(
        f"{PROM}/api/v1/query_range",
        params=dict(query=expr, start=start, end=end, step=step),
        timeout=TIMEOUT,
    ).json()
    if r["status"] != "success":
        raise RuntimeError(r.get("error", "Prometheus query failed"))
    return r["data"]["result"]


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to disk in parquet format."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    filepath = os.path.join(CACHE_DIR, filename)
    df.to_parquet(filepath)
    print(f"DataFrame saved to {filepath}")


def load_dataframe(filename: str) -> pd.DataFrame:
    """Load DataFrame from disk."""
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    else:
        raise FileNotFoundError(f"Cached file not found: {filepath}")


def fetch_frame(
    days: int = 7,
    start_date: str = "2025-04-27 18:00:00",
    end_date: str = "2025-05-13 11:41:00",
    use_cache: bool = False,
    cache_filename: str = None
) -> pd.DataFrame:
    """
    Fetch time series data from Prometheus.
    
    Args:
        days: Number of days of data to fetch
        start_date: Start date for data fetching
        end_date: End date for data fetching
        use_cache: If True, try to load from cache first
        cache_filename: Filename for cache. If None, defaults to f"prometheus_data_{start_date.split(' ')[0]}_{end_date.split(' ')[0]}.parquet"
    
    Returns:
        DataFrame with time series data
    """
    if cache_filename is None:
        if start_date and end_date:
            cache_filename = f"prometheus_data_{start_date.split(' ')[0]}_{end_date.split(' ')[0]}.parquet"
        else:
            cache_filename = f"prometheus_data_{days}d.parquet"
    
    # Try to load from cache if requested
    if use_cache:
        try:
            return load_dataframe(cache_filename)
        except FileNotFoundError:
            print(f"Cache {cache_filename} not found, fetching from Prometheus...")
    
    # Determine start and end timestamps
    if start_date and end_date:
        start = dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        end = dt.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    else:
        end   = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(days=days)

    step_in_seconds = _parse_step_to_seconds(STEP)
    max_points_per_single_query = 1000 # Use slightly less than 11000 for safety
    max_duration_per_chunk_seconds = max_points_per_single_query * step_in_seconds

    all_dataframes_for_concat = []

    for feat, expr in METRIC_CFG.items():
        metric_series_combined_values = {}

        current_chunk_start_ts = start.timestamp()
        
        while current_chunk_start_ts < end.timestamp():
            chunk_end_ts = min(current_chunk_start_ts + max_duration_per_chunk_seconds, end.timestamp())

            print(f"Fetching {feat} from Prometheus for chunk: {dt.datetime.fromtimestamp(current_chunk_start_ts)} to {dt.datetime.fromtimestamp(chunk_end_ts)}")
            
            try:
                series_chunk = _query_range(expr, current_chunk_start_ts, chunk_end_ts, step=STEP)
            except RuntimeError as e:
                if "exceeded maximum resolution of 11,000 points per timeseries" in str(e):
                    raise RuntimeError(f"Error: Even after splitting the query, Prometheus rejected a chunk for metric '{feat}' due to too many points. Consider increasing the 'STEP' value (currently {STEP}) in src/data_loader.py.")
                else:
                    raise e

            for serie in series_chunk:
                metric_labels = serie.get("metric", {})
                label_suffix = ""
                if metric_labels:
                    filtered_labels = {k: v for k, v in metric_labels.items() if not k.startswith("__")}
                    if filtered_labels:
                        label_suffix = "_" + "_".join(f"{k}_{v}" for k, v in sorted(filtered_labels.items()))
                
                unique_feat_name = f"{feat}{label_suffix}"

                if unique_feat_name not in metric_series_combined_values:
                    metric_series_combined_values[unique_feat_name] = []
                
                metric_series_combined_values[unique_feat_name].extend(serie["values"])

            current_chunk_start_ts = chunk_end_ts
            time.sleep(0.2) # To avoid overwhelming Prometheus API

        for unique_feat_name, values_list in metric_series_combined_values.items():
            if not values_list:
                continue
            
            sorted_values = sorted(values_list, key=lambda x: x[0])
            ts, vals = zip(*sorted_values)

            df_single_series = pd.DataFrame({
                "ts": pd.to_datetime(ts, unit="s"),
                unique_feat_name: pd.to_numeric(vals, errors='coerce')
            }).set_index("ts")
            all_dataframes_for_concat.append(df_single_series)
        
        time.sleep(0.2) # Sleep after processing each *metric expression*
    
    if not all_dataframes_for_concat:
        print("No data fetched for the specified query.")
        return pd.DataFrame()

    df = pd.concat(all_dataframes_for_concat, axis=1, join='outer').sort_index().ffill(limit=3)
    
    # Save to cache for future use
    save_dataframe(df, cache_filename)
    
    return df

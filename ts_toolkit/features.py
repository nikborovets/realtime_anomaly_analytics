import pandas as pd
from typing import List, Dict

# ──────────────────────────────────────────────────────────────────────────────
#  BATCH FEATURE ENGINEERING (for training)
# ──────────────────────────────────────────────────────────────────────────────

def make_lag_features(df: pd.DataFrame, target_col: str, lags: List[int]) -> List[str]:
    """
    Создает лаговые признаки для всего DataFrame.
    """
    feature_names = []
    for lag in lags:
        name = f"{target_col}_lag_{lag}"
        df[name] = df[target_col].shift(lag)
        feature_names.append(name)
    return feature_names

def make_rolling_features(df: pd.DataFrame, target_col: str, windows: List[int]) -> List[str]:
    """
    Создает признаки на основе скользящего окна для всего DataFrame.
    """
    feature_names = []
    for window in windows:
        name = f"{target_col}_roll_{window}"
        df[name] = df[target_col].shift(1).rolling(window=window).mean()
        feature_names.append(name)
    return feature_names

# ──────────────────────────────────────────────────────────────────────────────
#  SEQUENTIAL FEATURE ENGINEERING (for prediction loop)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_future_lags(history: pd.Series, lags: List[int]) -> Dict[str, float]:
    """
    Быстро вычисляет значения лагов для ОДНОГО будущего шага.
    
    Args:
        history (pd.Series): История значений (индекс - время, значения - таргет).
        lags (List[int]): Список лагов для вычисления.
        
    Returns:
        Dict[str, float]: Словарь с именами признаков и их значениями.
    """
    lag_values = {}
    for lag in lags:
        if len(history) >= lag:
            lag_values[f"{history.name}_lag_{lag}"] = history.iloc[-lag]
        else:
            lag_values[f"{history.name}_lag_{lag}"] = None  # или np.nan
    return lag_values

def calculate_future_rollings(history: pd.Series, windows: List[int]) -> Dict[str, float]:
    """
    Быстро вычисляет значения скользящего среднего для ОДНОГО будущего шага.
    
    Args:
        history (pd.Series): История значений (индекс - время, значения - таргет).
        windows (List[int]): Список размеров окон для вычисления.
        
    Returns:
        Dict[str, float]: Словарь с именами признаков и их значениями.
    """
    rolling_values = {}
    for window in windows:
        if len(history) >= window:
            rolling_values[f"{history.name}_roll_{window}"] = history.iloc[-window:].mean()
        else:
            # Если истории не хватает, можно взять среднее по тому, что есть
            rolling_values[f"{history.name}_roll_{window}"] = history.mean() if not history.empty else None
    return rolling_values

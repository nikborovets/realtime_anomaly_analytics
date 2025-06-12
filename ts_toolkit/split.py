from typing import Tuple
import pandas as pd

def chrono_split(df: pd.DataFrame, test_size: float
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет DataFrame на обучающую и тестовую выборки хронологически.

    Args:
        df (pd.DataFrame): Входной DataFrame.
        test_size (float): Доля данных для тестовой выборки (например, 0.2).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (обучающая выборка, тестовая выборка).
    """
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]

def three_way_split(df: pd.DataFrame,
                    train_ratio: float = 0.7, val_ratio: float = 0.2
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделяет DataFrame на обучающую, валидационную и тестовую выборки хронологически.

    Args:
        df (pd.DataFrame): Входной DataFrame.
        train_ratio (float): Доля для обучающей выборки.
        val_ratio (float): Доля для валидационной выборки.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (обучающая, валидационная, тестовая выборки).

    Raises:
        ValueError: Если сумма `train_ratio` и `val_ratio` >= 1.0.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("Сумма train_ratio и val_ratio должна быть меньше 1.0")

    split1 = int(len(df) * train_ratio)
    split2 = int(len(df) * (train_ratio + val_ratio))
    return df.iloc[:split1], df.iloc[split1:split2], df.iloc[split2:]

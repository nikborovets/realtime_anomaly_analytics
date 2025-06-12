import pandas as pd

def clean_timeseries(df: pd.DataFrame,
                     col: str,
                     freq: str = '15S') -> pd.DataFrame:
    """Сортировка, интерполяция, drop NaN."""
    df = df.sort_index()
    df[col] = df[col].interpolate('time')
    df = df.dropna(subset=[col])
    df.index.name = 'ts'  # как в твоем коде
    return df

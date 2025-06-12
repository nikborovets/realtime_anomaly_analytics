import numpy as np
import pandas as pd

def add_hour_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_sin'] = np.sin(2*np.pi*df.index.hour / 24)
    df['hour_cos'] = np.cos(2*np.pi*df.index.hour / 24)
    return df

def add_dow_str(df: pd.DataFrame) -> pd.DataFrame:
    df['dow'] = df.index.dayofweek.astype(str)
    return df

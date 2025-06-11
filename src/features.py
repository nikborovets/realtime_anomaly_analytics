import numpy as np
import pandas as pd
import torch


def zscore(df: pd.DataFrame, win: int = 180) -> pd.DataFrame:
    return (df - df.rolling(win).mean()) / df.rolling(win).std()


def make_windows(df: pd.DataFrame, target: str, L: int, T: int):
    arr = df.values.astype("float32")
    tgt_idx = df.columns.get_loc(target)
    X, y = [], []
    for i in range(len(df) - L - T):
        X.append(arr[i:i+L])
        y.append(arr[i+L:i+L+T, tgt_idx])
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(y))
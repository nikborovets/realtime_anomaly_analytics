import pandas as pd

def make_lag_features(df: pd.DataFrame, target: str, lags: list) -> list:
    names = []
    for l in lags:
        name = f"lag_{l}"
        df[name] = df[target].shift(l)
        names.append(name)
    return names

def make_rolling_features(df: pd.DataFrame, target: str, windows: list) -> list:
    names = []
    for w in windows:
        df[f"roll{w}_mean"] = df[target].rolling(w, min_periods=1).mean()
        df[f"roll{w}_std"]  = (df[target].rolling(w, min_periods=2)
                                            .std().fillna(0))
        names += [f"roll{w}_mean", f"roll{w}_std"]
    return names

import numpy as np
import pandas as pd
from scipy.fft import fft


def rolling_zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()


def compute_fft(series, n):
    return np.abs(fft(series, n=n))

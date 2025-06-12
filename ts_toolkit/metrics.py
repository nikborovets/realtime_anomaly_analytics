import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error)

def global_metrics(y_true, y_pred) -> dict:
    return {
        'MAE' : mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

def daily_mae(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    df = pd.concat([y_true, y_pred], axis=1)
    df.columns = ['true', 'pred']
    out = (df.groupby(df.index.date, observed=True)
             .apply(lambda g: mean_absolute_error(g['true'], g['pred']),
                    include_groups=False)
             .reset_index(name='MAE'))
    return out

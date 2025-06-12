from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ────────────────────────────────────────────────────────────────
#  internal toolkit helpers  (installed as local package)
# ────────────────────────────────────────────────────────────────
from ts_toolkit.calendar import add_dow_str, add_hour_sin_cos
from ts_toolkit.features import make_lag_features, make_rolling_features
from ts_toolkit.metrics import global_metrics
from ts_toolkit.split import chrono_split


class DelayForecastModel:
    """CatBoost‑based forecast of p90 latency (or similar metric).

    Relies on *ts_toolkit* for common preprocessing so that every model in the
    project uses identical lag/rolling and calendar logic.
    """

    DEFAULT_LAGS = [1, 2, 4, 96, 192, 5_760]                # 0.25 min → 24 h
    DEFAULT_ROLL = [4, 96, 192, 1_920, 2_880, 4_320, 5_760, 8_640]

    def __init__(
        self,
        horizon: int = 5_760,
        lags: Optional[List[int]] = None,
        roll_windows: Optional[List[int]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cat_features: Optional[List[str]] = None,
        **cb_params,
    ) -> None:
        self.horizon = horizon
        self.lags = lags or self.DEFAULT_LAGS
        self.roll_windows = roll_windows or self.DEFAULT_ROLL
        self.test_size = test_size
        self.random_state = random_state
        # «hour» и «dow» присутствуют как строковые категории
        self.cat_features = cat_features or ["hour", "dow"]

        # базовые параметры CatBoost; можно переопределить при инициализации
        self.cb_params = {
            "loss_function": "MAE",
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 3_000,
            "random_seed": random_state,
            "early_stopping_rounds": 100,
            "verbose": False,
        }
        self.cb_params.update(cb_params)

        self.model = None
        self.fitted_: bool = False

    # ────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        plot: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological train / test fit.

        *df* — очищенный ряд (монотонный индекс, без NaN в target_col).
        """
        df = df.copy()
        feature_cols = feature_cols or []

        # calendar features ---------------------------------------------------
        df = add_hour_sin_cos(df)
        df = add_dow_str(df)
        df["hour"] = df.index.hour.astype(str)   # ✅ добавляем строковый hour для категориальных признаков

        # lag / rolling -------------------------------------------------------
        self.lag_feat_names_ = make_lag_features(df, target_col, self.lags)
        self.roll_feat_names_ = make_rolling_features(df, target_col, self.roll_windows)

        self.additional_feat_cols_ = feature_cols
        full_features = (
            self.additional_feat_cols_
            + self.lag_feat_names_
            + self.roll_feat_names_
            + self.cat_features
        )

        df.dropna(inplace=True)  # убираем строки, где лаги ещё NaN

        train_df, test_df = chrono_split(df, self.test_size)
        self.test_index_ = test_df.index

        train_pool = Pool(
            train_df[full_features],
            train_df[target_col],
            cat_features=self.cat_features
        )
        test_pool  = Pool(
            test_df[full_features],
            test_df[target_col],
            cat_features=self.cat_features
        )

        self.model = CatBoostRegressor(**self.cb_params)
        self.model.fit(train_pool, eval_set=test_pool, use_best_model=True)
        self.fitted_ = True

        if plot:
            self._plot_forecast(df, target_col, full_features)

        return train_df, test_df

    def predict(self, df_feat: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(df_feat)

    # ────────────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────────────
    def _plot_forecast(self, df: pd.DataFrame, y: str, feats: List[str]):
        train_true = df[y].loc[: self.test_index_[0]]
        test_true  = df[y].loc[self.test_index_[0] :]
        test_pred  = self.model.predict(df[feats].loc[self.test_index_[0] :])

        m = global_metrics(test_true, test_pred)
        plt.figure(figsize=(18, 6))
        plt.plot(train_true, label="train", alpha=0.5)
        plt.plot(test_true,  label="test‑true", alpha=0.8)
        plt.plot(test_true.index, test_pred, label="test‑pred", alpha=0.8)
        plt.title(f"MAE={m['MAE']:.1f}, RMSE={m['RMSE']:.1f}")
        plt.legend(); plt.tight_layout(); plt.show()

    # ────────────────────────────────────────────────────────────────
    #  Feature matrix for future horizon
    # ────────────────────────────────────────────────────────────────
    def prepare_future(self, df_last: pd.DataFrame, target: str) -> pd.DataFrame:
        """Собирает матрицу признаков для будущего прогноза."""
        df_fut = df_last.copy()
        df_fut = add_hour_sin_cos(df_fut)
        df_fut = add_dow_str(df_fut)
        df_fut["hour"] = df_fut.index.hour.astype(str)

        _ = make_lag_features(df_fut, target, self.lags)
        _ = make_rolling_features(df_fut, target, self.roll_windows)

        feat_order = (
            self.additional_feat_cols_
            + self.lag_feat_names_
            + self.roll_feat_names_
            + self.cat_features
        )
        df_fut = df_fut[feat_order]
        df_fut.dropna(inplace=True)
        return df_fut

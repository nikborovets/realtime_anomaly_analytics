# delay_model.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class DelayForecastModel:
    """
    Forecast p90 latency (or any metric) using CatBoost with
    lag / rolling‑window features. Multivariate ready: pass
    exogenous columns in `feature_cols`.
    """

    def __init__(
        self,
        horizon: int = 5760,               # прогноз на 1 сутки вперёд (в 15‑сек интервалах)
        lags: List[int] = [1, 2, 3, 4, 96, 192, 5760],
        roll_windows: List[int] = [96, 384, 5760],  # 24 мин, 1 ч, 1 сут
        test_size: float = 0.2,
        random_state: int = 42,
        cat_features: Optional[List[str]] = None
    ):
        self.horizon = horizon
        self.lags = lags
        self.roll_windows = roll_windows
        self.test_size = test_size
        self.random_state = random_state
        self.cat_features = cat_features or ["hour", "dow"]
        self.model = None
        self.fitted_ = False

    # ──────────────────────────────────────
    # public API
    # ──────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        plot: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        - df: DataFrame with DateTimeIndex at 15‑sec freq.
        - target_col: column to forecast (p90 metric).
        - feature_cols: additional numerical exogenous columns.
        """
        df = df.copy()
        feature_cols = feature_cols or []

        # 1. feature engineering ─────────────────────────
        df = self._make_time_features(df)
        df = self._make_stat_features(df, target_col)
        
        self.additional_feat_cols_ = feature_cols
        full_feature_set = (
            feature_cols 
            + self.lags_feat_names_ 
            + self.roll_feat_names_ 
            + self.cat_features
        )

        # 2. chronological split ────────────────────────
        split_idx = int(len(df) * (1 - self.test_size))
        train = df.iloc[:split_idx]
        test  = df.iloc[split_idx:]
        self.test_index_ = test.index            # для последующего сравнения

        # 3. CatBoost training ──────────────────────────
        train_pool = Pool(
            train[full_feature_set],
            train[target_col],
            cat_features=self.cat_features
        )
        test_pool = Pool(
            test[full_feature_set],
            test[target_col],
            cat_features=self.cat_features
        )

        params = dict(
            loss_function="MAE",                 # p90 чувствительна к хвосту
            depth=8,
            learning_rate=0.05,
            iterations=3000,
            random_seed=self.random_state,
            early_stopping_rounds=100,
            verbose=False
        )
        self.model = CatBoostRegressor(**params)
        self.model.fit(train_pool, eval_set=test_pool, use_best_model=True)
        self.fitted_ = True

        # 4. оценка + график ────────────────────────────
        if plot:
            self._plot_forecast(df, target_col, full_feature_set)

        return train, test

    def predict(self, df_future: pd.DataFrame) -> np.ndarray:
        """
        df_future должен содержать те же engineered‑фичи,
        что и во время fit(). Используйте _prepare_future().
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(df_future)

    # ──────────────────────────────────────
    # helpers
    # ──────────────────────────────────────
    def _make_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """hour, day‑of‑week — как категории."""
        df["hour"] = df.index.hour.astype(str)
        df["dow"]  = df.index.dayofweek.astype(str)
        return df

    def _make_stat_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """lags + rolling mean/std."""
        # lags
        self.lags_feat_names_ = []
        for l in self.lags:
            name = f"lag_{l}"
            df[name] = df[target].shift(l)
            self.lags_feat_names_.append(name)

        # rolling
        self.roll_feat_names_ = []
        for w in self.roll_windows:
            m_name = f"roll{w}_mean"
            s_name = f"roll{w}_std"

            df[m_name] = df[target].rolling(w,   min_periods=1).mean()
            df[s_name] = (df[target]
                        .rolling(w, min_periods=2)  # std осмысленна от 2 точек
                        .std()
                        .fillna(0))                 # первые строки → 0 вместо NaN

            self.roll_feat_names_ += [m_name, s_name]


        df.dropna(inplace=True)
        return df

    def _plot_forecast(self, df, target, feat_set):
        train_true = df[target].loc[:self.test_index_[0]]
        test_true  = df[target].loc[self.test_index_[0]:]

        test_feat  = df[feat_set].loc[self.test_index_[0]:]
        test_pred  = self.model.predict(test_feat)

        mae = mean_absolute_error(test_true, test_pred)
        mse = mean_squared_error(test_true, test_pred)
        rmse = np.sqrt(mse)

        plt.figure(figsize=(18, 6))
        plt.plot(train_true.index, train_true, label="train", alpha=0.5)
        plt.plot(test_true.index,  test_true,  label="test-true", alpha=0.8)
        plt.plot(test_true.index,  test_pred,  label="test-pred", alpha=0.8)
        plt.title(f"MAE={mae:.1f}, RMSE={rmse:.1f}")
        plt.legend()
        plt.tight_layout()
        plt.show()


    # ──────────────────────────────────────
    # static factory for live horizon
    # ──────────────────────────────────────
    def prepare_future(self, df_last: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Возвращает матрицу признаков с ТОЧНО тем же порядком колонок,
        что и в момент fit().
        """
        df_future = df_last.copy()

        # если синус/косинус уже есть — оставляем, иначе создаём
        if 'hour_sin' not in df_future.columns:
            df_future['hour_sin'] = np.sin(2*np.pi*df_future.index.hour/24)
            df_future['hour_cos'] = np.cos(2*np.pi*df_future.index.hour/24)

        df_future = self._make_time_features(df_future)
        df_future = self._make_stat_features(df_future, target)

        # полный порядок, как в fit()
        feat_order = (
            self.additional_feat_cols_ +          # сохраняем из fit()
            self.lags_feat_names_ +
            self.roll_feat_names_ +
            self.cat_features
        )
        return df_future[feat_order]

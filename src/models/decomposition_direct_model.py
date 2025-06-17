# -----------------------  decomposition_direct_model.py  -----------------------
import os
from typing import List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression


class DecompositionDirectModel:
    """
    Direct multi‑step forecaster:
    ▸  log1p‑трансформация                        (stabilise variance)
    ▸  STL‑декомпозиция без утечки будущего       (robust=True)
    ▸  trend  → LinearRegression  (можно заменить spline/GB)
    ▸  seasonal pattern фиксируется (length = period)
    ▸  residuals → CatBoost
        –   MultiRMSE, если библиотека ≥ 1.2  
        –   иначе horizon отдельных моделей
    ▸  В отличие от recursive‑loop нет накопления ошибки.
    """

    LAGS_DEF  = [1, 2, 4, 96, 192, 5760]
    ROLL_DEF  = [4, 96, 192, 1920, 2880, 4320, 5760, 8640]

    def __init__(
        self,
        horizon: int = 240,
        period: int = 5760,                 # 1 день при шаге 15 с
        lags:   Optional[Sequence[int]] = None,
        rolls:  Optional[Sequence[int]] = None,
        cat_features: Optional[List[str]] = None,
        random_state: int = 42,
        **cb_params,
    ):
        self.horizon  = horizon
        self.period   = period
        self.lags     = list(lags or self.LAGS_DEF)
        self.rolls    = list(rolls or self.ROLL_DEF)
        self.cat_cols = cat_features or ["hour", "dow"]

        # --- деталь CatBoost ---
        base = dict(
            loss_function = "MultiRMSE",
            iterations    = 1500,
            depth         = 6,
            learning_rate = 0.05,
            l2_leaf_reg   = 20.0,
            random_seed   = random_state,
            verbose       = False,
            early_stopping_rounds = 100,
        )
        base.update(cb_params)
        self.cb_params = base

        self._trend_model      = LinearRegression()
        self._seasonal_pattern = None                            # ndarray[length = period]
        self._time0            = None
        self._step_sec         = None
        self._feature_names    = None

        self._multi_model: Optional[CatBoostRegressor] = None    # 1 модель на все шаги
        self._step_models:  List[CatBoostRegressor]   = []       # если MultiRMSE не доступен

        self.fitted_ = False

    # ------------------------------------------------------------------ #
    #                               FIT                                  #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        val_df: Optional[pd.DataFrame] = None,
    ):
        self._time0     = train_df.index[0]
        self._step_sec  = int((train_df.index[1] - train_df.index[0]).total_seconds())
        self.target_col = target_col

        # 1. STL (robust & one‑sided window → нет утечки)
        y_log = np.log1p(train_df[target_col])
        stl   = STL(y_log, period=self.period, robust=True)
        res   = stl.fit()
        trend, seas, resid = res.trend, res.seasonal, res.resid

        # 2. trend model
        t_idx = (trend.dropna().index - self._time0).total_seconds().to_numpy().reshape(-1, 1)
        self._trend_model.fit(t_idx, trend.dropna().values)

        # 3. seasonal pattern (первая дата = начало датасета)
        self._seasonal_pattern = seas.iloc[:self.period].to_numpy()

        # 4. residuals → supervised dataset
        resid_df = pd.DataFrame({target_col: resid}).dropna()
        X, Y = self._make_supervised(resid_df)

        self._feature_names = list(X.columns)
        train_pool = Pool(X, Y, cat_features=self.cat_cols)

        try:
            # CatBoost ≥ 1.2 умеет MultiRMSE – один fit для всех h
            self._multi_model = CatBoostRegressor(**self.cb_params)
            self._multi_model.fit(train_pool)
        except Exception:     # fallback: по модели на шаг
            self._multi_model = None
            for h in range(self.horizon):
                model_h = CatBoostRegressor(
                    **{**self.cb_params, "loss_function": "RMSE"}
                )
                y_h = Y[:, h]
                model_h.fit(train_pool, y_h)
                self._step_models.append(model_h)

        self.fitted_ = True
        return self

    # ------------------------------------------------------------------ #
    #                             PREDICT                                #
    # ------------------------------------------------------------------ #
    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Сначала вызовите .fit().")

        last_time  = df_hist.index[-1]
        future_idx = pd.date_range(
            start=last_time + pd.Timedelta(seconds=self._step_sec),
            periods=self.horizon,
            freq  = f"{self._step_sec}s",
        )

        # 1. trend + seasonality
        t_idx_f = (future_idx - self._time0).total_seconds().to_numpy().reshape(-1, 1)
        trend_f = self._trend_model.predict(t_idx_f)

        season_f_idx = ((future_idx - self._time0).total_seconds() / self._step_sec) % self.period
        seas_f = self._seasonal_pattern[season_f_idx.astype(int)]

        # 2. residual forecast (direct)
        hist_resid = self._last_residuals(df_hist)
        X_future   = self._make_features_for_last_window(hist_resid)

        if self._multi_model is not None:
            resid_pred = self._multi_model.predict(X_future)[0]
        else:   # собрать по‑одному
            resid_pred = np.array(
                [m.predict(X_future)[0] for m in self._step_models]
            )

        # 3. сложить и вернуться к исходному масштабу
        y_log_pred = trend_f + seas_f + resid_pred
        y_pred     = np.expm1(y_log_pred)
        return pd.Series(y_pred, index=future_idx, name="forecast")

    # ------------------------------------------------------------------ #
    #                     helpers: feature engineering                   #
    # ------------------------------------------------------------------ #
    # ──────────────────────────────────────────────────────────────────────
    # 1) _prepare_features  – временной индекс считаем от self._time0
    # ──────────────────────────────────────────────────────────────────────
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # календарные
        out["hour"]       = df.index.hour.astype(str)
        out["dow"]        = df.index.dayofweek.astype(str)
        out["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        out["time_idx_h"] = (df.index - self._time0).total_seconds() / 3600.0

        # лаги / роллы по residual
        for l in self.lags:
            out[f"lag_{l}"] = df[self.target_col].shift(l)
        for w in self.rolls:
            out[f"roll_mean_{w}"] = df[self.target_col].rolling(w).mean()
            out[f"roll_std_{w}"]  = df[self.target_col].rolling(w).std()

        return out.dropna()          # удаляем строки с NaN-ами

    # ──────────────────────────────────────────────────────────────────────
    # 2) _make_supervised  – выравнивание индекса feats и y_mult
    # ──────────────────────────────────────────────────────────────────────
    def _make_supervised(self, df: pd.DataFrame):
        """
        →  X (n_samples, n_features),  Y (n_samples, horizon)
        Строки берём только там, где есть и признаки, и все будущие цели.
        """
        feats = self._prepare_features(df)
        idx   = feats.index                       # только «чистые» строки

        # будущие значения цели, выровненные по тем же индексам
        y_cols = [df[self.target_col].shift(-h).loc[idx] for h in range(1, self.horizon + 1)]
        y_mult = np.column_stack(y_cols)

        # отсекаем последние horizon-строк, где появились NaN-ы после shift
        mask = ~np.isnan(y_mult).any(axis=1)

        return feats.iloc[mask], y_mult[mask]


    # последние residual для формирования окна предсказания
    def _last_residuals(self, df_hist: pd.DataFrame) -> pd.Series:
        y_log = np.log1p(df_hist[self.target_col])
        t_idx = (df_hist.index - self._time0).total_seconds().to_numpy().reshape(-1, 1)
        trend = self._trend_model.predict(t_idx)

        seas_idx = ((df_hist.index - self._time0).total_seconds() / self._step_sec) % self.period
        seas = self._seasonal_pattern[seas_idx.astype(int)]
        return pd.Series(y_log - trend - seas, index=df_hist.index, name=self.target_col)

    def _make_features_for_last_window(self, resid_hist: pd.Series) -> pd.DataFrame:
        """Строит последние признаки (одна строка) из history residuals."""
        df_feat = self._prepare_features(resid_hist.to_frame())
        return df_feat.iloc[[-1]].reindex(columns=self._feature_names, fill_value=np.nan)

    # ------------------------------------------------------------------ #
    #                         save / load                                #
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        if not self.fitted_:
            raise RuntimeError("Модель не обучена.")
        os.makedirs(path, exist_ok=True)
        if self._multi_model is not None:
            self._multi_model.save_model(os.path.join(path, "catboost_multi.cbm"))
        else:
            for i, m in enumerate(self._step_models):
                m.save_model(os.path.join(path, f"cb_step_{i+1}.cbm"))
        meta = self.__dict__.copy()
        # CatBoost объекты нельзя сериализовать joblib‑ом
        meta["_multi_model"] = None
        meta["_step_models"] = []
        joblib.dump(meta, os.path.join(path, "meta.joblib"))

    @classmethod
    def load(cls, path: str):
        meta = joblib.load(os.path.join(path, "meta.joblib"))
        inst = cls.__new__(cls)
        inst.__dict__.update(meta)

        if os.path.exists(os.path.join(path, "catboost_multi.cbm")):
            inst._multi_model = CatBoostRegressor()
            inst._multi_model.load_model(os.path.join(path, "catboost_multi.cbm"))
        else:
            inst._step_models = []
            for h in range(inst.horizon):
                m = CatBoostRegressor()
                fname = os.path.join(path, f"cb_step_{h+1}.cbm")
                if not os.path.exists(fname):
                    raise FileNotFoundError(fname)
                m.load_model(fname)
                inst._step_models.append(m)

        inst.fitted_ = True
        return inst

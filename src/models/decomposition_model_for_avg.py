import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose


class DecompositionImprovedTrendModel:
    """
    ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ.
    - Устранена ошибка TypeError при вызове .align().
    - Используется более надежный способ выравнивания данных через DataFrame.
    """

    DEFAULT_LAGS = [1, 2, 4, 96, 192, 5760]
    DEFAULT_ROLL = [4, 96, 192, 1920, 2880, 4320, 5760, 8640]

    def __init__(
        self,
        horizon: int = 96,
        period: int = 5760,
        lags: Optional[List[int]] = None,
        roll_windows: Optional[List[int]] = None,
        random_state: int = 42,
        cat_features: Optional[List[str]] = None,
        trend_model_type: str = 'local',
        trend_window_size: int = 960,
        **cb_params,
    ) -> None:
        self.horizon = horizon
        self.period = period
        self.lags = lags or self.DEFAULT_LAGS
        self.roll_windows = roll_windows or self.DEFAULT_ROLL
        self.random_state = random_state
        self.cat_features = cat_features or ["hour", "dow"]

        if trend_model_type not in ['global', 'local']:
            raise ValueError("trend_model_type должен быть 'global' или 'local'")
        self.trend_model_type = trend_model_type
        self.trend_window_size = trend_window_size

        self.trend_model_global = None
        self.last_trend_value_ = None
        self.local_trend_slope_ = None
        self.last_trend_time_numeric_ = None

        self.feature_names_ = []
        self.target_col_ = None
        self.train_start_time_ = None
        self.time_step_seconds_ = None
        self.seasonal_component_ = None
        self.model = None
        self.fitted_ = False

        self.cb_params = {
            "loss_function": "RMSE", "boosting_type": "Plain", "l2_leaf_reg": 10.0,
            "depth": 10, "learning_rate": 0.075, "iterations": 2500,
            "random_seed": random_state, "early_stopping_rounds": 150, "verbose": False,
        }
        self.cb_params.update(cb_params)

    def _predict_trend(self, index: pd.DatetimeIndex) -> pd.Series:
        time_idx_numeric = (index - self.train_start_time_).total_seconds().to_numpy()
        if self.trend_model_type == 'global':
            return pd.Series(self.trend_model_global.predict(time_idx_numeric.reshape(-1, 1)), index=index, name='trend_pred')
        elif self.trend_model_type == 'local':
            time_deltas = time_idx_numeric - self.last_trend_time_numeric_
            extrapolated_values = self.last_trend_value_ + self.local_trend_slope_ * time_deltas
            return pd.Series(extrapolated_values, index=index, name='trend_pred')

    def fit(self, train_df: pd.DataFrame, target_col: str, val_df: Optional[pd.DataFrame] = None):
        self.target_col_ = target_col
        self.train_start_time_ = train_df.index[0]
        self.time_step_seconds_ = (train_df.index[1] - train_df.index[0]).total_seconds()

        df_proc = train_df.copy()
        df_proc[target_col] = np.log1p(df_proc[target_col])
        
        decomposition = seasonal_decompose(df_proc[target_col], model='additive', period=self.period)
        trend_data = decomposition.trend.dropna()

        time_idx_numeric = (trend_data.index - self.train_start_time_).total_seconds().to_numpy().reshape(-1, 1)

        if self.trend_model_type == 'global':
            self.trend_model_global = LinearRegression()
            self.trend_model_global.fit(time_idx_numeric, trend_data)
        elif self.trend_model_type == 'local':
            window = min(len(trend_data), self.trend_window_size)
            local_trend_window = trend_data.iloc[-window:]
            local_time_idx = time_idx_numeric[-window:]
            local_model = LinearRegression()
            local_model.fit(local_time_idx, local_trend_window)
            self.local_trend_slope_ = local_model.coef_[0]
            self.last_trend_value_ = local_trend_window.iloc[-1]
            self.last_trend_time_numeric_ = local_time_idx[-1][0]
        
        self.seasonal_component_ = decomposition.seasonal.iloc[:self.period].to_numpy()

        # === ФИНАЛЬНЫЙ ИСПРАВЛЕННЫЙ БЛОК РАСЧЕТА ОСТАТКОВ ===
        # Создаем временный DataFrame для безопасного выравнивания по индексу
        temp_df = pd.DataFrame({
            'log_data': df_proc[target_col],
            'trend': self._predict_trend(train_df.index),
            'seasonal': decomposition.seasonal
        })
        # dropna() эффективно выполняет 'inner join' по всем трем рядам
        aligned_df = temp_df.dropna()
        # Вычисляем остатки на выровненных и очищенных данных
        new_residuals = aligned_df['log_data'] - aligned_df['trend'] - aligned_df['seasonal']
        # Создаем итоговый DataFrame остатков с правильным именем столбца
        resid_df = new_residuals.to_frame(name=target_col)
        # ========================================================

        if val_df is not None:
            val_start_time = val_df.index[0]
            train_resid_df = resid_df[resid_df.index < val_start_time]
            val_resid_df = resid_df[resid_df.index >= val_start_time]
        else:
            train_resid_df = resid_df
            val_resid_df = None

        X_train, y_train = self._prepare_data(train_resid_df, target_col, [])
        self.feature_names_ = list(X_train.columns)
        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        
        eval_set = None
        if val_resid_df is not None and not val_resid_df.empty:
            X_val, y_val = self._prepare_data(val_resid_df, target_col, [])
            if not X_val.empty:
                X_val = X_val.reindex(columns=self.feature_names_).fillna(0)
                eval_set = Pool(X_val, y_val, cat_features=self.cat_features)
        
        self.model = CatBoostRegressor(**self.cb_params)
        self.model.fit(train_pool, eval_set=eval_set, use_best_model=True if eval_set else False)
        self.fitted_ = True
        return self

    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Модель еще не обучена.")

        last_original_time = df_hist.index[-1]
        future_times = pd.date_range(
            start=last_original_time + pd.to_timedelta(self.time_step_seconds_, 's'),
            periods=self.horizon,
            freq=pd.to_timedelta(self.time_step_seconds_, 's')
        )

        trend_future = self._predict_trend(future_times)
        future_seasonal_indices = ((future_times - self.train_start_time_).total_seconds() / self.time_step_seconds_) % self.period
        seasonal_future = self.seasonal_component_[future_seasonal_indices.astype(int)]

        hist_proc = df_hist.copy()
        hist_proc[self.target_col_] = np.log1p(hist_proc[self.target_col_])
        
        temp_df_hist = pd.DataFrame({
            'log_data': hist_proc[self.target_col_],
            'trend': self._predict_trend(hist_proc.index),
            'seasonal': pd.Series(
                self.seasonal_component_[((hist_proc.index - self.train_start_time_).total_seconds() / self.time_step_seconds_).astype(int) % self.period],
                index=hist_proc.index
            )
        }).dropna()

        resid_hist = temp_df_hist['log_data'] - temp_df_hist['trend'] - temp_df_hist['seasonal']
        current_history = resid_hist.to_frame(name=self.target_col_)

        predictions_residuals = []
        for _ in range(self.horizon):
            X = self._prepare_features(current_history, self.target_col_, [])
            last_features = X.iloc[-1:]
            next_pred_residual = self.model.predict(last_features)[0]
            predictions_residuals.append(next_pred_residual)
            last_time = current_history.index[-1]
            next_time = last_time + pd.to_timedelta(self.time_step_seconds_, 's')
            new_row = pd.DataFrame({self.target_col_: [next_pred_residual]}, index=[next_time])
            current_history = pd.concat([current_history, new_row])

        last_known_residual = current_history[self.target_col_].iloc[-self.horizon - 1]
        smoothed_residuals = pd.Series(predictions_residuals).ewm(alpha=0.1).mean()
        initial_offset = last_known_residual - smoothed_residuals.iloc[0]
        final_residuals = smoothed_residuals + initial_offset

        final_prediction_log = trend_future.values + seasonal_future + final_residuals.to_numpy()
        predictions = np.expm1(final_prediction_log)
        predictions[predictions < 0] = 0
        return pd.Series(predictions, index=future_times, name="forecast")

    def _prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]):
        X = self._prepare_features(df, target_col, feature_cols)
        y = df[target_col].shift(-1)
        combined = pd.concat([X, y.rename("target")], axis=1).dropna()
        X_clean = combined.drop(columns='target')
        y_clean = combined['target']
        return X_clean, y_clean

    def _prepare_features(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> pd.DataFrame:
        df_copy = df.copy()
        df_out = pd.DataFrame(index=df_copy.index)
        df_out["hour"] = df_out.index.hour.astype(str)
        df_out['dow'] = df_out.index.dayofweek.astype(str)
        df_out['is_weekend'] = (df_out.index.dayofweek >= 5).astype(int)
        df_out['time_idx'] = (df_out.index - self.train_start_time_).total_seconds() / 3600.0
        for lag in self.lags:
            df_out[f'f_resid_lag_{lag}'] = df_copy[target_col].shift(lag)
        for w in self.roll_windows:
            df_out[f'f_resid_roll_mean_{w}'] = df_copy[target_col].rolling(w, min_periods=1).mean()
            df_out[f'f_resid_roll_std_{w}'] = df_copy[target_col].rolling(w, min_periods=1).std()
        if self.fitted_:
            return df_out.reindex(columns=self.feature_names_).fillna(0)
        else:
            return df_out.reindex(sorted(df_out.columns), axis=1)

    def save(self, path: str) -> None:
        """
        Saves the fitted model to a directory.
        
        The directory will contain:
        - 'catboost_model.cbm': The core CatBoost model.
        - 'model_metadata.joblib': Other parameters of the wrapper class.
        """
        if not self.fitted_:
            raise RuntimeError("Модель еще не обучена. Невозможно сохранить.")

        full_model_path = os.path.join("models", path)
        
        os.makedirs(full_model_path, exist_ok=True)

        self.model.save_model(os.path.join(full_model_path, "catboost_model.cbm"))

        metadata = self.__dict__.copy()
        del metadata["model"]
        
        joblib.dump(metadata, os.path.join(full_model_path, "model_metadata.joblib"))
        print(f"Модель сохранена в {full_model_path}")

    @classmethod
    def load(cls, path: str):
        """
        Loads a model from a directory.
        
        Method will load the model from the directory and return an instance of the class.
        """
        full_model_path = os.path.join("models", path)

        metadata_path = os.path.join(full_model_path, "model_metadata.joblib")
        catboost_model_path = os.path.join(full_model_path, "catboost_model.cbm")

        if not os.path.exists(metadata_path) or not os.path.exists(catboost_model_path):
            raise FileNotFoundError(f"Файлы модели не найдены в директории: {full_model_path}")

        metadata = joblib.load(metadata_path)
        
        instance = cls.__new__(cls)
        instance.__dict__.update(metadata)

        instance.model = CatBoostRegressor()
        instance.model.load_model(catboost_model_path)

        return instance
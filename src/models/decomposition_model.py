import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose


class DecompositionRecursiveModel:
    """
    A recursive forecasting model that decomposes the time series into trend,
    seasonal, and residual components.

    1.  The data is log-transformed.
    2.  `statsmodels.tsa.seasonal_decompose` splits the series into trend,
        seasonal, and residual parts.
    3.  A `LinearRegression` model is trained to extrapolate the trend.
    4.  The seasonal component is stored and repeated for future predictions.
    5.  A `CatBoostRegressor` is trained to recursively forecast the residuals.
    6.  The final forecast is the sum of the three components, which is then
        inverse-transformed back to the original scale.
    """

    DEFAULT_LAGS = [1, 2, 4, 96]
    DEFAULT_ROLL = [4, 12, 96]

    def __init__(
        self,
        horizon: int = 96,
        # Daily seasonality for 15-sec data: 24h * 60m * 60s / 15s = 5760
        period: int = 5760,
        lags: Optional[List[int]] = None,
        roll_windows: Optional[List[int]] = None,
        random_state: int = 42,
        cat_features: Optional[List[str]] = None,
        **cb_params,
    ) -> None:
        self.horizon = horizon
        self.period = period
        self.lags = lags or self.DEFAULT_LAGS
        self.roll_windows = roll_windows or self.DEFAULT_ROLL
        self.random_state = random_state
        self.cat_features = cat_features or ["hour", "dow"]
        
        self.feature_names_ = []
        self.target_col_ = None
        self.train_start_time_ = None
        self.time_step_seconds_ = None

        self.trend_model = LinearRegression()
        self.seasonal_component_ = None

        self.cb_params = {
            # Back to RMSE for stability
            "loss_function": "RMSE",
            "boosting_type": "Plain",
            # Increased regularization to prevent overfitting on noise
            "l2_leaf_reg": 10.0,
            "depth": 10,
            "learning_rate": 0.075,
            "iterations": 2500,
            "random_seed": random_state,
            "early_stopping_rounds": 150,
            "verbose": False,
        }
        self.cb_params.update(cb_params)

        self.model = None
        self.fitted_: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        val_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        """Decomposes the series and fits models on trend and residuals."""
        self.target_col_ = target_col
        feature_cols = feature_cols or []
        
        self.train_start_time_ = train_df.index[0]
        self.time_step_seconds_ = (train_df.index[1] - train_df.index[0]).total_seconds()

        # --- 1. DECOMPOSITION ---
        df_proc = train_df.copy()
        df_proc[target_col] = np.log1p(df_proc[target_col])
        
        decomposition = seasonal_decompose(df_proc[target_col], model='additive', period=self.period)
        
        # --- 2. TREND MODELING ---
        trend_data = decomposition.trend.dropna()
        time_idx_trend = (trend_data.index - self.train_start_time_).total_seconds().to_numpy().reshape(-1, 1)
        self.trend_model.fit(time_idx_trend, trend_data)

        # --- 3. SEASONAL COMPONENT ---
        self.seasonal_component_ = decomposition.seasonal.iloc[:self.period].to_numpy()

        # --- 4. RESIDUALS MODELING (CATBOOST) ---
        residuals = decomposition.resid.dropna()
        resid_df = pd.DataFrame(residuals).rename(columns={"resid": target_col})

        # We'll use the same train/val split on the residuals
        if val_df is not None:
            val_start_time = val_df.index[0]
            train_resid_df = resid_df[resid_df.index < val_start_time]
            val_resid_df = resid_df[resid_df.index >= val_start_time]
        else:
            train_resid_df = resid_df
            val_resid_df = None

        X_train, y_train = self._prepare_data(train_resid_df, target_col, feature_cols)
        self.feature_names_ = list(X_train.columns)
        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        
        eval_set = None
        if val_resid_df is not None and not val_resid_df.empty:
            combined_resid_df = pd.concat([train_resid_df, val_resid_df]).sort_index()
            X_all, y_all = self._prepare_data(combined_resid_df, target_col, feature_cols)
            
            valid_val_indices = val_resid_df.index.intersection(X_all.index)
            if not valid_val_indices.empty:
                X_val = X_all.loc[valid_val_indices].reindex(columns=self.feature_names_)
                y_val = y_all.loc[valid_val_indices]
                if not X_val.empty:
                    eval_set = Pool(X_val, y_val, cat_features=self.cat_features)

        self.model = CatBoostRegressor(**self.cb_params)
        self.model.fit(train_pool, eval_set=eval_set, use_best_model=True if eval_set else False)
        self.fitted_ = True
        return self

    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        """Forecasts by combining trend, seasonal, and residual predictions."""
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")

        # --- 1. PREPARE FUTURE TIMESTAMPS AND COMPONENTS ---
        last_original_time = df_hist.index[-1]
        future_times = pd.date_range(
            start=last_original_time + pd.to_timedelta(self.time_step_seconds_, 's'),
            periods=self.horizon,
            freq=pd.to_timedelta(self.time_step_seconds_, 's')
        )

        # Extrapolate trend
        time_idx_future = (future_times - self.train_start_time_).total_seconds().to_numpy().reshape(-1, 1)
        trend_future = self.trend_model.predict(time_idx_future)

        # Extrapolate seasonality
        future_seasonal_indices = ((future_times - self.train_start_time_).total_seconds() / self.time_step_seconds_) % self.period
        seasonal_future = self.seasonal_component_[future_seasonal_indices.astype(int)]

        # --- 2. RECURSIVELY PREDICT RESIDUALS ---
        # Get residuals from historical data to start the recursive process
        hist_proc = df_hist.copy()
        hist_proc[self.target_col_] = np.log1p(hist_proc[self.target_col_])
        
        time_idx_hist = (hist_proc.index - self.train_start_time_).total_seconds().to_numpy().reshape(-1, 1)
        trend_hist = self.trend_model.predict(time_idx_hist)
        
        hist_seasonal_indices = ((hist_proc.index - self.train_start_time_).total_seconds() / self.time_step_seconds_) % self.period
        seasonal_hist = self.seasonal_component_[hist_seasonal_indices.astype(int)]
        
        resid_hist = hist_proc[self.target_col_] - trend_hist - seasonal_hist
        
        current_history = pd.DataFrame({self.target_col_: resid_hist}, index=df_hist.index)

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

        # --- 2b. SMOOTH THE JUMP and DAMPEN OSCILLATIONS ---
        # Get the last known residual to smoothly transition from
        last_known_residual = current_history[self.target_col_].iloc[-self.horizon - 1]
        
        # Apply exponential smoothing to the residuals forecast
        # This makes the forecast start at the last known point and gradually decay
        smoothed_residuals = pd.Series(predictions_residuals).ewm(alpha=0.1).mean()
        
        # Adjust the smoothed series to start from the last known residual
        initial_offset = last_known_residual - smoothed_residuals.iloc[0]
        final_residuals = smoothed_residuals + initial_offset

        # --- 3. COMBINE AND INVERSE-TRANSFORM ---
        final_prediction_log = trend_future + seasonal_future + final_residuals.to_numpy()
        predictions = np.expm1(final_prediction_log)

        return pd.Series(predictions, index=future_times, name="forecast")

    def _prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]):
        """Creates feature matrix (X) and target vector (y) from residuals."""
        X = self._prepare_features(df, target_col, feature_cols)
        y = df[target_col].shift(-1)

        combined = pd.concat([X, y.rename("target")], axis=1).dropna()
        X_clean = combined[X.columns]
        y_clean = combined["target"]
        
        return X_clean, y_clean

    def _prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """Creates all features for the dataframe (from residuals)."""
        df_copy = df.copy()
        
        if feature_cols:
            df_out = df_copy[feature_cols].copy()
        else:
            df_out = pd.DataFrame(index=df_copy.index)

        # Calendar features
        df_out["hour"] = df_out.index.hour.astype(str)
        df_out['dow'] = df_out.index.dayofweek.astype(str)
        df_out['is_weekend'] = (df_out.index.dayofweek >= 5).astype(int)

        # Time trend feature can still be useful for residuals
        df_out['time_idx'] = (df_out.index - df_out.index[0]).total_seconds() / 3600.0

        # Features from residuals
        for lag in self.lags:
            df_out[f'f_resid_lag_{lag}'] = df_copy[target_col].shift(lag)

        for w in self.roll_windows:
            df_out[f'f_resid_roll_mean_{w}'] = df_copy[target_col].rolling(w).mean()
            df_out[f'f_resid_roll_std_{w}'] = df_copy[target_col].rolling(w).std()

        df_out = df_out.reindex(sorted(df_out.columns), axis=1)
        return df_out

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
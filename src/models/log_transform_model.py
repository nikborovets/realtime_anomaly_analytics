import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression

# ────────────────────────────────────────────────────────────────
#  internal toolkit helpers  (installed as local package)
# ────────────────────────────────────────────────────────────────
from ts_toolkit.calendar import add_dow_str, add_hour_sin_cos


class DetrendingLogRecursiveModel:
    """
    CatBoost-based recursive model that combines detrending with a log transform.
    
    1.  A simple linear trend is fit on the log-transformed data.
    2.  The CatBoost model is trained to predict the *residuals* (the data with
        the trend removed).
    3.  During prediction, the trend is extrapolated, and the model recursively
        predicts the future residuals.
    4.  The final forecast is the sum of the extrapolated trend and the
        predicted residuals, which is then inverse-transformed.
    """

    DEFAULT_LAGS = [1, 2, 4, 96, 192, 5_760]  # 0.25 min → 24 h
    DEFAULT_ROLL = [4, 96, 192, 1_920, 2_880, 4_320, 5_760, 8_640]

    def __init__(
        self,
        horizon: int = 96,
        lags: Optional[List[int]] = None,
        roll_windows: Optional[List[int]] = None,
        random_state: int = 42,
        cat_features: Optional[List[str]] = None,
        **cb_params,
    ) -> None:
        self.horizon = horizon
        self.lags = lags or self.DEFAULT_LAGS
        self.roll_windows = roll_windows or self.DEFAULT_ROLL
        self.random_state = random_state
        self.cat_features = cat_features or ["hour", "dow"]
        self.feature_names_ = []
        self.target_col_ = None

        # Trend model
        self.trend_model = LinearRegression()

        self.cb_params = {
            "loss_function": "RMSE",
            "boosting_type": "Plain",
            "l2_leaf_reg": 0.1,  # Reduced regularization to be "bolder"
            "depth": 8,
            "learning_rate": 0.075, # Slightly increased learning rate
            "iterations": 1500,     # Increased iterations
            "random_seed": random_state,
            "early_stopping_rounds": 150, # Increased patience
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
        """Prepares data and fits the detrending model."""
        self.target_col_ = target_col
        feature_cols = feature_cols or []
        
        # 1. Log-transform the target
        train_df_proc = train_df.copy()
        train_df_proc[target_col] = np.log1p(train_df_proc[target_col])
        if val_df is not None:
            val_df_proc = val_df.copy()
            val_df_proc[target_col] = np.log1p(val_df_proc[target_col])

        # 2. Fit and remove the linear trend
        # Create a time index for trend fitting
        time_idx_train = (train_df_proc.index - train_df_proc.index[0]).total_seconds().to_numpy().reshape(-1, 1)
        self.trend_model.fit(time_idx_train, train_df_proc[target_col])
        
        # Get trend predictions
        trend_train = self.trend_model.predict(time_idx_train)
        
        # Detrend by subtracting the trend from the log-transformed values
        train_df_proc[target_col] = train_df_proc[target_col] - trend_train

        if val_df is not None:
            time_idx_val = (val_df_proc.index - train_df_proc.index[0]).total_seconds().to_numpy().reshape(-1, 1)
            trend_val = self.trend_model.predict(time_idx_val)
            val_df_proc[target_col] = val_df_proc[target_col] - trend_val

        # 3. Prepare features and target (which are now residuals)
        X_train, y_train = self._prepare_data(train_df_proc, target_col, feature_cols)
        self.feature_names_ = list(X_train.columns)

        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        
        # 4. Prepare validation data if provided
        eval_set = None
        if val_df is not None:
            # Important: Combine the DETRENDED data for feature generation
            combined_df = pd.concat([train_df_proc, val_df_proc]).sort_index()
            X_all, y_all = self._prepare_data(combined_df, target_col, feature_cols)
            
            valid_val_indices = val_df_proc.index.intersection(X_all.index)
            
            if not valid_val_indices.empty:
                X_val = X_all.loc[valid_val_indices].reindex(columns=self.feature_names_)
                y_val = y_all.loc[valid_val_indices]
                
                if not X_val.empty:
                    eval_set = Pool(X_val, y_val, cat_features=self.cat_features)

        # 5. Train CatBoost on the residuals
        self.model = CatBoostRegressor(**self.cb_params)
        self.model.fit(train_pool, eval_set=eval_set, use_best_model=True if eval_set else False)
        self.fitted_ = True
        return self

    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        """Creates a forecast by combining a trend forecast and a recursive residual forecast."""
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        
        # 1. Log-transform and detrend the historical data
        current_history = df_hist.copy()
        time_idx_hist = (current_history.index - df_hist.index[0]).total_seconds().to_numpy().reshape(-1, 1)
        
        trend_hist = self.trend_model.predict(time_idx_hist)
        current_history[self.target_col_] = np.log1p(current_history[self.target_col_]) - trend_hist

        # --- Recursive prediction of residuals ---
        predictions_residuals = []
        time_step = current_history.index[1] - current_history.index[0]
        
        for _ in range(self.horizon):
            # Prepare features for the last point of the residual history
            X = self._prepare_features(current_history, self.target_col_, [])
            last_features = X.iloc[-1:]

            # Predict one step ahead (the residual)
            next_pred_residual = self.model.predict(last_features)[0]
            
            predictions_residuals.append(next_pred_residual)

            # Update history with the new predicted residual
            last_time = current_history.index[-1]
            next_time = last_time + time_step
            new_row = pd.DataFrame({self.target_col_: [next_pred_residual]}, index=[next_time])
            current_history = pd.concat([current_history, new_row])

        # 2. Extrapolate the trend into the future
        last_original_time = df_hist.index[-1]
        future_times = pd.date_range(
            start=last_original_time + time_step, 
            periods=self.horizon, 
            freq=time_step
        )
        time_idx_future = (future_times - df_hist.index[0]).total_seconds().to_numpy().reshape(-1, 1)
        trend_future = self.trend_model.predict(time_idx_future)

        # 3. Combine trend and residuals and inverse-transform
        final_prediction_log = trend_future + predictions_residuals
        predictions = np.expm1(final_prediction_log)
        
        return pd.Series(predictions, index=future_times, name="forecast")

    def _prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]):
        """Creates feature matrix (X) and target vector (y)."""
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
        """Creates all features for the dataframe."""
        df_copy = df.copy() 
        
        if feature_cols:
            df_out = df_copy[feature_cols].copy()
        else:
            df_out = pd.DataFrame(index=df_copy.index)

        # Calendar features
        df_out = add_hour_sin_cos(df_out)
        df_out = add_dow_str(df_out)
        df_out["hour"] = df_out.index.hour.astype(str)
        df_out['is_weekend'] = (df_out.index.dayofweek >= 5).astype(int)

        # Time trend feature - still useful for the residual model
        df_out['time_idx'] = (df_out.index - df_out.index[0]).total_seconds() / 3600.0

        # Spike features: rate of change from previous points
        df_out['f_target_diff_1'] = df_copy[target_col].diff(1)
        df_out['f_target_diff_2'] = df_copy[target_col].diff(2)
        df_out['f_target_diff_4'] = df_copy[target_col].diff(4)
        df_out['f_target_diff_96'] = df_copy[target_col].diff(96)

        # Volatility feature: rolling std of the spikes
        df_out['f_target_diff_1_roll_std_4'] = df_out['f_target_diff_1'].rolling(window=4).std()
        df_out['f_target_diff_1_roll_std_12'] = df_out['f_target_diff_1'].rolling(window=12).std()

        # Rolling max features to capture peak magnitudes
        df_out['f_target_roll_max_4'] = df_copy[target_col].rolling(window=4).max()
        df_out['f_target_roll_max_12'] = df_copy[target_col].rolling(window=12).max()

        # EWMA features
        df_out['f_target_ewm_mean_alpha_0_3'] = df_copy[target_col].ewm(alpha=0.3, adjust=False).mean()
        df_out['f_target_ewm_mean_alpha_0_1'] = df_copy[target_col].ewm(alpha=0.1, adjust=False).mean()

        # Lag features
        for lag in self.lags:
            df_out[f'f_target_lag_{lag}'] = df_copy[target_col].shift(lag)

        # Rolling mean features
        for w in self.roll_windows:
            df_out[f'f_target_roll_mean_{w}'] = df_copy[target_col].rolling(w).mean()

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
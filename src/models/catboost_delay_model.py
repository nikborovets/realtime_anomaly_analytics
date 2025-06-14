import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

# ────────────────────────────────────────────────────────────────
#  internal toolkit helpers  (installed as local package)
# ────────────────────────────────────────────────────────────────
from ts_toolkit.calendar import add_dow_str, add_hour_sin_cos
from ts_toolkit.features import (
    make_lag_features, 
    make_rolling_features,
    calculate_future_lags,
    calculate_future_rollings
)


class DelayForecastModel:
    """CatBoost‑based forecast of p90 latency using a multi-output strategy.
    
    This model predicts the entire forecast horizon at once to avoid error accumulation
    from recursive forecasting.
    """

    DEFAULT_LAGS = [1, 2, 4, 96, 192, 5_760]                # 0.25 min → 24 h
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
        self.target_col_ = None  # Запоминаем имя целевой колонки

        # CatBoost params for multi-output regression
        self.cb_params = {
            "loss_function": "MultiRMSE",  # Key change for multi-output
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 10, # Reduce iterations for faster initial training
            "random_seed": random_state,
            "early_stopping_rounds": 5,
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
        """Prepares data and fits the multi-output model."""
        self.target_col_ = target_col  # Сохраняем имя для predict
        feature_cols = feature_cols or []

        # Prepare features (X) and multi-target (Y) for training data
        X_train, Y_train = self._prepare_data(train_df, target_col, feature_cols)
        self.feature_names_ = list(X_train.columns)

        train_pool = Pool(X_train, Y_train, cat_features=self.cat_features)
        
        # Prepare validation data if provided
        eval_set = None
        if val_df is not None:
            # IMPORTANT: Use combined history to generate features for val set
            combined_df = pd.concat([train_df, val_df]).sort_index()
            X_all, Y_all = self._prepare_data(combined_df, target_col, feature_cols)
            
            # Select only the validation indices that survived the feature/target creation
            valid_val_indices = val_df.index.intersection(X_all.index)
            
            if not valid_val_indices.empty:
                X_val = X_all.loc[valid_val_indices].reindex(columns=self.feature_names_)
                Y_val = Y_all.loc[valid_val_indices]
                
                if not X_val.empty:
                    eval_set = Pool(X_val, Y_val, cat_features=self.cat_features)

        # Train the model
        self.model = CatBoostRegressor(**self.cb_params)
        self.model.fit(train_pool, eval_set=eval_set, use_best_model=True if eval_set else False)
        self.fitted_ = True
        return self

    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        """
        Creates a forecast for the next `horizon` steps based on history.
        """
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
            
        # 1. Prepare features for the last available data point
        X = self._prepare_features(df_hist, self.target_col_, [])
        last_features = X.iloc[-1:]

        # 2. Make a single prediction to get the entire forecast
        prediction_values = self.model.predict(last_features)[0]

        # 3. Create a timestamped Series for the forecast
        time_step = df_hist.index[1] - df_hist.index[0]
        last_time = df_hist.index[-1]
        future_times = pd.date_range(start=last_time + time_step, periods=self.horizon, freq=time_step)
        
        return pd.Series(prediction_values, index=future_times, name="forecast")

    def _prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]):
        """Creates feature matrix (X) and multi-target matrix (Y)."""
        # Create features (X)
        X = self._prepare_features(df, target_col, feature_cols)

        # Create multi-target matrix (Y) in a performant way
        targets = {
            f"target_{h}": df[target_col].shift(-h)
            for h in range(1, self.horizon + 1)
        }
        Y = pd.DataFrame(targets)

        # Align X and Y by dropping NaNs
        combined = pd.concat([X, Y], axis=1).dropna()
        X_clean = combined[X.columns]
        Y_clean = combined[Y.columns]
        
        return X_clean, Y_clean

    def _prepare_features(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """Creates all features for the dataframe."""
        df = df.copy()  # Explicitly create a copy
        df_out = df[feature_cols].copy()

        # calendar features
        df_out = add_hour_sin_cos(df_out)
        df_out = add_dow_str(df_out)
        df_out["hour"] = df_out.index.hour.astype(str)

        # lag / rolling features
        lag_names = make_lag_features(df, target_col, self.lags)
        roll_names = make_rolling_features(df, target_col, self.roll_windows)
        
        # Combine all features into the output dataframe
        df_out = pd.concat([df_out, df[lag_names + roll_names]], axis=1)

        return df_out

    def save(self, path: str) -> None:
        """Saves the fitted model to a directory.
        
        The directory will contain:
        - 'catboost_model.cbm': The core CatBoost model.
        - 'model_metadata.joblib': Other parameters of the wrapper class.
        """
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet. Cannot save.")

        os.makedirs(path, exist_ok=True)

        # 1. Save the CatBoost model using its native method
        self.model.save_model(os.path.join(path, "catboost_model.cbm"))

        # 2. Save the metadata of the wrapper
        metadata = self.__dict__.copy()
        del metadata["model"]  # The model itself is saved separately

        joblib.dump(metadata, os.path.join(path, "model_metadata.joblib"))
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Loads a model from a directory."""
        metadata_path = os.path.join(path, "model_metadata.joblib")
        catboost_model_path = os.path.join(path, "catboost_model.cbm")

        if not os.path.exists(metadata_path) or not os.path.exists(
            catboost_model_path
        ):
            raise FileNotFoundError(f"Model files not found in directory: {path}")

        # 1. Load metadata and create a shell instance
        metadata = joblib.load(metadata_path)
        instance = cls.__new__(cls)
        instance.__dict__.update(metadata)

        # 2. Load the CatBoost model into the instance
        instance.model = CatBoostRegressor()
        instance.model.load_model(catboost_model_path)

        return instance

import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

# ────────────────────────────────────────────────────────────────
#  internal toolkit helpers  (installed as local package)
# ────────────────────────────────────────────────────────────────
from ts_toolkit.calendar import add_dow_str, add_hour_sin_cos
from ts_toolkit.features import make_lag_features, make_rolling_features


class RecursiveDelayForecastModel:
    """CatBoost-based recursive forecast model for p90 latency.
    
    This model predicts one step at a time and uses the prediction as a feature 
    for the next step. It's generally more robust for series with trends 
    than a multi-output model.
    """

    DEFAULT_LAGS = [1, 2, 4, 96, 192, 5_760]  # 0.25 min → 24 h
    DEFAULT_ROLL = [4, 96, 192, 1_920, 2_880, 4_320, 5_760, 8_640]
    OUTLIER_UPPER_BOUND = 4723.295  # Based on IQR analysis from user

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

        # CatBoost params for single-output regression
        self.cb_params = {
            "loss_function": "RMSE",
            "boosting_type": "Plain",  # Use classic boosting for more aggressive trend following
            "l2_leaf_reg": 1,  # Weaken regularization to allow for higher peaks
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 1000,
            "random_seed": random_state,
            "early_stopping_rounds": 100,
            "verbose": False,
        }
        self.cb_params.update(cb_params)

        self.model = None
        self.fitted_: bool = False

    def _clip_outliers(self, series: pd.Series) -> pd.Series:
        """Clips outliers based on a fixed upper bound."""
        return series.clip(upper=self.OUTLIER_UPPER_BOUND)

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        val_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        """Prepares data and fits the single-output recursive model."""
        self.target_col_ = target_col
        feature_cols = feature_cols or []
        
        # 1. Clip outliers in the target variable
        train_df_proc = train_df.copy()
        train_df_proc[target_col] = self._clip_outliers(train_df_proc[target_col])
        if val_df is not None:
            val_df_proc = val_df.copy()
            val_df_proc[target_col] = self._clip_outliers(val_df_proc[target_col])

        # 2. Prepare features (X) and single-target (y)
        X_train, y_train = self._prepare_data(train_df_proc, target_col, feature_cols)
        self.feature_names_ = list(X_train.columns)

        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        
        # 3. Prepare validation data if provided
        eval_set = None
        if val_df is not None:
            combined_df = pd.concat([train_df_proc, val_df_proc]).sort_index()
            X_all, y_all = self._prepare_data(combined_df, target_col, feature_cols)
            
            valid_val_indices = val_df_proc.index.intersection(X_all.index)
            
            if not valid_val_indices.empty:
                X_val = X_all.loc[valid_val_indices].reindex(columns=self.feature_names_)
                y_val = y_all.loc[valid_val_indices]
                
                if not X_val.empty:
                    eval_set = Pool(X_val, y_val, cat_features=self.cat_features)

        # 4. Train the model
        self.model = CatBoostRegressor(**self.cb_params)
        self.model.fit(train_pool, eval_set=eval_set, use_best_model=True if eval_set else False)
        self.fitted_ = True
        return self

    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        """
        Creates a forecast for the next `horizon` steps using a recursive strategy.
        """
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
            
        # Start with a copy of the historical data, with outliers clipped
        current_history = df_hist.copy()
        current_history[self.target_col_] = self._clip_outliers(current_history[self.target_col_])

        predictions = []
        time_step = current_history.index[1] - current_history.index[0]
        
        for _ in range(self.horizon):
            # 1. Prepare features for the last point in the current history
            X = self._prepare_features(current_history, self.target_col_, [])
            last_features = X.iloc[-1:]

            # 2. Predict one step ahead
            next_pred = self.model.predict(last_features)[0]
            
            predictions.append(next_pred)

            # 3. Update history with the new prediction for the next iteration
            last_time = current_history.index[-1]
            next_time = last_time + time_step
            
            # Create a new row (as a DataFrame) to append to the history
            new_row = pd.DataFrame({self.target_col_: [next_pred]}, index=[next_time])
            
            # NOTE: This concat in a loop is not performant for very large horizons,
            # but is acceptable for typical forecast lengths.
            current_history = pd.concat([current_history, new_row])

        # 4. Create a timestamped Series for the final forecast
        last_original_time = df_hist.index[-1]
        future_times = pd.date_range(
            start=last_original_time + time_step, 
            periods=self.horizon, 
            freq=time_step
        )
        
        return pd.Series(predictions, index=future_times, name="forecast")

    def _prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]):
        """Creates feature matrix (X) and target vector (y) for single-step forecast."""
        X = self._prepare_features(df, target_col, feature_cols)
        
        # The target is the value at the next step
        y = df[target_col].shift(-1)

        # Align X and y by dropping rows with NaNs (mostly from feature generation)
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

        # Spike features: rate of change from previous points
        df_out['f_target_diff_1'] = df_copy[target_col].diff(1)
        df_out['f_target_diff_2'] = df_copy[target_col].diff(2)
        df_out['f_target_diff_4'] = df_copy[target_col].diff(4)
        df_out['f_target_diff_96'] = df_copy[target_col].diff(96)

        # Volatility feature: rolling std of the spikes
        df_out['f_target_diff_1_roll_std_4'] = df_out['f_target_diff_1'].rolling(window=4).std()

        # Rolling max features to capture peak magnitudes
        df_out['f_target_roll_max_4'] = df_copy[target_col].rolling(window=4).max()
        df_out['f_target_roll_max_12'] = df_copy[target_col].rolling(window=12).max()

        # EWMA features
        df_out['f_target_ewm_mean_alpha_0_3'] = df_copy[target_col].ewm(alpha=0.3, adjust=False).mean()
        df_out['f_target_ewm_mean_alpha_0_1'] = df_copy[target_col].ewm(alpha=0.1, adjust=False).mean()

        # Lag / rolling features from the target column
        lag_names = make_lag_features(df_copy, target_col, self.lags)
        roll_names = make_rolling_features(df_copy, target_col, self.roll_windows)
        
        feature_df = df_copy[lag_names + roll_names]

        # Combine all features into the output dataframe
        df_out = pd.concat([df_out, feature_df], axis=1)

        return df_out

    def save(self, path: str) -> None:
        """Saves the fitted model to a directory."""
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet. Cannot save.")

        os.makedirs(path, exist_ok=True)
        
        self.model.save_model(os.path.join(path, "catboost_model.cbm"))

        metadata = self.__dict__.copy()
        del metadata["model"]
        joblib.dump(metadata, os.path.join(path, "model_metadata.joblib"))
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Loads a model from a directory."""
        metadata_path = os.path.join(path, "model_metadata.joblib")
        catboost_model_path = os.path.join(path, "catboost_model.cbm")

        if not os.path.exists(metadata_path) or not os.path.exists(catboost_model_path):
            raise FileNotFoundError(f"Model files not found in directory: {path}")

        metadata = joblib.load(metadata_path)
        instance = cls.__new__(cls)
        instance.__dict__.update(metadata)

        instance.model = CatBoostRegressor()
        instance.model.load_model(catboost_model_path)

        return instance 
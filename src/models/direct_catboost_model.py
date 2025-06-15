import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from catboost import CatBoostRegressor, Pool
from tqdm.auto import tqdm

from ts_toolkit.calendar import add_dow_str, add_hour_sin_cos
from ts_toolkit.features import make_lag_features, make_rolling_features


class DirectDelayForecastModel:
    """
    CatBoost-based direct forecast model for p90 latency.

    This model trains a separate CatBoost model for each step in the forecast horizon.
    This strategy avoids the error accumulation problem inherent in recursive models,
    making it more robust for longer forecast horizons.
    """

    DEFAULT_LAGS = [1, 2, 4, 96, 192, 5_760]
    DEFAULT_ROLL = [4, 96, 192, 1_920, 2_880, 4_320, 5_760, 8_640]
    OUTLIER_UPPER_BOUND = 4723.295

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

        self.cb_params = {
            "loss_function": "RMSE",
            "boosting_type": "Plain",
            "l2_leaf_reg": 1,
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 100,  # A bit fewer iterations per model
            "random_seed": random_state,
            "early_stopping_rounds": 50,
            "verbose": False,
        }
        self.cb_params.update(cb_params)

        self.models: Dict[int, CatBoostRegressor] = {}
        self.fitted_: bool = False

    def _clip_outliers(self, series: pd.Series) -> pd.Series:
        return series.clip(upper=self.OUTLIER_UPPER_BOUND)

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        val_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.target_col_ = target_col
        feature_cols = feature_cols or []

        # 1. Clip outliers
        train_df_proc = train_df.copy()
        train_df_proc[target_col] = self._clip_outliers(train_df_proc[target_col])
        if val_df is not None:
            val_df_proc = val_df.copy()
            val_df_proc[target_col] = self._clip_outliers(val_df_proc[target_col])

        # 2. Prepare features once
        X = self._prepare_features(train_df_proc, target_col, feature_cols)
        self.feature_names_ = list(X.columns)

        for h in tqdm(range(1, self.horizon + 1), desc="Training direct models"):
            # 3. For each horizon `h`, create the target `y`
            y = train_df_proc[target_col].shift(-h)

            # 4. Align X and y
            combined = pd.concat([X, y.rename("target")], axis=1).dropna()
            X_train = combined[self.feature_names_]
            y_train = combined["target"]

            train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
            
            # Validation set (optional)
            eval_set = None
            # Note: For simplicity, validation is omitted here but can be added
            # by preparing features and targets for val_df similarly.

            # 5. Train and store a separate model for this horizon
            model = CatBoostRegressor(**self.cb_params)
            model.fit(train_pool, eval_set=eval_set)
            self.models[h] = model

        self.fitted_ = True
        print("Training complete.")
        return self

    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")

        # 1. Prepare features for the last available data point
        X = self._prepare_features(df_hist, self.target_col_, [])
        last_features = X.iloc[-1:]

        # 2. Predict each step of the horizon using its dedicated model
        predictions = []
        for h in range(1, self.horizon + 1):
            model = self.models[h]
            pred = model.predict(last_features)[0]
            predictions.append(pred)

        # 3. Create a timestamped Series for the forecast
        time_step = df_hist.index[1] - df_hist.index[0]
        last_time = df_hist.index[-1]
        future_times = pd.date_range(start=last_time + time_step, periods=self.horizon, freq=time_step)
        
        return pd.Series(predictions, index=future_times, name="forecast")

    def _prepare_features(
        self, df: pd.DataFrame, target_col: str, feature_cols: List[str]
    ) -> pd.DataFrame:
        df_copy = df.copy()
        
        if feature_cols:
            df_out = df_copy[feature_cols].copy()
        else:
            df_out = pd.DataFrame(index=df_copy.index)

        # Reusing the same rich feature set
        df_out = add_hour_sin_cos(df_out)
        df_out = add_dow_str(df_out)
        df_out["hour"] = df_out.index.hour.astype(str)
        df_out['is_weekend'] = (df_out.index.dayofweek >= 5).astype(int)
        
        df_out['f_target_diff_1'] = df_copy[target_col].diff(1)
        df_out['f_target_diff_2'] = df_copy[target_col].diff(2)
        df_out['f_target_diff_4'] = df_copy[target_col].diff(4)
        df_out['f_target_diff_96'] = df_copy[target_col].diff(96)
        
        df_out['f_target_diff_1_roll_std_4'] = df_out['f_target_diff_1'].rolling(window=4).std()
        
        df_out['f_target_roll_max_4'] = df_copy[target_col].rolling(window=4).max()
        df_out['f_target_roll_max_12'] = df_copy[target_col].rolling(window=12).max()
        
        df_out['f_target_ewm_mean_alpha_0_3'] = df_copy[target_col].ewm(alpha=0.3, adjust=False).mean()
        df_out['f_target_ewm_mean_alpha_0_1'] = df_copy[target_col].ewm(alpha=0.1, adjust=False).mean()
        
        lag_names = make_lag_features(df_copy, target_col, self.lags)
        roll_names = make_rolling_features(df_copy, target_col, self.roll_windows)
        
        feature_df = df_copy[lag_names + roll_names]
        df_out = pd.concat([df_out, feature_df], axis=1)

        return df_out.reindex(columns=self.feature_names_, fill_value=0) if self.fitted_ else df_out

    def save(self, path: str) -> None:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet. Cannot save.")
        os.makedirs(path, exist_ok=True)

        # Save each model in the ensemble
        for h, model in self.models.items():
            model.save_model(os.path.join(path, f"catboost_model_h{h}.cbm"))

        metadata = self.__dict__.copy()
        del metadata["models"]
        joblib.dump(metadata, os.path.join(path, "model_metadata.joblib"))
        print(f"Ensemble model saved to {path}")

    @classmethod
    def load(cls, path: str):
        metadata_path = os.path.join(path, "model_metadata.joblib")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found in directory: {path}")

        metadata = joblib.load(metadata_path)
        instance = cls.__new__(cls)
        instance.__dict__.update(metadata)

        # Load each model in the ensemble
        instance.models = {}
        for h in range(1, instance.horizon + 1):
            model_path = os.path.join(path, f"catboost_model_h{h}.cbm")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model for horizon {h} not found at {model_path}")
            
            model = CatBoostRegressor()
            model.load_model(model_path)
            instance.models[h] = model

        return instance 
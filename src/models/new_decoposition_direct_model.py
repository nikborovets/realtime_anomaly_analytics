import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

class DecompositionMultiOutputModel:
    """
    Улучшенная модель, использующая декомпозицию и прямую многоцелевую стратегию.

    1.  Данные логарифмируются.
    2.  `seasonal_decompose` разделяет ряд на тренд, сезонность и остатки.
    3.  Модель `LinearRegression` обучается для экстраполяции тренда.
    4.  Сезонный компонент сохраняется для будущего прогноза.
    5.  **Ключевое отличие:** `CatBoostRegressor` обучается предсказывать
        **сразу весь горизонт остатков** (например, 240 шагов) за один раз.
        Это устраняет проблему накопления ошибки рекурсивного прогноза.
    6.  Итоговый прогноз — это сумма трех компонентов с обратным преобразованием.
    """
    
    # Значения по умолчанию для лагов и окон
    DEFAULT_LAGS = [1, 2, 4, 96, 192, 240, 5760] # Добавлен лаг на горизонт
    DEFAULT_ROLL = [4, 96, 192, 1920, 2880, 4320, 5760, 8640]

    def __init__(
        self,
        horizon: int = 240,  # 1 час с частотой 15 сек = 240 шагов
        period: int = 5760,  # Суточная сезонность: 24*60*4=5760
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
        self.cat_features = cat_features or ["hour", "dow", "is_weekend"]
        
        self.feature_names_ = []
        self.target_col_ = None
        self.train_start_time_ = None
        self.time_step_seconds_ = None

        self.trend_model = LinearRegression()
        self.seasonal_component_ = None

        self.cb_params = {
            "loss_function": "MultiRMSE", # Специальная функция потерь для мульти-выхода
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 1500,
            "random_seed": random_state,
            "early_stopping_rounds": 100,
            "verbose": False,
        }
        self.cb_params.update(cb_params)

        self.model = CatBoostRegressor(**self.cb_params)
        self.fitted_: bool = False

    def _prepare_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Готовит фичи на основе истории данных (до момента прогноза)."""
        df_out = pd.DataFrame(index=df.index)

        # Календарные фичи
        df_out["hour"] = df.index.hour.astype(str)
        df_out['dow'] = df.index.dayofweek.astype(str)
        df_out['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df_out['day_of_year'] = df.index.dayofyear
        df_out['week_of_year'] = df.index.isocalendar().week.astype(int)

        # Временной тренд
        df_out['time_idx'] = (df.index - self.train_start_time_).total_seconds()

        # Фичи из остатков
        for lag in self.lags:
            df_out[f'f_resid_lag_{lag}'] = df[target_col].shift(lag)

        for w in self.roll_windows:
            df_out[f'f_resid_roll_mean_{w}'] = df[target_col].rolling(w, min_periods=2).mean()
            df_out[f'f_resid_roll_std_{w}'] = df[target_col].rolling(w, min_periods=2).std()

        return df_out.reindex(sorted(df_out.columns), axis=1)

    def _prepare_multi_output_target(self, series: pd.Series) -> pd.DataFrame:
        """
        Создает матрицу целевых переменных Y для multi-output обучения.
        Каждая строка соответствует моменту времени t, а столбцы - это
        значения остатков в моменты t+1, t+2, ..., t+horizon.
        """
        target_df = pd.DataFrame(index=series.index)
        for i in range(1, self.horizon + 1):
            target_df[f'target_{i}'] = series.shift(-i)
        return target_df

    def fit(self, train_df: pd.DataFrame, target_col: str, val_df: Optional[pd.DataFrame] = None):
        self.target_col_ = target_col
        self.train_start_time_ = train_df.index[0]
        self.time_step_seconds_ = (train_df.index[1] - train_df.index[0]).total_seconds()

        # --- 1. ДЕКОМПОЗИЦИЯ ---
        df_proc = train_df.copy()
        df_proc[target_col] = np.log1p(df_proc[target_col])
        
        decomposition = seasonal_decompose(df_proc[target_col], model='additive', period=self.period)
        
        # --- 2. МОДЕЛИРОВАНИЕ ТРЕНДА ---
        trend_data = decomposition.trend.dropna()
        time_idx_trend = (trend_data.index - self.train_start_time_).total_seconds().to_numpy().reshape(-1, 1)
        self.trend_model.fit(time_idx_trend, trend_data)

        # --- 3. СЕЗОННЫЙ КОМПОНЕНТ ---
        self.seasonal_component_ = decomposition.seasonal.iloc[:self.period].to_numpy()

        # --- 4. МОДЕЛИРОВАНИЕ ОСТАТКОВ (CATBOOST MULTI-OUTPUT) ---
        residuals = decomposition.resid.dropna()
        
        # Подготовка фичей (X) и целевой матрицы (Y)
        X = self._prepare_features(residuals.to_frame(name=target_col), target_col)
        Y = self._prepare_multi_output_target(residuals)

        # Выравнивание X и Y, удаление NaN, появившихся из-за shift и rolling
        combined = pd.concat([X, Y], axis=1).dropna()
        X_train = combined[X.columns]
        Y_train = combined[Y.columns]
        self.feature_names_ = list(X_train.columns)

        # Подготовка валидационного сета (если есть)
        eval_set = None
        if val_df is not None:
            # Валидация будет проводиться на последнем доступном блоке данных
            # Модель должна предсказать блок размером `horizon`
            val_start_idx = len(X_train) - self.horizon
            X_val = X_train.iloc[val_start_idx:]
            Y_val = Y_train.iloc[val_start_idx:]
            
            # Убираем валидационные данные из трейна
            X_train = X_train.iloc[:val_start_idx]
            Y_train = Y_train.iloc[:val_start_idx]

            eval_set = (X_val, Y_val)

        print(f"Обучение на {len(X_train)} сэмплах, валидация на {len(X_val) if eval_set else 0}.")
        self.model.fit(X_train, Y_train, cat_features=self.cat_features, eval_set=eval_set)
        
        self.fitted_ = True
        return self

    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Модель еще не обучена.")

        # --- 1. ПОДГОТОВКА БУДУЩИХ КОМПОНЕНТОВ (ТРЕНД И СЕЗОННОСТЬ) ---
        last_time = df_hist.index[-1]
        future_times = pd.date_range(
            start=last_time + pd.to_timedelta(self.time_step_seconds_, 's'),
            periods=self.horizon,
            freq=pd.to_timedelta(self.time_step_seconds_, 's')
        )

        # Экстраполяция тренда
        time_idx_future = (future_times - self.train_start_time_).total_seconds().to_numpy().reshape(-1, 1)
        trend_future = self.trend_model.predict(time_idx_future)

        # Экстраполяция сезонности
        start_seasonal_idx = int(((last_time - self.train_start_time_).total_seconds() / self.time_step_seconds_) + 1)
        seasonal_indices = np.arange(start_seasonal_idx, start_seasonal_idx + self.horizon) % self.period
        seasonal_future = self.seasonal_component_[seasonal_indices]
        
        # --- 2. ПРОГНОЗ ОСТАТКОВ ОДНИМ ВЫЗОВОМ ---
        # Сначала нужно получить остатки из переданной истории
        hist_proc = df_hist.copy()
        hist_proc[self.target_col_] = np.log1p(hist_proc[self.target_col_])
        
        time_idx_hist = (hist_proc.index - self.train_start_time_).total_seconds().to_numpy().reshape(-1, 1)
        trend_hist = self.trend_model.predict(time_idx_hist)
        
        hist_seasonal_indices = (np.arange(len(hist_proc)) + (hist_proc.index[0] - self.train_start_time_).total_seconds() / self.time_step_seconds_) % self.period
        seasonal_hist = self.seasonal_component_[hist_seasonal_indices.astype(int)]
        
        resid_hist = hist_proc[self.target_col_] - trend_hist - seasonal_hist

        # Готовим фичи для последней точки в истории
        X_features = self._prepare_features(resid_hist.to_frame(self.target_col_), self.target_col_)
        X_pred = X_features.iloc[-1:].reindex(columns=self.feature_names_, fill_value=0)

        # Делаем прогноз всего горизонта за раз
        residuals_future = self.model.predict(X_pred)[0]

        # --- 3. ОБЪЕДИНЕНИЕ И ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ ---
        final_prediction_log = trend_future + seasonal_future + residuals_future
        predictions = np.expm1(final_prediction_log)
        
        # Убедимся, что прогноз не отрицательный (задержка не может быть < 0)
        predictions[predictions < 0] = 0

        return pd.Series(predictions, index=future_times, name="forecast")
    
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
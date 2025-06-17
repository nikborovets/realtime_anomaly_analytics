import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

class _SeqDatasetDecomp(Dataset):
    """Dataset producing (feature_seq, residual_target_seq) pairs."""
    def __init__(self, features: np.ndarray, residuals: np.ndarray, seq_len: int, horizon: int):
        self.features = features.astype(np.float32)
        self.residuals = residuals.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_samples = features.shape[0] - seq_len - horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.residuals[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return x, y

class _LSTMHead(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, horizon: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

class LSTMDecompMultivariateForecast:
    """LSTM + сезонно-трендовая декомпозиция.

    1. Выделяем тренд (LinearRegression) и суточную сезонку (seasonal_decompose).
    2. Учим LSTM предсказывать остатки (residual) на горизонте H.
    3. Финальный прогноз = trend + seasonal + predicted residual, затем expm1.
    """

    def __init__(
        self,
        target_col: str,
        horizon: int = 900,
        seq_len: int = 5760,
        period: int = 5760,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 3,
        min_delta: float = 1e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: Optional[str] = None,
        use_amp: bool = True,
    ):
        self.target_col = target_col
        self.horizon = horizon
        self.seq_len = seq_len
        self.period = period
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA не доступна, но запрошен GPU")
        self.use_amp = use_amp and self.device.startswith("cuda")

        self.feature_cols_: List[str] = []
        self.scaler_X_ = None
        self.scaler_resid_ = None
        self.trend_model_ = None
        self.seasonal_pattern_: np.ndarray = None
        self.train_start_time_: pd.Timestamp = None
        self.time_step_: pd.Timedelta = None

        self.model = None
        self.fitted_ = False

    # ───────────────── fit ───────────────────
    def fit(self, df: pd.DataFrame, val_split: float = 0.1):
        df = df.sort_index()
        self.feature_cols_ = list(df.columns)
        if self.target_col not in df.columns:
            raise ValueError("target_col missing in dataframe")

        # basic impute
        df_filled = df.fillna(method="ffill").fillna(method="bfill")

        # log-scale target for stabilisation
        log_target = np.log1p(df_filled[self.target_col])

        # decompose
        decomposition = seasonal_decompose(log_target, period=self.period, model="additive")
        trend = decomposition.trend.dropna()
        # fit global linear trend
        time_idx = (trend.index - trend.index[0]).total_seconds().values.reshape(-1,1)
        self.trend_model_ = LinearRegression().fit(time_idx, trend.values)

        self.train_start_time_ = df_filled.index[0]
        self.time_step_ = df_filled.index[1] - df_filled.index[0]
        self.seasonal_pattern_ = decomposition.seasonal.iloc[: self.period].values  # length = period

        # residuals (aligned)
        aligned = pd.DataFrame({
            "log": log_target,
            "trend": pd.Series(self._predict_trend(df_filled.index), index=df_filled.index),
            "seasonal": decomposition.seasonal,
        }).dropna()
        residuals = aligned["log"] - aligned["trend"] - aligned["seasonal"]

        # ---------- scale X ----------
        # add cyc features
        feat_df = df_filled.copy()
        feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df.index.hour / 24)
        feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df.index.hour / 24)
        # feat_df["dow_sin"] = np.sin(2 * np.pi * feat_df.index.dayofweek / 7)
        # feat_df["dow_cos"] = np.cos(2 * np.pi * feat_df.index.dayofweek / 7)
        feat_df["time_idx"] = (feat_df.index - self.train_start_time_).total_seconds() / 3600

        X_array = feat_df.values
        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X_array)

        # scale residuals
        self.scaler_resid_ = StandardScaler()
        resid_scaled = self.scaler_resid_.fit_transform(residuals.values.reshape(-1,1)).flatten()

        # ensure same length (residual shorter due to nan drop)
        valid_idx = residuals.index
        X_scaled = X_scaled[df_filled.index.get_indexer(valid_idx)]
        # now len(X_scaled)==len(resid_scaled)

        # ensure proper device context (avoid illegal memory access)
        if self.device.startswith("cuda"):
            torch.cuda.set_device(int(self.device.split(":")[1]))

        # split train/val
        n_total = len(valid_idx)
        n_val = int(n_total * val_split)
        train_feat = X_scaled[: n_total - n_val]
        train_resid = resid_scaled[: n_total - n_val]
        val_feat = X_scaled[n_total - n_val - self.seq_len - self.horizon + 1 :]
        val_resid = resid_scaled[n_total - n_val - self.seq_len - self.horizon + 1 :]

        train_ds = _SeqDatasetDecomp(train_feat, train_resid, self.seq_len, self.horizon)
        val_ds = _SeqDatasetDecomp(val_feat, val_resid, self.seq_len, self.horizon) if n_val else None

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False) if val_ds else None

        n_features = train_feat.shape[1]
        self.model = _LSTMHead(n_features, self.hidden_size, self.num_layers, self.horizon, self.dropout).to(self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        loss_fn = nn.MSELoss()

        best_val = float("inf"); epochs_bad = 0

        for epoch in range(self.n_epochs):
            self.model.train(); epoch_loss=0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimiser.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    preds = self.model(x_batch)
                    loss = loss_fn(preds, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
                epoch_loss += loss.item() * x_batch.size(0)
            epoch_loss /= len(train_loader.dataset)

            if val_loader:
                self.model.eval(); val_loss=0.0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            p = self.model(x_val)
                            val_loss += loss_fn(p, y_val).item() * x_val.size(0)
                val_loss /= len(val_loader.dataset)
                # early stop logic
                if val_loss + self.min_delta < best_val:
                    best_val = val_loss; epochs_bad = 0
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                else:
                    epochs_bad += 1
                    if epochs_bad >= self.patience:
                        print(f"Early stop at epoch {epoch+1}")
                        self.model.load_state_dict(best_state)
                        break
                print(f"[Ep {epoch+1}/{self.n_epochs}] train={epoch_loss:.5f} val={val_loss:.5f}")
            else:
                print(f"[Ep {epoch+1}/{self.n_epochs}] train={epoch_loss:.5f}")

        self.fitted_ = True
        return self

    # ───────────────── predict ──────────────────
    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Model not fitted")
        if len(df_hist) < self.seq_len:
            raise ValueError("need longer history")

        df_hist = df_hist.sort_index()
        df_hist_filled = df_hist.fillna(method="ffill").fillna(method="bfill")

        # features for history
        feat_df = df_hist_filled.copy()
        feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df.index.hour / 24)
        feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df.index.hour / 24)
        # feat_df["dow_sin"] = np.sin(2 * np.pi * feat_df.index.dayofweek / 7)
        # feat_df["dow_cos"] = np.cos(2 * np.pi * feat_df.index.dayofweek / 7)
        feat_df["time_idx"] = (feat_df.index - self.train_start_time_).total_seconds() / 3600

        X_hist_scaled = self.scaler_X_.transform(feat_df.values)
        last_seq = X_hist_scaled[-self.seq_len:]
        x = torch.tensor(last_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            pred_resid_scaled = self.model(x).cpu().numpy().flatten()

        # inverse scale residual
        dummy = pred_resid_scaled.reshape(-1,1)
        resid_pred = self.scaler_resid_.inverse_transform(dummy).flatten()

        # trend future
        time_step = self.time_step_
        last_time = df_hist.index[-1]
        future_times = pd.date_range(start=last_time + time_step, periods=self.horizon, freq=time_step)
        trend_future = self._predict_trend(future_times)
        seasonal_idx = ((future_times - self.train_start_time_).total_seconds() / self.time_step_.total_seconds()).astype(int) % self.period
        seasonal_future = self.seasonal_pattern_[seasonal_idx]

        log_pred = trend_future + seasonal_future + resid_pred
        final_pred = np.expm1(log_pred)
        final_pred[final_pred < 0] = 0
        return pd.Series(final_pred, index=future_times, name="forecast")

    # ───────────────── helpers ──────────────────
    def _predict_trend(self, index: pd.DatetimeIndex) -> np.ndarray:
        time_delta = (index - self.train_start_time_).total_seconds()
        # TimedeltaIndex.total_seconds() → Float64Index; convert to np.ndarray before reshape
        time_idx = np.asarray(time_delta).reshape(-1, 1)
        return self.trend_model_.predict(time_idx)

    # save / load (optional later) 
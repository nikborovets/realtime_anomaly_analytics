import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


class _SeqDataset(Dataset):
    """Torch dataset that produces (sequence, target) pairs for multivariate TS."""

    def __init__(
        self,
        data: np.ndarray,
        target_idx: int,
        seq_len: int,
        horizon: int,
    ):
        self.data = data.astype(np.float32)
        self.target_idx = target_idx
        self.seq_len = seq_len
        self.horizon = horizon

        self.n_samples = data.shape[0] - seq_len - horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        x = self.data[start:end]  # (seq_len, n_features)
        y_start = end
        y_end = end + self.horizon
        # Only target variable
        y = self.data[y_start:y_end, self.target_idx]
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
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        last = out[:, -1, :]  # (batch, hidden)
        pred = self.head(last)  # (batch, horizon)
        return pred


class LSTMMultivariateForecast:
    """LSTM sequence-to-multi-horizon forecaster.

    Uses past `seq_len` steps of multivariate data to predict the next `horizon` values of `target_col`.
    """

    def __init__(
        self,
        target_col: str,
        horizon: int = 900,
        seq_len: int = 5760,
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
            raise RuntimeError(f"Запрошен GPU {self.device}, но CUDA недоступна")

        self.use_amp = use_amp and self.device.startswith("cuda")
        self.scaler_ = None
        self.model = None
        self.feature_cols_ = None
        self.target_idx_ = None
        self.fitted_ = False

    # ────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        val_split: float = 0.1,
    ):
        """Train the LSTM on the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Multivariate series with a DatetimeIndex.
        val_split : float, optional
            Fraction of data to keep for validation, by default 0.1.
        """
        # Ensure sorted by time
        df = df.sort_index()

        # Remember columns
        self.feature_cols_ = list(df.columns)
        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not in dataframe")
        self.target_idx_ = self.feature_cols_.index(self.target_col)

        # Impute missing values by ffill then bfill as quick fix
        df_imputed = df.fillna(method="ffill").fillna(method="bfill")

        # Scale all features
        self.scaler_ = StandardScaler()
        values = df_imputed.values
        values_scaled = self.scaler_.fit_transform(values)

        # Train/val split by time order
        n_total = values_scaled.shape[0]
        n_val = int(n_total * val_split)
        train_data = values_scaled[: n_total - n_val]
        val_data = values_scaled[n_total - n_val - self.seq_len - self.horizon + 1 :]
        # ^ include overlap so that val sequences can be formed

        train_ds = _SeqDataset(train_data, self.target_idx_, self.seq_len, self.horizon)
        val_ds = _SeqDataset(val_data, self.target_idx_, self.seq_len, self.horizon) if n_val > 0 else None

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False) if val_ds else None

        # Build model
        n_features = values_scaled.shape[1]
        self.model = _LSTMHead(
            n_features=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            horizon=self.horizon,
            dropout=self.dropout,
        ).to(self.device)

        # set correct default CUDA device to avoid cross-GPU ops
        if self.device.startswith("cuda"):
            torch.cuda.set_device(int(self.device.split(":")[1]))

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        epochs_bad = 0

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimiser.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    preds = self.model(x)
                    loss = loss_fn(preds, y)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
                epoch_loss += loss.item() * x.size(0)
            epoch_loss /= len(train_loader.dataset)

            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            preds_val = self.model(x_val)
                            val_loss += loss_fn(preds_val, y_val).item() * x_val.size(0)
                    val_loss /= len(val_loader.dataset)
                # early stopping check
                if val_loss + self.min_delta < best_val:
                    best_val = val_loss
                    epochs_bad = 0
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                else:
                    epochs_bad += 1
                    if epochs_bad >= self.patience:
                        print(f"Early stop at epoch {epoch+1}")
                        if val_loader:  # restore best
                            self.model.load_state_dict(best_state)
                        break
                print(f"[Epoch {epoch+1}/{self.n_epochs}] train_loss={epoch_loss:.5f} val_loss={val_loss:.5f}")
            else:
                print(f"[Epoch {epoch+1}/{self.n_epochs}] train_loss={epoch_loss:.5f}")

        self.fitted_ = True
        return self

    def predict(self, df_hist: pd.DataFrame) -> pd.Series:
        """Generate forecast for the next `horizon` steps using history sequence from df_hist."""
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")

        # Need last seq_len rows
        if len(df_hist) < self.seq_len:
            raise ValueError(f"Need at least seq_len={self.seq_len} history points for prediction")

        df_hist = df_hist.sort_index()
        last_seq = df_hist.iloc[-self.seq_len :].copy()
        last_seq = last_seq.fillna(method="ffill").fillna(method="bfill")
        last_seq_scaled = self.scaler_.transform(last_seq.values)
        x = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(x).cpu().numpy().flatten()

        # Inverse transform: we need to map back only target column
        # Build dummy array for inverse transform (one row, n_features)
        dummy = np.zeros((self.horizon, len(self.feature_cols_)))
        dummy[:, self.target_idx_] = pred_scaled
        dummy_inv = self.scaler_.inverse_transform(dummy)
        preds = dummy_inv[:, self.target_idx_]

        # Build index for future timestamps
        time_step = df_hist.index[1] - df_hist.index[0]
        last_time = df_hist.index[-1]
        future_times = pd.date_range(start=last_time + time_step, periods=self.horizon, freq=time_step)

        return pd.Series(preds, index=future_times, name="forecast")

    # ────────────────────────────────────────────────────────────────
    #  (Optional) Save / load helpers
    # ────────────────────────────────────────────────────────────────
    def save(self, path: str):
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet; cannot save.")
        os.makedirs(path, exist_ok=True)
        # Save torch model weights
        torch.save(self.model.state_dict(), os.path.join(path, "lstm_state.pt"))
        # Save metadata (scaler, config, etc.) via joblib
        metadata = {
            "scaler_": self.scaler_,
            "feature_cols_": self.feature_cols_,
            "target_col": self.target_col,
            "target_idx_": self.target_idx_,
            "horizon": self.horizon,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "device": self.device,  # save agnostic
        }
        joblib.dump(metadata, os.path.join(path, "metadata.joblib"))
        print(f"Модель сохранена в {path}")

    @classmethod
    def load(cls, path: str):
        metadata_path = os.path.join(path, "metadata.joblib")
        state_dict_path = os.path.join(path, "lstm_state.pt")
        if not os.path.exists(metadata_path) or not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"Файлы модели не найдены в директории: {path}")

        metadata = joblib.load(metadata_path)
        instance = cls(
            target_col=metadata["target_col"],
            horizon=metadata["horizon"],
            seq_len=metadata["seq_len"],
            hidden_size=metadata["hidden_size"],
            num_layers=metadata["num_layers"],
            dropout=metadata["dropout"],
            device=metadata.get("device", "cpu"),
        )
        instance.scaler_ = metadata["scaler_"]
        instance.feature_cols_ = metadata["feature_cols_"]
        instance.target_idx_ = metadata["target_idx_"]

        # Build model architecture and load weights
        n_features = len(instance.feature_cols_)
        instance.model = _LSTMHead(
            n_features=n_features,
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            horizon=instance.horizon,
            dropout=instance.dropout,
        )
        instance.model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
        instance.model.eval()
        instance.fitted_ = True
        return instance 
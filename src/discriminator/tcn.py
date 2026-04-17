from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..config import DEVICE, SEQUENCE_LENGTH


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = F.relu(self.norm2(self.conv2(h)))
        h = self.dropout(h)
        return h + res


class TCNDiscriminator(nn.Module):
    DILATIONS = [1, 2, 4, 8, 16, 32]
    KERNEL_SIZE = 3

    def __init__(self, in_channels: int = 2, hidden_channels: int = 64):
        super().__init__()
        blocks: List[nn.Module] = []
        ch_in = in_channels
        for d in self.DILATIONS:
            blocks.append(TCNBlock(ch_in, hidden_channels, self.KERNEL_SIZE, d))
            ch_in = hidden_channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        return self.head(x).squeeze(-1)


def build_tcn(sequence_length: int = SEQUENCE_LENGTH) -> TCNDiscriminator:
    return TCNDiscriminator(in_channels=2, hidden_channels=64)


def _compute_accuracy(model: TCNDiscriminator, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = (torch.sigmoid(model(xb)) >= 0.5).long()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)


def train_tcn(
    X_returns: np.ndarray,
    X_regimes: np.ndarray,
    y: np.ndarray,
    n_ensemble: int = 5,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> Dict:
    from sklearn.linear_model import LogisticRegression

    X = np.stack([X_returns, X_regimes], axis=-1).astype(np.float32)
    n = len(y)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_cal, y_cal = X[n_train + n_val :], y[n_train + n_val :]

    ds_train = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    ds_val = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
    ds_cal = TensorDataset(torch.tensor(X_cal), torch.tensor(y_cal, dtype=torch.long))
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size)
    loader_cal = DataLoader(ds_cal, batch_size=batch_size)

    models: List[TCNDiscriminator] = []
    memorisation_flags: List[bool] = []

    for seed_idx in range(n_ensemble):
        print(f"  Training TCN ensemble member {seed_idx + 1}/{n_ensemble}...")
        torch.manual_seed(seed_idx * 42)
        np.random.seed(seed_idx * 42)

        model = build_tcn().to(DEVICE)
        optim_ = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_, patience=3, factor=0.5
        )
        best_val_loss, wait, best_state = float("inf"), 0, None

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader_train:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                optim_.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim_.step()

            model.eval()
            val_loss, n_vb = 0.0, 0
            with torch.no_grad():
                for xb, yb in loader_val:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
                    val_loss += F.binary_cross_entropy_with_logits(model(xb), yb).item()
                    n_vb += 1
            val_loss /= max(n_vb, 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss, wait = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print(f"    Early stop at epoch {epoch + 1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        train_acc = _compute_accuracy(model, loader_train)
        val_acc = _compute_accuracy(model, loader_val)
        memorised = train_acc >= 0.90 and val_acc <= 0.60
        memorisation_flags.append(memorised)
        print(
            f"    train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
            f"memorised={'YES — will not deploy' if memorised else 'no'}"
        )
        models.append(model)

    deployed = [m for m, flag in zip(models, memorisation_flags) if not flag]
    if len(deployed) == 0:
        print("  WARNING: All ensemble members memorised. Deploying all with caution.")
        deployed = models

    print("  Platt scaling on calibration set...")
    all_probs_cal: List[np.ndarray] = []
    all_labels_cal: List[np.ndarray] = []
    for xb, yb in loader_cal:
        xb = xb.to(DEVICE)
        member_probs = []
        with torch.no_grad():
            for m in deployed:
                m.eval()
                member_probs.append(torch.sigmoid(m(xb)).cpu().numpy())
        all_probs_cal.append(np.mean(member_probs, axis=0))
        all_labels_cal.append(yb.numpy())

    cal_probs = np.concatenate(all_probs_cal)
    cal_labels = np.concatenate(all_labels_cal)

    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(cal_probs.reshape(-1, 1), cal_labels)

    cal_acc = ((cal_probs >= 0.5).astype(int) == cal_labels).mean()
    print(f"  Calibration accuracy (pre-Platt): {cal_acc:.3f}")
    print(f"  Deployed {len(deployed)}/{n_ensemble} ensemble members")

    return {
        "models": deployed,
        "platt_scaler": platt,
        "memorisation_flags": memorisation_flags,
        "n_deployed": len(deployed),
    }


def predict_tcn(
    X_returns: np.ndarray,
    X_regimes: np.ndarray,
    tcn_dict: Dict,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    models = tcn_dict["models"]
    platt = tcn_dict["platt_scaler"]
    X = np.stack([X_returns, X_regimes], axis=-1).astype(np.float32)
    ds = TensorDataset(torch.tensor(X))
    loader = DataLoader(ds, batch_size=batch_size)

    all_probs: List[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(DEVICE)
        member_out = []
        with torch.no_grad():
            for m in models:
                m.eval()
                member_out.append(torch.sigmoid(m(xb)).cpu().numpy())
        all_probs.append(np.mean(member_out, axis=0))

    raw_probs = np.concatenate(all_probs)
    calibrated = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    labels = (calibrated >= 0.5).astype(int)
    return labels, calibrated

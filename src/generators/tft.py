from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..config import DEVICE, N_REGIMES, WINDOW


class MixtureOfLogisticsHead(nn.Module):
    def __init__(self, d_in: int, n_components: int = 5):
        super().__init__()
        self.n_components = n_components
        self.proj = nn.Linear(d_in, n_components * 3)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.proj(x)
        means = out[:, : self.n_components]
        log_scales = out[:, self.n_components : 2 * self.n_components]
        logits = out[:, 2 * self.n_components :]
        return means, log_scales, logits

    @staticmethod
    def nll(
        y: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        y = y.unsqueeze(-1)
        centered = y - means
        s = torch.exp(log_scales) + 1e-8
        inv_s = 1.0 / s
        log_weights = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        log_pdf = -log_scales - centered * inv_s - 2.0 * F.softplus(-centered * inv_s)
        return -torch.logsumexp(log_pdf + log_weights, dim=-1).mean()

    def sample(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.softmax(logits, dim=-1)
        comp = torch.multinomial(weights, 1).squeeze(-1)
        mu = means[torch.arange(len(comp)), comp]
        s = torch.exp(log_scales[torch.arange(len(comp)), comp]) + 1e-8
        u = torch.rand_like(mu).clamp(1e-6, 1 - 1e-6)
        return mu + s * torch.log(u / (1 - u))


class TFTModel(nn.Module):
    def __init__(
        self,
        window: int = WINDOW,
        d_model: int = 64,
        d_regime: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_mix: int = 5,
    ):
        super().__init__()
        self.window = window
        self.d_total = d_model + d_regime

        self.return_proj = nn.Linear(1, d_model)
        self.regime_proj = nn.Linear(N_REGIMES, d_regime)
        self.pos_embed = nn.Parameter(torch.randn(1, window, self.d_total) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_total,
            nhead=n_heads,
            dim_feedforward=self.d_total * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = MixtureOfLogisticsHead(self.d_total, n_mix)

    def forward(
        self, windows: torch.Tensor, regimes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = windows.shape
        ret_emb = self.return_proj(windows.unsqueeze(-1))
        reg_oh = F.one_hot(regimes, N_REGIMES).float().unsqueeze(1).expand(-1, T, -1)
        reg_emb = self.regime_proj(reg_oh)
        x = torch.cat([ret_emb, reg_emb], dim=-1) + self.pos_embed[:, :T, :]
        x = self.encoder(x)
        return self.head(x[:, -1, :])


def train_tft(
    X_windows: np.ndarray,
    X_regimes: np.ndarray,
    y_targets: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 10,
    lr: float = 1e-3,
) -> Tuple[TFTModel, float]:
    n = len(y_targets)
    split = int(0.85 * n)
    ds_train = TensorDataset(
        torch.tensor(X_windows[:split]),
        torch.tensor(X_regimes[:split]),
        torch.tensor(y_targets[:split]),
    )
    ds_val = TensorDataset(
        torch.tensor(X_windows[split:]),
        torch.tensor(X_regimes[split:]),
        torch.tensor(y_targets[split:]),
    )
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size)

    model = TFTModel().to(DEVICE)
    optim_ = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, wait, best_state = float("inf"), 0, None

    for epoch in range(epochs):
        model.train()
        for xw, xr, yt in loader_train:
            xw, xr, yt = xw.to(DEVICE), xr.to(DEVICE), yt.to(DEVICE)
            means, log_scales, logits = model(xw, xr)
            loss = MixtureOfLogisticsHead.nll(yt, means, log_scales, logits)
            optim_.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xw, xr, yt in loader_val:
                xw, xr, yt = xw.to(DEVICE), xr.to(DEVICE), yt.to(DEVICE)
                m, ls, lo = model(xw, xr)
                val_loss += MixtureOfLogisticsHead.nll(yt, m, ls, lo).item()
        val_loss /= max(len(loader_val), 1)

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    residuals = []
    with torch.no_grad():
        for xw, xr, yt in loader_val:
            xw, xr, yt = xw.to(DEVICE), xr.to(DEVICE), yt.to(DEVICE)
            m, ls, lo = model(xw, xr)
            weights = torch.softmax(lo, dim=-1)
            pred = (weights * m).sum(dim=-1)
            residuals.append((yt - pred).cpu().numpy())
    residual_std = float(np.std(np.concatenate(residuals))) if residuals else 1e-4
    return model, residual_std


def tft_generate_next(
    window: np.ndarray,
    regime: int,
    tft_model: TFTModel,
    residual_std: float,
) -> float:
    tft_model.eval()
    with torch.no_grad():
        xw = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        xr = torch.tensor([regime], dtype=torch.long).to(DEVICE)
        means, log_scales, logits = tft_model(xw, xr)
        sample = tft_model.head.sample(means, log_scales, logits)
    noise = np.random.randn() * residual_std * 0.1
    return float(sample.cpu().item()) + noise

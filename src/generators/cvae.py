from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..config import DEVICE, N_REGIMES, WINDOW


class CVAEModel(nn.Module):
    def __init__(
        self,
        window: int = WINDOW,
        latent_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        enc_in = window + N_REGIMES + 1
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        dec_in = latent_dim + window + N_REGIMES
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self, window: torch.Tensor, regime_oh: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([window, regime_oh, target.unsqueeze(-1)], dim=-1)
        h = self.encoder(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(
        self, z: torch.Tensor, window: torch.Tensor, regime_oh: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([z, window, regime_oh], dim=-1)
        return self.decoder(x).squeeze(-1)

    def forward(
        self, window: torch.Tensor, regime: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        regime_oh = F.one_hot(regime, N_REGIMES).float()
        mu, logvar = self.encode(window, regime_oh, target)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z, window, regime_oh)
        return recon, mu, logvar


def cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4.0,
) -> Tuple[torch.Tensor, float]:
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, kl.item()


def train_cvae(
    X_windows: np.ndarray,
    X_regimes: np.ndarray,
    y_targets: np.ndarray,
    epochs: int = 60,
    batch_size: int = 256,
    beta: float = 4.0,
    lr: float = 1e-3,
    patience: int = 10,
) -> Tuple[CVAEModel, float]:
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

    model = CVAEModel().to(DEVICE)
    optim_ = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, wait, best_state = float("inf"), 0, None

    for epoch in range(epochs):
        model.train()
        epoch_kl = 0.0
        n_batches = 0
        for xw, xr, yt in loader_train:
            xw, xr, yt = xw.to(DEVICE), xr.to(DEVICE), yt.to(DEVICE)
            recon, mu, logvar = model(xw, xr, yt)
            loss, kl_val = cvae_loss(recon, yt, mu, logvar, beta)
            optim_.zero_grad()
            loss.backward()
            optim_.step()
            epoch_kl += kl_val
            n_batches += 1

        mean_kl = epoch_kl / max(n_batches, 1)
        if mean_kl < 0.05 and epoch > 5:
            beta = beta * 0.5

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xw, xr, yt in loader_val:
                xw, xr, yt = xw.to(DEVICE), xr.to(DEVICE), yt.to(DEVICE)
                recon, mu, logvar = model(xw, xr, yt)
                loss, _ = cvae_loss(recon, yt, mu, logvar, beta)
                val_loss += loss.item()
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
            regime_oh = F.one_hot(xr, N_REGIMES).float()
            z = torch.randn(xw.size(0), model.latent_dim, device=DEVICE)
            pred = model.decode(z, xw, regime_oh)
            residuals.append((yt - pred).cpu().numpy())
    residual_std = float(np.std(np.concatenate(residuals))) if residuals else 1e-4
    return model, residual_std


def cvae_generate_next(
    window: np.ndarray,
    regime: int,
    cvae_model: CVAEModel,
    residual_std: float,
) -> float:
    cvae_model.eval()
    with torch.no_grad():
        xw = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        regime_oh = F.one_hot(
            torch.tensor([regime], device=DEVICE), N_REGIMES
        ).float()
        z = torch.randn(1, cvae_model.latent_dim, device=DEVICE)
        pred = cvae_model.decode(z, xw, regime_oh)
    noise = np.random.randn() * residual_std * 0.1
    return float(pred.cpu().item()) + noise

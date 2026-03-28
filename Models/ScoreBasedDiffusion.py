from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def cosine_beta_schedule(T: int) -> torch.Tensor:
    s = 0.008
    steps = torch.linspace(0, T, T + 1)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0.0001, 0.999)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma(cond) * x + self.beta(cond)


class DiffusionDenoiser(nn.Module):
    def __init__(
        self,
        window: int = WINDOW,
        hidden_dim: int = 256,
        time_dim: int = 32,
        regime_dim: int = 16,
        T: int = 100,
    ):
        super().__init__()
        self.T = T

        self.ctx_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ctx_proj = nn.Linear(64, hidden_dim)

        self.time_embed = nn.Sequential(
            nn.Embedding(T, time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )

        self.regime_embed = nn.Linear(N_REGIMES, regime_dim)
        self.film = FiLM(regime_dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        noisy_y: torch.Tensor,
        t: torch.Tensor,
        window: torch.Tensor,
        regime: torch.Tensor,
    ) -> torch.Tensor:
  
        ctx = self.ctx_conv(window.unsqueeze(1))  # (B, 64, 1)
        ctx = self.ctx_proj(ctx.squeeze(-1))  

        t_emb = self.time_embed(t)  

        h = ctx + t_emb

        reg_oh = F.one_hot(regime, N_REGIMES).float()
        reg_emb = self.regime_embed(reg_oh)
        h = self.film(h, reg_emb)

        h = torch.cat([h, noisy_y.unsqueeze(-1)], dim=-1)
        return self.net(h).squeeze(-1)


class ScoreDiffusionModel:
    def __init__(self, T_train: int = 100, T_infer: int = 20):
        self.T_train = T_train
        self.T_infer = T_infer
        betas = cosine_beta_schedule(T_train)
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.denoiser = DiffusionDenoiser(T=T_train).to(DEVICE)
        self.alpha_bar_dev = self.alpha_bar.to(DEVICE)

    def q_sample(
        self, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        ab = self.alpha_bar_dev[t]
        return torch.sqrt(ab) * y0 + torch.sqrt(1 - ab) * noise

    def train_step(
        self,
        y0: torch.Tensor,
        window: torch.Tensor,
        regime: torch.Tensor,
        optim_: torch.optim.Optimizer,
    ) -> float:
        t = torch.randint(0, self.T_train, (y0.size(0),), device=DEVICE)
        noise = torch.randn_like(y0)
        noisy_y = self.q_sample(y0, t, noise)
        pred_noise = self.denoiser(noisy_y, t, window, regime)
        loss = F.mse_loss(pred_noise, noise)
        optim_.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        optim_.step()
        return loss.item()

    @torch.no_grad()
    def ddim_sample(
        self, window: torch.Tensor, regime: torch.Tensor, w: float = 2.0
    ) -> torch.Tensor:

        B = window.size(0)
        y = torch.randn(B, device=DEVICE)

        step_size = self.T_train // self.T_infer
        timesteps = list(range(self.T_train - 1, -1, -step_size))

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=DEVICE, dtype=torch.long)

            eps_cond = self.denoiser(y, t, window, regime)

            null_regime = torch.zeros(B, dtype=torch.long, device=DEVICE)
            eps_uncond = self.denoiser(y, t, window, null_regime)

            eps = (1 + w) * eps_cond - w * eps_uncond

            ab_t = self.alpha_bar_dev[t_val]
            pred_y0 = (y - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                ab_prev = self.alpha_bar_dev[t_prev]
                y = torch.sqrt(ab_prev) * pred_y0 + torch.sqrt(1 - ab_prev) * eps
            else:
                y = pred_y0
        return y


def train_diffusion(
    X_windows: np.ndarray,
    X_regimes: np.ndarray,
    y_targets: np.ndarray,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    cfg_drop_prob: float = 0.1,
) -> Tuple[ScoreDiffusionModel, float]:

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

    diff = ScoreDiffusionModel()
    optim_ = torch.optim.Adam(diff.denoiser.parameters(), lr=lr)

    for epoch in range(epochs):
        diff.denoiser.train()
        for xw, xr, yt in loader_train:
            xw, xr, yt = xw.to(DEVICE), xr.to(DEVICE), yt.to(DEVICE)
            # Classifier-free guidance: randomly zero out regime
            drop_mask = torch.rand(xr.size(0), device=DEVICE) < cfg_drop_prob
            xr = xr * (~drop_mask).long()
            diff.train_step(yt, xw, xr, optim_)

    diff.denoiser.eval()

    residuals = []
    with torch.no_grad():
        for xw, xr, yt in loader_val:
            xw, xr, yt = xw.to(DEVICE), xr.to(DEVICE), yt.to(DEVICE)
            pred = diff.ddim_sample(xw, xr, w=2.0)
            residuals.append((yt - pred).cpu().numpy())
    residual_std = float(np.std(np.concatenate(residuals))) if residuals else 1e-4
    return diff, residual_std


def diffusion_generate_next(
    window: np.ndarray,
    regime: int,
    diff_model: ScoreDiffusionModel,
    residual_std: float,
) -> float:
    diff_model.denoiser.eval()
    with torch.no_grad():
        xw = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        xr = torch.tensor([regime], dtype=torch.long).to(DEVICE)
        sample = diff_model.ddim_sample(xw, xr, w=2.0)
    noise = np.random.randn() * residual_std * 0.1
    return float(sample.cpu().item()) + noise

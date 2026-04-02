import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def cosine_beta_schedule(T: int) -> torch.Tensor:
    s = 0.008
    steps = torch.linspace(0, T, T + 1)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0.0001, 0.999)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.gamma(cond).unsqueeze(-1) 
        b = self.beta(cond).unsqueeze(-1)
        return g * x + b


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film = FiLM(cond_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class SequenceDenoiser(nn.Module):
    def __init__(
        self,
        seq_len: int = 252,
        channels: int = 128,
        n_blocks: int = 8,
        time_dim: int = 64,
        n_regimes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_regimes = n_regimes
        self.time_dim = time_dim

        self.input_proj = nn.Conv1d(1, channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

        self.regime_mlp = nn.Sequential(
            nn.Linear(n_regimes, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

        dilations = [2 ** (i % 4) for i in range(n_blocks)]
        self.blocks = nn.ModuleList(
            [ResidualBlock1d(channels, channels, d) for d in dilations]
        )

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(
        self,
        noisy_seq: torch.Tensor,
        t: torch.Tensor,
        regime: torch.Tensor,
    ) -> torch.Tensor:
   
        x = noisy_seq.unsqueeze(1)      
        x = self.input_proj(x)    

        t_emb = sinusoidal_embedding(t, self.time_dim)  
        cond = self.time_mlp(t_emb) + self.regime_mlp(
            F.one_hot(regime, self.n_regimes).float()
        )                                     

        for block in self.blocks:
            x = block(x, cond)

        return self.output_proj(x).squeeze(1)          


class ScoreDiffusionModel:
    def __init__(
        self,
        seq_len: int = 252,
        n_regimes: int = 3,
        T_train: int = 100,
        T_infer: int = 20,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.seq_len = seq_len
        self.T_train = T_train
        self.T_infer = T_infer
        self.device = device

        betas = cosine_beta_schedule(T_train)
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)

        self.denoiser = SequenceDenoiser(
            seq_len=seq_len, n_regimes=n_regimes
        ).to(device)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        ab = self.alpha_bar[t].unsqueeze(-1)   
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

    def train_step(
        self,
        x0: torch.Tensor,
        regime: torch.Tensor,
        optim: torch.optim.Optimizer,
    ) -> float:
        
        B = x0.size(0)
        t = torch.randint(0, self.T_train, (B,), device=self.device)
        noise = torch.randn_like(x0)
        noisy = self.q_sample(x0, t, noise)
        pred_noise = self.denoiser(noisy, t, regime)
        loss = F.mse_loss(pred_noise, noise)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        optim.step()
        
        return loss.item()

    @torch.no_grad()
    def ddim_sample(
        self,
        regime: torch.Tensor,
        n_samples: int | None = None,
        cfg_weight: float = 2.0,
    ) -> torch.Tensor:

        if n_samples is not None:
            B = n_samples
            regime = regime.expand(B)
        else:
            B = regime.size(0)

        x = torch.randn(B, self.seq_len, device=self.device)

        step_size = max(self.T_train // self.T_infer, 1)
        timesteps = list(range(self.T_train - 1, -1, -step_size))

        null_regime = torch.zeros(B, dtype=torch.long, device=self.device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=self.device, dtype=torch.long)

            eps_cond = self.denoiser(x, t, regime)
            eps_uncond = self.denoiser(x, t, null_regime)
            eps = (1 + cfg_weight) * eps_cond - cfg_weight * eps_uncond

            ab_t = self.alpha_bar[t_val]
            pred_x0 = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                ab_prev = self.alpha_bar[t_prev]
                x = (
                    torch.sqrt(ab_prev) * pred_x0
                    + torch.sqrt(1 - ab_prev) * eps
                )
            else:
                x = pred_x0

        return x  
        

def train_diffusion(
    sequences: np.ndarray,
    regimes: np.ndarray,
    seq_len: int = 252,
    n_regimes: int = 3,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    cfg_drop_prob: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> ScoreDiffusionModel:

    N = len(sequences)
    split = int(0.85 * N)

    ds_train = TensorDataset(
        torch.tensor(sequences[:split], dtype=torch.float32),
        torch.tensor(regimes[:split], dtype=torch.long),
    )
    ds_val = TensorDataset(
        torch.tensor(sequences[split:], dtype=torch.float32),
        torch.tensor(regimes[split:], dtype=torch.long),
    )
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size)

    model = ScoreDiffusionModel(
        seq_len=seq_len, n_regimes=n_regimes, device=device
    )
    optim = torch.optim.AdamW(model.denoiser.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.denoiser.train()
        for x0, reg in loader_train:
            x0, reg = x0.to(device), reg.to(device)
            # randomly zero-out regime
            drop_mask = torch.rand(reg.size(0), device=device) < cfg_drop_prob
            reg = reg * (~drop_mask).long()
            model.train_step(x0, reg, optim)
        scheduler.step()

        model.denoiser.eval()
        val_losses = []
        with torch.no_grad():
            for x0, reg in loader_val:
                x0, reg = x0.to(device), reg.to(device)
                B = x0.size(0)
                t = torch.randint(0, model.T_train, (B,), device=device)
                noise = torch.randn_like(x0)
                noisy = model.q_sample(x0, t, noise)
                pred = model.denoiser(noisy, t, reg)
                val_losses.append(F.mse_loss(pred, noise).item())
        avg_val = sum(val_losses) / max(len(val_losses), 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {
                k: v.cpu().clone()
                for k, v in model.denoiser.state_dict().items()
            }

        if (epoch + 1) % 10 == 0:
            print(f"  [Diffusion] epoch {epoch+1}/{epochs}  val_loss={avg_val:.6f}")

    if best_state is not None:
        model.denoiser.load_state_dict(best_state)
        model.denoiser.to(device)
    model.denoiser.eval()

    return model


def generate_sequences(
    model: ScoreDiffusionModel,
    regime: int,
    n_sequences: int = 10,
    cfg_weight: float = 2.0,
) -> np.ndarray:

    model.denoiser.eval()
    reg_tensor = torch.tensor([regime], dtype=torch.long, device=model.device)
    samples = model.ddim_sample(
        regime=reg_tensor, n_samples=n_sequences, cfg_weight=cfg_weight
    )
    return samples.cpu().numpy()

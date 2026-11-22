"""Generative models for mixed-type tabular data.

All three families share the same representation: a vector
``x = [standardised numeric | one-hot categorical]`` and the same class
conditioning mechanism (a learned label embedding).  This keeps capacity and
conditioning comparable; the differences are the generative objective
(variational bound vs. adversarial game vs. denoising score matching).

References
----------
Kingma & Welling, Auto-Encoding Variational Bayes, ICLR 2014.
Goodfellow et al., Generative Adversarial Nets, NeurIPS 2014.
Ho, Jain & Abbeel, Denoising Diffusion Probabilistic Models, NeurIPS 2020.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp_block(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2,
              dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class LabelEmbedding(nn.Module):
    def __init__(self, n_classes: int = 2, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(n_classes, dim)
        self.dim = dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.emb(y.long())


# --------------------------------------------------------------------------- #
# VAE
# --------------------------------------------------------------------------- #

class TabularVAE(nn.Module):
    def __init__(self, n_num: int, n_cat: int, n_classes: int = 2,
                 hidden: int = 256, latent: int = 16, emb_dim: int = 16):
        super().__init__()
        self.n_num, self.n_cat = n_num, n_cat
        self.latent = latent
        self.label = LabelEmbedding(n_classes, emb_dim)
        d_in = n_num + n_cat + emb_dim
        self.encoder = mlp_block(d_in, hidden, hidden, n_layers=2)
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = mlp_block(latent + emb_dim, hidden, n_num + n_cat, n_layers=2)

    def encode(self, num, cat, y):
        h = self.encoder(torch.cat([num, cat, self.label(y)], dim=1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def decode(self, z, y):
        out = self.decoder(torch.cat([z, self.label(y)], dim=1))
        return out[:, : self.n_num], out[:, self.n_num :]

    def forward(self, num, cat, y):
        mu, logvar = self.encode(num, cat, y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        num_hat, cat_logits = self.decode(z, y)
        return num_hat, cat_logits, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor, device):
        z = torch.randn(n, self.latent, device=device)
        return self.decode(z, y)


def vae_loss(model, num, cat, y, cat_slices, beta: float = 1.0):
    """ELBO: Gaussian reconstruction for numerics + per-feature CE for categories."""
    num_hat, cat_logits, mu, logvar = model(num, cat, y)
    recon = F.mse_loss(num_hat, num, reduction="sum") / num.shape[0]
    ce = cat_logits.new_zeros(())
    for sl in cat_slices:
        ce = ce + F.cross_entropy(cat_logits[:, sl], cat[:, sl].argmax(dim=1))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon + ce + beta * kl, {"recon": recon.detach(), "ce": ce.detach(), "kl": kl.detach()}


# --------------------------------------------------------------------------- #
# GAN (WGAN-GP)
# --------------------------------------------------------------------------- #

class TabularGenerator(nn.Module):
    def __init__(self, n_num: int, n_cat: int, n_classes: int = 2, z_dim: int = 32,
                 hidden: int = 256, emb_dim: int = 16):
        super().__init__()
        self.n_num, self.n_cat, self.z_dim = n_num, n_cat, z_dim
        self.label = LabelEmbedding(n_classes, emb_dim)
        self.net = mlp_block(z_dim + emb_dim, hidden, n_num + n_cat, n_layers=2)

    def forward(self, z, y):
        out = self.net(torch.cat([z, self.label(y)], dim=1))
        num = torch.tanh(out[:, : self.n_num]) * 3.0
        return num, out[:, self.n_num :]

    @torch.no_grad()
    def sample(self, n, y, device):
        z = torch.randn(n, self.z_dim, device=device)
        return self(z, y)


class TabularDiscriminator(nn.Module):
    def __init__(self, n_num: int, n_cat: int, n_classes: int = 2,
                 hidden: int = 256, emb_dim: int = 16):
        super().__init__()
        self.label = LabelEmbedding(n_classes, emb_dim)
        self.net = mlp_block(n_num + n_cat + emb_dim, hidden, 1, n_layers=2)

    def forward(self, num, cat, y):
        return self.net(torch.cat([num, cat, self.label(y)], dim=1)).squeeze(1)


def gradient_penalty(disc, real_num, real_cat, fake_num, fake_cat, y, device):
    b = real_num.shape[0]
    eps = torch.rand(b, 1, device=device)
    eps_num = eps
    eps_cat = eps
    inter_num = (eps_num * real_num + (1 - eps_num) * fake_num).requires_grad_(True)
    inter_cat = (eps_cat * real_cat + (1 - eps_cat) * fake_cat).requires_grad_(True)
    score = disc(inter_num, inter_cat, y)
    grads = torch.autograd.grad(
        outputs=score.sum(), inputs=[inter_num, inter_cat], create_graph=True
    )
    norm = torch.sqrt(sum(g.pow(2).sum(dim=1) for g in grads) + 1e-8)
    return ((norm - 1.0) ** 2).mean()


# --------------------------------------------------------------------------- #
# DDPM
# --------------------------------------------------------------------------- #

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000.0), dim // 2)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class TabularDenoiser(nn.Module):
    def __init__(self, n_num: int, n_cat: int, n_classes: int = 2,
                 hidden: int = 256, emb_dim: int = 16, t_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_num, self.n_cat = n_num, n_cat
        self.label = LabelEmbedding(n_classes, emb_dim)
        self.time = SinusoidalEmbedding(t_dim)
        self.inp = nn.Linear(n_num + n_cat + t_dim + emb_dim, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU())
             for _ in range(n_layers)]
        )
        self.out = nn.Linear(hidden, n_num + n_cat)
        self.act = nn.SiLU()

    def forward(self, x, t, y):
        h = self.inp(torch.cat([x, self.time(t), self.label(y)], dim=1))
        h = self.act(h)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


class GaussianDiffusion:
    """Linear-beta DDPM with DDIM sampling (Ho et al., 2020)."""

    def __init__(self, n_steps: int = 400, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str | torch.device = "cuda"):
        self.n_steps = n_steps
        self.device = torch.device(device)
        betas = torch.linspace(beta_start, beta_end, n_steps, device=self.device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_ab = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_ab = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise):
        return self.sqrt_ab[t].unsqueeze(1) * x0 + self.sqrt_1m_ab[t].unsqueeze(1) * noise

    def training_loss(self, model, x0, y):
        b = x0.shape[0]
        t = torch.randint(0, self.n_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t, y)
        return F.mse_loss(pred, noise)

    def training_loss_hybrid(self, model, x0, y, n_num: int, cat_slices):
        """Noise prediction for numeric dims, x0 classification for categoricals."""
        b = x0.shape[0]
        t = torch.randint(0, self.n_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t, y)
        loss_num = F.mse_loss(pred[:, :n_num], noise[:, :n_num])
        loss_cat = pred.new_zeros(())
        for sl in cat_slices:
            full = slice(n_num + sl.start, n_num + sl.stop)
            loss_cat = loss_cat + F.cross_entropy(pred[:, full], x0[:, full].argmax(dim=1))
        return loss_num + loss_cat

    @torch.no_grad()
    def ddim_sample_hybrid(self, model, n: int, y: torch.Tensor, n_num: int,
                           cat_slices, steps: int = 50):
        """DDIM sampling for the hybrid parameterisation.

        Returns (numeric samples, categorical probabilities).
        """
        x = torch.randn(n, n_num + model.n_cat, device=self.device)
        ts = torch.linspace(self.n_steps - 1, 0, steps, device=self.device).long()
        cat_probs = None
        for i, t in enumerate(ts):
            t_batch = torch.full((n,), int(t), device=self.device, dtype=torch.long)
            pred = model(x, t_batch, y)
            ab = self.alphas_cumprod[t]
            eps_num = pred[:, :n_num]
            x0_num = ((x[:, :n_num] - torch.sqrt(1 - ab) * eps_num) / torch.sqrt(ab)).clamp(-5.0, 5.0)
            cat_probs = torch.cat(
                [F.softmax(pred[:, n_num + sl.start : n_num + sl.stop], dim=1) for sl in cat_slices],
                dim=1,
            )
            x0 = torch.cat([x0_num, cat_probs], dim=1)
            eps_cat = (x[:, n_num:] - torch.sqrt(ab) * cat_probs) / torch.sqrt(1 - ab)
            if i < len(ts) - 1:
                ab_next = self.alphas_cumprod[ts[i + 1]]
                x_num = torch.sqrt(ab_next) * x0_num + torch.sqrt(1 - ab_next) * eps_num
                x_cat = torch.sqrt(ab_next) * cat_probs + torch.sqrt(1 - ab_next) * eps_cat
                x = torch.cat([x_num, x_cat], dim=1)
            else:
                x = x0
        return x[:, :n_num], cat_probs

    @torch.no_grad()
    def ddim_sample(self, model, n: int, y: torch.Tensor, n_num: int,
                    steps: int = 50) -> torch.Tensor:
        x = torch.randn(n, n_num + model.n_cat, device=self.device)
        ts = torch.linspace(self.n_steps - 1, 0, steps, device=self.device).long()
        for i, t in enumerate(ts):
            t_batch = torch.full((n,), int(t), device=self.device, dtype=torch.long)
            eps = model(x, t_batch, y)
            ab = self.alphas_cumprod[t]
            x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            if n_num:
                x0[:, :n_num] = x0[:, :n_num].clamp(-5.0, 5.0)
            x0[:, n_num:] = x0[:, n_num:].clamp(0.0, 1.0)
            if i < len(ts) - 1:
                ab_next = self.alphas_cumprod[ts[i + 1]]
                x = torch.sqrt(ab_next) * x0 + torch.sqrt(1 - ab_next) * eps
            else:
                x = x0
        return x


class EMA:
    """Exponential moving average of model parameters (used for DDPM sampling)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

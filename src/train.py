"""Training loops for the three tabular generators."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import PreparedData
from .models import (
    EMA,
    GaussianDiffusion,
    TabularDenoiser,
    TabularDiscriminator,
    TabularGenerator,
    TabularVAE,
    gradient_penalty,
    vae_loss,
)
from .utils import get_logger, save_json, set_seed

log = get_logger()


@dataclass
class TrainConfig:
    hidden: int = 256
    emb_dim: int = 16
    latent: int = 16
    z_dim: int = 32
    batch_size: int = 512
    vae_epochs: int = 90
    vae_lr: float = 2e-3
    gan_epochs: int = 30
    gan_lr: float = 1e-4
    n_critic: int = 2
    gp_lambda: float = 10.0
    ddpm_epochs: int = 100
    ddpm_lr: float = 2e-3
    ddpm_steps: int = 400
    ema_decay: float = 0.999
    ddim_steps: int = 50
    max_steps: int | None = None


def _batch_indices(n: int, bs: int, device, shuffle: bool = True, generator=None):
    if shuffle:
        idx = torch.randperm(n, device=device, generator=generator)
    else:
        idx = torch.arange(n, device=device)
    for i in range(0, n, bs):
        yield idx[i : i + bs]


def _hard_categorical(logits: torch.Tensor, cat_slices, tau: float = 1.0) -> torch.Tensor:
    outs = [F.gumbel_softmax(logits[:, sl], tau=tau, hard=True) for sl in cat_slices]
    return torch.cat(outs, dim=1)


def _to_tensors(data: PreparedData, device):
    tr = data.train.enc
    num = torch.tensor(tr.num, device=device)
    cat = torch.tensor(tr.cat, device=device)
    y = torch.tensor(tr.y, device=device)
    return num, cat, y


def train_vae(data: PreparedData, seed: int, cfg: TrainConfig, device) -> tuple[TabularVAE, dict]:
    set_seed(seed)
    n_num, n_cat = data.preprocessor.n_num, data.preprocessor.n_cat
    model = TabularVAE(n_num, n_cat, 2, cfg.hidden, cfg.latent, cfg.emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.vae_lr, weight_decay=1e-4)
    num, cat, y = _to_tensors(data, device)
    n = len(num)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.vae_epochs)
    history = []
    step = 0
    for epoch in range(cfg.vae_epochs):
        beta = min(1.0, (epoch + 1) / 10.0)
        ep = {"recon": 0.0, "ce": 0.0, "kl": 0.0, "n": 0}
        model.train()
        for idx in _batch_indices(n, cfg.batch_size, device):
            loss, parts = vae_loss(model, num[idx], cat[idx], y[idx],
                                   data.preprocessor.cat_slices, beta=beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in parts.items():
                ep[k] += float(v)
            ep["n"] += 1
            step += 1
            if cfg.max_steps and step >= cfg.max_steps:
                break
        sched.step()
        history.append({"epoch": epoch, **{k: v / max(ep["n"], 1) for k, v in ep.items() if k != "n"}})
        if epoch % 20 == 0 or epoch == cfg.vae_epochs - 1:
            log.info("  [VAE s%d] ep %3d recon=%.3f ce=%.3f kl=%.3f",
                     seed, epoch, history[-1]["recon"], history[-1]["ce"], history[-1]["kl"])
        if cfg.max_steps and step >= cfg.max_steps:
            break
    return model, {"history": history}


def train_gan(data: PreparedData, seed: int, cfg: TrainConfig, device) -> tuple[TabularGenerator, dict]:
    set_seed(seed + 101)
    n_num, n_cat = data.preprocessor.n_num, data.preprocessor.n_cat
    G = TabularGenerator(n_num, n_cat, 2, cfg.z_dim, cfg.hidden, cfg.emb_dim).to(device)
    D = TabularDiscriminator(n_num, n_cat, 2, cfg.hidden, cfg.emb_dim).to(device)
    opt_g = torch.optim.AdamW(G.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.9), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(D.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.9), weight_decay=1e-4)
    num, cat, y = _to_tensors(data, device)
    n = len(num)
    history = []
    step = 0
    for epoch in range(cfg.gan_epochs):
        ep = {"d": 0.0, "g": 0.0, "gp": 0.0, "n": 0}
        for idx in _batch_indices(n, cfg.batch_size, device):
            if len(idx) < 8:
                continue
            rn, rc, ry = num[idx], cat[idx], y[idx]
            b = len(idx)
            for _ in range(cfg.n_critic):
                z = torch.randn(b, cfg.z_dim, device=device)
                fn, fl = G(z, ry)
                fc = _hard_categorical(fl, data.preprocessor.cat_slices)
                d_real = D(rn, rc, ry)
                d_fake = D(fn.detach(), fc.detach(), ry)
                gp = gradient_penalty(D, rn, rc, fn.detach(), fc.detach(), ry, device)
                loss_d = d_fake.mean() - d_real.mean() + cfg.gp_lambda * gp
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0)
                opt_d.step()
            z = torch.randn(b, cfg.z_dim, device=device)
            fn, fl = G(z, ry)
            fc = _hard_categorical(fl, data.preprocessor.cat_slices)
            loss_g = -D(fn, fc, ry).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 10.0)
            opt_g.step()
            ep["d"] += float(loss_d); ep["g"] += float(loss_g); ep["gp"] += float(gp)
            ep["n"] += 1
            step += 1
            if cfg.max_steps and step >= cfg.max_steps:
                break
        history.append({"epoch": epoch, **{k: v / max(ep["n"], 1) for k, v in ep.items() if k != "n"}})
        if epoch % 10 == 0 or epoch == cfg.gan_epochs - 1:
            log.info("  [GAN s%d] ep %3d d=%.3f g=%.3f gp=%.3f", seed, epoch,
                     history[-1]["d"], history[-1]["g"], history[-1]["gp"])
        if cfg.max_steps and step >= cfg.max_steps:
            break
    return G, {"history": history}


def train_ddpm(data: PreparedData, seed: int, cfg: TrainConfig, device) -> tuple[TabularDenoiser, dict]:
    set_seed(seed + 202)
    n_num, n_cat = data.preprocessor.n_num, data.preprocessor.n_cat
    model = TabularDenoiser(n_num, n_cat, 2, cfg.hidden, cfg.emb_dim).to(device)
    diffusion = GaussianDiffusion(cfg.ddpm_steps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ddpm_lr, weight_decay=1e-4)
    num, cat, y = _to_tensors(data, device)
    x0_all = torch.cat([num, cat], dim=1)
    n = len(x0_all)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.ddpm_epochs)
    ema = EMA(model, decay=cfg.ema_decay)
    history = []
    step = 0
    for epoch in range(cfg.ddpm_epochs):
        ep_loss, nb = 0.0, 0
        model.train()
        for idx in _batch_indices(n, cfg.batch_size, device):
            loss = diffusion.training_loss_hybrid(model, x0_all[idx], y[idx],
                                                  n_num, data.preprocessor.cat_slices)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            ep_loss += float(loss)
            nb += 1
            step += 1
            if cfg.max_steps and step >= cfg.max_steps:
                break
        sched.step()
        history.append({"epoch": epoch, "loss": ep_loss / max(nb, 1)})
        if epoch % 25 == 0 or epoch == cfg.ddpm_epochs - 1:
            log.info("  [DDPM s%d] ep %3d loss=%.4f", seed, epoch, history[-1]["loss"])
        if cfg.max_steps and step >= cfg.max_steps:
            break
    ema.copy_to(model)
    return model, {"history": history}


def train_generator(kind: str, data: PreparedData, seed: int, cfg: TrainConfig,
                    device, run_dir: Path) -> dict:
    t0 = time.time()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Training %s on %s (seed=%d, n=%d)", kind, data.name, seed, len(data.train.enc.y))
    if kind == "vae":
        model, extra = train_vae(data, seed, cfg, device)
    elif kind == "gan":
        model, extra = train_gan(data, seed, cfg, device)
    elif kind == "ddpm":
        model, extra = train_ddpm(data, seed, cfg, device)
    else:
        raise ValueError(kind)
    elapsed = time.time() - t0
    ckpt = {
        "kind": kind,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": asdict(cfg),
        "meta": data.meta,
        "seed": seed,
    }
    torch.save(ckpt, run_dir / "model.pt")
    record = {"kind": kind, "dataset": data.name, "ratio": data.ratio, "seed": seed,
              "seconds": elapsed, "n_train": int(len(data.train.enc.y)), **extra}
    save_json(record, run_dir / "train_log.json")
    log.info("Finished %s seed=%d in %.1fs", kind, seed, elapsed)
    return record


def load_generator(kind: str, run_dir: Path, data: PreparedData, device):
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["cfg"])
    n_num, n_cat = data.preprocessor.n_num, data.preprocessor.n_cat
    if kind == "vae":
        model = TabularVAE(n_num, n_cat, 2, cfg.hidden, cfg.latent, cfg.emb_dim)
    elif kind == "gan":
        model = TabularGenerator(n_num, n_cat, 2, cfg.z_dim, cfg.hidden, cfg.emb_dim)
    elif kind == "ddpm":
        model = TabularDenoiser(n_num, n_cat, 2, cfg.hidden, cfg.emb_dim)
    else:
        raise ValueError(kind)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def sample_synthetic(kind: str, model, data: PreparedData, n: int, seed: int,
                     device, cfg: TrainConfig, temperature: float = 1.0,
                     hard: bool = False) -> "object":
    """Draw n class-1 (minority) synthetic rows, decoded to the raw feature space."""
    from .data import Encoding

    set_seed(seed + 777)
    pre = data.preprocessor
    y = torch.ones(n, dtype=torch.long, device=device)
    if kind == "vae":
        num_t, cat_t = model.sample(n, y, device)
        num, cat, mode = num_t.cpu().numpy(), cat_t.cpu().numpy(), "logits"
    elif kind == "gan":
        num_t, cat_t = model.sample(n, y, device)
        num, cat, mode = num_t.cpu().numpy(), cat_t.cpu().numpy(), "logits"
    elif kind == "ddpm":
        diffusion = GaussianDiffusion(cfg.ddpm_steps, device=device)
        num_t, probs = diffusion.ddim_sample_hybrid(model, n, y, pre.n_num,
                                                    pre.cat_slices, steps=cfg.ddim_steps)
        num, cat, mode = num_t.cpu().numpy(), probs.cpu().numpy(), "prob"
    else:
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    df = pre.to_frame(num, cat, mode=mode, temperature=temperature, rng=rng, hard=hard)
    enc = pre.transform(df)
    return df, Encoding(num=enc.num, cat=enc.cat, y=np.ones(n, dtype=np.int64))

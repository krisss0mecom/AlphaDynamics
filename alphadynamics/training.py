from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from .data import ProteinTrajectory, TransitionBatcher, as_tensor, residue_ids, temperature_tensor
from .models import AlphaDynamicsModel
from .utils import count_parameters


def train_transfer(
    proteins: list[ProteinTrajectory],
    model: AlphaDynamicsModel,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    log_every: int = 200,
    callback=None,
) -> AlphaDynamicsModel:
    batcher = TransitionBatcher(proteins, batch_size=batch_size, seed=seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    running = []
    for step in range(1, steps + 1):
        protein, x_np, y_np = batcher.sample()
        x = as_tensor(x_np, device)
        y = as_tensor(y_np, device)
        res_ids = residue_ids(protein.sequence, protein.n_residues, device)
        temp = temperature_tensor(protein.temperature_K, protein.n_residues, x.shape[0], device)
        loss = model.nll(x, y, residue_ids=res_ids, temperature=temp)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(float(loss.detach().cpu()))
        if callback and (step == 1 or step % log_every == 0 or step == steps):
            callback(step, float(np.mean(running)), protein.domain_id)
            running = []
    return model


def adapt_few_shot(
    base_model: AlphaDynamicsModel,
    protein: ProteinTrajectory,
    n_seed_frames: int,
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    seed: int = 42,
    train_full_model: bool = True,
) -> AlphaDynamicsModel:
    model = copy.deepcopy(base_model).to(device)
    if not train_full_model:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    seed_protein = protein.seed_train(n_seed_frames)
    if seed_protein.n_train < 2 or steps <= 0:
        return model
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    train = as_tensor(seed_protein.train, device)
    res_ids = residue_ids(seed_protein.sequence, seed_protein.n_residues, device)
    n_pairs = seed_protein.n_train - 1
    bsz = min(batch_size, n_pairs)
    model.train()
    for _ in range(steps):
        idx_np = rng.integers(0, n_pairs, size=bsz)
        idx = torch.from_numpy(idx_np).long().to(device)
        x = train[idx]
        y = train[idx + 1]
        temp = temperature_tensor(seed_protein.temperature_K, seed_protein.n_residues, x.shape[0], device)
        loss = model.nll(x, y, residue_ids=res_ids, temperature=temp)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
    return model


def save_checkpoint(
    path: str | Path,
    model: AlphaDynamicsModel,
    step: int,
    extra: dict | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "step": int(step),
        "n_params": count_parameters(model),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, p)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[AlphaDynamicsModel, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = AlphaDynamicsModel.from_config(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt

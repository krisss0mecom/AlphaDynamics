from __future__ import annotations

import numpy as np
import torch

from .data import ProteinTrajectory, as_tensor, residue_ids, temperature_tensor
from .metrics import canonical_jsd
from .models import AlphaDynamicsModel


@torch.no_grad()
def rollout(
    model: AlphaDynamicsModel,
    seed_frame: np.ndarray,
    n_steps: int,
    device: torch.device,
    sequence: str | None = None,
    kappa_mult: float = 1.0,
    greedy: bool = False,
    temperature_K: float | None = None,
) -> np.ndarray:
    model.eval()
    state = as_tensor(seed_frame[None], device)
    res_ids = residue_ids(sequence, seed_frame.shape[0], device)
    temp = temperature_tensor(temperature_K, seed_frame.shape[0], 1, device)
    out = np.empty((n_steps, seed_frame.shape[0], 2), dtype=np.float32)
    for i in range(n_steps):
        state = model.sample_next(
            state,
            residue_ids=res_ids,
            temperature=temp,
            kappa_mult=kappa_mult,
            greedy=greedy,
        )
        out[i] = state[0].detach().cpu().numpy()
    return out


@torch.no_grad()
def rollout_batch_few_shot(
    model: AlphaDynamicsModel,
    warmup_frames: np.ndarray,
    ensemble_size: int,
    n_steps: int,
    device: torch.device,
    sequence: str | None = None,
    kappa_mult: float = 1.0,
    greedy: bool = False,
    temperature_K: float | None = None,
) -> np.ndarray:
    """Few-shot reservoir washout rollout.

    Feeds REAL MD frames as teacher-forced primer (W steps), theta accumulates,
    then autonomous rollout from last warmup state for n_steps.

    warmup_frames: (W, N, 2) — real MD frames from val (single sequence)
    ensemble_size: number of stochastic rollout samples to spawn from end-of-warmup state
    n_steps: number of autonomous frames to generate
    Returns (B, n_steps, N, 2)
    """
    if warmup_frames.ndim != 3 or warmup_frames.shape[-1] != 2:
        raise ValueError(f"warmup_frames must be (W, N, 2), got {warmup_frames.shape}")
    model.eval()
    W, N, _ = warmup_frames.shape
    B = ensemble_size
    res_ids = residue_ids(sequence, N, device)
    temp = temperature_tensor(temperature_K, N, B, device)

    # Replicate warmup to ensemble dim: (W, B, N, 2)
    warmup_t = as_tensor(warmup_frames, device).unsqueeze(1).expand(W, B, N, 2).contiguous()

    # Teacher-forced warmup: feed each MD frame, theta accumulates
    from .models import build_features
    theta = None
    for t in range(W):
        x = warmup_t[t]
        features = build_features(
            x, residue_ids=res_ids,
            use_sequence=model.use_sequence,
            temperature=temp, use_temperature=model.use_temperature,
        )
        theta = model.core(features, theta_init=theta)

    # State at end of warmup: last real MD frame, replicated B-fold
    state = warmup_t[-1]

    # Autonomous rollout n_steps
    out = torch.empty((n_steps, B, N, 2), dtype=torch.float32, device=device)
    for i in range(n_steps):
        state, theta = model.sample_next_with_state(
            state, theta_prev=theta, residue_ids=res_ids, temperature=temp,
            kappa_mult=kappa_mult, greedy=greedy,
        )
        out[i] = state
    return out.permute(1, 0, 2, 3).contiguous().cpu().numpy()


@torch.no_grad()
def rollout_batch_stateful(
    model: AlphaDynamicsModel,
    seed_frames: np.ndarray,
    n_steps: int,
    device: torch.device,
    sequence: str | None = None,
    kappa_mult: float = 1.0,
    greedy: bool = False,
    temperature_K: float | None = None,
    burn_in: int = 0,
) -> np.ndarray:
    """Stateful (theta-persistent) rollout. Uses sample_next_with_state.

    burn_in: discard first `burn_in` autonomous frames (theta still accumulates).
    Returns (B, T_kept, N, 2) where T_kept = n_steps - burn_in.
    """
    if seed_frames.ndim != 3 or seed_frames.shape[-1] != 2:
        raise ValueError(f"seed_frames must be (B, N, 2), got {seed_frames.shape}")
    model.eval()
    B, N, _ = seed_frames.shape
    state = as_tensor(seed_frames, device)
    res_ids = residue_ids(sequence, N, device)
    temp = temperature_tensor(temperature_K, N, B, device)
    out = torch.empty((max(n_steps - burn_in, 0), B, N, 2), dtype=torch.float32, device=device)
    theta = None
    kept = 0
    for i in range(n_steps):
        state, theta = model.sample_next_with_state(
            state, theta_prev=theta, residue_ids=res_ids, temperature=temp,
            kappa_mult=kappa_mult, greedy=greedy,
        )
        if i >= burn_in:
            out[kept] = state
            kept += 1
    return out.permute(1, 0, 2, 3).contiguous().cpu().numpy()


@torch.no_grad()
def rollout_batch(
    model: AlphaDynamicsModel,
    seed_frames: np.ndarray,
    n_steps: int,
    device: torch.device,
    sequence: str | None = None,
    kappa_mult: float = 1.0,
    greedy: bool = False,
    temperature_K: float | None = None,
) -> np.ndarray:
    """Batched rollout. seed_frames: (B, N, 2) -> returns (B, T, N, 2).

    Concatenating along the time axis matches the per-member rollout
    semantic used by eval_seq_only_controls.py.
    """
    if seed_frames.ndim != 3 or seed_frames.shape[-1] != 2:
        raise ValueError(f"seed_frames must be (B, N, 2), got {seed_frames.shape}")
    model.eval()
    B, N, _ = seed_frames.shape
    state = as_tensor(seed_frames, device)
    res_ids = residue_ids(sequence, N, device)
    temp = temperature_tensor(temperature_K, N, B, device)
    out = torch.empty((n_steps, B, N, 2), dtype=torch.float32, device=device)
    for i in range(n_steps):
        state = model.sample_next(
            state,
            residue_ids=res_ids,
            temperature=temp,
            kappa_mult=kappa_mult,
            greedy=greedy,
        )
        out[i] = state
    # (T, B, N, 2) -> (B, T, N, 2) so concatenation along time is trivial
    return out.permute(1, 0, 2, 3).contiguous().cpu().numpy()


def rollout_report(
    model: AlphaDynamicsModel,
    protein: ProteinTrajectory,
    n_steps: int,
    device: torch.device,
    n_bins: int = 36,
    kappa_mult: float = 1.0,
    greedy: bool = False,
) -> tuple[np.ndarray, dict]:
    roll = rollout(
        model,
        seed_frame=protein.val[0],
        n_steps=n_steps,
        device=device,
        sequence=protein.sequence,
        kappa_mult=kappa_mult,
        greedy=greedy,
        temperature_K=protein.temperature_K,
    )
    rep = canonical_jsd(roll, protein.val, n_bins=n_bins)
    return roll, rep.to_dict()

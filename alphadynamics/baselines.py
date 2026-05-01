from __future__ import annotations

import numpy as np

from .data import ProteinTrajectory
from .metrics import canonical_jsd


def _wrap(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def identity_rollout(protein: ProteinTrajectory, n_steps: int) -> np.ndarray:
    frame = protein.val[0]
    return np.repeat(frame[None], n_steps, axis=0).astype(np.float32)


def gaussian_step_rollout(protein: ProteinTrajectory, n_steps: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    deltas = _wrap(protein.train[1:] - protein.train[:-1])
    mu = deltas.mean(axis=0)
    std = deltas.std(axis=0) + 1e-4
    state = protein.val[0].copy()
    out = np.empty((n_steps, protein.n_residues, 2), dtype=np.float32)
    for i in range(n_steps):
        state = _wrap(state + rng.normal(mu, std).astype(np.float32))
        out[i] = state
    return out


def ar1_rollout(protein: ProteinTrajectory, n_steps: int, seed: int = 42) -> np.ndarray:
    """Circular AR(1)-style baseline using sin/cos features per torsion."""
    rng = np.random.default_rng(seed)
    train = protein.train
    x = np.concatenate([np.sin(train[:-1]), np.cos(train[:-1])], axis=-1)
    y = np.concatenate([np.sin(train[1:]), np.cos(train[1:])], axis=-1)
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    xtx = x_flat.T @ x_flat + 1e-3 * np.eye(x_flat.shape[1])
    w = np.linalg.solve(xtx, x_flat.T @ y_flat)
    resid = y_flat - x_flat @ w
    sigma = resid.std(axis=0) + 1e-4
    state = protein.val[0].copy()
    out = np.empty((n_steps, protein.n_residues, 2), dtype=np.float32)
    for i in range(n_steps):
        feat = np.concatenate([np.sin(state), np.cos(state)], axis=-1).reshape(1, -1)
        pred = feat @ w + rng.normal(0.0, sigma, size=(1, sigma.size))
        pred = pred.reshape(protein.n_residues, 4)
        sin_part = pred[:, :2]
        cos_part = pred[:, 2:]
        state = np.arctan2(sin_part, cos_part).astype(np.float32)
        out[i] = state
    return out


def evaluate_baselines(protein: ProteinTrajectory, n_steps: int, n_bins: int, seed: int = 42) -> dict:
    rows = {}
    for name, fn in [
        ("identity", lambda: identity_rollout(protein, n_steps)),
        ("gaussian_step", lambda: gaussian_step_rollout(protein, n_steps, seed=seed)),
        ("ar1_circular", lambda: ar1_rollout(protein, n_steps, seed=seed)),
    ]:
        roll = fn()
        rows[name] = canonical_jsd(roll, protein.val, n_bins=n_bins).to_dict()
    return rows


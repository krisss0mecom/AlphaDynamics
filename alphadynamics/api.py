"""High-level inference API.

This is the entry point most users want::

    from alphadynamics import predict_torsion_ensemble
    traj = predict_torsion_ensemble("AAAY", n_ensemble=16, rollout_steps=2500)

For finer control over the model, use :func:`alphadynamics.load_pretrained`
and the lower-level rollout functions in :mod:`alphadynamics.rollout`.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .data import as_tensor, residue_ids, temperature_tensor
from .rollout import rollout
from .weights import load_pretrained


_AA_VOCAB = set("ACDEFGHIKLMNPQRSTVWYX")


def _check_sequence(sequence: str) -> str:
    seq = sequence.strip().upper()
    bad = sorted(set(seq) - _AA_VOCAB)
    if bad:
        raise ValueError(
            f"sequence contains non-standard residues {bad!r}; "
            f"allowed alphabet is {sorted(_AA_VOCAB)}"
        )
    if len(seq) < 1:
        raise ValueError("sequence is empty")
    return seq


def _seed_frame(n_residues: int, rng: np.random.Generator) -> np.ndarray:
    """Random uniform torsion seed in [-pi, pi]."""
    return rng.uniform(-math.pi, math.pi, size=(n_residues, 2)).astype(np.float32)


def predict_torsion_ensemble(
    sequence: str,
    *,
    model_name: str = "ad_transfer_v2_clean",
    n_ensemble: int = 16,
    rollout_steps: int = 2500,
    seed: int | None = 42,
    device: str | torch.device | None = None,
    kappa_mult: float = 1.0,
    temperature_K: float | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Predict an ensemble of torsion-angle trajectories for a sequence.

    Parameters
    ----------
    sequence : str
        One-letter amino-acid string (e.g. ``"AAAY"``). Length determines the
        number of residues. ``X`` is allowed for unknown.
    model_name : str
        Pretrained checkpoint to use. Default: ``"ad_transfer_v2_clean"``.
    n_ensemble : int
        Number of independent rollouts (each with a different seed frame).
    rollout_steps : int
        Number of trajectory frames to predict per ensemble member.
    seed : int or None
        Base RNG seed for reproducibility. ``None`` = nondeterministic.
    device : str, torch.device, or None
        ``"cuda"`` / ``"cpu"`` / specific device; ``None`` = auto.
    kappa_mult : float
        Multiplier on the von Mises concentration parameters. ``1.0`` =
        standard. Lower = wider sampling.
    temperature_K : float or None
        Optional temperature conditioning (Kelvin) for temperature-aware models.
    show_progress : bool
        Show a progress message.

    Returns
    -------
    np.ndarray of shape ``(n_ensemble, rollout_steps, n_residues, 2)`` and
    dtype ``float32``. Last axis is ``[phi, psi]`` in radians, wrapped to
    ``[-pi, pi]``.
    """
    seq = _check_sequence(sequence)
    n_res = len(seq)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    model = load_pretrained(model_name, device=device_t, eval_mode=True)

    rng = np.random.default_rng(seed)
    output = np.empty((n_ensemble, rollout_steps, n_res, 2), dtype=np.float32)

    total_steps = n_ensemble * rollout_steps

    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(
            total=total_steps,
            desc=f"  predict {seq}",
            unit="step",
            unit_scale=False,
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.1,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]",
        )
        progress_callback = lambda n=1: pbar.update(n)
    else:
        pbar = None
        progress_callback = None

    try:
        for k in range(n_ensemble):
            seed_frame = _seed_frame(n_res, rng)
            traj = rollout(
                model,
                seed_frame,
                n_steps=rollout_steps,
                device=device_t,
                sequence=seq,
                kappa_mult=kappa_mult,
                greedy=False,
                temperature_K=temperature_K,
                progress_callback=progress_callback,
            )
            output[k] = traj
    finally:
        if pbar is not None:
            pbar.close()

    return output

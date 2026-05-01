from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


EPS = 1e-12


@dataclass
class JSDReport:
    mode: str
    mean_jsd: float
    per_residue_jsd: list[float]
    n_bins: int
    n_rollout_frames: int
    n_gt_frames: int
    gt_source: str
    smoothing: str
    includes_train_in_target: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _hist2d(x: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=(edges, edges))
    if hist.sum() == 0:
        return np.full((n_bins, n_bins), 1.0 / (n_bins * n_bins), dtype=np.float64)
    return (hist / hist.sum()).astype(np.float64)


def histogram2d(
    angles_2d: np.ndarray,
    n_bins: int,
    mode: Literal["canonical", "visual"] = "canonical",
) -> np.ndarray:
    hist = _hist2d(angles_2d, n_bins)
    if mode == "canonical":
        return hist
    if mode == "visual":
        if gaussian_filter is None:
            raise RuntimeError("visual mode requires scipy")
        smooth = gaussian_filter(hist * hist.size, sigma=1.0, mode="wrap")
        return smooth / smooth.sum()
    raise ValueError(f"unknown metric mode: {mode}")


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64) + EPS
    q = q.astype(np.float64) + EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5 * (kl_pm + kl_qm))


def per_residue_jsd(
    rollout: np.ndarray,
    target: np.ndarray,
    n_bins: int = 36,
    mode: Literal["canonical", "visual"] = "canonical",
) -> list[float]:
    if rollout.ndim != 3 or target.ndim != 3:
        raise ValueError("rollout and target must be shaped (T, N, 2)")
    if rollout.shape[1] != target.shape[1]:
        raise ValueError("rollout and target must have same residue count")
    out = []
    for r in range(rollout.shape[1]):
        p = histogram2d(rollout[:, r, :], n_bins=n_bins, mode=mode)
        q = histogram2d(target[:, r, :], n_bins=n_bins, mode=mode)
        out.append(jsd(p, q))
    return out


def canonical_jsd(rollout: np.ndarray, val: np.ndarray, n_bins: int = 36) -> JSDReport:
    vals = per_residue_jsd(rollout, val, n_bins=n_bins, mode="canonical")
    return JSDReport(
        mode="canonical",
        mean_jsd=float(np.mean(vals)),
        per_residue_jsd=[float(v) for v in vals],
        n_bins=int(n_bins),
        n_rollout_frames=int(rollout.shape[0]),
        n_gt_frames=int(val.shape[0]),
        gt_source="held_out_val",
        smoothing="none",
        includes_train_in_target=False,
    )


def visual_jsd(rollout: np.ndarray, train: np.ndarray, val: np.ndarray, n_bins: int = 36) -> JSDReport:
    target = np.concatenate([train, val], axis=0)
    vals = per_residue_jsd(rollout, target, n_bins=n_bins, mode="visual")
    return JSDReport(
        mode="visual",
        mean_jsd=float(np.mean(vals)),
        per_residue_jsd=[float(v) for v in vals],
        n_bins=int(n_bins),
        n_rollout_frames=int(rollout.shape[0]),
        n_gt_frames=int(target.shape[0]),
        gt_source="train_plus_val",
        smoothing="gaussian_sigma_1.0",
        includes_train_in_target=True,
    )


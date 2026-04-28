"""AR(1) circular baseline for the aligned audit.

Predict x_{t+1} = wrap(x_t + delta) where delta ~ von Mises(mu_delta, kappa_delta)
per torsion.  Trained by NLL on training pairs.  This is the minimal stochastic
baseline that respects torus periodicity but uses *no* learnable feature
embedding — strictly weaker than the MLP baseline.

Trained per domain on the same train split as the v1 audit, evaluated on the
same val split.  Reports per-domain AR(1) NLL alongside MLP and PF_t4.

Usage:
  python src/ar1_baseline.py \\
      --data_dir mdcath_real_data/mdcath_348K \\
      --out results/ar1_baseline_aligned40.json \\
      --steps 4000 --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
TWO_PI = 2 * math.pi


def wrap(a):
    return a - TWO_PI * torch.round(a / TWO_PI)


class AR1Circular(nn.Module):
    """Per-torsion AR(1): x_{t+1} ~ von Mises(x_t + mu_delta, kappa)."""

    def __init__(self, A: int):
        super().__init__()
        self.A = A
        # mu_delta in [-pi, pi]; we parametrize via raw, wrap on use
        self.mu_delta = nn.Parameter(torch.zeros(A))
        # log_kappa for stability (kappa = exp(log_kappa)).
        self.log_kappa = nn.Parameter(torch.zeros(A))

    def neg_log_lik(self, x_curr: torch.Tensor, x_next: torch.Tensor):
        """x_curr,x_next: (T, A) angles in radians."""
        delta = wrap(x_next - x_curr - self.mu_delta)
        kappa = self.log_kappa.exp()
        # log p(delta | mu=0, kappa) = kappa cos(delta) - log(2pi I_0(kappa))
        # Use torch's modified bessel via VonMises
        vm = torch.distributions.VonMises(torch.zeros_like(delta), kappa.expand_as(delta))
        return -vm.log_prob(delta).sum(dim=-1).mean()


def train_ar1(train_flat: torch.Tensor, val_flat: torch.Tensor,
              steps: int, lr: float, device: str):
    A = train_flat.shape[1]
    model = AR1Circular(A).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = train_flat.shape[0] - 1

    for s in range(steps):
        idx = torch.randint(0, n, (min(512, n),), device=device)
        x_curr = train_flat[idx]
        x_next = train_flat[idx + 1]
        nll = model.neg_log_lik(x_curr, x_next)
        opt.zero_grad()
        nll.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_nll = model.neg_log_lik(val_flat[:-1], val_flat[1:]).item()
        train_nll = model.neg_log_lik(train_flat[:-1], train_flat[1:]).item()
    return model, train_nll, val_nll


def load_dihedrals(npz_path: Path):
    """Returns dict with 'train' and 'val' as np arrays of (T, N, 2)."""
    d = np.load(npz_path)
    return {
        "domain_id": str(d["domain_id"]),
        "N": int(d["N"]),
        "identity_deg": float(d.get("identity_deg", float("nan"))),
        "train": d["train"].astype(np.float32),
        "val": d["val"].astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pattern", default="*_dihedrals.npz")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    files = sorted(Path(args.data_dir).glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {args.data_dir}")
        sys.exit(1)
    print(f"Found {len(files)} domain files in {args.data_dir}")

    results = []
    for fpath in files:
        info = load_dihedrals(fpath)
        train = torch.from_numpy(info["train"]).reshape(info["train"].shape[0], -1).to(device)
        val = torch.from_numpy(info["val"]).reshape(info["val"].shape[0], -1).to(device)
        if len(train) < 10 or len(val) < 10:
            print(f"[skip] {info['domain_id']} — too few frames")
            continue
        t0 = time.time()
        model, train_nll, val_nll = train_ar1(train, val, args.steps, args.lr, device)
        elapsed = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
        results.append({
            "domain_id": info["domain_id"],
            "N": info["N"],
            "identity_deg": info["identity_deg"],
            "train_size": int(train.shape[0]),
            "val_size": int(val.shape[0]),
            "ar1_train_nll": float(train_nll),
            "ar1_val_nll": float(val_nll),
            "ar1_params": int(n_params),
            "ar1_train_seconds": float(elapsed),
        })
        print(f"  {info['domain_id']}: AR1 val NLL={val_nll:.2f} (n={n_params}, t={elapsed:.1f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "experiment": "ar1_baseline_aligned",
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "data_dir": args.data_dir,
        "steps": args.steps,
        "lr": args.lr,
        "device": device,
        "n_domains": len(results),
        "results": results,
    }, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()

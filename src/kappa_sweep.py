"""Kappa-multiplier calibration sweep on the rollout audit subset.

For each domain in {1lwjA03, 1kwgA03, 1vq8L01} train PhaseFlow_t4 once
(seed 42), then run a 2500-step rollout for each kappa multiplier in
{1, 5, 10, 20, 30, 50, 100} and compute Ramachandran JSD vs val.

Replaces the heuristic κ×30 with a calibrated table.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chain_model import chain_log_prob
from train_real import ChainPhaseFlowVar


TWO_PI = 2 * math.pi


def prob_histogram(angles_2d, n_bins=36):
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    H, _, _ = np.histogram2d(angles_2d[:, 0], angles_2d[:, 1], bins=[edges, edges])
    s = H.sum()
    return H / s if s > 0 else H


def jsd_2d(P, Q, eps=1e-12):
    P = P + eps; Q = Q + eps
    P /= P.sum(); Q /= Q.sum()
    M = 0.5 * (P + Q)
    def _kl(a, b): return (a * (np.log(a) - np.log(b))).sum()
    return 0.5 * (_kl(P, M) + _kl(Q, M))


def per_residue_jsd(traj_a, traj_b, n_bins=36):
    R = traj_a.shape[1]
    out = np.empty(R)
    for r in range(R):
        Pa = prob_histogram(traj_a[:, r], n_bins)
        Pb = prob_histogram(traj_b[:, r], n_bins)
        out[r] = jsd_2d(Pa, Pb)
    return out


def train_pf(train_flat, steps, lr, t_max, K, device):
    A = train_flat.shape[1]; N = A // 2
    model = ChainPhaseFlowVar(N=N, n_components=K, t_max=t_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = train_flat.shape[0] - 1
    for s in range(steps):
        idx = torch.randint(0, n, (min(512, n),), device=device)
        x = train_flat[idx]; y = train_flat[idx + 1]
        log_pi, mu, kappa = model(x)
        nll = -chain_log_prob(y, log_pi, mu, kappa).mean()
        opt.zero_grad(); nll.backward(); opt.step()
    return model


@torch.no_grad()
def rollout_pf(model, seed_state, n_steps, kappa_mult):
    state = seed_state.clone()
    A = state.shape[1]
    out = torch.empty(n_steps, A, device=state.device)
    for t in range(n_steps):
        log_pi, mu, kappa = model(state)
        k_scaled = kappa * kappa_mult
        pi = log_pi.exp()
        comp = torch.multinomial(pi, 1).squeeze(-1)
        bidx = torch.arange(1, device=state.device)
        nxt = torch.distributions.VonMises(mu[bidx, comp], k_scaled[bidx, comp]).sample()
        out[t] = nxt[0]; state = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--kappa_mults", nargs="+", type=float, default=[1, 5, 10, 20, 30, 50, 100])
    ap.add_argument("--rollout_steps", type=int, default=2500)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--t_max", type=float, default=4.0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    out = {
        "experiment": "kappa_sweep_aligned3",
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "data_dir": args.data_dir, "kappa_mults": args.kappa_mults,
        "rollout_steps": args.rollout_steps, "K": args.K, "t_max": args.t_max,
        "results": [],
    }

    for dom in args.domains:
        npz = Path(args.data_dir) / f"{dom}_dihedrals.npz"
        if not npz.exists():
            npz = Path(args.data_dir) / f"{dom}_T348_dihedrals.npz"
        if not npz.exists(): continue
        d = np.load(npz)
        train = d["train"].astype(np.float32); val = d["val"].astype(np.float32)
        train_flat = torch.from_numpy(train.reshape(train.shape[0], -1)).to(device)
        val_flat = torch.from_numpy(val.reshape(val.shape[0], -1)).to(device)

        torch.manual_seed(42); np.random.seed(42)
        print(f"  {dom} training PF (K={args.K}, t_max={args.t_max})...")
        model = train_pf(train_flat, args.steps, 2e-3, args.t_max, args.K, device)

        per_kappa = {}
        for km in args.kappa_mults:
            torch.manual_seed(42); np.random.seed(42)
            roll = rollout_pf(model, val_flat[0:1], args.rollout_steps, km).cpu().numpy()
            roll = roll.reshape(args.rollout_steps, val.shape[1], 2)
            jsd = per_residue_jsd(roll, val)
            per_kappa[str(km)] = {"mean_jsd": float(jsd.mean()),
                                   "per_residue_jsd": jsd.tolist()}
            print(f"    {dom} κ×{km}: mean JSD = {jsd.mean():.4f}")
        out["results"].append({"domain_id": dom, "N": int(d["N"]), "per_kappa": per_kappa})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

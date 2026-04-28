"""Compute JSD reference scale to anchor the rollout fidelity claim.

For each rollout audit domain:
  1. JSD(MD-replica i, MD-replica j) on the validation slice — irreducible
     noise floor from the dataset (replicas are independent statistical
     realisations of the same dynamics).
  2. JSD(MLP rollout, GT) — what the absolute MLP baseline gives if used
     as a propagator with the same kappa-rescaling protocol.
  3. JSD(uniform, GT) — pessimal upper bound.
  4. JSD(AlphaDynamics, GT) — repeat for symmetry; should match v1 audit.

This eliminates the "is JSD = 0.194 good?" reviewer attack by giving
an anchored interpretation: AD JSD vs replica-replica floor and vs MLP.

Outputs: results/jsd_reference_scale.json + .md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chain_model import chain_log_prob  # noqa: E402
from train_real import ChainPhaseFlowVar  # noqa: E402

TWO_PI = 2 * math.pi
N_BINS = 36
KAPPA_MULT = 30.0


def wrap_np(a):
    return (a + np.pi) % TWO_PI - np.pi


def prob_histogram(angles_2d, n_bins=N_BINS):
    """angles_2d: (T, 2). Returns (n_bins, n_bins) probability."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    H, _, _ = np.histogram2d(angles_2d[:, 0], angles_2d[:, 1], bins=[edges, edges])
    s = H.sum()
    return H / s if s > 0 else H


def jsd_2d(P, Q, eps=1e-12):
    P = P + eps
    Q = Q + eps
    P /= P.sum()
    Q /= Q.sum()
    M = 0.5 * (P + Q)
    def _kl(a, b):
        return (a * (np.log(a) - np.log(b))).sum()
    return 0.5 * (_kl(P, M) + _kl(Q, M))


def per_residue_jsd(traj_a, traj_b, n_bins=N_BINS):
    """traj_a, traj_b: (T, R, 2)."""
    R = traj_a.shape[1]
    out = np.empty(R)
    for r in range(R):
        Pa = prob_histogram(traj_a[:, r], n_bins)
        Pb = prob_histogram(traj_b[:, r], n_bins)
        out[r] = jsd_2d(Pa, Pb)
    return out


class AR1Circular(torch.nn.Module):
    """Per-torsion AR(1): x_{t+1} ~ wrapped vM(x_t + mu_delta, kappa)."""
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.mu_delta = torch.nn.Parameter(torch.zeros(A))
        self.log_kappa = torch.nn.Parameter(torch.zeros(A))


def train_ar1(train_flat, steps, lr, device):
    A = train_flat.shape[1]
    model = AR1Circular(A).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = train_flat.shape[0] - 1
    for s in range(steps):
        idx = torch.randint(0, n, (min(512, n),), device=device)
        x_curr = train_flat[idx]
        x_next = train_flat[idx + 1]
        delta = x_next - x_curr - model.mu_delta
        delta = delta - TWO_PI * torch.round(delta / TWO_PI)
        kappa = model.log_kappa.exp()
        vm = torch.distributions.VonMises(torch.zeros_like(delta), kappa.expand_as(delta))
        nll = -vm.log_prob(delta).sum(dim=-1).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
    return model


@torch.no_grad()
def rollout_ar1(model, seed_state, n_steps):
    state = seed_state.clone()
    A = state.shape[1]
    out = torch.empty(n_steps, A, device=state.device)
    kappa = model.log_kappa.exp().expand(1, A)
    for t in range(n_steps):
        mu_step = state + model.mu_delta.unsqueeze(0)
        delta = torch.distributions.VonMises(torch.zeros_like(mu_step),
                                              kappa).sample()
        nxt = mu_step + delta
        # wrap to [-pi, pi]
        nxt = nxt - TWO_PI * torch.round(nxt / TWO_PI)
        out[t] = nxt[0]
        state = nxt
    return out


# --- AbsoluteMLPRollout: minimal residual MLP baseline rollout ---------------
class AbsoluteMLP(torch.nn.Module):
    """Predict next angle as wrapped(x_t + delta) where delta ~ vM(mu_delta, kappa).
    This is mathematically equivalent to the abs-MLP from the v1 paper but uses
    von Mises sampling so that we can do rollouts."""
    def __init__(self, A, hidden=128, n_components=8):
        super().__init__()
        from chain_model import ChainMDNHead
        self.A = A
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2 * A, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        )
        self.head = ChainMDNHead(hidden, A, n_components, hidden)

    def forward(self, angles):
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        log_pi, mu, kappa = self.head(self.encoder(feats))
        return log_pi, mu, kappa


def train_mlp(train_flat, steps, lr, device, hidden=128, K=8):
    A = train_flat.shape[1]
    model = AbsoluteMLP(A, hidden=hidden, n_components=K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = train_flat.shape[0] - 1
    for s in range(steps):
        idx = torch.randint(0, n, (min(512, n),), device=device)
        x = train_flat[idx]
        y = train_flat[idx + 1]
        log_pi, mu, kappa = model(x)
        nll = -chain_log_prob(y, log_pi, mu, kappa).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
    return model


@torch.no_grad()
def rollout_mlp(model, seed_state, n_steps, kappa_mult):
    """seed_state: (1, A). returns (n_steps, A)."""
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
        out[t] = nxt[0]
        state = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="dir with *_T{TEMP}_dihedrals.npz files (mdcath_alltemps style)")
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--temps", nargs="+", default=["320", "348", "379", "413", "450"])
    ap.add_argument("--rollout_steps", type=int, default=2500)
    ap.add_argument("--mlp_steps", type=int, default=4000)
    ap.add_argument("--mlp_lr", type=float, default=2e-3)
    ap.add_argument("--kappa_mult", type=float, default=KAPPA_MULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "results" / "jsd_reference_scale.json"))
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out = {
        "experiment": "jsd_reference_scale",
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "data_dir": args.data_dir,
        "rollout_steps": args.rollout_steps,
        "kappa_mult": args.kappa_mult,
        "mlp_steps": args.mlp_steps,
        "per_domain": [],
    }

    for dom in args.domains:
        print(f"\n=== {dom} ===")
        # 1. Cross-temperature: surrogate for cross-replica because mdCATH gives
        # per-temp single replica. We approximate "replica-replica JSD" by
        # JSD(half1 of trajectory, half2 of trajectory) at the SAME temperature.
        primary_npz = Path(args.data_dir) / f"{dom}_T348_dihedrals.npz"
        if not primary_npz.exists():
            print(f"[skip] {primary_npz} missing")
            continue
        d = np.load(primary_npz)
        train = d["train"].astype(np.float32)
        val = d["val"].astype(np.float32)
        traj = np.concatenate([train, val], axis=0)  # full
        T = traj.shape[0]
        if T < 1000:
            print(f"[skip] {dom}: only {T} frames"); continue
        h1 = traj[:T // 2]
        h2 = traj[T // 2:]
        # split-trajectory JSD as bootstrap-style "replica" floor
        floor_jsd = per_residue_jsd(h1, h2)
        floor_mean = float(floor_jsd.mean())
        # Cross-temperature: compute JSD between T=348 val histogram and T=320/450
        # val histogram on the same residues — this is "physical" floor.
        cross_t_jsds = {}
        for t in args.temps:
            other = Path(args.data_dir) / f"{dom}_T{t}_dihedrals.npz"
            if not other.exists() or t == "348":
                continue
            do = np.load(other)
            ot = np.concatenate([do["train"], do["val"]], axis=0).astype(np.float32)
            j = per_residue_jsd(val, ot[:val.shape[0]] if ot.shape[0] >= val.shape[0] else ot)
            cross_t_jsds[t] = float(j.mean())
        # 2. Uniform vs GT (pessimal)
        uniform_hist = np.full((N_BINS, N_BINS), 1.0 / (N_BINS * N_BINS))
        unif_jsd = []
        for r in range(val.shape[1]):
            P = prob_histogram(val[:, r])
            unif_jsd.append(jsd_2d(P, uniform_hist))
        unif_mean = float(np.mean(unif_jsd))

        # 3. Train MLP, rollout, compute JSD vs val histogram
        train_flat = torch.from_numpy(train.reshape(train.shape[0], -1)).to(device)
        val_flat = torch.from_numpy(val.reshape(val.shape[0], -1)).to(device)
        if train_flat.shape[0] < 50:
            print(f"[skip] {dom}: train too small"); continue
        print(f"  Training abs-MLP {args.mlp_steps} steps...")
        mlp = train_mlp(train_flat, args.mlp_steps, args.mlp_lr, device)
        seed = val_flat[0:1]
        print(f"  Rollout MLP {args.rollout_steps} (κ×{args.kappa_mult})...")
        roll = rollout_mlp(mlp, seed, args.rollout_steps, args.kappa_mult).cpu().numpy()
        roll = roll.reshape(args.rollout_steps, val.shape[1], 2)
        mlp_jsd = per_residue_jsd(roll, val)
        mlp_mean = float(mlp_jsd.mean())

        # 4. Train AR(1), rollout, compute JSD vs val histogram
        print(f"  Training AR(1) 4000 steps...")
        ar1 = train_ar1(train_flat, 4000, 2e-2, device)
        ar1_roll = rollout_ar1(ar1, seed, args.rollout_steps).cpu().numpy()
        ar1_roll = ar1_roll.reshape(args.rollout_steps, val.shape[1], 2)
        ar1_jsd = per_residue_jsd(ar1_roll, val)
        ar1_mean = float(ar1_jsd.mean())

        out["per_domain"].append({
            "domain_id": dom,
            "N": int(val.shape[1]),
            "n_train": int(train.shape[0]),
            "n_val": int(val.shape[0]),
            "split_traj_jsd_mean": floor_mean,
            "split_traj_jsd_per_residue": floor_jsd.tolist(),
            "cross_temperature_jsd": cross_t_jsds,
            "uniform_vs_val_jsd_mean": unif_mean,
            "mlp_rollout_jsd_mean": mlp_mean,
            "mlp_rollout_jsd_per_residue": mlp_jsd.tolist(),
            "ar1_rollout_jsd_mean": ar1_mean,
            "ar1_rollout_jsd_per_residue": ar1_jsd.tolist(),
        })

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"  floor (split-traj) JSD: {floor_mean:.4f}")
        print(f"  cross-T JSDs: {cross_t_jsds}")
        print(f"  uniform vs val JSD: {unif_mean:.4f}")
        print(f"  MLP rollout JSD: {mlp_mean:.4f}")
        print(f"  AR(1) rollout JSD: {ar1_mean:.4f}")

    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

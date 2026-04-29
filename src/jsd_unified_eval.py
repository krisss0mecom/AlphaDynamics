"""Unified JSD evaluator — single canonical metric for all models.

Eliminates the v2 paper Table 4 inconsistency where AD JSD used a smoothed
train+val GT histogram while Timewarp JSD used raw val histogram. This
script re-evaluates ALL models under the same protocol.

Canonical protocol:
  - GT histogram: held-out val slice ONLY (no train leakage)
  - n_bins = 36 (same as ramachandran_energy_v2)
  - NO Gaussian smoothing (raw counts)
  - per-residue 2D JSD, then mean across residues

Re-evaluates:
  1. Timewarp 4AA rollouts -- recomputes JSD from saved positions
     (timewarp_rollout_npz/{pep}_timewarp_rollout.npz)
  2. AlphaDynamics rollouts -- retrains per peptide and rolls out
     (uses train_real.ChainPhaseFlowVar with the canonical training protocol)
  3. Optional: few-shot curve and distillation models can use the same
     evaluator by importing canonical_jsd().

Outputs:
  results/head_to_head_4aa_unified_metric.json
  results/head_to_head_4aa_unified_metric.md
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
import mdtraj as md

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chain_model import chain_log_prob
from train_real import ChainPhaseFlowVar


N_BINS_CANON = 36
TWO_PI = 2 * math.pi


def canonical_histogram(angles_2d: np.ndarray, n_bins: int = N_BINS_CANON):
    """No smoothing. angles_2d: (T, 2) in radians."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    H, _, _ = np.histogram2d(angles_2d[:, 0], angles_2d[:, 1], bins=[edges, edges])
    s = H.sum()
    return (H / s) if s > 0 else H


def canonical_jsd_2d(P, Q, eps=1e-12):
    P = P + eps; Q = Q + eps
    P /= P.sum(); Q /= Q.sum()
    M = 0.5 * (P + Q)
    def _kl(a, b): return (a * (np.log(a) - np.log(b))).sum()
    return 0.5 * (_kl(P, M) + _kl(Q, M))


def canonical_jsd_per_residue(rollout: np.ndarray, gt: np.ndarray, n_bins: int = N_BINS_CANON):
    """rollout, gt: (T, R, 2). Returns (R,) array of JSDs."""
    R = rollout.shape[1]
    out = np.empty(R)
    for r in range(R):
        P = canonical_histogram(rollout[:, r], n_bins)
        Q = canonical_histogram(gt[:, r], n_bins)
        out[r] = canonical_jsd_2d(P, Q)
    return out


def positions_to_torsions(positions: np.ndarray, pdb_path: Path) -> np.ndarray:
    top = md.load_pdb(str(pdb_path)).topology
    traj = md.Trajectory(positions.astype(np.float32), top)
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)
    phi_res = np.array([top.atom(int(r[1])).residue.index for r in phi_idx])
    psi_res = np.array([top.atom(int(r[1])).residue.index for r in psi_idx])
    common = sorted(set(phi_res.tolist()) & set(psi_res.tolist()))
    phi_lookup = {int(r): i for i, r in enumerate(phi_res)}
    psi_lookup = {int(r): i for i, r in enumerate(psi_res)}
    phi_cols = [phi_lookup[r] for r in common]
    psi_cols = [psi_lookup[r] for r in common]
    return np.stack([phi[:, phi_cols], psi[:, psi_cols]], axis=-1).astype(np.float32)


def train_pf_canonical(train_data, steps, lr, t_max, K, device, seed=42, batch=256):
    """Identical to ramachandran_energy_v2.train_model()."""
    A = train_data.shape[1]
    N = A // 2
    torch.manual_seed(seed)
    model = ChainPhaseFlowVar(N=N, n_osc=64, n_components=K, hidden=128,
                                t_max=t_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n_pairs = train_data.shape[0] - 1
    if n_pairs <= 0:
        return model
    model.train()
    for s in range(1, steps + 1):
        bsz = min(batch, max(1, n_pairs))
        idx = torch.randint(0, n_pairs, (bsz,), device=device)
        x = train_data[idx]; y = train_data[idx + 1]
        log_pi, mu, kappa = model(x)
        nll = -chain_log_prob(y, log_pi, mu, kappa).mean()
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
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


def evaluate_timewarp(rollout_npz: Path, pdb_path: Path, val_torsions: np.ndarray):
    """Re-compute Timewarp JSD under canonical metric from saved positions."""
    arr = np.load(rollout_npz)
    tw_torsions = positions_to_torsions(arr["positions"], pdb_path)
    R_val = val_torsions.shape[1]
    if tw_torsions.shape[1] != R_val:
        tw_torsions = tw_torsions[:, :R_val]
    jsd = canonical_jsd_per_residue(tw_torsions, val_torsions)
    return float(jsd.mean()), jsd.tolist()


def evaluate_alphadynamics(train_full: torch.Tensor, val_torsions: np.ndarray,
                            steps: int, rollout_steps: int, kappa_mult: float,
                            t_max: float, K: int, lr: float, device: str,
                            n_residues: int):
    model = train_pf_canonical(train_full, steps, lr, t_max, K, device, seed=42)
    val_flat = torch.from_numpy(val_torsions.reshape(val_torsions.shape[0], -1)).to(device)
    torch.manual_seed(42); np.random.seed(42)
    roll = rollout_pf(model, val_flat[0:1], rollout_steps, kappa_mult).cpu().numpy()
    roll = roll.reshape(rollout_steps, n_residues, 2)
    jsd = canonical_jsd_per_residue(roll, val_torsions)
    return float(jsd.mean()), jsd.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="dir with {pep}_dihedrals.npz (train/val torsions)")
    ap.add_argument("--timewarp_rollout_dir", required=True,
                    help="dir with {pep}_timewarp_rollout.npz (raw positions)")
    ap.add_argument("--pdb_dir", required=True,
                    help="dir with {pep}-traj-state0.pdb")
    ap.add_argument("--peptides", nargs="+", default=["AAAY", "AACE", "AAEW"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--rollout_steps", type=int, default=2500)
    ap.add_argument("--kappa_mult", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--t_max", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    out = {
        "experiment": "head_to_head_4aa_unified_metric",
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "metric": {
            "name": "canonical_held_out_val_jsd",
            "n_bins": N_BINS_CANON,
            "smoothing": "none",
            "ground_truth": "held_out_val_only",
            "comment": "Identical evaluator for AD and Timewarp (no smoothing, val-only GT)",
        },
        "rollout_steps": args.rollout_steps,
        "kappa_mult": args.kappa_mult,
        "results": [],
    }

    for peptide in args.peptides:
        npz = Path(args.data_dir) / f"{peptide}_dihedrals.npz"
        if not npz.exists():
            npz = Path(args.data_dir) / f"{peptide}_T348_dihedrals.npz"
        rollout_npz = Path(args.timewarp_rollout_dir) / f"{peptide}_timewarp_rollout.npz"
        pdb = Path(args.pdb_dir) / f"{peptide}-traj-state0.pdb"
        if not (npz.exists() and rollout_npz.exists() and pdb.exists()):
            print(f"[skip] {peptide}: missing inputs")
            continue
        d = np.load(npz)
        train_full = d["train"].astype(np.float32)
        val = d["val"].astype(np.float32)
        train_flat = torch.from_numpy(train_full.reshape(train_full.shape[0], -1)).to(device)
        n_res = val.shape[1]
        print(f"\n=== {peptide} ({n_res} residues, train={train_full.shape[0]}, val={val.shape[0]}) ===")

        # Timewarp re-eval
        tw_jsd_mean, tw_jsd_pr = evaluate_timewarp(rollout_npz, pdb, val)
        print(f"  Timewarp (re-eval, canonical metric): mean JSD = {tw_jsd_mean:.4f}")

        # AlphaDynamics rerun under canonical metric
        ad_jsd_mean, ad_jsd_pr = evaluate_alphadynamics(
            train_flat, val, args.steps, args.rollout_steps, args.kappa_mult,
            args.t_max, args.K, args.lr, device, n_res)
        print(f"  AlphaDynamics (full train, canonical metric): mean JSD = {ad_jsd_mean:.4f}")

        ratio = tw_jsd_mean / ad_jsd_mean if ad_jsd_mean > 0 else float("inf")
        verdict = "AD wins" if ad_jsd_mean < tw_jsd_mean else "Timewarp wins"
        print(f"  Ratio TW/AD = {ratio:.2f}× — {verdict}")

        out["results"].append({
            "peptide": peptide,
            "N_residues": int(n_res),
            "timewarp_jsd_canonical": tw_jsd_mean,
            "timewarp_jsd_per_residue": tw_jsd_pr,
            "alphadynamics_jsd_canonical": ad_jsd_mean,
            "alphadynamics_jsd_per_residue": ad_jsd_pr,
            "ratio_tw_over_ad": ratio,
            "verdict": verdict,
        })
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))

    # Aggregate + Markdown
    if out["results"]:
        means_tw = [r["timewarp_jsd_canonical"] for r in out["results"]]
        means_ad = [r["alphadynamics_jsd_canonical"] for r in out["results"]]
        m_tw = sum(means_tw) / len(means_tw)
        m_ad = sum(means_ad) / len(means_ad)
        out["aggregate"] = {
            "mean_timewarp_jsd": m_tw,
            "mean_alphadynamics_jsd": m_ad,
            "mean_ratio_tw_over_ad": m_tw / m_ad if m_ad > 0 else float("inf"),
        }
        Path(args.out).write_text(json.dumps(out, indent=2))

        md = ROOT / "results" / (Path(args.out).stem + ".md")
        lines = [
            "# Head-to-head AD vs Timewarp on 4AA-large/test — UNIFIED METRIC",
            "",
            "Both AlphaDynamics and Microsoft Timewarp 2500-step rollouts evaluated",
            "under the same canonical Ramachandran JSD:",
            "",
            f"- GT histogram: **held-out val only** (no train leakage)",
            f"- bins: {N_BINS_CANON} per axis",
            "- smoothing: **none** (raw counts)",
            "- per-residue 2D JSD averaged across residues",
            "",
            "This corrects the v2 Table 4 inconsistency where AD used smoothed",
            "train+val GT (paper v2 commit fb355be) while Timewarp used raw val GT.",
            "",
            "| Peptide | AD JSD | Timewarp JSD | TW / AD |",
            "|---|---:|---:|---:|",
        ]
        for r in out["results"]:
            lines.append(f"| {r['peptide']} | **{r['alphadynamics_jsd_canonical']:.4f}** "
                         f"| {r['timewarp_jsd_canonical']:.4f} | "
                         f"{r['ratio_tw_over_ad']:.2f}× |")
        lines.append(f"| **Mean** | **{m_ad:.4f}** | **{m_tw:.4f}** | **{m_tw/m_ad:.2f}×** |")
        md.write_text("\n".join(lines) + "\n")
        print(f"\n[saved] {args.out}")
        print(f"[saved] {md}")


if __name__ == "__main__":
    main()

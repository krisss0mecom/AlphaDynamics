"""Train ChainMLP and ChainPhaseFlow on REAL protein data (mdshare).

Datasets:
  - pentapeptide_dihedrals.npz: WLALL pentapeptide, N=4 residues, 30K frames @ 100ps
  - ala2_1ps_dihedrals.npz:    Ala2, N=1 residue, 800K train @ 1ps stride

Same training loop as scaling_study.py but using real (not synthetic) data.
Includes warmup ablation: PhaseFlow with n_steps ∈ {16, 32, 64} ODE integration steps.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from chain_model import (ChainMLP, ChainPhaseFlow, ChainPhaseEncoder,
                         ChainMDNHead, chain_log_prob, chain_sample,
                         ChainPhaseFlowFunc)
from train_chain import evaluate, train_one, wrap


def load_real(npz_path, device):
    d = np.load(npz_path)
    tr = torch.tensor(d["train"], dtype=torch.float32, device=device)
    va = torch.tensor(d["val"], dtype=torch.float32, device=device)
    if tr.dim() == 2:  # Ala2 single dihedral pair, shape (T, 2)
        N = 1
        tr = tr.unsqueeze(1)  # (T, 1, 2)
        va = va.unsqueeze(1)
    else:
        N = tr.shape[1]
    print(f"  {npz_path}: train {tr.shape}, val {va.shape}, N={N}")
    return tr, va, N


def flatten(x):
    T_ = x.shape[0]
    return x.reshape(T_, -1)


def identity_baseline(val_flat):
    diff = wrap(val_flat[:-1] - val_flat[1:])
    err = torch.sqrt((diff ** 2).mean(-1))
    return math.degrees(err.mean().item())


# Custom PhaseFlow with configurable t_span (warmup ablation)
class ChainPhaseFlowVar(torch.nn.Module):
    def __init__(self, N, n_osc=64, n_components=8, hidden=128, t_max=1.0):
        super().__init__()
        self.N = N
        self.A = 2 * N
        self.encoder = ChainPhaseEncoder(self.A, n_osc=n_osc, t_span=(0.0, t_max))
        self.head = ChainMDNHead(2 * n_osc, self.A, n_components, hidden)

    def forward(self, angles):
        return self.head(self.encoder(angles))


def run(name, model_factory, train_flat, val_flat, device,
        steps=8000, batch=1024, lr=2e-3, save_dir="runs/real"):
    torch.manual_seed(42)
    model = model_factory().to(device)
    return train_one(model, train_flat, val_flat, device, name,
                     steps=steps, batch=batch, lr=lr,
                     eval_every=4000, log_every=2000,
                     save_dir=save_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="real_data/pentapeptide_dihedrals.npz")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--out", default="real_results.md")
    p.add_argument("--warmup_sweep", action="store_true",
                   help="Run PhaseFlow with multiple t_max values")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tr, va, N = load_real(args.data, device)
    train_flat = flatten(tr)
    val_flat = flatten(va)
    id_err = identity_baseline(val_flat)
    print(f"Identity baseline: {id_err:.2f}°")

    results = []

    # MLP baseline
    h, p_count = run("MLP", lambda: ChainMLP(N=N, n_components=8, hidden=128),
                     train_flat, val_flat, device, steps=args.steps,
                     save_dir=f"runs/real_{Path(args.data).stem}_mlp")
    results.append(("MLP", p_count, h[-1]))

    # PhaseFlow with default t_max=1.0
    h, p_count = run("PhaseFlow", lambda: ChainPhaseFlow(N=N, n_components=8, hidden=128),
                     train_flat, val_flat, device, steps=args.steps,
                     save_dir=f"runs/real_{Path(args.data).stem}_phaseflow")
    results.append(("PhaseFlow_t1", p_count, h[-1]))

    # Warmup sweep — multiple t_max values
    if args.warmup_sweep:
        for t_max in [2.0, 4.0, 8.0]:
            label = f"PhaseFlow_t{int(t_max)}"
            h, p_count = run(label,
                             lambda tm=t_max: ChainPhaseFlowVar(N=N, t_max=tm),
                             train_flat, val_flat, device, steps=args.steps,
                             save_dir=f"runs/real_{Path(args.data).stem}_t{int(t_max)}")
            results.append((label, p_count, h[-1]))

    # Write report
    lines = [
        f"# REAL data: {args.data}",
        "",
        f"N residues: {N}, identity baseline: {id_err:.2f}°",
        "",
        "| Model | Params | NLL | Mode° | Best-of-10° |",
        "|---|---|---|---|---|",
    ]
    for name, params, m in results:
        lines.append(f"| {name} | {params:,} | {m['nll']:.4f} | "
                     f"{m['mode_deg']:.2f} | **{m['best10_deg']:.2f}** |")
    lines += ["", "Identity baseline above. Lower NLL = better calibration.", ""]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nReport: {args.out}")

    print("\n=== SUMMARY ===")
    print(f"Identity: {id_err:.2f}°")
    for name, params, m in results:
        print(f"  {name:<16} params={params:,} NLL={m['nll']:.4f} "
              f"best10={m['best10_deg']:.2f}°")


if __name__ == "__main__":
    main()

"""K-sweep ablation: vary mixture-of-von-Mises components K on representative
audit domains. Reports val NLL across K ∈ {2, 4, 8, 16, 32}."""
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


def train_pf(train_flat, val_flat, K, steps, lr, t_max, device):
    A = train_flat.shape[1]
    N = A // 2
    model = ChainPhaseFlowVar(N=N, n_components=K, t_max=t_max).to(device)
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
    model.eval()
    with torch.no_grad():
        log_pi, mu, kappa = model(val_flat[:-1])
        val_nll = -chain_log_prob(val_flat[1:], log_pi, mu, kappa).mean().item()
    return val_nll, sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--Ks", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--t_max", type=float, default=4.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out = {
        "experiment": "k_sweep_ablation",
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "data_dir": args.data_dir,
        "Ks": args.Ks,
        "t_max": args.t_max,
        "steps": args.steps,
        "results": [],
    }

    for dom in args.domains:
        npz = Path(args.data_dir) / f"{dom}_dihedrals.npz"
        if not npz.exists():
            npz = Path(args.data_dir) / f"{dom}_T348_dihedrals.npz"
        if not npz.exists():
            print(f"[skip] {dom} data missing"); continue
        d = np.load(npz)
        train = d["train"].astype(np.float32)
        val = d["val"].astype(np.float32)
        train_flat = torch.from_numpy(train.reshape(train.shape[0], -1)).to(device)
        val_flat = torch.from_numpy(val.reshape(val.shape[0], -1)).to(device)
        per_K = {}
        for K in args.Ks:
            t0 = time.time()
            torch.manual_seed(42)
            np.random.seed(42)
            val_nll, n_params = train_pf(train_flat, val_flat, K, args.steps,
                                          args.lr, args.t_max, device)
            elapsed = time.time() - t0
            per_K[str(K)] = {"val_nll": val_nll, "n_params": n_params, "seconds": elapsed}
            print(f"  {dom} K={K}: val NLL={val_nll:.2f} (n={n_params}, t={elapsed:.1f}s)")
        out["results"].append({"domain_id": dom, "N": int(d["N"]), "per_K": per_K})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

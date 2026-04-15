"""Train ChainMLP and ChainPhaseFlow on synthetic chain MD, compare."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from chain_model import (ChainMLP, ChainPhaseFlow, chain_log_prob, chain_sample)

TWO_PI = 2.0 * math.pi


def wrap(a):
    return a - TWO_PI * torch.round(a / TWO_PI)


def load_chain(data_path, device):
    d = np.load(data_path)
    tr = torch.tensor(d["train"], dtype=torch.float32, device=device)  # (T, N, 2)
    va = torch.tensor(d["val"], dtype=torch.float32, device=device)
    N = int(d["N"])
    return tr, va, N


def flatten(angles):
    """Convert (T, N, 2) → (T, 2N): [phi_1, psi_1, phi_2, psi_2, ...]."""
    T_, N, _ = angles.shape
    return angles.reshape(T_, 2 * N)


@torch.no_grad()
def evaluate(model, val_flat, device, n_samples=10, chunk=2048):
    model.eval()
    T_ = len(val_flat) - 1
    A = val_flat.shape[1]

    nll_sum = 0.0
    err_mode_sum = 0.0
    err_best_sum = 0.0
    for s in range(0, T_, chunk):
        e = min(s + chunk, T_)
        inp = val_flat[s:e]
        tgt = val_flat[s + 1:e + 1]

        log_pi, mu, kappa = model(inp)
        lp = chain_log_prob(tgt, log_pi, mu, kappa)
        nll_sum += -lp.sum().item()

        # Mode: argmax component, take its (mu) as point prediction
        am = log_pi.argmax(dim=-1)  # (B,)
        bidx = torch.arange(len(inp), device=device)
        mode_pred = mu[bidx, am]  # (B, A)
        err_mode = torch.sqrt(((wrap(mode_pred - tgt)) ** 2).mean(-1))
        err_mode_sum += err_mode.sum().item()

        samples = chain_sample(log_pi, mu, kappa, n_samples=n_samples)  # (B, ns, A)
        diff = wrap(samples - tgt.unsqueeze(1))
        per_sample = torch.sqrt((diff ** 2).mean(-1))  # (B, ns)
        err_best_sum += per_sample.min(dim=-1).values.sum().item()

    model.train()
    return {
        "nll": nll_sum / T_,
        "mode_deg": math.degrees(err_mode_sum / T_),
        "best10_deg": math.degrees(err_best_sum / T_),
    }


def train_one(model, train_flat, val_flat, device, name,
              steps, batch, lr, eval_every, log_every, save_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Training {name} ({n_params:,} params) ===")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    N_frames = len(train_flat) - 1
    history = []
    best_nll = float("inf")
    t0 = time.time()
    model.train()

    for step in range(1, steps + 1):
        idx = torch.randint(0, N_frames, (batch,), device=device)
        inp = train_flat[idx]
        tgt = train_flat[idx + 1]

        log_pi, mu, kappa = model(inp)
        lp = chain_log_prob(tgt, log_pi, mu, kappa)
        loss = -lp.mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0:
            sps = step / (time.time() - t0)
            print(f"  {name:<12} step {step:5d}/{steps} | loss={loss.item():.4f} | "
                  f"lr={opt.param_groups[0]['lr']:.2e} | {sps:.1f} sps")

        if step % eval_every == 0 or step == steps:
            m = evaluate(model, val_flat, device)
            history.append({"step": step, **m})
            marker = ""
            if m["nll"] < best_nll:
                best_nll = m["nll"]
                marker = " ** best"
                torch.save({"model_state": model.state_dict(),
                            "step": step, "metrics": m, "params": n_params},
                           Path(save_dir) / "best.pt")
            print(f"  [EVAL {name}] step={step} NLL={m['nll']:.4f} "
                  f"mode={m['mode_deg']:.2f}° best10={m['best10_deg']:.2f}°{marker}")

    return history, n_params


def identity_baseline(val_flat):
    T_ = len(val_flat) - 1
    diff = wrap(val_flat[:-1] - val_flat[1:])
    err = torch.sqrt((diff ** 2).mean(-1))
    return math.degrees(err.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="chain_data.npz")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n_components", type=int, default=8)
    p.add_argument("--n_osc", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr, va, N = load_chain(args.data, device)
    print(f"Chain N={N} residues ({2*N} angles), train {len(tr):,}, val {len(va):,}")
    train_flat = flatten(tr)
    val_flat = flatten(va)

    id_err = identity_baseline(val_flat)
    print(f"\nIdentity baseline (pred=input): {id_err:.2f}°")

    # --- MLP baseline ---
    torch.manual_seed(args.seed)
    mlp = ChainMLP(N=N, n_components=args.n_components,
                   hidden=args.hidden).to(device)
    mlp_hist, mlp_params = train_one(
        mlp, train_flat, val_flat, device, "MLP",
        args.steps, args.batch, args.lr, args.eval_every, args.log_every,
        save_dir="runs/chain_mlp")

    # --- Phase-Flow ---
    torch.manual_seed(args.seed)
    pf = ChainPhaseFlow(N=N, n_osc=args.n_osc,
                        n_components=args.n_components,
                        hidden=args.hidden).to(device)
    pf_hist, pf_params = train_one(
        pf, train_flat, val_flat, device, "PhaseFlow",
        args.steps, args.batch, args.lr, args.eval_every, args.log_every,
        save_dir="runs/chain_phaseflow")

    # Write chain_results.md
    out = Path("chain_results.md")
    lines = [
        f"# Chain N={N} — MLP vs PhaseFlow",
        "",
        f"Synthetic Langevin MD, {2*N} angles, "
        f"stride gives mean |step| ≈ 10° per frame.",
        "",
        f"**Identity baseline: {id_err:.2f}°**",
        "",
        "| Model | Params | Final NLL | Mode° | Best-of-10° |",
        "|---|---|---|---|---|",
        f"| ChainMLP | {mlp_params:,} | {mlp_hist[-1]['nll']:.4f} | "
        f"{mlp_hist[-1]['mode_deg']:.2f} | "
        f"**{mlp_hist[-1]['best10_deg']:.2f}** |",
        f"| ChainPhaseFlow | {pf_params:,} | {pf_hist[-1]['nll']:.4f} | "
        f"{pf_hist[-1]['mode_deg']:.2f} | "
        f"**{pf_hist[-1]['best10_deg']:.2f}** |",
        "",
        "## Training curves",
        "",
        "### MLP baseline",
        "| step | NLL | mode° | best10° |",
        "|---|---|---|---|",
    ]
    for h in mlp_hist:
        lines.append(f"| {h['step']} | {h['nll']:.4f} | "
                     f"{h['mode_deg']:.2f} | {h['best10_deg']:.2f} |")
    lines += ["", "### PhaseFlow (primes + phyllotaxis + RK4 + MDN)",
              "| step | NLL | mode° | best10° |",
              "|---|---|---|---|"]
    for h in pf_hist:
        lines.append(f"| {h['step']} | {h['nll']:.4f} | "
                     f"{h['mode_deg']:.2f} | {h['best10_deg']:.2f} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {out.resolve()}")

    print(f"\n=== SUMMARY ===")
    print(f"Identity:     {id_err:.2f}°")
    print(f"MLP:          NLL={mlp_hist[-1]['nll']:.4f}  best10={mlp_hist[-1]['best10_deg']:.2f}°")
    print(f"PhaseFlow:    NLL={pf_hist[-1]['nll']:.4f}  best10={pf_hist[-1]['best10_deg']:.2f}°")


if __name__ == "__main__":
    main()

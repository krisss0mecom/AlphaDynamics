"""Run AlphaDynamics + MLP on converted mdCATH domains.

The default path points to freshly aligned 348 K phi/psi pairs. Use
``--out_prefix`` for audit runs so partial reruns do not overwrite the
historical 37-domain tables.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import os, glob, math, time, json
import numpy as np
import torch

from chain_model import ChainMLP, chain_log_prob, chain_sample
from train_real import ChainPhaseFlowVar

TWO_PI = 2 * math.pi
def wrap(a): return a - TWO_PI * torch.round(a / TWO_PI)


def train_model(model, train_flat, val_flat, device, steps=4000, batch=512, lr=2e-3, name=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    N_train = len(train_flat) - 1
    model.train()
    best_nll = float("inf")
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N_train, (batch,), device=device)
        x = train_flat[idx]; y = train_flat[idx + 1]
        log_pi, mu, kappa = model(x)
        lp = chain_log_prob(y, log_pi, mu, kappa)
        loss = -lp.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    # Final eval on all val
    model.eval()
    with torch.no_grad():
        log_pi, mu, kappa = model(val_flat[:-1])
        nll = -chain_log_prob(val_flat[1:], log_pi, mu, kappa).mean().item()
        am = log_pi.argmax(dim=-1)
        bidx = torch.arange(len(val_flat) - 1, device=device)
        mode = mu[bidx, am]
        mode_err = math.degrees(torch.sqrt((wrap(mode - val_flat[1:]) ** 2).mean(-1)).mean().item())
        samples = chain_sample(log_pi, mu, kappa, n_samples=10)
        best10 = math.degrees(torch.sqrt((wrap(samples - val_flat[1:].unsqueeze(1)) ** 2).mean(-1)).min(-1).values.mean().item())
    return {"nll": nll, "mode_deg": mode_err, "best10_deg": best10,
            "params": sum(p.numel() for p in model.parameters()),
            "time_sec": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "mdcath_real_data" / "mdcath_348K"))
    parser.add_argument("--results_dir", default=str(ROOT / "results"))
    parser.add_argument("--out_prefix", default="mdcath_benchmark_results")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_domains", type=int, default=0,
                        help="Limit number of domains for smoke/audit runs; 0 means all")
    parser.add_argument("--allow_legacy_npz", action="store_true",
                        help="Allow npz files without dihedral_alignment metadata")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    DATA_DIR = Path(args.data_dir)
    RESULTS_DIR = Path(args.results_dir)
    RESULTS_DIR.mkdir(exist_ok=True)
    files = sorted(glob.glob(str(DATA_DIR / "*_dihedrals.npz")))
    if args.max_domains > 0:
        files = files[:args.max_domains]
    print(f"Found {len(files)} datasets")

    all_results = []
    for f in files:
        d = np.load(f)
        alignment = str(d["dihedral_alignment"]) if "dihedral_alignment" in d else "legacy_or_unknown"
        if alignment != "common_residue_index" and not args.allow_legacy_npz:
            raise ValueError(
                f"{f} lacks common_residue_index alignment metadata. "
                "Regenerate it with src/mdcath_convert_v3.py --force, or pass "
                "--allow_legacy_npz for historical-only reruns."
            )
        domain_id = str(d['domain_id'])
        N = int(d['N'])
        train = torch.tensor(d['train'], dtype=torch.float32, device=device).reshape(d['train'].shape[0], -1)
        val = torch.tensor(d['val'], dtype=torch.float32, device=device).reshape(d['val'].shape[0], -1)
        diff = wrap(val[:-1] - val[1:])
        id_err = math.degrees(torch.sqrt((diff ** 2).mean()).item())

        print(f"\n=== {domain_id} (N={N}, train {len(train)}, val {len(val)}, identity={id_err:.1f}°) ===")

        results_domain = {"domain_id": domain_id, "N": N, "identity_deg": id_err,
                          "train_size": len(train), "val_size": len(val), "models": {}}

        # MLP baseline
        torch.manual_seed(42)
        model = ChainMLP(N=N, n_components=8, hidden=128).to(device)
        r = train_model(model, train, val, device, steps=args.steps, batch=args.batch, name="MLP")
        print(f"  MLP       NLL={r['nll']:.3f}  mode={r['mode_deg']:.1f}°  best10={r['best10_deg']:.1f}°  ({r['params']:,} p, {r['time_sec']:.0f}s)")
        results_domain["models"]["MLP"] = r

        # AlphaDynamics t=1 (safer default for high-entropy data based on our findings)
        torch.manual_seed(42)
        model = ChainPhaseFlowVar(N=N, n_osc=64, n_components=8, hidden=128, t_max=1.0).to(device)
        r = train_model(model, train, val, device, steps=args.steps, batch=args.batch, name="PF_t1")
        print(f"  PF t=1    NLL={r['nll']:.3f}  mode={r['mode_deg']:.1f}°  best10={r['best10_deg']:.1f}°  ({r['params']:,} p, {r['time_sec']:.0f}s)")
        results_domain["models"]["PhaseFlow_t1"] = r

        # AlphaDynamics t=4 (in case we have structure)
        torch.manual_seed(42)
        model = ChainPhaseFlowVar(N=N, n_osc=64, n_components=8, hidden=128, t_max=4.0).to(device)
        r = train_model(model, train, val, device, steps=args.steps, batch=args.batch, name="PF_t4")
        print(f"  PF t=4    NLL={r['nll']:.3f}  mode={r['mode_deg']:.1f}°  best10={r['best10_deg']:.1f}°  ({r['params']:,} p, {r['time_sec']:.0f}s)")
        results_domain["models"]["PhaseFlow_t4"] = r

        all_results.append(results_domain)

        # PATCH 2026-04-25: incremental save after each domain
        # so power-loss/crash doesn't lose all completed results
        json_path_partial = RESULTS_DIR / f"{args.out_prefix}.json"
        with open(json_path_partial, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [partial save] {len(all_results)}/{len(files)} domains -> {json_path_partial}")

    # Save
    json_path = RESULTS_DIR / f"{args.out_prefix}.json"
    md_path = RESULTS_DIR / f"{args.out_prefix}.md"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Markdown report
    lines = ["# mdCATH unified benchmark — aligned 348 K domains",
             "",
             f"Data directory: `{DATA_DIR}`",
             "",
             f"Training steps per model: {args.steps}, batch: {args.batch}",
             "",
             "Protocol: CHARMM36m + TIP3P water, 348 K, 5 replicas per domain.",
             "Phi/psi pairs are aligned by common residue index.",
             "",
             "| Domain | N | Identity° | MLP NLL | PF_t1 NLL | PF_t4 NLL | Best model | ΔNLL (best PF - MLP) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in all_results:
        m = r["models"]
        mlp_nll = m["MLP"]["nll"]
        pf1 = m["PhaseFlow_t1"]["nll"]
        pf4 = m["PhaseFlow_t4"]["nll"]
        best_pf = min(pf1, pf4)
        best_name = "PF_t1" if pf1 < pf4 else "PF_t4"
        delta = best_pf - mlp_nll
        winner = best_name if best_pf < mlp_nll else "MLP"
        lines.append(f"| {r['domain_id']} | {r['N']} | {r['identity_deg']:.1f} | "
                     f"{mlp_nll:.3f} | {pf1:.3f} | {pf4:.3f} | "
                     f"**{winner}** | {delta:+.3f} |")

    # Aggregate
    mlp_nlls = [r["models"]["MLP"]["nll"] for r in all_results]
    pf_best_nlls = [min(r["models"]["PhaseFlow_t1"]["nll"], r["models"]["PhaseFlow_t4"]["nll"]) for r in all_results]
    delta = np.array(pf_best_nlls) - np.array(mlp_nlls)
    wins = int((delta < 0).sum())
    lines += ["",
              f"## Summary",
              f"- **AlphaDynamics wins: {wins}/{len(all_results)} domains**",
              f"- Mean ΔNLL: {delta.mean():+.3f} nats (negative = PhaseFlow better)",
              f"- MLP mean NLL: {np.mean(mlp_nlls):.3f}",
              f"- PhaseFlow best mean NLL: {np.mean(pf_best_nlls):.3f}"]

    with open(md_path, 'w') as f:
        f.write("\n".join(lines))
    print("\n\n=== FINAL REPORT ===\n")
    print("\n".join(lines))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()

"""Audit AlphaDynamics against a true temporal-context baseline.

The residual MLP audit is useful as a parametrization sanity check, but it is
still a pointwise model. This script adds a recurrent baseline that sees a
window of previous torsion frames before predicting the next frame. That is the
baseline family reviewers are more likely to accept as a stronger temporal
competitor.
"""
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
sys.path.insert(0, str(ROOT / "src"))

from chain_model import ChainMDNHead, ChainMLP, chain_log_prob, chain_sample
from train_real import ChainPhaseFlowVar

TWO_PI = 2.0 * math.pi


def wrap(angles):
    return angles - TWO_PI * torch.round(angles / TWO_PI)


def scalar_to_str(value):
    if hasattr(value, "shape") and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


class TemporalGRUBaseline(nn.Module):
    """GRU baseline over the last W torsion frames."""

    def __init__(self, N, window=8, n_components=8, hidden=128, layers=1, dropout=0.0):
        super().__init__()
        self.N = N
        self.A = 2 * N
        self.window = window
        self.gru = nn.GRU(
            input_size=2 * self.A,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = ChainMDNHead(hidden, self.A, n_components, hidden)

    def forward(self, seq_angles):
        feats = torch.cat([torch.sin(seq_angles), torch.cos(seq_angles)], dim=-1)
        out, _ = self.gru(feats)
        return self.head(out[:, -1])


def make_windows(flat, idx, window):
    offsets = torch.arange(window, device=flat.device)
    return flat[idx.unsqueeze(1) + offsets.unsqueeze(0)]


def validation_nll_pointwise(model, val_flat, window):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = val_flat[window - 1:-1]
        y = val_flat[window:]
        log_pi, mu, kappa = model(x)
        nll = -chain_log_prob(y, log_pi, mu, kappa).mean().item()
    if was_training:
        model.train()
    return nll


def validation_nll_temporal(model, val_flat, window):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        n_val = len(val_flat) - window
        idx = torch.arange(n_val, device=val_flat.device)
        x = make_windows(val_flat, idx, window)
        y = val_flat[window:]
        log_pi, mu, kappa = model(x)
        nll = -chain_log_prob(y, log_pi, mu, kappa).mean().item()
    if was_training:
        model.train()
    return nll


def clone_state_dict_to_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def eval_pointwise(model, val_flat, device, window, eval_samples):
    model.eval()
    with torch.no_grad():
        x = val_flat[window - 1:-1]
        y = val_flat[window:]
        log_pi, mu, kappa = model(x)
        nll = -chain_log_prob(y, log_pi, mu, kappa).mean().item()
        am = log_pi.argmax(dim=-1)
        bidx = torch.arange(len(x), device=device)
        mode = mu[bidx, am]
        mode_err = math.degrees(torch.sqrt((wrap(mode - y) ** 2).mean(-1)).mean().item())
        best10 = None
        if eval_samples > 0:
            samples = chain_sample(log_pi, mu, kappa, n_samples=eval_samples)
            best10 = math.degrees(
                torch.sqrt((wrap(samples - y.unsqueeze(1)) ** 2).mean(-1))
                .min(-1)
                .values.mean()
                .item()
            )
    return nll, mode_err, best10


def eval_temporal(model, val_flat, device, window, eval_samples):
    model.eval()
    with torch.no_grad():
        n_val = len(val_flat) - window
        idx = torch.arange(n_val, device=device)
        x = make_windows(val_flat, idx, window)
        y = val_flat[window:]
        log_pi, mu, kappa = model(x)
        nll = -chain_log_prob(y, log_pi, mu, kappa).mean().item()
        am = log_pi.argmax(dim=-1)
        bidx = torch.arange(n_val, device=device)
        mode = mu[bidx, am]
        mode_err = math.degrees(torch.sqrt((wrap(mode - y) ** 2).mean(-1)).mean().item())
        best10 = None
        if eval_samples > 0:
            samples = chain_sample(log_pi, mu, kappa, n_samples=eval_samples)
            best10 = math.degrees(
                torch.sqrt((wrap(samples - y.unsqueeze(1)) ** 2).mean(-1))
                .min(-1)
                .values.mean()
                .item()
            )
    return nll, mode_err, best10


def train_pointwise(model, train_flat, val_flat, device, steps, batch, lr, window, eval_every, eval_samples):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n_train = len(train_flat) - window
    if n_train <= 0:
        raise ValueError(f"Need more than {window} training frames")

    t0 = time.time()
    best_nll = float("inf")
    best_step = 0
    best_state = None
    last_eval_nll = None
    model.train()

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (batch,), device=device)
        x = train_flat[idx + window - 1]
        y = train_flat[idx + window]
        log_pi, mu, kappa = model(x)
        loss = -chain_log_prob(y, log_pi, mu, kappa).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            current_nll = validation_nll_pointwise(model, val_flat, window)
            last_eval_nll = current_nll
            if current_nll < best_nll:
                best_nll = current_nll
                best_step = step
                best_state = clone_state_dict_to_cpu(model)

    if eval_every > 0 and best_state is not None:
        model.load_state_dict(best_state)

    nll, mode_err, best10 = eval_pointwise(model, val_flat, device, window, eval_samples)
    return {
        "nll": nll,
        "selected_step": best_step if eval_every > 0 else steps,
        "best_nll": best_nll if eval_every > 0 else nll,
        "last_eval_nll": last_eval_nll if eval_every > 0 else nll,
        "eval_every": eval_every,
        "mode_deg": mode_err,
        "best_of_samples_deg": best10,
        "params": sum(p.numel() for p in model.parameters()),
        "time_sec": time.time() - t0,
    }


def train_temporal(model, train_flat, val_flat, device, steps, batch, lr, window, eval_every, eval_samples):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n_train = len(train_flat) - window
    if n_train <= 0:
        raise ValueError(f"Need more than {window} training frames")

    t0 = time.time()
    best_nll = float("inf")
    best_step = 0
    best_state = None
    last_eval_nll = None
    model.train()

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (batch,), device=device)
        x = make_windows(train_flat, idx, window)
        y = train_flat[idx + window]
        log_pi, mu, kappa = model(x)
        loss = -chain_log_prob(y, log_pi, mu, kappa).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            current_nll = validation_nll_temporal(model, val_flat, window)
            last_eval_nll = current_nll
            if current_nll < best_nll:
                best_nll = current_nll
                best_step = step
                best_state = clone_state_dict_to_cpu(model)

    if eval_every > 0 and best_state is not None:
        model.load_state_dict(best_state)

    nll, mode_err, best10 = eval_temporal(model, val_flat, device, window, eval_samples)
    return {
        "nll": nll,
        "selected_step": best_step if eval_every > 0 else steps,
        "best_nll": best_nll if eval_every > 0 else nll,
        "last_eval_nll": last_eval_nll if eval_every > 0 else nll,
        "eval_every": eval_every,
        "mode_deg": mode_err,
        "best_of_samples_deg": best10,
        "params": sum(p.numel() for p in model.parameters()),
        "time_sec": time.time() - t0,
    }


def find_files(data_dir, domains, max_domains):
    if domains:
        files = []
        missing = []
        for domain in domains:
            candidates = sorted(glob.glob(str(data_dir / f"{domain}*_dihedrals.npz")))
            if candidates:
                files.append(candidates[0])
            else:
                missing.append(domain)
        if missing:
            raise FileNotFoundError(f"Missing domains in {data_dir}: {', '.join(missing)}")
    else:
        files = sorted(glob.glob(str(data_dir / "*_dihedrals.npz")))

    if max_domains > 0:
        files = files[:max_domains]
    if not files:
        raise FileNotFoundError(f"No *_dihedrals.npz files found in {data_dir}")
    return [Path(f) for f in files]


def load_domain(path, device, allow_legacy_npz):
    data = np.load(path)
    alignment = scalar_to_str(data["dihedral_alignment"]) if "dihedral_alignment" in data else "legacy_or_unknown"
    if alignment != "common_residue_index" and not allow_legacy_npz:
        raise ValueError(
            f"{path} lacks common_residue_index alignment metadata. "
            "Use aligned converter output or pass --allow_legacy_npz for historical-only checks."
        )

    train_np = data["train"]
    val_np = data["val"]
    domain_id = scalar_to_str(data["domain_id"]) if "domain_id" in data else path.name.split("_")[0]
    N = int(data["N"]) if "N" in data else int(train_np.shape[1])
    train = torch.tensor(train_np, dtype=torch.float32, device=device).reshape(train_np.shape[0], -1)
    val = torch.tensor(val_np, dtype=torch.float32, device=device).reshape(val_np.shape[0], -1)
    diff = wrap(val[:-1] - val[1:])
    identity_deg = math.degrees(torch.sqrt((diff**2).mean()).item())
    return domain_id, N, train, val, identity_deg, alignment


def metric_value(metrics, key="nll"):
    return metrics[key] if metrics is not None else float("nan")


def save_report(results, run_meta, results_dir, out_prefix):
    json_path = results_dir / f"{out_prefix}.json"
    md_path = results_dir / f"{out_prefix}.md"
    payload = {"run": run_meta, "results": results}
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Temporal-context baseline audit",
        "",
        "Purpose: test AlphaDynamics against a true sequence-context baseline.",
        "",
        f"Data directory: `{run_meta['data_dir']}`",
        f"Window: {run_meta['window']} frames",
        f"Steps/model: {run_meta['steps']}, batch: {run_meta['batch']}, seeds: {run_meta['seeds']}",
        f"Validation selection: {'best checkpoint every ' + str(run_meta['eval_every']) + ' steps' if run_meta['eval_every'] else 'final checkpoint only'}",
        f"Device: {run_meta['device']}",
        "",
        "| Domain | Seed | N | Identity° | MLP abs NLL | Temporal GRU NLL | Best PF NLL | Best PF | Winner vs GRU | ΔNLL PF-GRU |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]

    pf_vs_gru = []
    pf_vs_abs = []
    gru_vs_abs = []
    abs_nlls = []
    gru_nlls = []
    pf_nlls = []

    for row in results:
        models = row["models"]
        abs_nll = metric_value(models.get("MLP_abs"))
        gru_nll = metric_value(models.get("TemporalGRU"))
        pf_items = [(name, metrics["nll"]) for name, metrics in models.items() if name.startswith("PhaseFlow_t")]
        if pf_items:
            best_pf_name, best_pf_nll = min(pf_items, key=lambda item: item[1])
        else:
            best_pf_name, best_pf_nll = "NA", float("nan")
        delta_gru = best_pf_nll - gru_nll
        winner = best_pf_name if delta_gru < 0 else "TemporalGRU"

        pf_vs_gru.append(delta_gru < 0)
        pf_vs_abs.append(best_pf_nll < abs_nll)
        gru_vs_abs.append(gru_nll < abs_nll)
        abs_nlls.append(abs_nll)
        gru_nlls.append(gru_nll)
        pf_nlls.append(best_pf_nll)

        lines.append(
            f"| {row['domain_id']} | {row['seed']} | {row['N']} | {row['identity_deg']:.1f} | "
            f"{abs_nll:.3f} | {gru_nll:.3f} | {best_pf_nll:.3f} | {best_pf_name} | "
            f"**{winner}** | {delta_gru:+.3f} |"
        )

    if results:
        delta = np.array(pf_nlls) - np.array(gru_nlls)
        delta_abs = np.array(pf_nlls) - np.array(abs_nlls)
        gru_delta_abs = np.array(gru_nlls) - np.array(abs_nlls)
        lines += [
            "",
            "## Summary",
            f"- PhaseFlow wins vs temporal GRU: **{sum(pf_vs_gru)}/{len(results)}** domain-seed runs.",
            f"- PhaseFlow wins vs absolute MLP: **{sum(pf_vs_abs)}/{len(results)}** domain-seed runs.",
            f"- Temporal GRU improves over absolute MLP: **{sum(gru_vs_abs)}/{len(results)}** domain-seed runs.",
            f"- Mean MLP abs NLL: {np.mean(abs_nlls):.3f}",
            f"- Mean temporal GRU NLL: {np.mean(gru_nlls):.3f}",
            f"- Mean best PhaseFlow NLL: {np.mean(pf_nlls):.3f}",
            f"- Mean ΔNLL PhaseFlow-GRU: {delta.mean():+.3f} nats (negative favors PhaseFlow).",
            f"- Mean ΔNLL PhaseFlow-absolute: {delta_abs.mean():+.3f} nats (negative favors PhaseFlow).",
            f"- Mean ΔNLL GRU-absolute: {gru_delta_abs.mean():+.3f} nats (negative favors GRU).",
        ]

    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "mdcath_real_data" / "mdcath_348K"))
    parser.add_argument("--results_dir", default=str(ROOT / "results"))
    parser.add_argument("--out_prefix", default="temporal_baseline_audit")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--max_domains", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--phaseflow_tmax", nargs="+", type=float, default=[4.0])
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["mlp_abs", "temporal_gru", "phaseflow"],
                        choices=["mlp_abs", "temporal_gru", "phaseflow"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow_legacy_npz", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    files = find_files(data_dir, args.domains, args.max_domains)
    print(f"Device: {device}")
    print(f"Found {len(files)} datasets")

    run_meta = {
        "data_dir": str(data_dir),
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "eval_every": args.eval_every,
        "seeds": args.seeds,
        "phaseflow_tmax": args.phaseflow_tmax,
        "window": args.window,
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "n_components": args.n_components,
        "eval_samples": args.eval_samples,
        "models": args.models,
        "device": str(device),
    }

    all_results = []
    for path in files:
        domain_id, N, train, val, identity_deg, alignment = load_domain(path, device, args.allow_legacy_npz)
        print(f"\n=== {domain_id} (N={N}, train={len(train)}, val={len(val)}, identity={identity_deg:.1f}°, {alignment}) ===")

        for seed in args.seeds:
            print(f"  seed {seed}")
            row = {
                "domain_id": domain_id,
                "source_file": str(path),
                "seed": seed,
                "N": N,
                "identity_deg": identity_deg,
                "train_size": len(train),
                "val_size": len(val),
                "window": args.window,
                "alignment": alignment,
                "models": {},
            }

            if "mlp_abs" in args.models:
                set_seed(seed, device)
                model = ChainMLP(N=N, n_components=args.n_components, hidden=args.hidden).to(device)
                metrics = train_pointwise(model, train, val, device, args.steps, args.batch, args.lr, args.window, args.eval_every, args.eval_samples)
                row["models"]["MLP_abs"] = metrics
                print(f"    MLP_abs       NLL={metrics['nll']:.3f} step={metrics['selected_step']} mode={metrics['mode_deg']:.1f}°")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if "temporal_gru" in args.models:
                set_seed(seed, device)
                model = TemporalGRUBaseline(
                    N=N,
                    window=args.window,
                    n_components=args.n_components,
                    hidden=args.hidden,
                    layers=args.layers,
                    dropout=args.dropout,
                ).to(device)
                metrics = train_temporal(model, train, val, device, args.steps, args.batch, args.lr, args.window, args.eval_every, args.eval_samples)
                row["models"]["TemporalGRU"] = metrics
                print(f"    TemporalGRU   NLL={metrics['nll']:.3f} step={metrics['selected_step']} mode={metrics['mode_deg']:.1f}°")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if "phaseflow" in args.models:
                for t_max in args.phaseflow_tmax:
                    set_seed(seed, device)
                    model = ChainPhaseFlowVar(
                        N=N,
                        n_osc=64,
                        n_components=args.n_components,
                        hidden=args.hidden,
                        t_max=t_max,
                    ).to(device)
                    metrics = train_pointwise(model, train, val, device, args.steps, args.batch, args.lr, args.window, args.eval_every, args.eval_samples)
                    key = f"PhaseFlow_t{t_max:g}"
                    row["models"][key] = metrics
                    print(f"    {key:<13} NLL={metrics['nll']:.3f} step={metrics['selected_step']} mode={metrics['mode_deg']:.1f}°")
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            all_results.append(row)
            json_path, md_path = save_report(all_results, run_meta, results_dir, args.out_prefix)
            print(f"    [saved] {len(all_results)} domain-seed runs -> {json_path}")

    _, md_path = save_report(all_results, run_meta, results_dir, args.out_prefix)
    print(f"\nFinal report: {md_path}")


if __name__ == "__main__":
    main()

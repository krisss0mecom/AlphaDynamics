#!/usr/bin/env python3
"""AlphaDynamics CLI MVP.

This is a thin product wrapper around the audited research scripts. It does
not invent a second training path; it makes the current convert -> train ->
rollout -> report workflow reproducible from one entry point.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    rendered = " ".join(cmd)
    print(f"$ {rendered}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def add_common_io(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the underlying command without executing it")


def cmd_convert(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON, str(ROOT / "src" / "mdcath_convert_v3.py"),
        "--bench_dir", args.bench_dir,
        "--out_dir", args.out_dir,
    ]
    if args.force:
        cmd.append("--force")
    run_command(cmd, dry_run=args.dry_run)


def cmd_train(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON, str(ROOT / "src" / "mdcath_benchmark.py"),
        "--data_dir", args.data_dir,
        "--results_dir", args.results_dir,
        "--out_prefix", args.out_prefix,
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--device", args.device,
    ]
    if args.max_domains:
        cmd += ["--max_domains", str(args.max_domains)]
    if args.allow_legacy_npz:
        cmd.append("--allow_legacy_npz")
    run_command(cmd, dry_run=args.dry_run)


def cmd_rollout(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON, str(ROOT / "src" / "ramachandran_energy_v2.py"),
        "--data_dir", args.data_dir,
        "--results_dir", args.results_dir,
        "--fig_dir", args.fig_dir,
        "--out_prefix", args.out_prefix,
        "--temp", str(args.temp),
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--rollout_steps", str(args.rollout_steps),
        "--kappa_mult", str(args.kappa_mult),
        "--device", args.device,
        "--domains", *args.domains,
    ]
    if args.allow_legacy_npz:
        cmd.append("--allow_legacy_npz")
    run_command(cmd, dry_run=args.dry_run)


def cmd_kappa_sweep(args: argparse.Namespace) -> None:
    for kappa_mult in args.kappa_mult:
        token = str(kappa_mult).replace(".", "p")
        cmd = [
            PYTHON, str(ROOT / "src" / "ramachandran_energy_v2.py"),
            "--data_dir", args.data_dir,
            "--results_dir", args.results_dir,
            "--fig_dir", args.fig_dir,
            "--out_prefix", f"{args.out_prefix}_kappa{token}",
            "--temp", str(args.temp),
            "--steps", str(args.steps),
            "--batch", str(args.batch),
            "--rollout_steps", str(args.rollout_steps),
            "--kappa_mult", str(kappa_mult),
            "--device", args.device,
            "--domains", *args.domains,
        ]
        if args.allow_legacy_npz:
            cmd.append("--allow_legacy_npz")
        run_command(cmd, dry_run=args.dry_run)


def cmd_strong_baseline(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON, str(ROOT / "src" / "strong_baseline_audit.py"),
        "--data_dir", args.data_dir,
        "--results_dir", args.results_dir,
        "--out_prefix", args.out_prefix,
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--eval_every", str(args.eval_every),
        "--hidden", str(args.hidden),
        "--n_components", str(args.n_components),
        "--device", args.device,
        "--phaseflow_tmax", *[str(v) for v in args.phaseflow_tmax],
        "--seeds", *[str(v) for v in args.seeds],
    ]
    if args.domains:
        cmd += ["--domains", *args.domains]
    if args.max_domains:
        cmd += ["--max_domains", str(args.max_domains)]
    if args.allow_legacy_npz:
        cmd.append("--allow_legacy_npz")
    run_command(cmd, dry_run=args.dry_run)


def cmd_temporal_baseline(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON, str(ROOT / "src" / "temporal_baseline_audit.py"),
        "--data_dir", args.data_dir,
        "--results_dir", args.results_dir,
        "--out_prefix", args.out_prefix,
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--eval_every", str(args.eval_every),
        "--hidden", str(args.hidden),
        "--layers", str(args.layers),
        "--dropout", str(args.dropout),
        "--n_components", str(args.n_components),
        "--eval_samples", str(args.eval_samples),
        "--window", str(args.window),
        "--device", args.device,
        "--phaseflow_tmax", *[str(v) for v in args.phaseflow_tmax],
        "--seeds", *[str(v) for v in args.seeds],
        "--models", *args.models,
    ]
    if args.domains:
        cmd += ["--domains", *args.domains]
    if args.max_domains:
        cmd += ["--max_domains", str(args.max_domains)]
    if args.allow_legacy_npz:
        cmd.append("--allow_legacy_npz")
    run_command(cmd, dry_run=args.dry_run)


def scalar_to_str(value: Any) -> str:
    if hasattr(value, "shape") and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def cmd_doctor(args: argparse.Namespace) -> None:
    checks = [
        ("python", platform.python_version(), True),
        ("repo", str(ROOT), ROOT.exists()),
    ]
    for module in [
        "torch",
        "torchdiffeq",
        "numpy",
        "scipy",
        "mdtraj",
        "h5py",
        "matplotlib",
        "huggingface_hub",
    ]:
        spec = importlib.util.find_spec(module)
        checks.append((module, "importable" if spec else "missing", spec is not None))

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        import torch

        checks.append(("torch version", torch.__version__, True))
        checks.append(("cuda available", str(torch.cuda.is_available()), True))
        if torch.cuda.is_available():
            checks.append(("cuda device", torch.cuda.get_device_name(0), True))

    for path in [
        ROOT / "results" / "mdcath_aligned20_4000step_cpu.json",
        ROOT / "results" / "mdcath_aligned20_n100_4000step_gpu.json",
        ROOT / "results" / "ramachandran_aligned3_4000step_gpu.json",
        ROOT / "results" / "ramachandran_aligned3_n98_4000step_gpu.json",
    ]:
        checks.append((str(path.relative_to(ROOT)), "present" if path.exists() else "missing", path.exists()))

    ok = True
    for name, detail, passed in checks:
        ok = ok and passed
        mark = "OK" if passed else "MISSING"
        print(f"{mark:8} {name}: {detail}")

    if not ok and args.strict:
        raise SystemExit(1)


def cmd_validate_data(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*_dihedrals.npz"))
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        raise FileNotFoundError(f"No *_dihedrals.npz files found in {data_dir}")

    rows = []
    failures = []
    for path in files:
        data = np.load(path)
        alignment = (
            scalar_to_str(data["dihedral_alignment"])
            if "dihedral_alignment" in data
            else "missing"
        )
        domain_id = scalar_to_str(data["domain_id"]) if "domain_id" in data else path.stem
        train = data["train"]
        val = data["val"]
        n_value = int(data["N"]) if "N" in data else int(train.shape[1])
        residue_indices = data["residue_indices"] if "residue_indices" in data else None
        issues = []
        if alignment != "common_residue_index":
            issues.append(f"alignment={alignment}")
        if train.ndim != 3 or train.shape[-1] != 2:
            issues.append(f"train_shape={train.shape}")
        if val.ndim != 3 or val.shape[-1] != 2:
            issues.append(f"val_shape={val.shape}")
        if train.ndim >= 2 and train.shape[1] != n_value:
            issues.append(f"N={n_value}, train_N={train.shape[1]}")
        if residue_indices is None:
            issues.append("missing_residue_indices")
        elif len(residue_indices) != n_value:
            issues.append(f"residue_indices={len(residue_indices)}, N={n_value}")
        rows.append((domain_id, n_value, train.shape[0], val.shape[0], alignment, issues))
        if issues:
            failures.append((path, issues))

    print("| Domain | N | Train frames | Val frames | Alignment | Status |")
    print("|---|---:|---:|---:|---|---|")
    for domain_id, n_value, train_len, val_len, alignment, issues in rows:
        status = "OK" if not issues else "; ".join(issues)
        print(f"| {domain_id} | {n_value} | {train_len} | {val_len} | {alignment} | {status} |")

    print(f"\nChecked {len(rows)} files in {data_dir}")
    if failures:
        print(f"Failed validation: {len(failures)} files")
        if args.strict:
            raise SystemExit(1)
    else:
        print("All checked files passed the aligned torsion data contract.")


def summarize_nll(path: Path) -> dict[str, float | int | str]:
    rows = json.loads(path.read_text())
    mlp = [r["models"]["MLP"]["nll"] for r in rows]
    pf = [
        min(r["models"]["PhaseFlow_t1"]["nll"], r["models"]["PhaseFlow_t4"]["nll"])
        for r in rows
    ]
    wins = sum(1 for m, p in zip(mlp, pf) if p < m)
    return {
        "name": path.stem,
        "domains": len(rows),
        "N": rows[0]["N"] if rows else 0,
        "wins": wins,
        "mlp_mean": sum(mlp) / len(mlp),
        "pf_mean": sum(pf) / len(pf),
        "ratio": sum(mlp) / sum(pf),
        "ad_params": rows[0]["models"]["PhaseFlow_t4"].get("params", 0) if rows else 0,
        "mlp_params": rows[0]["models"]["MLP"].get("params", 0) if rows else 0,
    }


def summarize_rollout(path: Path) -> dict[str, float | int | str]:
    data = json.loads(path.read_text())
    rows = list(data.values())
    return {
        "name": path.stem,
        "domains": len(rows),
        "N": rows[0]["N"] if rows else 0,
        "jsd": sum(r["jsd_mean"] for r in rows) / len(rows),
        "emd": sum(r["emd_mean_deg"] for r in rows) / len(rows),
        "dg": sum(r["dG_basin_mean_kcal"] for r in rows) / len(rows),
        "pop": sum(r["pop_err_mean"] for r in rows) / len(rows),
    }


def summarize_strong_baseline(path: Path) -> dict[str, float | int | str]:
    data = json.loads(path.read_text())
    rows = data.get("results", [])
    if not rows:
        return {
            "name": path.stem,
            "runs": 0,
            "pf_res_wins": 0,
            "pf_abs_wins": 0,
            "res_abs_wins": 0,
            "abs_mean": 0.0,
            "res_mean": 0.0,
            "pf_mean": 0.0,
        }
    abs_nll = []
    res_nll = []
    pf_nll = []
    for row in rows:
        models = row["models"]
        best_pf = min(
            metrics["nll"]
            for name, metrics in models.items()
            if name.startswith("PhaseFlow_t")
        )
        abs_value = models["MLP_abs"]["nll"]
        res_value = models["MLP_residual"]["nll"]
        abs_nll.append(abs_value)
        res_nll.append(res_value)
        pf_nll.append(best_pf)
    return {
        "name": path.stem,
        "runs": len(rows),
        "pf_res_wins": sum(1 for p, r in zip(pf_nll, res_nll) if p < r),
        "pf_abs_wins": sum(1 for p, a in zip(pf_nll, abs_nll) if p < a),
        "res_abs_wins": sum(1 for r, a in zip(res_nll, abs_nll) if r < a),
        "abs_mean": sum(abs_nll) / len(abs_nll),
        "res_mean": sum(res_nll) / len(res_nll),
        "pf_mean": sum(pf_nll) / len(pf_nll),
    }


def summarize_temporal_baseline(path: Path) -> dict[str, float | int | str]:
    data = json.loads(path.read_text())
    rows = data.get("results", [])
    if not rows:
        return {
            "name": path.stem,
            "runs": 0,
            "pf_gru_wins": 0,
            "pf_abs_wins": 0,
            "gru_abs_wins": 0,
            "abs_mean": 0.0,
            "gru_mean": 0.0,
            "pf_mean": 0.0,
        }
    abs_nll = []
    gru_nll = []
    pf_nll = []
    for row in rows:
        models = row["models"]
        best_pf = min(
            metrics["nll"]
            for name, metrics in models.items()
            if name.startswith("PhaseFlow_t")
        )
        abs_value = models["MLP_abs"]["nll"]
        gru_value = models["TemporalGRU"]["nll"]
        abs_nll.append(abs_value)
        gru_nll.append(gru_value)
        pf_nll.append(best_pf)
    return {
        "name": path.stem,
        "runs": len(rows),
        "pf_gru_wins": sum(1 for p, g in zip(pf_nll, gru_nll) if p < g),
        "pf_abs_wins": sum(1 for p, a in zip(pf_nll, abs_nll) if p < a),
        "gru_abs_wins": sum(1 for g, a in zip(gru_nll, abs_nll) if g < a),
        "abs_mean": sum(abs_nll) / len(abs_nll),
        "gru_mean": sum(gru_nll) / len(gru_nll),
        "pf_mean": sum(pf_nll) / len(pf_nll),
    }


def cmd_report(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    lines = [
        "# AlphaDynamics audit report",
        "",
        "Generated from existing JSON result files.",
        "",
        "## One-step NLL",
        "",
        "| Run | Domains | N | Wins | Mean MLP NLL | Mean AD NLL | Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for prefix in args.nll_prefix:
        row = summarize_nll(results_dir / f"{prefix}.json")
        lines.append(
            f"| `{row['name']}` | {row['domains']} | {row['N']} | "
            f"{row['wins']}/{row['domains']} | {row['mlp_mean']:.3f} | "
            f"{row['pf_mean']:.3f} | {row['ratio']:.2f}x |"
        )

    lines += [
        "",
        "## Model Size",
        "",
        "| Run | MLP params | AlphaDynamics params |",
        "|---|---:|---:|",
    ]
    for prefix in args.nll_prefix:
        row = summarize_nll(results_dir / f"{prefix}.json")
        lines.append(
            f"| `{row['name']}` | {int(row['mlp_params']):,} | "
            f"{int(row['ad_params']):,} |"
        )

    lines += [
        "",
        "## Rollout Free Energy",
        "",
        "| Run | Domains | N | Mean JSD | Mean EMD | Mean abs dG_basin | Mean pop err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for prefix in args.rollout_prefix:
        row = summarize_rollout(results_dir / f"{prefix}.json")
        lines.append(
            f"| `{row['name']}` | {row['domains']} | {row['N']} | "
            f"{row['jsd']:.3f} | {row['emd']:.1f} deg | "
            f"{row['dg']:.3f} kcal/mol | {row['pop']:.3f} |"
        )

    if args.strong_prefix:
        lines += [
            "",
            "## Strong Baseline",
            "",
            "| Run | Domain-seed runs | PF wins vs residual MLP | PF wins vs absolute MLP | Residual wins vs absolute MLP | Mean abs MLP NLL | Mean residual MLP NLL | Mean AD NLL |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for prefix in args.strong_prefix:
            path = results_dir / f"{prefix}.json"
            if not path.exists():
                lines.append(f"| `{prefix}` | 0 | missing | missing | missing | 0.000 | 0.000 | 0.000 |")
                continue
            row = summarize_strong_baseline(path)
            lines.append(
                f"| `{row['name']}` | {row['runs']} | "
                f"{row['pf_res_wins']}/{row['runs']} | "
                f"{row['pf_abs_wins']}/{row['runs']} | "
                f"{row['res_abs_wins']}/{row['runs']} | "
                f"{row['abs_mean']:.3f} | {row['res_mean']:.3f} | "
                f"{row['pf_mean']:.3f} |"
            )

    if args.temporal_prefix:
        lines += [
            "",
            "## Temporal Baseline",
            "",
            "| Run | Domain-seed runs | PF wins vs temporal GRU | PF wins vs absolute MLP | GRU wins vs absolute MLP | Mean abs MLP NLL | Mean temporal GRU NLL | Mean AD NLL |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for prefix in args.temporal_prefix:
            path = results_dir / f"{prefix}.json"
            if not path.exists():
                lines.append(f"| `{prefix}` | 0 | missing | missing | missing | 0.000 | 0.000 | 0.000 |")
                continue
            row = summarize_temporal_baseline(path)
            lines.append(
                f"| `{row['name']}` | {row['runs']} | "
                f"{row['pf_gru_wins']}/{row['runs']} | "
                f"{row['pf_abs_wins']}/{row['runs']} | "
                f"{row['gru_abs_wins']}/{row['runs']} | "
                f"{row['abs_mean']:.3f} | {row['gru_mean']:.3f} | "
                f"{row['pf_mean']:.3f} |"
            )

    lines += [
        "",
        "## Product Readiness",
        "",
        "- Reproducible command surface: `alphadynamics` / `python src/alphadynamics_cli.py`.",
        "- Data contract: aligned `.npz` files with `dihedral_alignment=common_residue_index`.",
        "- Current evidence: aligned one-step NLL audit, limited rollout audit, and optional residual-baseline audit.",
        "- Current limitation: rollout calibration still uses a sampling-time kappa multiplier.",
    ]

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    if args.dry_run:
        print("\n".join(lines))
        print(f"\n[dry-run] would write {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlphaDynamics product CLI: diagnose, validate, train, rollout, baseline, report"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert", help="Convert mdCATH H5 files to aligned torsion npz")
    p.add_argument("--bench-dir", default=str(ROOT / "mdcath_raw"))
    p.add_argument("--out-dir", default=str(ROOT / "mdcath_real_data" / "mdcath_348K"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("train", help="Train/evaluate MLP and AlphaDynamics NLL benchmark")
    add_common_io(p)
    p.add_argument("--data-dir", default=str(ROOT / "mdcath_real_data" / "mdcath_348K"))
    p.add_argument("--out-prefix", default="alphadynamics_train")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-domains", type=int, default=0)
    p.add_argument("--allow-legacy-npz", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("rollout", help="Train AlphaDynamics and evaluate rollout free energy")
    add_common_io(p)
    p.add_argument("--data-dir", default=str(ROOT / "mdcath_real_data" / "mdcath_alltemps"))
    p.add_argument("--fig-dir", default=str(ROOT / "paper" / "figures"))
    p.add_argument("--out-prefix", default="alphadynamics_rollout")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--temp", type=int, default=348)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=2500)
    p.add_argument("--kappa-mult", type=float, default=30.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--allow-legacy-npz", action="store_true")
    p.set_defaults(func=cmd_rollout)

    p = sub.add_parser("kappa-sweep", help="Run rollout audits across kappa sampling multipliers")
    add_common_io(p)
    p.add_argument("--data-dir", default=str(ROOT / "mdcath_real_data" / "mdcath_alltemps"))
    p.add_argument("--fig-dir", default=str(ROOT / "paper" / "figures"))
    p.add_argument("--out-prefix", default="alphadynamics_kappa_sweep")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--temp", type=int, default=348)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=2500)
    p.add_argument("--kappa-mult", nargs="+", type=float, default=[1.0, 5.0, 10.0, 20.0, 30.0, 50.0])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--allow-legacy-npz", action="store_true")
    p.set_defaults(func=cmd_kappa_sweep)

    p = sub.add_parser("strong-baseline", help="Audit against residual/autoregressive MLP")
    add_common_io(p)
    p.add_argument("--data-dir", default=str(ROOT / "mdcath_real_data" / "mdcath_348K"))
    p.add_argument("--out-prefix", default="strong_baseline_audit")
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--max-domains", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--phaseflow-tmax", nargs="+", type=float, default=[1.0, 4.0])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--allow-legacy-npz", action="store_true")
    p.set_defaults(func=cmd_strong_baseline)

    p = sub.add_parser("temporal-baseline", help="Audit against a true temporal GRU context baseline")
    add_common_io(p)
    p.add_argument("--data-dir", default=str(ROOT / "mdcath_real_data" / "mdcath_348K"))
    p.add_argument("--out-prefix", default="temporal_baseline_audit")
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--max-domains", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--phaseflow-tmax", nargs="+", type=float, default=[4.0])
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--eval-samples", type=int, default=10)
    p.add_argument("--models", nargs="+", default=["mlp_abs", "temporal_gru", "phaseflow"],
                   choices=["mlp_abs", "temporal_gru", "phaseflow"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--allow-legacy-npz", action="store_true")
    p.set_defaults(func=cmd_temporal_baseline)

    p = sub.add_parser("doctor", help="Check environment and shipped result artifacts")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if required checks fail")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("validate-data", help="Validate converted aligned torsion npz files")
    p.add_argument("--data-dir", default=str(ROOT / "mdcath_real_data" / "mdcath_348K"))
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--strict", action="store_true", help="Exit non-zero if any file fails validation")
    p.set_defaults(func=cmd_validate_data)

    p = sub.add_parser("report", help="Create a compact Markdown report from result JSON")
    add_common_io(p)
    p.add_argument("--output", default=str(ROOT / "results" / "alphadynamics_audit_report.md"))
    p.add_argument("--nll-prefix", nargs="+", default=[
        "mdcath_aligned20_4000step_cpu",
        "mdcath_aligned20_n100_4000step_gpu",
    ])
    p.add_argument("--rollout-prefix", nargs="+", default=[
        "ramachandran_aligned3_4000step_gpu",
        "ramachandran_aligned3_n98_4000step_gpu",
    ])
    p.add_argument("--strong-prefix", nargs="*", default=[
        "strong_baseline_3dom_3seed_4000step_cuda",
    ])
    p.add_argument("--temporal-prefix", nargs="*", default=[])
    p.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

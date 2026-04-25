#!/usr/bin/env python3
"""AlphaDynamics CLI MVP.

This is a thin product wrapper around the audited research scripts. It does
not invent a second training path; it makes the current convert -> train ->
rollout -> report workflow reproducible from one entry point.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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
        description="AlphaDynamics CLI MVP: convert, train, rollout, report"
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
    p.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Compute publication-grade statistical tests for the v1 aligned audit.

Adds Wilcoxon signed-rank, sign test, paired-t, geometric mean, bootstrap CI for
ratio-of-means.  Operates on the v1 result JSONs without re-training.

Inputs:
  results/mdcath_aligned20_4000step_cpu.json     (N=48, 20 domains)
  results/mdcath_aligned20_n100_4000step_gpu.json (N=98, 20 domains)

Output:
  results/audit_statistics.json
  results/audit_statistics.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_pairs(path: Path, model_a: str, model_b: str):
    with open(path) as f:
        domains = json.load(f)
    pairs = []
    for d in domains:
        models = d.get("models", {})
        if model_a in models and model_b in models:
            pairs.append({
                "domain_id": d["domain_id"],
                "N": d["N"],
                "identity_deg": d.get("identity_deg"),
                f"nll_{model_a}": models[model_a]["nll"],
                f"nll_{model_b}": models[model_b]["nll"],
                f"params_{model_a}": models[model_a].get("params"),
                f"params_{model_b}": models[model_b].get("params"),
            })
    return pairs


def stats_block(label: str, pairs: list, key_a: str, key_b: str):
    a = np.array([p[key_a] for p in pairs])
    b = np.array([p[key_b] for p in pairs])
    n = len(a)
    diff = a - b
    wins_b = int((b < a).sum())  # b wins when smaller NLL

    # Wilcoxon signed-rank (two-sided)
    try:
        wilcoxon = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        wilcoxon_stat = float(wilcoxon.statistic)
        wilcoxon_p = float(wilcoxon.pvalue)
    except Exception as e:
        wilcoxon_stat = float("nan")
        wilcoxon_p = float("nan")

    # Sign test (binomial)
    sign_p = 2 * stats.binom.cdf(min(wins_b, n - wins_b), n, 0.5) if n else float("nan")

    # Paired t-test on log NLLs (more robust to scale)
    log_diff = np.log(a) - np.log(b)
    try:
        t = stats.ttest_rel(np.log(a), np.log(b))
        t_stat = float(t.statistic)
        t_p = float(t.pvalue)
    except Exception:
        t_stat = float("nan")
        t_p = float("nan")

    # Bootstrap CI for ratio-of-means
    rng = np.random.default_rng(42)
    n_boot = 10000
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ratios[i] = a[idx].mean() / b[idx].mean()
    ratio_ci_low = float(np.percentile(ratios, 2.5))
    ratio_ci_high = float(np.percentile(ratios, 97.5))

    return {
        "label": label,
        "n_domains": int(n),
        f"mean_nll_{key_a.removeprefix('nll_')}": float(a.mean()),
        f"mean_nll_{key_b.removeprefix('nll_')}": float(b.mean()),
        "ratio_of_means": float(a.mean() / b.mean()),
        "ratio_ci_95": [ratio_ci_low, ratio_ci_high],
        "geometric_mean_per_domain_ratio": float(np.exp((np.log(a) - np.log(b)).mean())),
        f"wins_{key_b.removeprefix('nll_')}": wins_b,
        "wilcoxon_W": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": float(sign_p),
        "paired_t_log_stat": t_stat,
        "paired_t_log_p": t_p,
        "log_diff_mean": float(log_diff.mean()),
        "log_diff_std": float(log_diff.std(ddof=1) if len(log_diff) > 1 else 0.0),
    }


def main():
    files = [
        ("N=48 aligned (20 domains)", RESULTS / "mdcath_aligned20_4000step_cpu.json"),
        ("N=98 aligned (20 domains)", RESULTS / "mdcath_aligned20_n100_4000step_gpu.json"),
    ]

    out = {"timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    out["per_size_class"] = []
    all_pairs = []
    for label, path in files:
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        pairs = load_pairs(path, "MLP", "PhaseFlow_t4")
        if not pairs:
            continue
        all_pairs.extend(pairs)
        block = stats_block(label, pairs, "nll_MLP", "nll_PhaseFlow_t4")
        out["per_size_class"].append(block)
        print(f"\n=== {label} ===")
        for k, v in block.items():
            print(f"  {k}: {v}")

    # Combined across all 40
    combined = stats_block("Combined aligned audit (40 domains)",
                           all_pairs, "nll_MLP", "nll_PhaseFlow_t4")
    out["combined"] = combined
    print(f"\n=== Combined ===")
    for k, v in combined.items():
        print(f"  {k}: {v}")

    out_json = RESULTS / "audit_statistics.json"
    out_md = RESULTS / "audit_statistics.md"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_json}")

    # Human-readable Markdown summary
    lines = [
        "# Aligned audit — statistical tests (paper-ready)",
        "",
        "Computed from `results/mdcath_aligned20_4000step_cpu.json` and",
        "`results/mdcath_aligned20_n100_4000step_gpu.json`. Pairs (MLP, PhaseFlow_t4) per domain.",
        "",
        "| Size class | n | Mean MLP | Mean AD | Ratio (95% CI) | Wins | Wilcoxon W (p) | Sign test p | Paired-t on log (p) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in out["per_size_class"] + [combined]:
        ci = b["ratio_ci_95"]
        lines.append(
            f"| {b['label']} | {b['n_domains']} | "
            f"{b['mean_nll_MLP']:.2f} | {b['mean_nll_PhaseFlow_t4']:.2f} | "
            f"**{b['ratio_of_means']:.2f}×** ({ci[0]:.2f}–{ci[1]:.2f}) | "
            f"{b['wins_PhaseFlow_t4']}/{b['n_domains']} | "
            f"{b['wilcoxon_W']:.0f} (p={b['wilcoxon_p']:.2e}) | "
            f"p={b['sign_test_p']:.2e} | "
            f"t={b['paired_t_log_stat']:.2f} (p={b['paired_t_log_p']:.2e}) |"
        )
    lines += [
        "",
        "**Interpretation.**  Wilcoxon two-sided signed-rank tests the median paired NLL difference.",
        "Sign test counts how often AD beats MLP without using magnitudes.",
        "Paired-t on log NLLs gives a parametric significance check robust to scale.",
        "Bootstrap (10000 resamples) gives the 95% CI on ratio-of-means.",
        "Geometric-mean per-domain ratio is more robust to outliers than ratio-of-means.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()

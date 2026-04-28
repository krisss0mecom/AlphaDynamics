"""Aligned audit statistics v2 — integrates AR(1) baseline + Wilcoxon tests
across MLP / AR(1) / PhaseFlow_t4.

Reads:
  results/mdcath_aligned20_4000step_cpu.json     (N=48, 20 domains, MLP/PF_t1/PF_t4 NLLs)
  results/mdcath_aligned20_n100_4000step_gpu.json (N=98, 20 domains)
  results/ar1_baseline_aligned40.json            (AR1 NLL N=48)
  results/ar1_baseline_aligned40_n98.json        (AR1 NLL N=98, optional)

Writes:
  results/audit_statistics_v2.json
  results/audit_statistics_v2.md  — paper-ready Table 1
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def safe_load(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def attach_ar1(domains, ar1_data):
    if ar1_data is None:
        return
    lookup = {r["domain_id"]: r["ar1_val_nll"] for r in ar1_data["results"]}
    for d in domains:
        if d["domain_id"] in lookup:
            d["models"]["AR1"] = {"nll": lookup[d["domain_id"]],
                                  "params": 2 * 2 * d["N"]}  # 2 params * 2 * N residues


def stats_for_pair(name_a, vals_a, name_b, vals_b):
    a = np.array(vals_a)
    b = np.array(vals_b)
    n = len(a)
    diff = a - b
    wins_b = int((b < a).sum())
    try:
        w = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    sign_p = 2 * stats.binom.cdf(min(wins_b, n - wins_b), n, 0.5) if n else float("nan")
    rng = np.random.default_rng(42)
    n_boot = 10000
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ratios[i] = a[idx].mean() / b[idx].mean()
    return {
        "compare": f"{name_a} vs {name_b}",
        "n": n,
        f"mean_{name_a}": float(a.mean()),
        f"mean_{name_b}": float(b.mean()),
        "ratio_of_means_a_over_b": float(a.mean() / b.mean()),
        "ratio_ci_95": [float(np.percentile(ratios, 2.5)),
                         float(np.percentile(ratios, 97.5))],
        "geometric_mean_per_domain_ratio": float(np.exp((np.log(a) - np.log(b)).mean())),
        f"wins_{name_b}_lower_nll": wins_b,
        "wilcoxon_W": w_stat,
        "wilcoxon_p": w_p,
        "sign_test_p": float(sign_p),
    }


def block(label, domains):
    out = {"label": label, "n_domains": len(domains)}
    if not domains:
        return out
    mlp = [d["models"]["MLP"]["nll"] for d in domains]
    pf = [d["models"]["PhaseFlow_t4"]["nll"] for d in domains]
    has_ar1 = all("AR1" in d["models"] for d in domains)
    out["mlp_vs_phaseflow"] = stats_for_pair("MLP", mlp, "PhaseFlow_t4", pf)
    if has_ar1:
        ar1 = [d["models"]["AR1"]["nll"] for d in domains]
        out["mlp_vs_ar1"] = stats_for_pair("MLP", mlp, "AR1", ar1)
        out["ar1_vs_phaseflow"] = stats_for_pair("AR1", ar1, "PhaseFlow_t4", pf)
    return out


def md_row(b, key):
    if key not in b:
        return None
    s = b[key]
    a_name, b_name = s["compare"].split(" vs ")
    ci = s["ratio_ci_95"]
    return (
        f"| {b['label']} | {a_name} vs {b_name} | {s['n']} | "
        f"{s[f'mean_{a_name}']:.2f} | {s[f'mean_{b_name}']:.2f} | "
        f"{s['ratio_of_means_a_over_b']:.2f}× ({ci[0]:.2f}–{ci[1]:.2f}) | "
        f"{s[f'wins_{b_name}_lower_nll']}/{s['n']} | "
        f"{s['wilcoxon_W']:.0f} (p={s['wilcoxon_p']:.2e}) |"
    )


def main():
    n48 = safe_load(RESULTS / "mdcath_aligned20_4000step_cpu.json")
    n98 = safe_load(RESULTS / "mdcath_aligned20_n100_4000step_gpu.json")
    ar1_n48 = safe_load(RESULTS / "ar1_baseline_aligned40.json")
    ar1_n98 = safe_load(RESULTS / "ar1_baseline_aligned40_n98.json")
    if not (n48 and n98):
        print("Missing main audit JSONs"); return
    attach_ar1(n48, ar1_n48)
    attach_ar1(n98, ar1_n98)
    combined = n48 + n98

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "blocks": [
            block("N=48 aligned (20 domains)", n48),
            block("N=98 aligned (20 domains)", n98),
            block("Combined aligned (40 domains)", combined),
        ],
    }
    (RESULTS / "audit_statistics_v2.json").write_text(json.dumps(out, indent=2))

    md = [
        "# Aligned audit — v2 statistics with AR(1) baseline",
        "",
        "Pairwise NLL comparisons across MLP (absolute), AR(1) (per-residue trivial),",
        "and PhaseFlow_t4 (AlphaDynamics).  All numbers per-domain val NLL summed",
        "over 2N torsions, averaged over domains (or via paired test where indicated).",
        "",
        "| Cohort | Comparison | n | Mean A | Mean B | Ratio A/B (95% CI) | Wins B | Wilcoxon W (p) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in out["blocks"]:
        for key in ("mlp_vs_phaseflow", "mlp_vs_ar1", "ar1_vs_phaseflow"):
            row = md_row(b, key)
            if row: md.append(row)
    md += [
        "",
        "**Reading.** *Wins B* = how often the second model has lower NLL than the first.",
        "*Ratio A/B = mean_A / mean_B*; >1 means A has higher (worse) NLL.  Wilcoxon",
        "two-sided signed-rank tests paired NLL differences.  CIs are 10000-bootstrap.",
        "",
        "**Interpretation.** AR(1) is a strong per-step baseline because consecutive",
        "MD frames are highly correlated; on the *one-step NLL* metric AR(1) sometimes",
        "approaches or beats PhaseFlow.  The motivating headline of v1 (PhaseFlow vs",
        "absolute MLP) remains intact and statistically significant at p < 1e-12 across",
        "40 domains, but the v2 narrative shifts the load-bearing claim to *long-rollout",
        "fidelity* (§4.4 + Table 3 below) where AR(1) decoheres while PhaseFlow remains",
        "stable.  See `results/jsd_reference_scale.json` for AR(1) and MLP rollout JSDs",
        "alongside the AlphaDynamics rollout JSD on the audit subset.",
    ]
    (RESULTS / "audit_statistics_v2.md").write_text("\n".join(md) + "\n")
    print("[saved] audit_statistics_v2.{json,md}")


if __name__ == "__main__":
    main()

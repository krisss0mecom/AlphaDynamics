# Aligned audit — statistical tests (paper-ready)

Computed from `results/mdcath_aligned20_4000step_cpu.json` and
`results/mdcath_aligned20_n100_4000step_gpu.json`. Pairs (MLP, PhaseFlow_t4) per domain.

| Size class | n | Mean MLP | Mean AD | Ratio (95% CI) | Wins | Wilcoxon W (p) | Sign test p | Paired-t on log (p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N=48 aligned (20 domains) | 20 | 871.81 | 113.85 | **7.66×** (6.13–9.84) | 20/20 | 0 (p=1.91e-06) | p=1.91e-06 | t=20.00 (p=3.17e-14) |
| N=98 aligned (20 domains) | 20 | 519.49 | 102.17 | **5.08×** (4.56–5.87) | 20/20 | 0 (p=1.91e-06) | p=1.91e-06 | t=25.36 (p=4.08e-16) |
| Combined aligned audit (40 domains) | 40 | 695.65 | 108.01 | **6.44×** (5.45–7.75) | 40/40 | 0 (p=1.82e-12) | p=1.82e-12 | t=28.72 (p=7.58e-28) |

**Interpretation.**  Wilcoxon two-sided signed-rank tests the median paired NLL difference.
Sign test counts how often AD beats MLP without using magnitudes.
Paired-t on log NLLs gives a parametric significance check robust to scale.
Bootstrap (10000 resamples) gives the 95% CI on ratio-of-means.
Geometric-mean per-domain ratio is more robust to outliers than ratio-of-means.

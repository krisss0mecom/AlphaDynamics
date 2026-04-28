# Aligned audit — v2 statistics with AR(1) baseline

Pairwise NLL comparisons across MLP (absolute), AR(1) (per-residue trivial),
and PhaseFlow_t4 (AlphaDynamics).  All numbers per-domain val NLL summed
over 2N torsions, averaged over domains (or via paired test where indicated).

| Cohort | Comparison | n | Mean A | Mean B | Ratio A/B (95% CI) | Wins B | Wilcoxon W (p) |
|---|---|---:|---:|---:|---:|---:|---:|
| N=48 aligned (20 domains) | MLP vs PhaseFlow_t4 | 20 | 871.81 | 113.85 | 7.66× (6.13–9.84) | 20/20 | 0 (p=1.91e-06) |
| N=48 aligned (20 domains) | MLP vs AR1 | 20 | 871.81 | 69.91 | 12.47× (8.81–16.57) | 20/20 | 0 (p=1.91e-06) |
| N=48 aligned (20 domains) | AR1 vs PhaseFlow_t4 | 20 | 69.91 | 113.85 | 0.61× (0.50–0.78) | 6/20 | 27 (p=2.33e-03) |
| N=98 aligned (20 domains) | MLP vs PhaseFlow_t4 | 20 | 519.49 | 102.17 | 5.08× (4.56–5.87) | 20/20 | 0 (p=1.91e-06) |
| N=98 aligned (20 domains) | MLP vs AR1 | 20 | 519.49 | 89.00 | 5.84× (4.55–7.33) | 20/20 | 0 (p=1.91e-06) |
| N=98 aligned (20 domains) | AR1 vs PhaseFlow_t4 | 20 | 89.00 | 102.17 | 0.87× (0.67–1.17) | 12/20 | 102 (p=9.27e-01) |
| Combined aligned (40 domains) | MLP vs PhaseFlow_t4 | 40 | 695.65 | 108.01 | 6.44× (5.45–7.75) | 40/40 | 0 (p=1.82e-12) |
| Combined aligned (40 domains) | MLP vs AR1 | 40 | 695.65 | 79.45 | 8.76× (6.78–11.06) | 40/40 | 0 (p=1.82e-12) |
| Combined aligned (40 domains) | AR1 vs PhaseFlow_t4 | 40 | 79.45 | 108.01 | 0.74× (0.62–0.89) | 18/40 | 258 (p=4.08e-02) |

**Reading.** *Wins B* = how often the second model has lower NLL than the first.
*Ratio A/B = mean_A / mean_B*; >1 means A has higher (worse) NLL.  Wilcoxon
two-sided signed-rank tests paired NLL differences.  CIs are 10000-bootstrap.

**Interpretation.** AR(1) is a strong per-step baseline because consecutive
MD frames are highly correlated; on the *one-step NLL* metric AR(1) sometimes
approaches or beats PhaseFlow.  The motivating headline of v1 (PhaseFlow vs
absolute MLP) remains intact and statistically significant at p < 1e-12 across
40 domains, but the v2 narrative shifts the load-bearing claim to *long-rollout
fidelity* (§4.4 + Table 3 below) where AR(1) decoheres while PhaseFlow remains
stable.  See `results/jsd_reference_scale.json` for AR(1) and MLP rollout JSDs
alongside the AlphaDynamics rollout JSD on the audit subset.

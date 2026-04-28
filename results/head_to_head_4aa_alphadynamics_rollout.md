# Ramachandran free energy — AlphaDynamics aligned rollout vs ground truth

Data directory: `/workspace/AlphaDynamics/timewarp_real_data/4AA-large_test3`
Domains: AAAY, AACE, AAEW
Training: 4000 steps, batch 256, device cuda
Rollout: 2500 steps, κ×30

Metrics:
- **JSD**: Jensen-Shannon divergence between P(φ,ψ)_model and P_gt (nats). Range [0, ln2≈0.693]. Lower=better.
- **EMD**: avg marginal Wasserstein distance (°). Measures spatial displacement of density. Lower=better.
- **|ΔG_basin|**: avg error on G at GT-basin centers (kcal/mol). <1 = within thermal kT.
- **Pop err**: avg |P_basin_model - P_basin_gt| over top-3 basins. <0.1 = well-calibrated populations.

| Domain | N | JSD mean | JSD max | EMD mean (°) | \|ΔG_basin\| (kcal) | Pop err |
|---|---|---|---|---|---|---|
| AAAY | 2 | 0.085 | 0.101 | 20.3 | 0.111 | 0.084 |
| AACE | 2 | 0.067 | 0.074 | 19.6 | 0.355 | 0.107 |
| AAEW | 2 | 0.134 | 0.179 | 30.7 | 0.783 | 0.195 |

## Interpretation guidelines

| Metric | Excellent | Good | Poor |
|---|---|---|---|
| JSD | <0.1 | 0.1–0.3 | >0.4 |
| EMD (°) | <10 | 10–30 | >50 |
| \|ΔG_basin\| (kcal) | <0.5 | 0.5–1.5 | >2.0 |
| Pop err | <0.05 | 0.05–0.15 | >0.25 |

Figures: `paper/figures/head_to_head_4aa_alphadynamics_rollout_{domain}.png`
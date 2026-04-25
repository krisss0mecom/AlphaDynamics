# Ramachandran free energy — AlphaDynamics aligned rollout vs ground truth

Data directory: `mdcath_real_data/mdcath_alltemps`
Domains: 4ktyB04, 1w36F02, 2hoxA01
Training: 4000 steps, batch 128, device cuda
Rollout: 2500 steps, κ×30

Metrics:
- **JSD**: Jensen-Shannon divergence between P(φ,ψ)_model and P_gt (nats). Range [0, ln2≈0.693]. Lower=better.
- **EMD**: avg marginal Wasserstein distance (°). Measures spatial displacement of density. Lower=better.
- **|ΔG_basin|**: avg error on G at GT-basin centers (kcal/mol). <1 = within thermal kT.
- **Pop err**: avg |P_basin_model - P_basin_gt| over top-3 basins. <0.1 = well-calibrated populations.

| Domain | N | JSD mean | JSD max | EMD mean (°) | \|ΔG_basin\| (kcal) | Pop err |
|---|---|---|---|---|---|---|
| 4ktyB04 | 98 | 0.127 | 0.352 | 12.3 | 0.801 | 0.059 |
| 1w36F02 | 98 | 0.122 | 0.501 | 11.3 | 1.222 | 0.065 |
| 2hoxA01 | 98 | 0.266 | 0.493 | 30.1 | 2.186 | 0.151 |

## Interpretation guidelines

| Metric | Excellent | Good | Poor |
|---|---|---|---|
| JSD | <0.1 | 0.1–0.3 | >0.4 |
| EMD (°) | <10 | 10–30 | >50 |
| \|ΔG_basin\| (kcal) | <0.5 | 0.5–1.5 | >2.0 |
| Pop err | <0.05 | 0.05–0.15 | >0.25 |

Figures: `paper/figures/ramachandran_aligned3_n98_4000step_gpu_{domain}.png`
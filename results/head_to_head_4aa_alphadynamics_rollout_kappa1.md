# Ramachandran free energy — AlphaDynamics aligned rollout vs ground truth

Data directory: `/workspace/AlphaDynamics/timewarp_real_data/4AA-large_test3`
Domains: AAAY, AACE, AAEW
Training: 4000 steps, batch 256, device cuda
Rollout: 2500 steps, κ×1

Metrics:
- **JSD**: Jensen-Shannon divergence between P(φ,ψ)_model and P_gt (nats). Range [0, ln2≈0.693]. Lower=better.
- **EMD**: avg marginal Wasserstein distance (°). Measures spatial displacement of density. Lower=better.
- **|ΔG_basin|**: avg error on G at GT-basin centers (kcal/mol). <1 = within thermal kT.
- **Pop err**: avg |P_basin_model - P_basin_gt| over top-3 basins. <0.1 = well-calibrated populations.

| Domain | N | JSD mean | JSD max | EMD mean (°) | \|ΔG_basin\| (kcal) | Pop err |
|---|---|---|---|---|---|---|
| AAAY | 2 | 0.014 | 0.020 | 6.7 | 0.085 | 0.038 |
| AACE | 2 | 0.016 | 0.017 | 7.0 | 0.090 | 0.036 |
| AAEW | 2 | 0.013 | 0.014 | 6.8 | 0.151 | 0.041 |

## Interpretation guidelines

| Metric | Excellent | Good | Poor |
|---|---|---|---|
| JSD | <0.1 | 0.1–0.3 | >0.4 |
| EMD (°) | <10 | 10–30 | >50 |
| \|ΔG_basin\| (kcal) | <0.5 | 0.5–1.5 | >2.0 |
| Pop err | <0.05 | 0.05–0.15 | >0.25 |

Figures: `paper/figures/head_to_head_4aa_alphadynamics_rollout_kappa1_{domain}.png`
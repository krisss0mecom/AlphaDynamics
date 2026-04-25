# Ramachandran free energy — AlphaDynamics aligned rollout vs ground truth

Data directory: `mdcath_real_data/mdcath_alltemps`
Domains: 1lwjA03, 1kwgA03, 1vq8L01
Training: 4000 steps, batch 512, device cuda
Rollout: 2500 steps, κ×30

Metrics:
- **JSD**: Jensen-Shannon divergence between P(φ,ψ)_model and P_gt (nats). Range [0, ln2≈0.693]. Lower=better.
- **EMD**: avg marginal Wasserstein distance (°). Measures spatial displacement of density. Lower=better.
- **|ΔG_basin|**: avg error on G at GT-basin centers (kcal/mol). <1 = within thermal kT.
- **Pop err**: avg |P_basin_model - P_basin_gt| over top-3 basins. <0.1 = well-calibrated populations.

| Domain | N | JSD mean | JSD max | EMD mean (°) | \|ΔG_basin\| (kcal) | Pop err |
|---|---|---|---|---|---|---|
| 1lwjA03 | 48 | 0.143 | 0.278 | 11.9 | 0.956 | 0.067 |
| 1kwgA03 | 48 | 0.138 | 0.362 | 13.9 | 1.136 | 0.070 |
| 1vq8L01 | 48 | 0.300 | 0.451 | 35.9 | 1.977 | 0.141 |

## Interpretation guidelines

| Metric | Excellent | Good | Poor |
|---|---|---|---|
| JSD | <0.1 | 0.1–0.3 | >0.4 |
| EMD (°) | <10 | 10–30 | >50 |
| \|ΔG_basin\| (kcal) | <0.5 | 0.5–1.5 | >2.0 |
| Pop err | <0.05 | 0.05–0.15 | >0.25 |

Figures: `paper/figures/ramachandran_aligned3_4000step_gpu_{domain}.png`

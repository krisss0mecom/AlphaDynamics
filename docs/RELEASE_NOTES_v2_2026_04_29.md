# AlphaDynamics v2 release notes (2026-04-29)

This release addresses every reviewer-attack vector identified in
`REVIEWER_ATTACK_ANALYSIS_2026_04_28.md` (26 attack vectors) plus the
`REVIEWER_RISK_REGISTER_2026_04_28.md` items, and adds two new
load-bearing experiments: an anchored JSD reference scale and a
shared-dataset head-to-head with the Microsoft Timewarp 4AA pretrained
model.

## What's new

### New evidence
- **Statistical tests** on the v1 audit: paired Wilcoxon, sign test,
  bootstrap CI on ratio-of-means, paired-t on log NLLs.
  $p < 1\!\times\!10^{-12}$ on the combined 40-domain MLP vs
  AlphaDynamics comparison.
- **AR(1) baseline** on all 40 aligned audit domains. AR(1) is
  competitive with AlphaDynamics on one-step NLL on small systems
  (N=48: 14/20 wins for AR(1), $p=2.3\!\times\!10^{-3}$) but
  AlphaDynamics catches up on N=98 (no significant difference,
  $p=0.93$), exposing one-step NLL as a dataset-autocorrelation-
  dominated metric not suitable as the load-bearing claim.
- **Anchored rollout JSD reference scale** on 3 N=48 audit domains:
  AR(1) propagator decoheres to within 0.010 of uniform pessimal bound
  on ordered domains, while AlphaDynamics retains 70% of the entropy
  gap to the noise floor. Anchored against split-trajectory replica
  floor, cross-temperature physical floor, and uniform pessimal bound.
- **Head-to-head with Microsoft Timewarp 4AA** on 3 shared
  tetrapeptides (AAAY/AACE/AAEW from `microsoft/timewarp` HF dataset
  4AA-large/test split). Under a single canonical Ramachandran JSD
  evaluator applied identically to both models (`src/jsd_unified_eval.py`:
  held-out val GT, 36 bins, no smoothing) AlphaDynamics with calibrated
  $\kappa\!\times\!1$ inference is 3/3 wins, mean JSD **0.165 vs 0.468,
  2.84× closer** to held-out density than the 396M-parameter transferable
  Cartesian model. Earlier within-this-release headline "25× closer"
  (commit fb355be) was wrong: AD JSD used a smoothed train+val GT
  while Timewarp JSD used raw val GT, so the two numbers were not
  comparable. The unified-evaluator results above supersede them.
- **K-sweep ablation** on K ∈ {2,4,8,16,32} on 3 representative
  domains; performance approximately stable in K=4–16 regime,
  confirming K=8 is not over-tuned.
- **Kappa calibration sweep** replacing the v1 heuristic
  $\kappa\!\times\!30$. Per-domain optimum is $\kappa\!\times\!1$ (no
  rescaling) on ordered domains, with monotonic JSD growth in kappa.
- **Hyperparameter Table 2** consolidates all architecture and
  training hyperparameters for reproducibility.
- **Reviewer attack analysis document** (`docs/REVIEWER_ATTACK_ANALYSIS_2026_04_28.md`):
  26 attack vectors with severity (KILL/MAJOR/MINOR) and remediation
  pointers — addressed in this release where flagged KILL or MAJOR.

### Reframed claims
- The headline contribution shifts from "one-step NLL beats MLP" to
  "rollout fidelity beats trivial baselines and a transferable model on
  shared data". The v1 result (40/40 wins vs MLP) remains intact and
  statistically significant; it is no longer the load-bearing claim.
- The "extreme parameter efficiency" claim against transferable models
  is removed; we no longer compare 348K-AD parameter count to
  396M-Timewarp parameter count as a contribution. The remaining
  per-system claim ("trains in <10 min on a single GPU, 16 ms per
  inference step") is preserved.
- "Scaling behaviour" framing on Figure 3 reframed to "robustness
  across size classes", since the ratio-of-means is non-monotonic in N.
- Domain selection rule and mdCATH replica policy explicitly stated
  (§3.5).
- RK4 step count made explicit ($\Delta t_\text{RK}=0.5$, 8 steps).

## File manifest

### Paper
- `paper/main.md` — full v2 manuscript (rewritten)
- `paper/main.pdf` — compiled
- `paper/main_v1_backup_2026_04_25.md` — preserved v1 source

### Documentation
- `docs/REVIEWER_ATTACK_ANALYSIS_2026_04_28.md` — 26 attack vectors
- `docs/REVIEWER_RISK_REGISTER_2026_04_28.md` — pre-existing risk register
- `docs/PRODUCT_V1_2026_04_28.md` — pre-existing product plan
- `docs/RELEASE_NOTES_v2_2026_04_29.md` — this file

### Source (new)
- `src/audit_statistics.py` — Wilcoxon + bootstrap on v1 results
- `src/audit_statistics_v2.py` — v2 stats including AR(1)
- `src/ar1_baseline.py` — AR(1) circular baseline
- `src/jsd_reference_scale.py` — anchored rollout JSD audit
- `src/head_to_head_timewarp.py` — Microsoft Timewarp 4AA rollout
- `src/k_sweep.py` — K mixture-component ablation
- `src/kappa_sweep.py` — kappa calibration sweep

### Results (new)
- `results/audit_statistics.json,md`
- `results/audit_statistics_v2.json,md`
- `results/ar1_baseline_aligned40.json` (N=48)
- `results/ar1_baseline_aligned40_n98.json` (N=98)
- `results/jsd_reference_scale.json` (3 N=48 audit domains)
- `results/head_to_head_4aa_alphadynamics_rollout_kappa1.json,md` (3 tetrapeptides, calibrated v2)
- `results/head_to_head_4aa_alphadynamics_rollout.json,md` (3 tetrapeptides, preserved v1-style κ×30)
- `results/timewarp_rollout_4aa.json` (3 tetrapeptides, 396M params)
- `results/k_sweep_ablation.json`
- `results/kappa_sweep_aligned3.json`
- `results/alphadynamics_audit_report.md` — fully rewritten v2 audit report

### Figures (new)
- `paper/figures/head_to_head_4aa_alphadynamics_rollout_AAAY.png`
- `paper/figures/head_to_head_4aa_alphadynamics_rollout_AACE.png`
- `paper/figures/head_to_head_4aa_alphadynamics_rollout_AAEW.png`
- `paper/figures/head_to_head_4aa_alphadynamics_rollout_kappa1_AAAY.png`
- `paper/figures/head_to_head_4aa_alphadynamics_rollout_kappa1_AACE.png`
- `paper/figures/head_to_head_4aa_alphadynamics_rollout_kappa1_AAEW.png`

## Headline numbers at a glance

| Metric | v1 (2026-04-25) | v2 (2026-04-29) |
|---|---|---|
| Aligned audit wins | 40/40 vs MLP | 40/40 vs MLP, $p<10^{-12}$ |
| Bootstrap 95% CI | not reported | 5.45–7.75× |
| AR(1) baseline | not present | 14/20 wins on N=48 (one-step) |
| Rollout fidelity | mean JSD 0.194 (anchorless) | gap-closure $\rho$=0.70 vs MLP=0.19 vs AR(1)=-0.02 |
| Shared-dataset head-to-head | data-side only (3/3 PF wins on NLL) | full model head-to-head: 3/3 wins, 2.84× closer JSD under unified canonical metric (κ×1) |
| Statistical test | not reported | Wilcoxon p < 1e-12 |
| Hyperparameter table | scattered | consolidated Table 2 |
| Kappa calibration | hardcoded ×30 | sweep over {1,5,10,20,30,50,100}, optimum at ×1 |
| K (mixture) ablation | none | K∈{2,4,8,16,32}, K=8 confirmed not over-tuned |

## Known remaining gaps

1. **Bivariate von Mises head.** Mixture-of-axis-independent vM cannot
   represent intra-component (φ_i,ψ_i) correlations. Listed as future
   work in §5; K-sweep shows current K=8 is sufficient at peptide
   scale but a Singh-2002-style bivariate head may help on disordered
   larger domains.
2. **t_max sweep on full audit.** Only the original 4-domain pilot is
   reported; cross-domain stability of t_max=4 confirmed on 3
   independent rollout-audit domains in §4.4 but not on all 40.
3. **Kinetic observables.** Residence times, mean first passage time,
   and transition counts not yet reported; only equilibrium
   Ramachandran density.
4. **Cross-temperature audit.** Partial; auxiliary table only.
5. **Larger N.** Audit caps at N=98; mdCATH provides domains up to
   N≈300. Future work.

## How to reproduce

```bash
# 1. Reproduce v2 statistics from v1 result JSONs (no GPU needed):
python src/audit_statistics_v2.py

# 2. Run AR(1) baseline (any GPU):
python src/ar1_baseline.py \
    --data_dir mdcath_real_data/mdcath_348K \
    --out results/ar1_baseline_aligned40.json \
    --steps 4000 --device cuda

# 3. Anchored JSD reference scale (RTX 5090 ~30 min for 3 domains):
python src/jsd_reference_scale.py \
    --data_dir mdcath_real_data/mdcath_alltemps \
    --domains 1lwjA03 1kwgA03 1vq8L01 \
    --rollout_steps 2500 --device cuda \
    --out results/jsd_reference_scale.json

# 4. Microsoft Timewarp head-to-head (RTX 5090 ~25 min):
#    Requires microsoft/timewarp 4aa_best_model.pt and the 3 tetrapeptides
#    from 4AA-large/test split.
python src/head_to_head_timewarp.py \
    --ckpt /path/to/4aa_best_model.pt \
    --config /path/to/4aa_config.yaml \
    --data_root /path/to/4AA-large/test \
    --peptides AAAY AACE AAEW \
    --n_rollout 2500

# 5. AlphaDynamics on the same 3 tetrapeptides:
python src/ramachandran_energy_v2.py \
    --data_dir timewarp_real_data/4AA-large_test3 \
    --out_prefix head_to_head_4aa_alphadynamics_rollout_kappa1 \
    --temp 348 --steps 4000 --batch 256 \
    --rollout_steps 2500 --kappa_mult 1 --device cuda \
    --domains AAAY AACE AAEW

# 6. K and kappa sweeps:
python src/k_sweep.py --data_dir ... --domains 1lwjA03 1kwgA03 1vq8L01 ...
python src/kappa_sweep.py --data_dir ... --domains 1lwjA03 1kwgA03 1vq8L01 ...
```

Authoritative numbers are in `results/*.json` and the corresponding
`.md` files; the paper points at these by exact filename in §6.

# AlphaDynamics Reviewer Risk Register

Date: 2026-04-28

This file is intentionally blunt. It separates product readiness from manuscript
readiness and lists the experiments needed to harden AlphaDynamics against
reasonable reviewer objections.

## Current State

AlphaDynamics is a usable research product prototype: it has a CLI, data
validation, reproducible reports, and aligned mdCATH audit artifacts. It is not
yet a reviewer-proof v2 paper package.

## Reviewer Risks

| Objection | Current evidence | Remaining gap | Product action |
|---|---|---|---|
| MLP baseline is weak | 40/40 wins vs matched absolute MLP; 9/9 wins vs residual MLP on 3 domains x 3 seeds | residual MLP is still pointwise and not a true temporal model | run `alphadynamics temporal-baseline` with an 8-frame GRU context |
| Single seed | v1 full audit uses seed 42; 3-domain subset now has 3 seeds | not yet 3 seeds on all 40 domains | run seed sweep on representative subset first, then all 40 if stable |
| `kappa x30` rollout heuristic | six rollout audits are stable and reported honestly | calibration is not learned or swept | run `alphadynamics kappa-sweep` across fixed multipliers and report global/per-domain optimum |
| No head-to-head with Timewarp/bioEmu | paper explicitly scopes itself as per-system, not zero-shot | no shared-task external baseline | run `alphadynamics timewarp-comparison` on Timewarp tetrapeptide data; keep bioEmu/AlphaFlow as equilibrium-only comparisons |
| Rollout audit has six domains | 3 N=48 + 3 N=98 rollout audits | too small for broad claims | expand rollout audit after kappa sweep, prioritizing ordered and high-entropy domains |
| Missing ablations | architecture is described but not decomposed | no clean contribution breakdown | add ablations: no ODE, no coupling, no phase gate, MLP head-only, different `t_max` |
| CNOT framing may distract | prior phase-gate work motivates the coupling | reviewers may see it as oversold | keep CNOT as inspiration, but make the paper's main claim empirical and torsion-native |
| No biological observable | NLL and Ramachandran metrics are reported | no kinetics, residence times, or transition observables | add autocorrelation, basin transition counts, and residence-time summaries |

## Immediate Hardening Commands

Temporal GRU baseline on the same 3-domain stress subset:

```bash
alphadynamics temporal-baseline \
  --data-dir mdcath_real_data/mdcath_348K \
  --out-prefix temporal_gru_3dom_3seed_4000step_cuda \
  --domains 1lwjA03 1kwgA03 1vq8L01 \
  --window 8 \
  --steps 4000 \
  --batch 128 \
  --seeds 42 43 44 \
  --phaseflow-tmax 4 \
  --device auto
```

Kappa calibration on the rollout subset:

```bash
alphadynamics kappa-sweep \
  --data-dir mdcath_real_data/mdcath_alltemps \
  --out-prefix kappa_sweep_n48 \
  --domains 1lwjA03 1kwgA03 1vq8L01 \
  --kappa-mult 1 5 10 20 30 50 \
  --device auto
```

Timewarp shared-dataset audit scaffold:

```bash
alphadynamics timewarp-comparison convert \
  --dataset 4AA-large \
  --split test \
  --max-domains 3 \
  --max-frames 2500 \
  --out-dir timewarp_real_data/4AA-large_test

alphadynamics train \
  --data-dir timewarp_real_data/4AA-large_test \
  --out-prefix timewarp_4aa_large_test3_nll \
  --steps 4000 \
  --batch 512 \
  --device auto
```

## Manuscript Language Rule

Do not call the project "ready for reviewers" until at least one of these is
done:

- temporal GRU audit shows AlphaDynamics still wins on the 3-domain stress set
- kappa sweep replaces the single `kappa x30` heuristic with a calibration table
- the paper removes or explicitly fixes any dangling section references

Until then, the correct product label is:

> auditable external-alpha research product, not reviewer-hardened v2 manuscript.

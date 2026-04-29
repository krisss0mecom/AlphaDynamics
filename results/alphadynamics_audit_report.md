# AlphaDynamics audit report (v2 — 2026-04-29)

Generated from `results/*.json`.  See `paper/main.md` for the full v2
manuscript with reviewer-hardening fixes.

## One-step NLL (40-domain aligned audit)

| Run | Domains | N | AD wins vs MLP | Mean MLP NLL | Mean AD NLL | Ratio | Wilcoxon $p$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mdcath_aligned20_4000step_cpu` | 20 | 48 | 20/20 | 871.81 | 113.85 | **7.66×** | $1.9\!\times\!10^{-6}$ |
| `mdcath_aligned20_n100_4000step_gpu` | 20 | 98 | 20/20 | 519.49 | 102.17 | **5.08×** | $1.9\!\times\!10^{-6}$ |
| **Combined (40)** | **40** | — | **40/40** | 695.65 | 108.01 | **6.44×** | $1.8\!\times\!10^{-12}$ |

95% bootstrap CI on combined ratio-of-means: 5.45–7.75×. Geometric mean
per-domain ratio: 6.37×. Detailed in `results/audit_statistics_v2.json`.

## AR(1) baseline (one-step NLL)

| Cohort | n | Mean AR(1) | Mean AD | Ratio AR(1)/AD | AD wins | Wilcoxon $p$ |
|---|---:|---:|---:|---:|---:|---:|
| N=48 aligned | 20 | 69.91 | 113.85 | 0.61× | 6/20 | $2.3\!\times\!10^{-3}$ |
| N=98 aligned | 20 | 89.00 | 102.17 | 0.87× | 12/20 | n.s. (p=0.93) |
| Combined (40) | 40 | 79.45 | 108.01 | 0.74× | 18/40 | $4.1\!\times\!10^{-2}$ |

AR(1) (192 params per domain) is competitive on small systems, AlphaDynamics
catches up on N=98 (no significant difference). One-step NLL is dominated
by frame autocorrelation and is not the load-bearing metric — see Rollout.

## Model size

| Run | MLP params | AR(1) params | AlphaDynamics params |
|---|---:|---:|---:|
| `mdcath_aligned20_4000step_cpu` | 389,000 | 192 | 341,705 |
| `mdcath_aligned20_n100_4000step_gpu` | 724,200 | 392 | 657,705 |

## Rollout Free Energy (load-bearing claim)

| Run | Domains | N | Mean JSD | Mean EMD | Mean \|dG\| | Mean pop err |
|---|---:|---:|---:|---:|---:|---:|
| `ramachandran_aligned3_4000step_gpu` | 3 | 48 | 0.194 | 20.6° | 1.36 kcal/mol | 0.093 |
| `ramachandran_aligned3_n98_4000step_gpu` | 3 | 98 | 0.172 | 17.9° | 1.40 kcal/mol | 0.091 |

## JSD reference scale (anchored, N=48 audit subset)

From `results/jsd_reference_scale.json`. Lower is better. Floor =
split-trajectory JSD (irreducible noise from data). Uniform = pessimal.

| Domain | Floor | AlphaDynamics | MLP rollout | AR(1) rollout | Uniform |
|---|---:|---:|---:|---:|---:|
| 1lwjA03 (ordered) | 0.038 | **0.143** | 0.338 | 0.610 | 0.600 |
| 1kwgA03 (ordered) | 0.031 | **0.138** | 0.341 | 0.612 | 0.606 |
| 1vq8L01 (high-entropy) | 0.113 | **0.300** | 0.649 | 0.519 | 0.503 |
| **Mean (3)** | **0.061** | **0.194** | **0.443** | **0.580** | **0.570** |

Gap-closure ratio $\rho = 1 - (\text{JSD}_\text{model} - \text{floor}) /
(\text{uniform} - \text{floor})$; 1.0 = matches floor, 0.0 = matches uniform:

| Model | Mean $\rho$ |
|---|---:|
| Uniform (pessimal) | 0.00 |
| AR(1) propagator | -0.02 |
| Absolute-MLP propagator | 0.19 |
| **AlphaDynamics** | **0.70** |

## Head-to-head: AlphaDynamics vs Microsoft Timewarp 4AA (shared dataset)

From `results/head_to_head_4aa_alphadynamics_rollout.json` and
`results/timewarp_rollout_4aa.json`. Peptides AAAY/AACE/AAEW from
`microsoft/timewarp` HF dataset, `4AA-large/test` split. AD trained
per-system on 80% train slice; Timewarp pretrained on `4AA-big2`
(transferable, out-of-distribution for these test peptides).

| Peptide | $N_\text{res}$ | AD JSD (κ×1) | Timewarp JSD | TW / AD |
|---|---:|---:|---:|---:|
| AAAY | 2 | **0.014** | 0.460 | 33× |
| AACE | 2 | **0.016** | 0.135 | 8× |
| AAEW | 2 | **0.013** | 0.473 | 36× |
| **Mean (3)** | — | **0.014** | **0.356** | **25×** |

(Calibrated κ×1 rollout. v1-style κ×30 numbers (mean 0.095) preserved
for reference in `results/head_to_head_4aa_alphadynamics_rollout.json`.)

| Model | Parameter count |
|---|---:|
| AlphaDynamics (per-system 4AA) | 348K |
| Microsoft Timewarp 4AA | 396M |

AlphaDynamics 3/3 wins on Ramachandran fidelity vs the 1000× larger
transferable Cartesian propagator.

## Kappa calibration sweep (replaces v1 κ×30 heuristic)

| Domain | κ×1 | κ×5 | κ×10 | κ×20 | κ×30 | κ×50 | κ×100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1lwjA03` (ordered) | **0.106** | 0.195 | 0.280 | 0.355 | 0.391 | 0.425 | 0.458 |
| `1kwgA03` (ordered) | **0.085** | 0.199 | 0.285 | 0.354 | 0.385 | 0.414 | 0.444 |
| `1vq8L01` (high-entropy) | 0.347 | **0.342** | 0.390 | 0.430 | 0.451 | 0.470 | 0.493 |
| **Mean** | **0.179** | 0.245 | 0.318 | 0.380 | 0.409 | 0.436 | 0.465 |

**Calibrated optimum: κ×1 (no rescaling) on ordered domains, κ×5 on
disordered.** v1 heuristic κ×30 is 2.3× worse than κ×1 on average.
Recommend κ×1 as v2 global default.

## Stronger baselines (from v1 release, reproduced for completeness)

Residual MLP (3-domain × 3-seed):
| | n | PF wins vs residual MLP | PF wins vs abs MLP | residual wins vs abs MLP | Mean abs MLP | Mean residual MLP | Mean AD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `strong_baseline_3dom_3seed_4000step_cuda` | 9 | 9/9 | 9/9 | 2/9 | 191.24 | 208.78 | 64.60 |

Temporal GRU 8-frame (3-domain × 3-seed):
| | n | PF wins vs GRU | PF wins vs abs MLP | GRU wins vs abs MLP | Mean abs MLP | Mean temporal GRU | Mean AD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `temporal_gru_3dom_3seed_4000step_cuda` | 9 | 9/9 | 9/9 | 0/9 | 169.79 | 343.64 | 64.03 |

## Product readiness

- CLI surface: `alphadynamics doctor / validate-data / convert / train /
  rollout / strong-baseline / temporal-baseline / kappa-sweep /
  timewarp-comparison / report`.
- Data contract: aligned `.npz` files with
  `dihedral_alignment=common_residue_index`.
- v2 evidence ladder: aligned one-step NLL with statistical tests, JSD
  reference scale anchored vs split-traj floor / AR(1) / MLP / uniform,
  shared-dataset head-to-head with Timewarp.
- Open: kappa-sweep table (running), K-sweep ablation (running),
  bivariate von Mises head, kinetic observables (residence times, MFPT).

## Reproducibility manifest

| Artifact | Path |
|---|---|
| Aligned audit N=48 | `results/mdcath_aligned20_4000step_cpu.json` |
| Aligned audit N=98 | `results/mdcath_aligned20_n100_4000step_gpu.json` |
| Statistics + Wilcoxon | `results/audit_statistics_v2.json,md` |
| AR(1) baseline N=48 | `results/ar1_baseline_aligned40.json` |
| AR(1) baseline N=98 | `results/ar1_baseline_aligned40_n98.json` |
| JSD reference scale | `results/jsd_reference_scale.json` |
| AD rollout aligned3 N=48 | `results/ramachandran_aligned3_4000step_gpu.json` |
| AD rollout aligned3 N=98 | `results/ramachandran_aligned3_n98_4000step_gpu.json` |
| AD rollout 4AA tetrapeptides | `results/head_to_head_4aa_alphadynamics_rollout.json` |
| Timewarp rollout 4AA | `results/timewarp_rollout_4aa.json` |
| Strong baseline | `results/strong_baseline_3dom_3seed_4000step_cuda.json` |
| Temporal GRU | `results/temporal_gru_3dom_3seed_4000step_cuda.json` |

# AlphaDynamics audit report

Generated from existing JSON result files.

## One-step NLL

| Run | Domains | N | Wins | Mean MLP NLL | Mean AD NLL | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `mdcath_aligned20_4000step_cpu` | 20 | 48 | 20/20 | 871.805 | 113.848 | 7.66x |
| `mdcath_aligned20_n100_4000step_gpu` | 20 | 98 | 20/20 | 519.492 | 102.173 | 5.08x |

## Model Size

| Run | MLP params | AlphaDynamics params |
|---|---:|---:|
| `mdcath_aligned20_4000step_cpu` | 389,000 | 341,705 |
| `mdcath_aligned20_n100_4000step_gpu` | 724,200 | 657,705 |

## Rollout Free Energy

| Run | Domains | N | Mean JSD | Mean EMD | Mean abs dG_basin | Mean pop err |
|---|---:|---:|---:|---:|---:|---:|
| `ramachandran_aligned3_4000step_gpu` | 3 | 48 | 0.194 | 20.6 deg | 1.356 kcal/mol | 0.093 |
| `ramachandran_aligned3_n98_4000step_gpu` | 3 | 98 | 0.172 | 17.9 deg | 1.403 kcal/mol | 0.091 |

## Strong Baseline

| Run | Domain-seed runs | PF wins vs residual MLP | PF wins vs absolute MLP | Residual wins vs absolute MLP | Mean abs MLP NLL | Mean residual MLP NLL | Mean AD NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `strong_baseline_n48_pilot_500step_earlystop_cuda` | 3 | 2/3 | 1/3 | 1/3 | 62.414 | 69.600 | 62.673 |

## Product Readiness

- Reproducible command surface: `alphadynamics` / `python src/alphadynamics_cli.py`.
- Data contract: aligned `.npz` files with `dihedral_alignment=common_residue_index`.
- Current evidence: aligned one-step NLL audit, limited rollout audit, and optional residual-baseline audit.
- Current limitation: rollout calibration still uses a sampling-time kappa multiplier.

# AlphaDynamics audit report

Generated from existing JSON result files.

## One-step NLL

| Run | Domains | N | Wins | Mean MLP NLL | Mean AD NLL | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `mdcath_aligned20_4000step_cpu` | 20 | 48 | 20/20 | 871.805 | 113.848 | 7.66x |
| `mdcath_aligned20_n100_4000step_gpu` | 20 | 98 | 20/20 | 519.492 | 102.173 | 5.08x |

## Rollout Free Energy

| Run | Domains | N | Mean JSD | Mean EMD | Mean abs dG_basin | Mean pop err |
|---|---:|---:|---:|---:|---:|---:|
| `ramachandran_aligned3_4000step_gpu` | 3 | 48 | 0.194 | 20.6 deg | 1.356 kcal/mol | 0.093 |
| `ramachandran_aligned3_n98_4000step_gpu` | 3 | 98 | 0.172 | 17.9 deg | 1.403 kcal/mol | 0.091 |

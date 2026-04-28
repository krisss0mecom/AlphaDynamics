# Strong baseline audit

Purpose: test whether AlphaDynamics still wins against a residual/autoregressive MLP baseline.

Data directory: `mdcath_real_data/mdcath_348K`
Steps/model: 500, batch: 64, seeds: [42]
Validation selection: best checkpoint every 100 steps
Device: cuda

| Domain | Seed | N | Identity° | MLP abs NLL | MLP residual NLL | Best PF NLL | Best PF | Winner vs residual | ΔNLL PF-residual |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1lwjA03 | 42 | 48 | 25.0 | 27.907 | 47.185 | 28.193 | PhaseFlow_t1 | **PhaseFlow_t1** | -18.992 |
| 1kwgA03 | 42 | 48 | 25.0 | 19.995 | 44.924 | 19.585 | PhaseFlow_t1 | **PhaseFlow_t1** | -25.339 |
| 1vq8L01 | 42 | 48 | 54.5 | 139.340 | 116.692 | 140.240 | PhaseFlow_t1 | **MLP_residual** | +23.548 |

## Summary
- PhaseFlow wins vs residual MLP: **2/3** domain-seed runs.
- PhaseFlow wins vs absolute MLP: **1/3** domain-seed runs.
- Residual MLP improves over absolute MLP: **1/3** domain-seed runs.
- Mean MLP abs NLL: 62.414
- Mean MLP residual NLL: 69.600
- Mean best PhaseFlow NLL: 62.673
- Mean ΔNLL PhaseFlow-residual: -6.928 nats (negative favors PhaseFlow).
- Mean ΔNLL PhaseFlow-absolute: +0.259 nats (negative favors PhaseFlow).
- Mean ΔNLL residual-absolute: +7.186 nats (negative favors residual MLP).

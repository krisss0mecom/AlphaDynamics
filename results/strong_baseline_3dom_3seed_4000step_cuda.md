# Strong baseline audit

Purpose: test whether AlphaDynamics still wins against a residual/autoregressive MLP baseline.

Data directory: `mdcath_real_data/mdcath_348K`
Steps/model: 4000, batch: 128, seeds: [42, 43, 44]
Validation selection: final checkpoint only
Device: cuda

| Domain | Seed | N | Identity° | MLP abs NLL | MLP residual NLL | Best PF NLL | Best PF | Winner vs residual | ΔNLL PF-residual |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1lwjA03 | 42 | 48 | 25.0 | 129.983 | 146.743 | 33.240 | PhaseFlow_t4 | **PhaseFlow_t4** | -113.503 |
| 1lwjA03 | 43 | 48 | 25.0 | 121.294 | 155.036 | 32.932 | PhaseFlow_t4 | **PhaseFlow_t4** | -122.104 |
| 1lwjA03 | 44 | 48 | 25.0 | 126.159 | 137.044 | 31.783 | PhaseFlow_t4 | **PhaseFlow_t4** | -105.261 |
| 1kwgA03 | 42 | 48 | 25.0 | 156.345 | 142.444 | 21.833 | PhaseFlow_t4 | **PhaseFlow_t4** | -120.611 |
| 1kwgA03 | 43 | 48 | 25.0 | 95.344 | 102.838 | 21.133 | PhaseFlow_t4 | **PhaseFlow_t4** | -81.704 |
| 1kwgA03 | 44 | 48 | 25.0 | 73.291 | 105.228 | 21.086 | PhaseFlow_t4 | **PhaseFlow_t4** | -84.142 |
| 1vq8L01 | 42 | 48 | 54.5 | 346.308 | 351.543 | 139.637 | PhaseFlow_t4 | **PhaseFlow_t4** | -211.905 |
| 1vq8L01 | 43 | 48 | 54.5 | 368.723 | 295.000 | 139.837 | PhaseFlow_t4 | **PhaseFlow_t4** | -155.164 |
| 1vq8L01 | 44 | 48 | 54.5 | 303.723 | 443.159 | 139.944 | PhaseFlow_t4 | **PhaseFlow_t4** | -303.215 |

## Summary
- PhaseFlow wins vs residual MLP: **9/9** domain-seed runs.
- PhaseFlow wins vs absolute MLP: **9/9** domain-seed runs.
- Residual MLP improves over absolute MLP: **2/9** domain-seed runs.
- Mean MLP abs NLL: 191.241
- Mean MLP residual NLL: 208.782
- Mean best PhaseFlow NLL: 64.603
- Mean ΔNLL PhaseFlow-residual: -144.179 nats (negative favors PhaseFlow).
- Mean ΔNLL PhaseFlow-absolute: -126.638 nats (negative favors PhaseFlow).
- Mean ΔNLL residual-absolute: +17.541 nats (negative favors residual MLP).

# Temporal-context baseline audit

Purpose: test AlphaDynamics against a true sequence-context baseline.

Data directory: `mdcath_real_data/mdcath_348K`
Window: 8 frames
Steps/model: 4000, batch: 128, seeds: [42, 43, 44]
Validation selection: final checkpoint only
Device: cuda

| Domain | Seed | N | Identity° | MLP abs NLL | Temporal GRU NLL | Best PF NLL | Best PF | Winner vs GRU | ΔNLL PF-GRU |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1lwjA03 | 42 | 48 | 25.0 | 147.805 | 288.894 | 28.008 | PhaseFlow_t4 | **PhaseFlow_t4** | -260.886 |
| 1lwjA03 | 43 | 48 | 25.0 | 126.988 | 233.124 | 31.349 | PhaseFlow_t4 | **PhaseFlow_t4** | -201.776 |
| 1lwjA03 | 44 | 48 | 25.0 | 154.383 | 332.968 | 32.462 | PhaseFlow_t4 | **PhaseFlow_t4** | -300.506 |
| 1kwgA03 | 42 | 48 | 25.0 | 95.447 | 246.188 | 22.225 | PhaseFlow_t4 | **PhaseFlow_t4** | -223.963 |
| 1kwgA03 | 43 | 48 | 25.0 | 75.001 | 202.182 | 21.270 | PhaseFlow_t4 | **PhaseFlow_t4** | -180.911 |
| 1kwgA03 | 44 | 48 | 25.0 | 91.039 | 220.084 | 21.620 | PhaseFlow_t4 | **PhaseFlow_t4** | -198.464 |
| 1vq8L01 | 42 | 48 | 54.5 | 259.967 | 393.371 | 140.392 | PhaseFlow_t4 | **PhaseFlow_t4** | -252.979 |
| 1vq8L01 | 43 | 48 | 54.5 | 311.903 | 506.023 | 139.563 | PhaseFlow_t4 | **PhaseFlow_t4** | -366.460 |
| 1vq8L01 | 44 | 48 | 54.5 | 265.554 | 669.895 | 139.365 | PhaseFlow_t4 | **PhaseFlow_t4** | -530.530 |

## Summary
- PhaseFlow wins vs temporal GRU: **9/9** domain-seed runs.
- PhaseFlow wins vs absolute MLP: **9/9** domain-seed runs.
- Temporal GRU improves over absolute MLP: **0/9** domain-seed runs.
- Mean MLP abs NLL: 169.787
- Mean temporal GRU NLL: 343.637
- Mean best PhaseFlow NLL: 64.028
- Mean ΔNLL PhaseFlow-GRU: -279.608 nats (negative favors PhaseFlow).
- Mean ΔNLL PhaseFlow-absolute: -105.759 nats (negative favors PhaseFlow).
- Mean ΔNLL GRU-absolute: +173.849 nats (negative favors GRU).

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

## Summary
- PhaseFlow wins vs temporal GRU: **5/5** domain-seed runs.
- PhaseFlow wins vs absolute MLP: **5/5** domain-seed runs.
- Temporal GRU improves over absolute MLP: **0/5** domain-seed runs.
- Mean MLP abs NLL: 119.925
- Mean temporal GRU NLL: 260.671
- Mean best PhaseFlow NLL: 27.063
- Mean ΔNLL PhaseFlow-GRU: -233.608 nats (negative favors PhaseFlow).
- Mean ΔNLL PhaseFlow-absolute: -92.862 nats (negative favors PhaseFlow).
- Mean ΔNLL GRU-absolute: +140.747 nats (negative favors GRU).

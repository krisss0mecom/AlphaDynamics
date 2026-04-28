# mdCATH unified benchmark — aligned 348 K domains

Data directory: `timewarp_real_data/4AA-large_test3`

Training steps per model: 4000, batch: 512

Protocol: CHARMM36m + TIP3P water, 348 K, 5 replicas per domain.
Phi/psi pairs are aligned by common residue index.

| Domain | N | Identity° | MLP NLL | PF_t1 NLL | PF_t4 NLL | Best model | ΔNLL (best PF - MLP) |
|---|---|---|---|---|---|---|---|
| AAAY | 2 | 22.9 | 26.406 | 15.165 | 2.783 | **PF_t4** | -23.623 |
| AACE | 2 | 19.9 | 23.646 | 15.193 | 1.699 | **PF_t4** | -21.947 |
| AAEW | 2 | 22.1 | 22.055 | 20.012 | 2.901 | **PF_t4** | -19.154 |

## Summary
- **AlphaDynamics wins: 3/3 domains**
- Mean ΔNLL: -21.575 nats (negative = PhaseFlow better)
- MLP mean NLL: 24.036
- PhaseFlow best mean NLL: 2.461
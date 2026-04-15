# AlphaDynamics

Neural propagator for molecular dynamics prediction on protein torus T^N.
Input: current protein conformation (dihedral angles φ, ψ per residue).
Output: probability distribution over the next frame's conformation.

**Author:** Krzysztof Gwozdz
**Started:** 2026-04-14

## What it does

AlphaDynamics predicts how a folded protein moves in time — the MD counterpart
to AlphaFold (which predicts static structure from sequence).

- Input: torsion angles (φ, ψ) of all residues at time t
- Output: mixture-of-von-Mises distribution over angles at time t+dt
- Core architecture: phase oscillators coupled via CNOT-style interactions,
  evolved by torchdiffeq RK4 adjoint ODE solver
- Model size: ~350K parameters (6,920× smaller than Microsoft Timewarp)
- Inference speed: ~16 ms per frame on RTX 5090

## Headline results

### mdCATH unified benchmark — 37 protein domains

Uniform protocol (CHARMM36m + TIP3P water at 348K, 5 replicas × 440 frames
per domain) on 37 randomly-selected 50-residue CATH domains from the
mdCATH dataset (Mirarchi et al., Sci Data 2024, 5398 proteins).

| Stat | Value |
|---|---|
| Domains | 37 |
| **AlphaDynamics wins** | **37/37 (100%)** |
| Mean ΔNLL vs MLP | **−472 nats** |
| AlphaDynamics mean NLL | **108.0** |
| MLP mean NLL | 580.0 |
| AlphaDynamics vs MLP ratio | **5.4× better** on average |

Best ratio: 1lwjA03 (11.2× better than MLP).
P-value that 37/37 is random: ≈ 7.3 × 10⁻¹².

See [results/mdcath_benchmark_results.md](results/mdcath_benchmark_results.md)
for full per-domain table.

### Rollout stability (2500-step autoregressive)

On 5 domains, 2500 autoregressive frames (= length of original mdCATH replica):
no trajectory explosion (mean step drift −32° — steps shrink slightly, not grow).
Mean per-residue Ramachandran KL vs ground truth: 1.84.

See [results/mdcath_rollout_results.md](results/mdcath_rollout_results.md).

## Empirical laws observed

**Law 1 — Warmup time matches protein scale:**
Optimum ODE integration time t_max depends on chain length N and data
temporal correlations. On mdCATH at 50 residues, 348K, 1ps stride, t=4
is optimal. Too short (t=1) → oscillators don't synchronize. Too long
(t=8) → dynamics overshoot.

**Law 2 — Advantage scales with protein ordering:**
The win ratio (MLP NLL / AlphaDynamics NLL) is inversely proportional to
the identity baseline (natural frame-to-frame change). Well-ordered
proteins (small step) give the largest advantage. Fast/disordered proteins
(large step) give smaller advantage but AlphaDynamics still wins.

## Architecture

```
dφ_i/dt = ω_i + Σ_j W_ij · cos(φ_j) · sin(φ_j − φ_i) + a · sin(φ_anchor_i − φ_i)
```

- **ω_i**: prime-based natural frequencies [2.11, 1.31, 0.67, 0.31, 0.17] rad/s
  cycled across N oscillators (incommensurable → no mutual resonance,
  KAM-friendly)
- **W_ij**: learnable asymmetric N×N coupling matrix (CNOT-inspired
  efficient decomposition)
- **φ_anchor_i**: golden phyllotaxis (2π/φ²·i mod 2π − π) — Weyl
  equidistribution on S¹, breaks symmetry heterogeneously
- **Integrator**: torchdiffeq RK4 adjoint, integration horizon t_max (tuned)
- **Output head**: 8-component mixture of von Mises densities on T^N
  (axis-independent within each mixture component)

## Directory layout

```
AlphaDynamics/
├── README.md                  — this file
├── requirements.txt
├── src/                       — model + training + eval code
│   ├── chain_model.py           — ChainMLP + ChainPhaseFlow (AlphaDynamics)
│   ├── train_real.py            — training loop (dataset agnostic)
│   ├── train_chain.py           — training helpers
│   ├── chain_md.py              — synthetic Langevin MD generator
│   ├── rollout_eval.py          — KL divergence, per-residue evaluation
│   ├── mdcath_convert_v3.py     — mdCATH HDF5 → dihedral npz
│   ├── mdcath_benchmark.py      — unified 37-domain benchmark runner
│   └── mdcath_rollout_test.py   — 2500-step rollout stability test
├── results/
│   ├── mdcath_benchmark_results.md
│   ├── mdcath_benchmark_results.json
│   ├── mdcath_rollout_results.md
│   └── mdcath_rollout_results.json
├── docs/                      — daily research logs (Polish)
│   ├── EKSPERYMENTY_2026_04_14.txt   — every experiment with numbers
│   ├── ODKRYCIA_2026_04_14.txt       — chronological discoveries
│   ├── DECYZJE_2026_04_14.txt        — architectural decisions + why
│   ├── PORAZKI_2026_04_14.txt        — failures + lessons
│   └── STRATEGIA_2026_04_15.txt      — publication / collaboration strategy
└── data/                      — how to obtain data (raw data not committed)
```

## Reproducing results

```bash
pip install -r requirements.txt

# 1. Download mdCATH domains (50-residue subset)
python src/mdcath_download.py  # TODO: extract from scripts

# 2. Convert HDF5 → dihedrals
python src/mdcath_convert_v3.py

# 3. Run unified benchmark
python src/mdcath_benchmark.py

# 4. Rollout stability
python src/mdcath_rollout_test.py
```

## Related work

- **AlphaFold 2/3** (DeepMind) — static structure prediction (different task).
- **Timewarp** (Klein et al., NeurIPS 2023, Microsoft) — Cartesian normalizing
  flow for peptide dynamics (396M params).
- **AlphaFlow / ESMFlow** (Jing et al., MIT, 2024) — flow matching on
  conformational ensembles (different task).
- **MDGen** (Jing et al., MIT, 2024) — autoregressive MD in Cartesian.
- **AlphaFold-MSA-subsampling** (Wayment-Steele et al.) — hack AF2 via
  reduced MSA to get ensembles (different task: states, not trajectories).
- **AlphaFold-Metainference** (Vendruscolo lab, Cambridge 2024) — NMR-
  restrained ensemble from AF2.

AlphaDynamics occupies a distinct niche: **continuous temporal propagation
of torus dynamics** with minimal parameters and ODE-based inductive bias.

## Status

- [x] mdCATH 37-domain unified benchmark — 37/37 wins
- [x] Multi-seed validation on pentapeptide (p=0.0001)
- [x] Rollout stability (no explosion, moderate distribution preservation)
- [ ] Scaling to larger proteins (N=100, 150, 200 residues)
- [ ] Direct head-to-head vs Timewarp, AlphaFlow
- [ ] arXiv preprint
- [ ] NeurIPS ML4Sci / ICLR workshop submission

## Data

Raw mdCATH trajectories are not committed (3.3 TB total, 200 MB per
domain). See `data/README.md` (TODO) for download instructions via
Hugging Face `compsciencelab/mdCATH`.

## License

To be decided.

## Citation

Manuscript in preparation. Please do not cite without contacting the
author first.

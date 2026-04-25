# AlphaDynamics

Per-system neural propagator for molecular dynamics emulation on the protein
torsion torus T^N.
Input: current protein conformation (dihedral angles φ, ψ per residue).
Output: probability distribution over the next frame's conformation.

**Author:** Krzysztof Gwozdz
**Started:** 2026-04-14

## What it does

AlphaDynamics learns a fast surrogate of a specific protein trajectory. Given
seed MD data for one folded protein/domain, it trains a compact model that
predicts the next-step distribution in backbone torsion space and can generate
autoregressive rollouts for analysis.

It is not yet a zero-shot sequence-to-dynamics model. A future sequence- or
structure-conditioned version must be validated with sequence-identity splits
and external baselines before making that claim.

- Input: torsion angles (φ, ψ) of all residues at time t
- Output: mixture-of-von-Mises distribution over angles at time t+dt
- Core architecture: phase oscillators coupled via CNOT-style interactions,
  evolved by torchdiffeq RK4 adjoint ODE solver
- Model size: ~350K parameters per protein/domain for the v1 full-chain model
- Inference speed: ~16 ms per frame on RTX 5090

## Headline results

**Current publication-grade status:** the aligned audit is the defensible v1
result: 20 mdCATH domains at N=48, 20 domains at N=98, plus 3+3 aligned
rollout/free-energy audits. The converter now aligns φ and ψ by common residue
index and stores `residue_indices` and
`dihedral_alignment=common_residue_index`.

Validated aligned local inputs currently exist for 20 N=48 mdCATH domains at
348 K, 20 N=98 mdCATH domains at 348 K, and the matching all-temperature
rollout inputs. The smoke test
`results/mdcath_aligned5_smoke.md` verifies the new benchmark path only; it is
not a trained scientific result. A short CPU audit with 100 training steps is
available at `results/mdcath_aligned5_100step_cpu.md`; it is a sanity-check,
not a publication-grade replacement for the full 4000-step benchmark.

### Aligned mdCATH N≈50 audit — 20 domains, 4000 steps

Fresh phi/psi-aligned rerun on the 20 locally available N≈50 mdCATH domains
at 348 K:

| Domains | N used | Steps | Device | Win rate | Ratio of means | Mean ΔNLL |
|---:|---:|---:|---|---:|---:|---:|
| 20 | 48 | 4000 | CPU | **20/20** | **7.66×** | **-757.96** |

All 20 input `.npz` files have `dihedral_alignment=common_residue_index`.
Full table: [results/mdcath_aligned20_4000step_cpu.md](results/mdcath_aligned20_4000step_cpu.md).

### Historical mdCATH benchmark — pre-alignment, superseded

These tables were generated before the phi/psi residue-index alignment audit.
They remain useful as development history, but the current manuscript uses the
aligned 20+20 domain audit above instead.

| Size class | N residues | Domains | Win rate | Mean ratio | Best ratio |
|---|---|---|---|---|---|
| Small | 50 | 37 | **37/37** | 5.4× | 11.2× (1lwjA03) |
| **Medium** | **100** | **20** | **20/20** | **6.3×** | **21.7× (5cmbA02)** |
| **Total** | — | **57** | **57/57 (100%)** | — | — |

P-value that 57/57 is random: ≈ 7 × 10⁻¹⁸.

**Superseded scaling note:** the pre-alignment run suggested the mean ratio
increased from 5.4× at N=50 to 6.3× at N=100. The aligned audit changes that
statement: N=48 has a 7.66× ratio of means and N=98 has 5.08×. The defensible
claim is that AlphaDynamics' advantage persists at larger N and rollout
fidelity does not visibly degrade, not that the NLL ratio monotonically grows
with chain length.

Full tables:
- [results/mdcath_benchmark_results.md](results/mdcath_benchmark_results.md) — N=50 (37 domains)
- [results/mdcath_N100_results.md](results/mdcath_N100_results.md) — N=100 (20 domains)

### Rollout stability (2500-step autoregressive)

On 5 domains, 2500 autoregressive frames (= length of original mdCATH replica):
no trajectory explosion (mean step drift −32° — steps shrink slightly, not grow).
Mean per-residue Ramachandran KL vs ground truth: 1.84.

See [results/mdcath_rollout_results.md](results/mdcath_rollout_results.md).

### Aligned rollout free-energy audit — 3 domains, GPU

Fresh aligned 2500-step rollouts with `κ×30` on three representative domains:

| Domains | Training | Rollout | Mean JSD | Mean EMD | Mean \|ΔG_basin\| | Mean pop err |
|---:|---|---|---:|---:|---:|---:|
| 3 | 4000 steps, batch 512, CUDA | 2500 steps | 0.194 | 20.6° | 1.356 kcal/mol | 0.093 |

Ordered domains are good (`1lwjA03`, `1kwgA03`: JSD ≈ 0.14, population error
≈ 0.07). The disordered domain `1vq8L01` is the honest limitation
(JSD 0.300, EMD 35.9°, |ΔG_basin| 1.98 kcal/mol).

Full table: [results/ramachandran_aligned3_4000step_gpu.md](results/ramachandran_aligned3_4000step_gpu.md).

### Aligned N=100 scaling audit — 20 domains, GPU

Fresh aligned one-step NLL audit at the larger size class (N=98 common
residues, mdCATH 348 K), trained for 4000 steps per model with batch
256 on CUDA:

| Domains | Win rate | Mean MLP NLL | Mean PF_t4 NLL | Ratio of means |
|---:|---:|---:|---:|---:|
| 20 | **20/20 (100%)** | 519.5 | **102.2** | **5.08×** |

PhaseFlow $t_\text{max}=4$ wins all 20 domains. Best margins: `4ktyB04`
(9.8×), `2dhkA01` and `1w36F02` (8.3×). The full table with per-domain
identity, MLP, PF_t1, PF_t4 NLLs is in
[results/mdcath_aligned20_n100_4000step_gpu.md](results/mdcath_aligned20_n100_4000step_gpu.md).

### Aligned N=98 rollout free-energy audit — 3 domains, GPU

Fresh aligned 2500-step rollouts with `κ×30` on three representative N=98 domains:

| Domains | Training | Rollout | Mean JSD | Mean EMD | Mean \|ΔG_basin\| | Mean pop err |
|---:|---|---|---:|---:|---:|---:|
| 3 | 4000 steps, batch 128, CUDA | 2500 steps | 0.172 | 17.9° | 1.403 kcal/mol | 0.092 |

Two ordered domains are good (`4ktyB04`: JSD 0.127, pop err 0.059;
`1w36F02`: JSD 0.122, pop err 0.065). The disordered domain
`2hoxA01` is the honest limitation (JSD 0.266, EMD 30.1°,
|ΔG_basin| 2.19 kcal/mol, pop err 0.151).

Rollout fidelity at N=98 is **marginally better** than at N=48 (N=48
mean JSD 0.194 vs N=98 mean JSD 0.172), suggesting AlphaDynamics scales
to larger proteins without rollout degradation.

Full table: [results/ramachandran_aligned3_n98_4000step_gpu.md](results/ramachandran_aligned3_n98_4000step_gpu.md).

Release/audit documentation:
- [docs/AUDIT_MANIFEST_2026_04_25.md](docs/AUDIT_MANIFEST_2026_04_25.md)
- [docs/PREPRINT_PACKAGE_2026_04_25.md](docs/PREPRINT_PACKAGE_2026_04_25.md)

## Empirical laws observed

**Law 1 — Warmup time matches protein scale:**
Optimum ODE integration time t_max depends on chain length N and data
temporal correlations. Historical runs favored t=4 on the N≈50 mdCATH
benchmark, but the aligned 100-step audit favored t=1 on the five local
domains. Treat t_max as a hyperparameter until the full aligned rerun
settles it.

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
│   ├── mdcath_benchmark.py      — aligned mdCATH benchmark runner
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

# 1. Download mdCATH domains into mdcath_raw/data/
# Example shown in data/README.md via huggingface_hub

# 2. Convert HDF5 → aligned dihedrals
python src/mdcath_convert_v3.py \
  --bench_dir mdcath_raw \
  --out_dir mdcath_real_data/mdcath_348K

# 3. Run an audit benchmark without overwriting historical tables
python src/mdcath_benchmark.py \
  --data_dir mdcath_real_data/mdcath_348K \
  --out_prefix mdcath_aligned5_results \
  --device cpu

# Or run the aligned five-domain audit end to end
DEVICE=cpu STEPS=4000 BATCH=512 src/run_aligned5_benchmark.sh

# 4. Rollout stability
python src/mdcath_rollout_test.py
```

## CLI MVP

The product wrapper keeps the audited scripts behind one command surface:

```bash
# Convert mdCATH H5 files to aligned torsion npz
python src/alphadynamics_cli.py convert \
  --bench-dir mdcath_raw \
  --out-dir mdcath_real_data/mdcath_348K

# Train/evaluate the one-step NLL benchmark
python src/alphadynamics_cli.py train \
  --data-dir mdcath_real_data/mdcath_348K \
  --out-prefix mdcath_aligned20_4000step_cpu \
  --steps 4000 \
  --batch 512 \
  --device auto

# Train rollout model and evaluate Ramachandran free-energy fidelity
python src/alphadynamics_cli.py rollout \
  --data-dir mdcath_real_data/mdcath_alltemps \
  --out-prefix ramachandran_aligned3_4000step_gpu \
  --domains 1lwjA03 1kwgA03 1vq8L01 \
  --steps 4000 \
  --batch 512 \
  --device auto

# Build compact Markdown summary from existing JSON result files
python src/alphadynamics_cli.py report \
  --output results/alphadynamics_audit_report.md
```

Every execution subcommand supports `--dry-run` to print the underlying audited
script call before launching a long job.

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

- [x] Historical mdCATH 37-domain unified benchmark at N≈50 — superseded by aligned subset
- [x] Historical mdCATH 20-domain scaling benchmark at N≈100 — superseded by aligned N=98 rerun
- [x] Aligned mdCATH N≈50 benchmark subset — 20 domains, 20/20 wins
- [x] Cross-temperature all-temperature data regenerated with alignment metadata
- [x] Rollout stability test (no explosion, moderate distribution preservation)
- [x] Aligned rollout/free-energy audit — 3 N=48 domains, κ×30
- [x] Aligned N=98 scaling audit — 20 domains, 20/20 wins, 5.08× ratio
- [x] Aligned N=98 rollout audit — 3 domains, comparable fidelity to N=48
- [x] Converter fixed to align φ/ψ by residue index
- [x] CLI MVP wrapper — convert, train, rollout, report
- [x] v1 preprint package prepared — aligned 20+20 NLL and 3+3 rollout audit
- [ ] Remaining N≈50 aligned rerun domains, if raw H5 files are downloaded
- [ ] Rollout fidelity without κ-rescaling or honest v1 limitation
- [ ] Scaling to N=150, N=200 residues
- [ ] Direct head-to-head vs Timewarp, AlphaFlow, bioEmu
- [ ] CASP Refinement targets (CASP15 / CASP16)
- [ ] arXiv preprint
- [ ] NeurIPS ML4Sci / ICLR workshop submission

## Data

Raw mdCATH trajectories are not committed (3.3 TB total, 200 MB per
domain). See `data/README.md` for download instructions via Hugging Face
`compsciencelab/mdCATH` and for the aligned `.npz` file format used by the
audited benchmarks.

## License

To be decided.

## Citation

Manuscript in preparation. Please do not cite without contacting the
author first.

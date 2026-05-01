# AlphaDynamics

[![DOI](https://zenodo.org/badge/1211339504.svg)](https://doi.org/10.5281/zenodo.19788564)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![License: CC BY 4.0](https://img.shields.io/badge/Manuscript-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

**Compact phase-flow neural propagator for protein torsion dynamics.**

This repository ships **two complementary releases** of AlphaDynamics:

| Release          | Type            | What it gives you                                                    | Status                     |
| ---------------- | --------------- | -------------------------------------------------------------------- | -------------------------- |
| **v0.3.0 (latest, 2026-05-01)** | sequence-only product | `pip install alphadynamics` → predict torsion ensembles from a sequence string, no MD seed required | Beta product release       |
| **v2.0-preprint (2026-04-29)** | per-system paper      | per-protein surrogate trained from seed MD; full reviewer-hardening audit, Zenodo DOI | Published preprint (paper) |

Both releases share the same phase-flow architecture; **v0.3.0** focuses on
broad usability (one model, any sequence) and **v2** focuses on per-protein
fidelity (one model per protein, audited against MD).

---

## v0.3.0 — sequence-only product (2026-05-01)

> 2.39× lower JSD than Microsoft Timewarp · 3000× fewer parameters · 64 phase oscillators · `pip install alphadynamics`

A tiny (~123K parameter) neural propagator that, given only a protein
sequence, predicts an ensemble of torsion-angle (φ, ψ) trajectories matching
the marginal Ramachandran density of long-timescale molecular dynamics
simulations. On the canonical 4AA benchmark it produces densities that are
**2.39× closer to ground-truth MD** than Microsoft Research's Timewarp model
(396M parameters), at roughly **3000× fewer parameters**.

This is a free and open contribution to the protein-dynamics community.

### Quickstart

```bash
pip install alphadynamics
```

That installs the code. Pretrained weights (a few MB) are downloaded
automatically on first use into `~/.cache/alphadynamics/weights/`.

### Predict from the command line

```bash
alphadynamics predict --sequence AAAY --n-ensemble 16 --rollout-steps 2500 -o aaay.npz
```

Output `aaay.npz` has shape `(16, 2500, 4, 2)` — ensemble × time × residues ×
[phi, psi] in radians.

### Predict from Python

```python
from alphadynamics import predict_torsion_ensemble

traj = predict_torsion_ensemble(
    "AAAY",
    n_ensemble=16,
    rollout_steps=2500,
    seed=42,          # deterministic
)
print(traj.shape)     # (16, 2500, 4, 2)
```

### Other CLI commands

```bash
alphadynamics info       # banner, headline metric, credits
alphadynamics models     # list available pretrained weights
alphadynamics version
```

### Headline result (v0.3.0)

Canonical Ramachandran Jensen-Shannon divergence (36 bins, no smoothing,
held-out validation as ground truth) on the canonical 4AA test set, averaged
over three peptides AAAY, AACE, AAEW:

| Model                  | Params | Mean JSD | 4AA wins | Notes                           |
| ---------------------- | -----: | -------: | :------: | ------------------------------- |
| Microsoft Timewarp     |  396 M |    0.468 |   0 / 3  | published baseline (research)   |
| **AlphaDynamics v0.3** |  123 K |  **0.196** | **3 / 3** | **2.39× lower**, **3000× smaller** |

On longer peptides the improvement narrows but the model remains competitive
at a tiny fraction of the parameter count:

| Test set       | Mean JSD |
| -------------- | -------: |
| 4AA (3 peps)   |    0.196 |
| mdCATH N≈48    |    0.276 |
| mdCATH N≈98    |    0.389 |

**Honest caveats.** The headline metric is *density* match — the model
captures the marginal Ramachandran distribution well but its kinetic
fingerprints (autocorrelation, dwell-time distribution, transition matrix)
do not yet reproduce MD at the same level of precision. v0.3.0 is best
read today as a *density surrogate*, not a kinetics surrogate.

### How v0.3.0 works (one paragraph)

A residue's torsion state `(φ, ψ)` is treated as a phase pair. Conditioned
on the amino-acid identity, position, and current angles, an MLP emits
per-residue oscillator parameters: an intrinsic frequency, a coupling
matrix, and an anchor phase. A phase-flow ODE then integrates the joint
state of 64 coupled oscillators with classical RK4 over a fixed horizon
`t_max=4.0` (8 substeps). The integrated phase state is decoded into a
mixture of axis-independent von Mises distributions per residue, from
which the next torsion frame is sampled. Rolled out autoregressively, this
defines a transferable sequence-only propagator over the torsion torus.

The code lives in [`alphadynamics/`](alphadynamics/). Weights are hosted
on the [GitHub Releases page](https://github.com/krisss0mecom/AlphaDynamics/releases)
and downloaded on demand by `alphadynamics.weights.load_pretrained`.

---

## v2.0-preprint — per-system paper (2026-04-29)

The v2 release is a per-protein surrogate: AlphaDynamics trains a
348K-parameter phase-flow model **per protein domain** from seed MD and
predicts the next-step distribution over backbone torsion angles. In the
v2 (2026-04-29) audit it beats:

- a matched **MLP baseline on 40/40 domains** (paired Wilcoxon $p<10^{-12}$,
  6.44× ratio-of-means; 95% bootstrap CI 5.45–7.75×),
- a **trivial AR(1) baseline in long rollouts** (gap-closure ratio 0.70 vs
  AR(1)'s 0.00, anchored against the split-trajectory replica floor),
- the **396M-parameter Microsoft Timewarp 4AA model on 3/3 shared
  tetrapeptides** from the public `microsoft/timewarp` 4AA-large/test split,
  under a single canonical Ramachandran JSD evaluator applied identically
  to both models (held-out val GT, 36 bins, no smoothing): mean JSD
  **0.165 vs 0.468, 2.84× closer** to held-out density, using the
  calibrated κ×1 rollout.

![Aligned mdCATH NLL audit: AlphaDynamics vs MLP](paper/figures/fig1_scatter.png)

### v2 30-second summary

- **Task:** learn a per-protein molecular-dynamics surrogate in φ/ψ torsion space.
- **Model:** coupled phase oscillators + neural ODE + mixture-of-von-Mises head.
- **One-step NLL:** 40/40 wins vs MLP, $p<10^{-12}$. AR(1) is competitive on
  small systems; AlphaDynamics catches up on N=98 and pulls ahead on rollout.
- **Rollout fidelity (load-bearing claim):** 70% gap-closure to noise floor,
  vs MLP rollout 19%, AR(1) -2% (decohered toward uniform), uniform 0%.
- **Shared-dataset head-to-head:** 3/3 wins vs Microsoft Timewarp 4AA model
  on out-of-training tetrapeptides under unified canonical JSD;
  **2.84× closer** to held-out density (calibrated κ×1).
- **Scope (v2):** per-system surrogate trained from seed MD, not a
  zero-shot sequence-to-dynamics model. (For sequence-only, use **v0.3.0**.)

**Author:** Krzysztof Gwozdz
**Started:** 2026-04-14
**Preprint DOI (v2, 2026-04-29):** [10.5281/zenodo.19877815](https://doi.org/10.5281/zenodo.19877815)
**Concept DOI (all versions):** [10.5281/zenodo.19788564](https://doi.org/10.5281/zenodo.19788564)

### What v2 does

AlphaDynamics learns a fast surrogate of a specific protein trajectory.
Given seed MD data for one folded protein/domain, it trains a compact
model that predicts the next-step distribution in backbone torsion
space and can generate autoregressive rollouts for analysis.

The v2 release is *not* a zero-shot sequence-to-dynamics model. For that,
use the **v0.3.0** sequence-only product above.

- Input: torsion angles (φ, ψ) of all residues at time t
- Output: mixture-of-von-Mises distribution over angles at time t+dt
- Core architecture: phase oscillators coupled via CNOT-style interactions,
  evolved by torchdiffeq RK4 adjoint ODE solver
- Model size: ~350K parameters per protein/domain for the v1 full-chain model
- Inference speed: ~16 ms per frame on RTX 5090

### v2 headline results

**Current publication-grade status:** the aligned audit is the defensible v1
result: 20 mdCATH domains at N=48, 20 domains at N=98, plus 3+3 aligned
rollout/free-energy audits. The converter now aligns φ and ψ by common residue
index and stores `residue_indices` and
`dihedral_alignment=common_residue_index`.

Validated aligned local inputs currently exist for 20 N=48 mdCATH domains at
348 K, 20 N=98 mdCATH domains at 348 K, and the matching all-temperature
rollout inputs. Smoke tests and short undertrained audits are excluded from the
public release; the shipped result tables below are the publication-grade
4000-step audits.

#### Aligned mdCATH N≈50 audit — 20 domains, 4000 steps

Fresh phi/psi-aligned rerun on the 20 locally available N≈50 mdCATH domains
at 348 K:

| Domains | N used | Steps | Device | Win rate | Ratio of means | Mean ΔNLL |
|---:|---:|---:|---|---:|---:|---:|
| 20 | 48 | 4000 | CPU | **20/20** | **7.66×** | **-757.96** |

All 20 input `.npz` files have `dihedral_alignment=common_residue_index`.
Full table: [results/mdcath_aligned20_4000step_cpu.md](results/mdcath_aligned20_4000step_cpu.md).

#### Aligned rollout free-energy audit — 3 domains, GPU

Fresh aligned 2500-step rollouts with `κ×30` on three representative domains:

| Domains | Training | Rollout | Mean JSD | Mean EMD | Mean \|ΔG_basin\| | Mean pop err |
|---:|---|---|---:|---:|---:|---:|
| 3 | 4000 steps, batch 512, CUDA | 2500 steps | 0.194 | 20.6° | 1.356 kcal/mol | 0.093 |

Ordered domains are good (`1lwjA03`, `1kwgA03`: JSD ≈ 0.14, population error
≈ 0.07). The disordered domain `1vq8L01` is the honest limitation
(JSD 0.300, EMD 35.9°, |ΔG_basin| 1.98 kcal/mol).

Full table: [results/ramachandran_aligned3_4000step_gpu.md](results/ramachandran_aligned3_4000step_gpu.md).

#### Aligned N=100 scaling audit — 20 domains, GPU

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

#### Aligned N=98 rollout free-energy audit — 3 domains, GPU

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

### v2 empirical laws observed

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

### v2 architecture

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

---

## Repository layout

```
AlphaDynamics/
├── README.md                            — this file
├── LICENSE                              — Apache 2.0 (code)
├── LICENSE-MANUSCRIPT.md                — CC BY 4.0 (paper)
├── NOTICE                               — author lineage and attribution
├── CITATION.cff                         — citation metadata
├── pyproject.toml                       — pip install alphadynamics
│
├── alphadynamics/                       — v0.3.0 sequence-only product (NEW)
│   ├── __init__.py                       — public API + banner
│   ├── api.py                            — predict_torsion_ensemble
│   ├── cli.py                            — `alphadynamics predict / info / models`
│   ├── banner.py                         — ASCII logo + author credit
│   ├── weights.py                        — lazy download from GitHub Releases
│   ├── ad_init.py                        — von Mises mixture prior
│   ├── models.py                         — phase-flow ODE propagator
│   ├── rollout.py                        — autoregressive rollout
│   ├── training.py                       — training loops
│   ├── data.py                           — protein trajectory loader
│   ├── metrics.py                        — canonical Ramachandran JSD
│   └── baselines.py                      — AR(1), Gaussian-step, identity
│
├── src/                                 — v2 paper code (per-system)
│   ├── alphadynamics_cli.py               — legacy product CLI
│   ├── chain_model.py                     — ChainMLP + ChainPhaseFlow
│   ├── train_real.py                      — training utilities
│   ├── chain_md.py                        — synthetic Langevin MD generator
│   ├── rollout_eval.py                    — autoregressive rollout + metrics
│   ├── ramachandran_energy_v2.py          — Ramachandran free-energy audit
│   └── ... (additional audit + benchmark scripts)
│
├── paper/
│   ├── main.md                            — manuscript source
│   ├── main.pdf                           — compiled preprint
│   ├── references.bib
│   ├── make_figures.py
│   └── figures/                           — fig1/fig2/fig3 + ramachandran panels
│
├── results/                             — aligned audit artifacts (v2 paper)
├── docs/                                — preprint package & audit notes
└── data/                                — how to obtain mdCATH (raw not committed)
```

---

## Reproducing the v2 paper

```bash
pip install -e .[paper]

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

# 4. Ramachandran free-energy rollout audit
python src/ramachandran_energy_v2.py
```

The legacy v2 CLI (`src/alphadynamics_cli.py`) is preserved for paper
reproducibility. The new pip-installable CLI under `alphadynamics.cli`
is the v0.3.0 sequence-only product surface.

The productization plan and research expansion ladder are documented in
[docs/PRODUCT_V1_2026_04_28.md](docs/PRODUCT_V1_2026_04_28.md).
The reviewer hardening checklist is tracked in
[docs/REVIEWER_RISK_REGISTER_2026_04_28.md](docs/REVIEWER_RISK_REGISTER_2026_04_28.md).

---

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

---

## Status

- [x] v2 preprint (2026-04-29) on Zenodo, DOI [10.5281/zenodo.19877815](https://doi.org/10.5281/zenodo.19877815)
- [x] v0.3.0 sequence-only product release (2026-05-01) — `pip install alphadynamics`
- [x] Aligned mdCATH N≈50 benchmark — 20 domains, 20/20 wins
- [x] Aligned mdCATH N=98 scaling audit — 20 domains, 5.08× ratio
- [x] Aligned N=98 rollout audit — comparable fidelity to N=48
- [x] Head-to-head vs Microsoft Timewarp 4AA — 3/3 wins (per-system, calibrated κ×1)
- [x] Sequence-only head-to-head vs Microsoft Timewarp 4AA — 3/3 wins (transferable, v0.3.0)
- [x] Statistical tests on aligned audit (Wilcoxon, bootstrap CI, AR(1) baseline)
- [x] Anchored JSD reference scale vs floor / uniform / AR(1) / MLP rollout
- [x] K-sweep ablation on mixture components
- [x] Editable package metadata — `pip install -e .` exposes `alphadynamics`
- [ ] Bivariate von Mises head (Singh et al. 2002)
- [ ] Kinetic observables (residence times, MFPT)
- [ ] Scaling to N=150, N=200 residues
- [ ] Head-to-head vs AlphaFlow, bioEmu, MDGen
- [ ] CASP Refinement targets (CASP15 / CASP16)
- [ ] arXiv preprint
- [ ] NeurIPS ML4Sci / ICLR workshop submission

---

## Data

Raw mdCATH trajectories are not committed (3.3 TB total, 200 MB per
domain). See `data/README.md` for download instructions via Hugging Face
`compsciencelab/mdCATH` and for the aligned `.npz` file format used by the
audited benchmarks.

---

## License

Source code is licensed under the Apache License 2.0; see [LICENSE](LICENSE).
Author and lineage attribution is in [NOTICE](NOTICE) — please preserve it
in any redistribution or derivative work, as required by Section 4 of the
Apache 2.0 license.

The manuscript, paper figures, result tables, and documentation are licensed
under CC BY 4.0; see [LICENSE-MANUSCRIPT.md](LICENSE-MANUSCRIPT.md).

---

## Citation

If you use AlphaDynamics in academic work, please cite the relevant release.
A `CITATION.cff` file is included so GitHub's "Cite this repository" button
generates the right entry automatically.

**For the v2 paper (per-system, peer-reviewable preprint):**

```bibtex
@misc{gwozdz2026alphadynamics,
  author       = {Gwóźdź, Krzysztof},
  title        = {{AlphaDynamics}: A Per-System Phase-Flow Propagator for
                  Protein Torsion Dynamics with Calibrated Rollout Fidelity},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v2.0-preprint-2026-04-29},
  doi          = {10.5281/zenodo.19877815},
  url          = {https://doi.org/10.5281/zenodo.19877815}
}
```

**For the v0.3.0 sequence-only product release:**

```bibtex
@software{gwozdz2026alphadynamicsproduct,
  author  = {Gwozdz, Krzysztof},
  title   = {AlphaDynamics: Compact sequence-only neural propagator
             for protein torsion dynamics},
  year    = {2026},
  url     = {https://github.com/krisss0mecom/AlphaDynamics},
  license = {Apache-2.0},
  version = {0.3.0}
}
```

---

## Author

**Krzysztof Gwozdz** — independent researcher, Poland
<krisss0gwo@gmail.com>

AlphaDynamics is the protein-dynamics application of a multi-year research
program on phase-oscillator computation across hardware (REZON), formal
phase computing, and neuroscience. See [NOTICE](NOTICE) for the full
research lineage.

This project is released as a gift to the protein-dynamics community.

---

## Acknowledgements

- Microsoft Research's Timewarp paper and codebase, used as the
  comparison baseline.
- The mdCATH consortium for long-timescale MD trajectories.
- The 4AA-large test set from `microsoft/timewarp` used in the canonical
  benchmark.

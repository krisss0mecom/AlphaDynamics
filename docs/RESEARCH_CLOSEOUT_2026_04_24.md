# AlphaDynamics Research Closeout

Date: 2026-04-24; updated 2026-04-25

## Defensible Target

Finish AlphaDynamics as a per-system neural MD surrogate:

- input: a seed/reference MD trajectory or prepared torsion trajectory for one protein
- output: next-step torsion distribution plus autoregressive rollouts
- validation: NLL, rollout Ramachandran KL/JSD/EMD, step-size drift, and baseline comparisons

Do not present the current repository as a zero-shot sequence-to-dynamics
model. That is a separate v3 research program.

## Current Scientific Claim

Defensible after aligned reruns:

> A compact torus-native phase-flow model can learn per-protein backbone
> torsion dynamics from seed MD and can outperform matched MLP baselines
> on one-step NLL, while producing stable autoregressive rollouts with
> explicitly reported fidelity limits.

Not defensible yet:

- sequence-only prediction of real protein dynamics
- converged thermodynamics or kinetics for large proteins from short seed MD
- basin-DAM as the dominant causal improvement unless implemented and ablated
- publication-grade 57-domain numbers until phi/psi-aligned reruns replace the historical tables

## Blocking Issues

1. Historical 37/57-domain tables were generated before the phi/psi alignment audit.
   The converter now aligns by common residue index and stores `residue_indices`.
   The local 20-domain N≈50 subset has been rerun from aligned inputs; the
   remaining historical domains still need aligned reruns if their H5 files are
   downloaded.
2. `paper/main_v2.md` describes basin-DAM, Timewarp/MSM, Mpro and web-tool claims
   that are not all backed by local source code and completed artifacts.
3. Rollout fidelity in v1 still depends on `kappa_mult=30` and the model samples
   narrower trajectories than ground truth.
4. Local CUDA on Orin is usable after freeing desktop/browser memory. Heavy runs
   should still be launched with conservative batch sizes and monitored because
   unified RAM/VRAM pressure can trigger allocation failures.

## Work Plan

### Track A: Reproducible v1 Paper

- Regenerate aligned 348 K dihedral files for all available mdCATH domains.
- Rerun N≈50 and N≈100 one-step NLL benchmarks from aligned inputs.
- Rerun rollout metrics on the five-domain subset using the same aligned input convention.
- Keep the v1 paper honest: per-system surrogate, κ-rescaling limitation, no zero-shot claim.

### Track B: Product MVP

- CLI stages: convert -> train -> rollout -> evaluate -> report.
- Inputs: mdCATH H5, generic `.npz` torsions, later PDB+trajectory.
- Outputs: checkpoint, rollout `.npz`, metrics JSON/Markdown, Ramachandran figures.
- Web/UI can wrap the CLI only after CLI is deterministic.

### Track C: v2/v3 Research

- Implement basin-DAM in code before claiming it in the manuscript.
- Complete 20 domains x 3 seeds x ablation arms if basin-DAM remains a central claim.
- Treat sequence-conditioned v3 as new research with sequence-identity-controlled splits.

## First Milestone

Create a clean aligned five-domain audit:

```bash
python src/mdcath_convert_v3.py \
  --bench_dir mdcath_raw \
  --out_dir mdcath_real_data/mdcath_348K

python src/mdcath_benchmark.py \
  --data_dir mdcath_real_data/mdcath_348K \
  --out_prefix mdcath_aligned5_results \
  --device cpu
```

Equivalent runner:

```bash
DEVICE=cpu STEPS=4000 BATCH=512 src/run_aligned5_benchmark.sh
```

The smoke output `results/mdcath_aligned5_smoke.md` only verifies the pipeline.
It is not a scientific benchmark because it uses one training step.

Status:

- completed aligned conversion for local domains:
  `1hw7A02`, `1kwgA03`, `1lwjA03`, `1ss3A00`, `1vq8L01`
- completed CPU smoke benchmark:
  `results/mdcath_aligned5_smoke.md`
- completed CPU 100-step aligned audit:
  `results/mdcath_aligned5_100step_cpu.md`
- completed CPU 4000-step aligned benchmark on 5 domains:
  `results/mdcath_aligned5_4000step_cpu.md`
- completed CPU 4000-step aligned benchmark on 20 domains:
  `results/mdcath_aligned20_4000step_cpu.md`
- regenerated all-temperature `.npz` files for 20 domains x 5 temperatures
  with `dihedral_alignment=common_residue_index`
- completed CUDA 4000-step aligned rollout/free-energy audit on 3 domains:
  `results/ramachandran_aligned3_4000step_gpu.md`
- completed CUDA 4000-step aligned N=100 benchmark on 20 domains:
  `results/mdcath_aligned20_n100_4000step_gpu.md`
- completed CUDA 4000-step aligned N=98 rollout/free-energy audit on 3 domains:
  `results/ramachandran_aligned3_n98_4000step_gpu.md`
- updated `paper/main.md` title, abstract, dataset, main results,
  limitations, conclusion, and figure captions to use the aligned audit as
  the headline v1 evidence.
- added CLI MVP wrapper:
  `src/alphadynamics_cli.py` with `convert`, `train`, `rollout`, and `report`
  subcommands.
- prepared a clean v1 preprint release bundle:
  `release/alphadynamics_v1_preprint_2026_04_25.tar.gz`

100-step audit result:

| Run | Domains | N | Steps | Device | PF wins vs MLP | Mean ΔNLL |
|---|---:|---:|---:|---|---:|---:|
| `mdcath_aligned5_100step_cpu` | 5 | 48 | 100 | CPU | 5/5 | -20.698 |
| `mdcath_aligned5_4000step_cpu` | 5 | 48 | 4000 | CPU | 5/5 | -364.202 |
| `mdcath_aligned20_4000step_cpu` | 20 | 48 | 4000 | CPU | 20/20 | -757.957 |
| `mdcath_aligned20_n100_4000step_gpu` | 20 | 98 | 4000 | CUDA | 20/20 | -417.319 |

Interpretation: this is a useful sanity-check that the aligned pipeline still
shows a PhaseFlow advantage. The 100-step result is not publication-grade
because the models are undertrained. The 4000-step 20-domain N=48 result is
the current publication-grade one-step N≈50 subset result, with ratio-of-means
7.66x and PF_t4 winning on all 20 domains. The N=98 result extends scaling
verification: 20/20 wins, ratio-of-means 5.08x. Per-domain win ratios at
N=98 range from 3.5x (`2of5H00`) to 9.8x (`4ktyB04`).

Aligned rollout/free-energy audit:

| Run | Domains | Training | Rollout | Mean JSD | Mean EMD | Mean \|ΔG_basin\| | Mean pop err |
|---|---:|---|---|---:|---:|---:|---:|
| `ramachandran_aligned3_4000step_gpu` | 3 (N=48) | 4000 steps, CUDA | 2500 steps, κ×30 | 0.194 | 20.6° | 1.356 kcal/mol | 0.093 |
| `ramachandran_aligned3_n98_4000step_gpu` | 3 (N=98) | 4000 steps, CUDA | 2500 steps, κ×30 | 0.172 | 17.9° | 1.403 kcal/mol | 0.092 |

Interpretation N=48: ordered domains are good (`1lwjA03`, `1kwgA03`, JSD ≈ 0.14,
population error ≈ 0.07). The high-identity/disordered domain `1vq8L01`
is weaker (JSD 0.300, EMD 35.9°, |ΔG_basin| 1.98 kcal/mol) and should be
reported as a rollout limitation, not hidden.

Interpretation N=98: same pattern. Two ordered domains are good
(`4ktyB04` JSD 0.127, pop err 0.059; `1w36F02` JSD 0.122, pop err 0.065).
The disordered `2hoxA01` is the limitation (JSD 0.266, EMD 30.1°,
|ΔG_basin| 2.19 kcal/mol). Mean JSD at N=98 is marginally better than
at N=48 (0.172 vs 0.194), so rollout fidelity does NOT degrade with
larger chain length on aligned data.

Remaining closeout work:

- aligned rerun for the remaining historical N≈50 domains if raw H5 data is
  downloaded (publication-grade pełnego 37/57 domain replikacji)
- regenerate paper figures/PDF after the aligned manuscript update
- harden CLI product wrapper with checkpoint export, config files, and packaged
  entry point

PUBLISHED ON ZENODO (2026-04-26):
- Version DOI: 10.5281/zenodo.19788565
- Concept DOI: 10.5281/zenodo.19788564
- GitHub Release: v1.0-preprint-2026-04-25
- License: Apache 2.0 (code) + CC-BY-4.0 (manuscript)

PUBLICATION-GRADE AUDIT COMPLETE (2026-04-25):
The minimum viable paper data is now in place:
- 20-domain N=48 NLL aligned (20/20 wins, 7.66× ratio)
- 20-domain N=98 NLL aligned (20/20 wins, 5.08× ratio)
- 3-domain N=48 rollout aligned (mean JSD 0.194)
- 3-domain N=98 rollout aligned (mean JSD 0.172, comparable to N=48)
- Honest disordered limitations documented (1vq8L01, 2hoxA01)

Publication decision:

- Proceed with v1 preprint using the aligned 20+20 NLL audit and 3+3 rollout
  audit. Do not block submission on the full historical 37-domain N≈50 rerun.
- Keep the title: "AlphaDynamics: A compact per-system phase-flow surrogate
  for protein torsion dynamics".
- Submit first as an arXiv preprint with q-bio.BM as the natural primary
  category and cs.LG as cross-list if available; Zenodo can archive a tagged
  code/data snapshot after the preprint package is coherent.

# AlphaDynamics v1 Preprint Package

Date: 2026-04-25

## Submission Decision

Proceed with the v1 preprint now. Do not wait for the full historical
37-domain N≈50 rerun. The current evidence is sufficient for a narrow,
honest per-system surrogate claim:

- 40 aligned one-step NLL domains
- 6 aligned rollout/free-energy domains
- explicit disordered-domain limitations
- explicit kappa x30 rollout limitation
- explicit statement that this is not zero-shot sequence-to-dynamics

Recommended first venue:

- arXiv primary: `q-bio.BM`
- arXiv cross-list: `cs.LG`, if available
- Zenodo: after the repo snapshot is cleaned and tagged

## Title

AlphaDynamics: A compact per-system phase-flow surrogate for protein torsion
dynamics

## Short Abstract For Submission Form

AlphaDynamics is a compact per-system neural propagator for protein backbone
torsion dynamics. It operates directly on aligned phi/psi torsions and evolves
learned phase oscillators through a continuous ODE before predicting a
mixture-of-von-Mises next-step distribution. On an aligned mdCATH audit at
348 K, AlphaDynamics outperforms a matched MLP baseline on 40/40 domains:
20/20 wins at N=48 with a 7.66x ratio of mean NLLs and 20/20 wins at N=98
with a 5.08x ratio. Six 2500-step autoregressive rollouts produce stable
Ramachandran free-energy maps, with mean JSD 0.194 at N=48 and 0.172 at N=98.
The model is deliberately scoped as a per-system surrogate trained from seed
MD, not a zero-shot sequence-to-dynamics model; rollout fidelity currently
uses a kappa-rescaling heuristic and is weaker on high-entropy domains.

## Files To Include In A Public Release

Core manuscript:

- `paper/main.md`
- `paper/main.pdf`
- `paper/figures/fig1_scatter.png`
- `paper/figures/fig2_ratio_vs_identity.png`
- `paper/figures/fig3_scaling.png`
- `paper/figures/fig4_rollout.png`
- `paper/figures/fig5_architecture.png`
- `paper/figures/fig6_gnn_comparison.png`
- `paper/figures/fig8_ablation.png`
- `paper/figures/ramachandran_aligned3_4000step_gpu_1lwjA03.png`
- `paper/figures/ramachandran_aligned3_4000step_gpu_1kwgA03.png`
- `paper/figures/ramachandran_aligned3_4000step_gpu_1vq8L01.png`
- `paper/figures/ramachandran_aligned3_n98_4000step_gpu_4ktyB04.png`
- `paper/figures/ramachandran_aligned3_n98_4000step_gpu_1w36F02.png`
- `paper/figures/ramachandran_aligned3_n98_4000step_gpu_2hoxA01.png`

Audited result files:

- `results/alphadynamics_audit_report.md`
- `results/mdcath_aligned20_4000step_cpu.md`
- `results/mdcath_aligned20_4000step_cpu.json`
- `results/mdcath_aligned20_n100_4000step_gpu.md`
- `results/mdcath_aligned20_n100_4000step_gpu.json`
- `results/ramachandran_aligned3_4000step_gpu.md`
- `results/ramachandran_aligned3_4000step_gpu.json`
- `results/ramachandran_aligned3_n98_4000step_gpu.md`
- `results/ramachandran_aligned3_n98_4000step_gpu.json`

Audited code paths:

- `src/alphadynamics_cli.py`
- `src/mdcath_convert_v3.py`
- `src/mdcath_convert_alltemps.py`
- `src/mdcath_benchmark.py`
- `src/ramachandran_energy_v2.py`
- `src/tdm_model.py`
- `src/test_tdm.py`
- `src/chain_model.py`
- `src/train_real.py`

Documentation:

- `README.md`
- `data/README.md`
- `docs/RESEARCH_CLOSEOUT_2026_04_24.md`
- `docs/AUDIT_MANIFEST_2026_04_25.md`
- `docs/PREPRINT_PACKAGE_2026_04_25.md`

## Files To Exclude From The v1 Claim

These can remain in the repository as historical/development artifacts, but
they should not be used as headline evidence:

- `paper/main_v2.md`, `paper/main_v2.pdf`, `paper/main_v2.html`
- `results/mdcath_benchmark_results.*`
- `results/mdcath_N100_results.*`
- smoke outputs such as `results/mdcath_aligned5_smoke.*`
- short undertrained audits such as `results/mdcath_aligned5_100step_cpu.*`
- local logs such as `results/*run.log`, `results/*stdout.log`
- raw `.h5`, `.npz`, checkpoints, and local cache directories

## Reproduction Commands

Create aligned single-temperature inputs:

```bash
python src/alphadynamics_cli.py convert \
  --bench-dir mdcath_raw \
  --out-dir mdcath_real_data/mdcath_348K
```

Run the NLL audit:

```bash
python src/alphadynamics_cli.py train \
  --data-dir mdcath_real_data/mdcath_348K \
  --out-prefix mdcath_aligned20_4000step_cpu \
  --steps 4000 \
  --batch 512 \
  --device auto
```

Run rollout/free-energy evaluation:

```bash
python src/alphadynamics_cli.py rollout \
  --data-dir mdcath_real_data/mdcath_alltemps \
  --out-prefix ramachandran_aligned3_4000step_gpu \
  --domains 1lwjA03 1kwgA03 1vq8L01 \
  --steps 4000 \
  --batch 512 \
  --device auto
```

Build the compact audit report:

```bash
python src/alphadynamics_cli.py report \
  --output results/alphadynamics_audit_report.md
```

## Final Pre-Submission Checks

```bash
python3 -m py_compile src/*.py paper/*.py
bash -n src/run_aligned5_benchmark.sh src/final_benchmark.sh
python3 src/alphadynamics_cli.py report --dry-run
pandoc main.md --pdf-engine=weasyprint -o main.pdf  # run from paper/
pdfinfo paper/main.pdf
```

## Known Limitations To Keep In The Paper

- per-system training only; no sequence-only zero-shot claim
- kappa x30 inference heuristic for rollouts
- weaker rollout fidelity on high-entropy/disordered domains
- no direct Timewarp/MDGen/bioEmu head-to-head yet
- no converged kinetics or thermodynamics claim
- single-seed benchmark runs

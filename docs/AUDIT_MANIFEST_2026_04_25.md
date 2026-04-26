# AlphaDynamics v1 Audit Manifest

Date: 2026-04-25

This manifest identifies the files that support the v1 aligned audit claim.
The historical pre-alignment benchmark tables are not part of the headline
evidence.

## Headline Claim

AlphaDynamics is a compact per-system phase-flow surrogate for protein torsion
dynamics. On aligned mdCATH inputs, it wins against a matched MLP baseline on:

- 20/20 N=48 domains, ratio of means 7.66x
- 20/20 N=98 domains, ratio of means 5.08x

Aligned rollout/free-energy audits cover six domains:

- 3 N=48 domains, mean JSD 0.194
- 3 N=98 domains, mean JSD 0.172

The model is not claimed to be zero-shot from sequence and does not yet provide
converged thermodynamics or kinetics. Rollout fidelity uses kappa x30 sampling
and reports disordered domains as explicit limitations.

## Primary Manuscript

| SHA256 | Path |
|---|---|
| `c1e1ae3154008ac382b53c72fb72de9e8f5f3b3793a3135dfcf756b07cb74004` | `paper/main.md` |
| `ebdf0719663dcdc8a298f03318def89ae03205c27afbeb476c17987dbe9860fd` | `paper/main.pdf` |

## Primary Figures

| SHA256 | Path |
|---|---|
| `b17b3dc6a29c5159e3eab6123c8c35a47b6df235535ca23dd008c0cb307f5b4a` | `paper/figures/fig1_scatter.png` |
| `ac5d8aa5c010b6fa76f48ed584b1c876410aaa12740b5bf9f51e93c814b857aa` | `paper/figures/fig2_ratio_vs_identity.png` |
| `5f61f16d18572448bcd66f0cee5e24c1337292682480cbfda6d51c96aa97bba9` | `paper/figures/fig3_scaling.png` |
| `260d3d7df56915a6dc2b639d423e029f14e982b65f3563278ccce0112654d3ea` | `paper/figures/ramachandran_aligned3_4000step_gpu_1lwjA03.png` |
| `25528078ca70eaff1841e042c25bdcdfeab99a4365fbfd327a1f672ed9edee9e` | `paper/figures/ramachandran_aligned3_n98_4000step_gpu_4ktyB04.png` |

## Result Tables

| SHA256 | Path |
|---|---|
| `a6378fd3531322d4de1e5d6342c80f7a2e2db89fb514f17c464f5d6766c809a7` | `results/mdcath_aligned20_4000step_cpu.json` |
| `b2431ef90224e8a34f2777692f4f5cade32d350363b31c1a25d86d6d15d58981` | `results/mdcath_aligned20_4000step_cpu.md` |
| `15fdca641fa8cfd40be39d46d0b8289e4bb0cd356a9abc00967703014a4423dd` | `results/mdcath_aligned20_n100_4000step_gpu.json` |
| `de58e5fc4ecce5313f68bf137830ac38518de4553e900a8053b5be5b5b29af52` | `results/mdcath_aligned20_n100_4000step_gpu.md` |
| `ede105c86d637073bcc0a3f4faa590322ef634e5797404e05192542f383129c5` | `results/ramachandran_aligned3_4000step_gpu.json` |
| `77bb6dcedc4aee5c1f364b3f13521f1fd560506bb6db366f83c74379dc642500` | `results/ramachandran_aligned3_4000step_gpu.md` |
| `b8932c92a26f135c671578fbe45909a78485cccde848196b998561fb585b744c` | `results/ramachandran_aligned3_n98_4000step_gpu.json` |
| `bb615179549692a0284174082087a502af8a0137337e27db1a70af4d5af1475f` | `results/ramachandran_aligned3_n98_4000step_gpu.md` |

## Audited Code Paths

| SHA256 | Path |
|---|---|
| `4d48f89f42ad28d1a592520bc282131d0aee46f0d860ec61f4d3d832f11e0565` | `src/alphadynamics_cli.py` |
| `ef3e75d06c323a5b6b79410bc0d614273a6289105fc46c21d6f811167fbe97d3` | `src/mdcath_benchmark.py` |
| `2cfc6166f8fe79bd5daae0a9da5b9399630b8ba1cfd3051321bf29d66d9919a6` | `src/mdcath_convert_v3.py` |
| `4a890455f3b3086753188fa81694c388b7674c0e9f8528fbc74bd3c3d2e11313` | `src/mdcath_convert_alltemps.py` |
| `747a4c5f987d0980c09e4fe307f810015b7d3441b6ce09f83c55c79b0d349089` | `src/ramachandran_energy_v2.py` |

## Verification Commands

```bash
python3 -m py_compile src/*.py paper/*.py
python3 src/alphadynamics_cli.py report --dry-run
pdfinfo paper/main.pdf
```

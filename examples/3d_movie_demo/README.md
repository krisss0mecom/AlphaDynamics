# 3D Movie Pipeline Demo

Complete sequence-to-3D-trajectory pipeline using `alphadynamics` v0.4.0+.

## What's here

```
3d_movie_demo/
├── README.md             # this file
├── viewer.html           # 3Dmol.js standalone web viewer
├── make_demo.sh          # reproducible regen script
├── klvffae_amyloid.pdb   # 7 aa, amyloid β16-22 (β-aggregating)
├── trpcage_minifold.pdb  # 20 aa, classic mini-fold benchmark
└── aaay_4aa.pdb          # 4 aa, paper benchmark
```

Each PDB has 50 frames (subsampled from 200) showing one ensemble member's
backbone trajectory.

## Try it (3 ways)

### 1. Browser — zero install

The standalone `viewer.html` loads PDB files via 3Dmol.js. Two options:

**Locally:**
```bash
cd examples/3d_movie_demo
python -m http.server 8000
# open http://localhost:8000/viewer.html
```

**Or via GitHub Pages** (if enabled): just open the link to `viewer.html`
on `github.io/<user>/AlphaDynamics/examples/3d_movie_demo/`.

You get an interactive 3D viewer with:
- Switch between 3 sample peptides
- Play / pause animation
- Frame slider
- Speed control

### 2. PyMOL / VMD / ChimeraX

```bash
pymol klvffae_amyloid.pdb       # PyMOL
vmd trpcage_minifold.pdb         # VMD
chimerax aaay_4aa.pdb            # ChimeraX
```

In PyMOL, animate with the `play` command.

### 3. Generate from scratch

```bash
pip install -U alphadynamics    # >=0.4.0 with rebuild subcommand
bash make_demo.sh
```

Takes ~1-2 minutes on CPU. Each peptide:
1. Runs torsion prediction (`alphadynamics predict`) → `.npz`
2. Reconstructs 3D backbone (`alphadynamics rebuild`) → `.pdb`
3. Reports per-frame Rg and end-to-end distance

## What you're looking at

These are torsion-angle ensembles converted to 3D backbone coordinates
using NeRF (Parsons 2005) with Engh-Huber 1991 standard bond geometry.

Each ensemble member is one independent rollout from the same starting
state. The *distribution* across frames represents the equilibrium
ensemble of conformations the peptide would visit during long
molecular-dynamics simulation.

**Backbone heavy atoms only** (N, Cα, C, O). No side chains, no
hydrogens — for those you'd need additional tools (Rosetta, OpenMM, etc).

**Diagnostic visualization**, not high-resolution structure prediction.
Torsion errors accumulate along the chain; for long peptides
(N > ~50) end-to-end displacement may be substantial.

## Per-peptide notes

### KLVFFAE (7 aa)
Central segment of amyloid β-peptide (Aβ16-22). β-aggregating in vitro,
forms fibrils. Model should produce extended/PPII-rich ensemble.

Expected: Rg ~5 Å, end-to-end ~13 Å (mix of compact and extended states).

### Trp-cage NLYIQWLKDGGPSSGRPPPS (20 aa)
Designed mini-protein, smallest known stable α-helix fold (Neidigh,
Fesinmeyer & Andersen 2002). Native: helix res 2-9, PPII C-term.

Expected: Rg ~10 Å, end-to-end varies (folded ~7 Å, unfolded ~30+ Å).

### AAAY (4 aa)
Used as 4AA benchmark in AlphaDynamics paper v2 (DOI 10.5281/zenodo.19877815).
JSD vs MD ground truth: AlphaDynamics 0.196, Microsoft Timewarp 0.468 (3000× larger model).

Expected: Rg ~3-4 Å, end-to-end ~6-12 Å.

## Architecture

```
sequence "KLVFFAE"
   │
   ▼  alphadynamics predict
   │    (78K-param phase-oscillator model;
   │     phase-flow ODE on torus S¹×S¹)
   │
torsion trajectory (E, T, N, 2)  in φ,ψ
   │
   ▼  alphadynamics rebuild
   │    (deterministic NeRF; no ML)
   │
multi-model PDB (T, N, 4 atoms)
   │
   ▼  3Dmol.js / PyMOL / VMD / ChimeraX
   │
3D backbone movie
```

## Resources

- [Main repo](https://github.com/krisss0mecom/AlphaDynamics)
- [PyPI package](https://pypi.org/project/alphadynamics/)
- [HuggingFace model](https://huggingface.co/krissss0/alphadynamics)
- [Try in browser (Gradio Space)](https://huggingface.co/spaces/krissss0/alphadynamics)
- [Paper v2 (Zenodo DOI)](https://doi.org/10.5281/zenodo.19877815)

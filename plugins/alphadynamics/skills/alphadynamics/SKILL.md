---
name: alphadynamics
description: Run AlphaDynamics protein torsion-dynamics prediction. Use when user wants to predict, analyze, or compare backbone dihedral (phi/psi) ensembles for a peptide / short protein sequence — e.g. "predict torsions for AAAY", "what conformations does KLVFFAE adopt", "compare AAAY vs AAAW", "Ramachandran for [sequence]". The Python package `alphadynamics` (v0.3.1+) is already pip-installed system-wide.
---

# AlphaDynamics — predict protein torsion ensembles

This skill is for running the locally-installed `alphadynamics` package
(authored by Krzysztof Gwozdz, the user). It predicts an ensemble of
(phi, psi) trajectories for a peptide sequence — useful for:

- Quick Ramachandran ensemble analysis
- Comparing sequence variants / mutants
- Estimating alpha/beta/PPII basin populations
- Sanity-checking before launching real MD

## Workflow

### 1. Get sequence from user

If user gave a 1-letter amino-acid sequence (e.g. `AAAY`, `KLVFFAE`),
use it directly. If unclear, ask once. Length 4-100 is the trained range;
warn if shorter or longer.

### 2. Run prediction

Use the CLI directly (it auto-downloads weights from GitHub Releases
on first use, into `~/.cache/alphadynamics/weights/`):

```bash
alphadynamics predict --sequence <SEQ> --n-ensemble <N> --rollout-steps <T> --device cpu -o /tmp/<SEQ>.npz
```

**Sane defaults** (use unless user asks otherwise):

- Quick exploration: `--n-ensemble 4 --rollout-steps 200` (~10 sec on CPU)
- Standard: `--n-ensemble 16 --rollout-steps 2500` (~5-15 min on Jetson CPU,
  faster on GPU/x86)

Prefer `--device cpu` on Jetson Orin (8GB unified RAM is fragile for GPU torch).
Prefer `--device cuda` on dedicated NVIDIA boxes.

### 3. Analyze output

The output is an `.npz` file with `torsions` of shape
`(ensemble, time, residues, [phi, psi])`, all in radians.

Inline Python analysis the user will care about:

```python
import numpy as np
d = np.load('/tmp/<SEQ>.npz', allow_pickle=True)
t = d['torsions']
phi_deg = np.degrees(t[..., 0].flatten())
psi_deg = np.degrees(t[..., 1].flatten())

def basin(plo, phi_, slo, shi):
    return ((phi_deg >= plo) & (phi_deg <= phi_) &
            (psi_deg >= slo) & (psi_deg <= shi)).mean() * 100

print(f"alpha-helix R    : {basin(-130,-30,-90,30):.1f}%")
print(f"beta-sheet       : {basin(-180,-90,70,180):.1f}%")
print(f"PPII extended    : {basin(-90,-30,100,180):.1f}%")
print(f"alpha-helix L    : {basin(30,100,-10,90):.1f}%  (sterically forbidden)")
```

Report the basin populations to the user with a one-line interpretation
(e.g. "Dominantly PPII / extended; some alpha-helix; alpha-L below 2%
which means model honours steric exclusion correctly").

### 4. Optional: plot Ramachandran

If user explicitly asks for a plot AND matplotlib is available, save a
2D histogram PNG. Otherwise skip — terminal output is enough.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.hist2d(phi_deg, psi_deg, bins=50, range=[[-180,180],[-180,180]], cmap="viridis")
plt.xlabel("phi (deg)"); plt.ylabel("psi (deg)")
plt.title(f"AlphaDynamics — {seq}")
plt.colorbar(label="count")
plt.savefig("/tmp/<SEQ>_rama.png", dpi=120)
```

### 5. Compare two sequences

If user wants to compare (e.g. wild-type vs mutant), run prediction for
each, then report basin populations side by side. Highlight which basin
shifted most.

## Hard limits — be honest with the user

The user IS the author of this model. Don't oversell. Be honest about:

- **Density only, not kinetics.** Model captures *where* the peptide spends
  time on the Ramachandran torus, not *when* transitions happen. Don't
  claim residence times or transition rates.
- **Trained on 4-98 residues.** Reliability degrades outside this range;
  N>200 is unreliable.
- **Backbone only.** No side-chain rotamers, no Cartesian xyz, no docking.
- **Mean JSD on 4AA test:** 0.196 vs Microsoft Timewarp 0.468 (2.39x
  closer). On longer N=48 peptides JSD ~0.276; on N=98 ~0.389.

## Available models

```
ad_transfer_v2_clean (default)  — main propagator, 78K params
ad_init_full_1477               — von Mises mixture prior, 45K params
```

Use the default unless user requests `ad_init` for prior-only inspection.

## Resources

- Source: https://github.com/krisss0mecom/AlphaDynamics
- PyPI: https://pypi.org/project/alphadynamics/
- Paper v2 (per-system, related work): Zenodo DOI 10.5281/zenodo.19877815

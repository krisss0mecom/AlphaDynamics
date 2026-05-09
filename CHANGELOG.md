# Changelog

All notable changes to AlphaDynamics will be documented in this file.

## v0.4.3 — 2026-05-09 (final Windows hotfix)

### Fixed
- **Real source of `charmap codec cant encode` on Windows**: `->` in `geometry.py:trajectory_to_pdb()` REMARK header. The PDB file was opened without explicit encoding, so Windows used cp1252 (charmap) which doesnt support U+2192. Crash position 43 = exact location of arrow.
- v0.4.2 only fixed CLI prints; geometry.py still wrote -> in the PDB header itself.
- Now: `open(out_pdb, "w", encoding="utf-8")` + ASCII `->` in REMARK.

### Verified
- cp1252 strict-encoding simulation now writes PDB without exception.

## v0.4.2 — 2026-05-09 (Windows compatibility hotfix)

### Fixed
- **Windows charmap (cp1252) crash on PDB generation**. v0.4.1 used Unicode arrow (→) and Å in print statements, which crashed on default Windows console encoding (cp1252). Symptom: `(3D PDB generation skipped: charmap codec cant encode character)`. Result: PDB never written.
- Replaced Unicode arrows with ASCII `->` in interactive output.
- Replaced Å (Angstrom symbol) with `A` in unit labels.
- Replaced Cα with Ca in user-facing prints.
- Added `sys.stdout.reconfigure(encoding="utf-8")` on Windows for safety.

### Notes
- Same v0.4.1 functionality otherwise.
- HTML/PNG plot files are unaffected (plotly/matplotlib write Unicode directly to file, no terminal encoding issue).

## v0.4.1 — 2026-05-09 (same day patch)

### Changed — auto-generate PDB
- `alphadynamics predict` now **automatically writes both `.npz` and `.pdb`** by default.
  Previously you had to run `alphadynamics rebuild` separately. This was confusing UX.
- Interactive prompt also auto-generates PDB after npz.
- New flag `--no-pdb` to skip PDB generation if you only want torsions.
- New flag `--pdb-out` to override PDB output path.
- New flag `--pdb-frames` (default: 50) to subsample trajectory in PDB.

### Notes
- PDB uses ensemble member 0; subsampled to 50 frames for compact file size.
- Diagnostics (Rg, end-to-end) printed automatically.
- Link to live online viewer printed for convenience.

## v0.4.0 — 2026-05-09

### Added — 3D backbone reconstruction
- New module `alphadynamics.geometry`: deterministic NeRF reconstruction
  (Parsons 2005) of backbone heavy atoms (N, Cα, C, O) from torsion
  trajectories with Engh-Huber 1991 standard bond geometry.
- Functions: `torsions_to_backbone()`, `trajectory_to_pdb()`,
  `radius_of_gyration()`, `end_to_end_distance()`,
  `trajectory_diagnostics()`.
- New CLI subcommand `alphadynamics rebuild`:
  ```
  alphadynamics rebuild rollout.npz -s KLVFFAE -o backbone.pdb --diagnostics
  ```
- Output: multi-model PDB with backbone heavy atoms, openable in
  PyMOL / VMD / ChimeraX. Per-frame Rg and end-to-end distance
  diagnostics.
- 13 new unit tests in `tests/test_geometry.py`.
- README section "3D backbone reconstruction" with quickstart.

### Notes
- Backbone-only (no side chains, no hydrogens).
- Torsion errors accumulate along the chain; for long peptides
  (N > ~50) end-to-end displacement may be substantial. Use as
  diagnostic visualization, not as high-resolution structure prediction.
- Defaults: ω = 180° (all-trans). For cis-proline supply per-residue
  `omega_deg` array.
- Terminal angles `phi[0]` and `psi[-1]` are undefined geometrically and
  are replaced with conventional defaults (-60° / +120°).

## v0.3.9 — 2026-05-04
- Robust checkpoint loader with AD-Init seeding
- Scope warning for sequences > 20 aa

## v0.3.0 → v0.3.8
- Open-source release of sequence-only product
- Interactive prompt mode
- HTML Ramachandran plots (plotly)
- Auto-update check on import
- Pre-flight write checks

See git log for details on intermediate versions.

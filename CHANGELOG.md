# Changelog

All notable changes to AlphaDynamics will be documented in this file.

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

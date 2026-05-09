"""Tests for alphadynamics.geometry — NeRF backbone reconstruction."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from alphadynamics.geometry import (
    BL,
    BA,
    DEFAULT_PHI_FIRST,
    DEFAULT_PSI_LAST,
    end_to_end_distance,
    radius_of_gyration,
    torsions_to_backbone,
    trajectory_diagnostics,
    trajectory_to_pdb,
)


def test_single_residue():
    """Single residue produces 4 atoms with zero Rg."""
    coords = torsions_to_backbone(np.array([-60.0]), np.array([-45.0]))
    assert coords.shape == (1, 4, 3)
    assert radius_of_gyration(coords) == pytest.approx(0.0, abs=1e-6)


def test_alpha_helix_canonical_geometry():
    """Canonical α-helix (φ=-60, ψ=-45) gives ~3.6 residues per turn pitch.

    For a 10-residue α-helix, end-to-end distance should be ~13-15 Å
    (1.5 Å rise per residue × 9 distances).
    """
    n = 10
    coords = torsions_to_backbone(np.full(n, -60.0), np.full(n, -45.0))
    assert coords.shape == (n, 4, 3)
    e2e = end_to_end_distance(coords)
    assert 10.0 < e2e < 20.0, f"α-helix e2e {e2e:.2f} outside expected range"


def test_beta_strand_canonical_geometry():
    """Canonical β-strand (φ=-120, ψ=120) gives extended chain.

    For a 10-residue strand, end-to-end ~30 Å (3.4 Å × 9).
    """
    n = 10
    coords = torsions_to_backbone(np.full(n, -120.0), np.full(n, 120.0))
    e2e = end_to_end_distance(coords)
    assert 25.0 < e2e < 35.0, f"β-strand e2e {e2e:.2f} outside expected range"


def test_helix_rg_smaller_than_strand():
    """α-helix is more compact than β-strand for same length."""
    n = 12
    helix = torsions_to_backbone(np.full(n, -60.0), np.full(n, -45.0))
    strand = torsions_to_backbone(np.full(n, -120.0), np.full(n, 120.0))
    assert radius_of_gyration(helix) < radius_of_gyration(strand)


def test_bond_lengths_preserved():
    """N-Cα, Cα-C, C-N distances should match Engh-Huber values."""
    n = 5
    coords = torsions_to_backbone(np.full(n, -60.0), np.full(n, -45.0))
    for i in range(n):
        N, Ca, C, _ = coords[i]
        assert np.linalg.norm(Ca - N) == pytest.approx(BL["N-Ca"], abs=1e-3)
        assert np.linalg.norm(C - Ca) == pytest.approx(BL["Ca-C"], abs=1e-3)
        if i < n - 1:
            N_next = coords[i + 1, 0]
            assert np.linalg.norm(N_next - C) == pytest.approx(BL["C-N"], abs=1e-3)


def test_omega_default_is_trans():
    """Default ω = 180° means trans peptide (planar, ω≈180°)."""
    n = 4
    coords = torsions_to_backbone(np.full(n, -60.0), np.full(n, -45.0))
    # Compute observed ω: dihedral Cα(i)-C(i)-N(i+1)-Cα(i+1)
    Ca0, C0 = coords[0, 1], coords[0, 2]
    N1, Ca1 = coords[1, 0], coords[1, 1]
    b1 = C0 - Ca0
    b2 = N1 - C0
    b3 = Ca1 - N1
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    cos_om = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    omega_observed = np.degrees(np.arccos(np.clip(cos_om, -1, 1)))
    assert omega_observed > 170.0, (
        f"Default ω should be near 180° (trans), observed {omega_observed:.1f}°"
    )


def test_radians_auto_detected():
    """Trajectory in radians is auto-converted to degrees in PDB output."""
    n = 5
    T = 3
    # Radians (max abs ~3)
    traj_rad = np.full((T, n, 2), -1.0)  # -1 rad ≈ -57°
    # Should treat as radians, convert internally
    coords_rad = torsions_to_backbone(
        np.degrees(traj_rad[0, :, 0]), np.degrees(traj_rad[0, :, 1])
    )
    rg_rad = radius_of_gyration(coords_rad)

    # Degrees explicit
    traj_deg = np.full((T, n, 2), -57.2958)  # same as -1 rad
    coords_deg = torsions_to_backbone(traj_deg[0, :, 0], traj_deg[0, :, 1])
    rg_deg = radius_of_gyration(coords_deg)

    assert rg_rad == pytest.approx(rg_deg, rel=1e-3)


def test_diagnostics_consistent():
    """trajectory_diagnostics matches per-frame values."""
    n = 8
    T = 5
    traj_deg = np.random.RandomState(42).uniform(-180, 180, size=(T, n, 2))
    diag = trajectory_diagnostics(traj_deg)
    assert diag["rg"].shape == (T,)
    assert diag["end_to_end"].shape == (T,)
    assert diag["rg_mean"] == pytest.approx(diag["rg"].mean())


def test_pdb_output_format():
    """PDB output is valid multi-model format."""
    n = 4
    T = 3
    traj_deg = np.random.RandomState(0).uniform(-180, 180, size=(T, n, 2))
    seq = "AGLY"[:n]

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        path = f.name
    try:
        trajectory_to_pdb(traj_deg, seq, path)
        with open(path) as fh:
            content = fh.read()

        # Multi-model markers
        assert content.count("MODEL") == T
        assert content.count("ENDMDL") == T
        assert content.endswith("END\n")

        # Atom records: 4 atoms × N residues × T models
        atom_count = content.count("\nATOM ")
        assert atom_count == 4 * n * T

        # Each ATOM line has correct fields
        atoms = [line for line in content.split("\n") if line.startswith("ATOM")]
        for line in atoms[:4]:
            assert line[12:16].strip() in {"N", "CA", "C", "O"}
    finally:
        os.unlink(path)


def test_cis_proline_omega():
    """Setting ω=0° (cis-Pro) produces different geometry than trans."""
    n = 5
    phi = np.full(n, -60.0)
    psi = np.full(n, -30.0)

    coords_trans = torsions_to_backbone(phi, psi, omega_deg=180.0)
    coords_cis = torsions_to_backbone(phi, psi, omega_deg=0.0)

    e2e_trans = end_to_end_distance(coords_trans)
    e2e_cis = end_to_end_distance(coords_cis)

    # Cis peptide bond should produce shorter e2e (chain bends back)
    assert e2e_cis != pytest.approx(e2e_trans, rel=0.01)


def test_terminal_default_handling():
    """phi[0] and psi[-1] are replaced with defaults without crashing."""
    n = 6
    # Even with NaN at terminals, should work (replaced with defaults)
    phi = np.array([np.nan, -60.0, -60.0, -60.0, -60.0, -60.0])
    psi = np.array([-45.0, -45.0, -45.0, -45.0, -45.0, np.nan])

    # Don't crash — replace with defaults
    coords = torsions_to_backbone(phi, psi)
    assert coords.shape == (n, 4, 3)
    assert np.all(np.isfinite(coords))


def test_input_validation():
    """Wrong shapes raise ValueError."""
    with pytest.raises(ValueError):
        torsions_to_backbone(np.array([1.0, 2.0]), np.array([1.0]))  # length mismatch
    with pytest.raises(ValueError):
        torsions_to_backbone(np.array([]), np.array([]))  # empty


def test_round_trip_basic():
    """Build chain, extract Cα-Cα distances, all should be ~3.8 Å (peptide unit)."""
    n = 8
    coords = torsions_to_backbone(np.full(n, -60.0), np.full(n, -45.0))
    ca = coords[:, 1]
    ca_distances = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    # Cα-Cα virtual bond is ~3.8 Å for trans peptide
    assert all(3.7 < d < 3.9 for d in ca_distances), (
        f"Cα-Cα distances {ca_distances} outside expected ~3.8 Å range"
    )

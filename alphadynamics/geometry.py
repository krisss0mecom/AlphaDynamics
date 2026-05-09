"""3D backbone reconstruction from torsion angles.

Deterministic NeRF (Natural Extension Reference Frame) — converts
(phi, psi, omega) per residue + standard backbone bond geometry into
xyz coordinates for backbone heavy atoms (N, Cα, C, O).

This is **post-processing** for AlphaDynamics torsion outputs, not a
structure prediction model. Backbone errors accumulate over long chains;
use as diagnostic visualization, not high-resolution prediction.

References
----------
Parsons et al. 2005, "Practical conversion from torsion space to Cartesian
space for in silico protein synthesis", J. Comput. Chem. 26: 1063-1068.

Engh & Huber 1991, "Accurate bond and angle parameters for X-ray protein
structure refinement", Acta Crystallogr. A 47: 392-400.

Usage
-----
>>> from alphadynamics import predict_torsion_ensemble
>>> from alphadynamics.geometry import trajectory_to_pdb
>>> traj = predict_torsion_ensemble("KLVFFAE", n_ensemble=4, rollout_steps=200)
>>> trajectory_to_pdb(traj[0], "KLVFFAE", "klvffae.pdb")
>>> # Open in PyMOL: pymol klvffae.pdb
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np

# ─── Engh & Huber 1991 standard backbone geometry ──────────────────────────
# Bond lengths (Å)
BL = {
    "N-Ca": 1.458,
    "Ca-C": 1.525,
    "C-N":  1.329,
    "C=O":  1.231,
}
# Bond angles (degrees)
BA = {
    "N-Ca-C": 111.0,   # tau
    "Ca-C-N": 116.2,
    "C-N-Ca": 121.7,
    "O=C-N":  123.0,
    "O=C-Ca": 120.8,
}

# Defaults for terminal residues (where phi_1 / psi_N are undefined)
DEFAULT_PHI_FIRST = -60.0   # α-helix region (canonical default)
DEFAULT_PSI_LAST  = +120.0  # extended region (canonical default)

# Standard one-letter to three-letter AA code (with X = UNK)
_AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    "X": "UNK",
}


def _place_next(
    A: np.ndarray, B: np.ndarray, C: np.ndarray,
    bond_length: float, bond_angle_deg: float, dihedral_deg: float,
) -> np.ndarray:
    """Place atom D given last 3 anchors (A, B, C) and internal coords.

    Parsons 2005 NeRF formulation. Returns position of atom D such that:
      |C-D| = bond_length
      angle(B-C-D) = bond_angle_deg
      dihedral(A-B-C-D) = dihedral_deg

    All angles in degrees.
    """
    angle_rad = np.radians(180.0 - bond_angle_deg)
    dih_rad = np.radians(dihedral_deg)

    bc = (C - B) / np.linalg.norm(C - B)
    ab = B - A
    n = np.cross(ab, bc)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        # collinear — use arbitrary perpendicular axis
        n = np.array([0.0, 0.0, 1.0]) if abs(bc[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n = n - bc * np.dot(n, bc)
        n /= np.linalg.norm(n)
    else:
        n /= n_norm
    m = np.cross(n, bc)

    D_local = bond_length * np.array([
        np.cos(angle_rad),
        np.cos(dih_rad) * np.sin(angle_rad),
        np.sin(dih_rad) * np.sin(angle_rad),
    ])
    R = np.column_stack([bc, m, n])
    return C + R @ D_local


def _place_oxygen(
    N_next: np.ndarray, Ca_curr: np.ndarray, C_curr: np.ndarray,
) -> np.ndarray:
    """Place carbonyl O atom: planar with N-Ca-C, opposite from N(next).

    Geometry: O lies in the peptide plane, on the opposite side of the
    C-Ca/C-N(next) bisector. Distance |C=O| = 1.231 Å.
    """
    inward = (Ca_curr - C_curr) + (N_next - C_curr)
    inward /= np.linalg.norm(inward)
    return C_curr - BL["C=O"] * inward


def torsions_to_backbone(
    phi_deg: np.ndarray,
    psi_deg: np.ndarray,
    omega_deg: Union[np.ndarray, float, None] = None,
) -> np.ndarray:
    """Reconstruct backbone heavy atoms from torsion angles.

    Parameters
    ----------
    phi_deg : (N,) array
        Backbone phi angles in degrees. ``phi_deg[0]`` is undefined (no
        preceding C atom); replaced with :data:`DEFAULT_PHI_FIRST`.
    psi_deg : (N,) array
        Backbone psi angles in degrees. ``psi_deg[-1]`` is undefined (no
        following N atom); replaced with :data:`DEFAULT_PSI_LAST`.
    omega_deg : (N,) array, scalar, or None
        Peptide bond omega angles. ``None`` defaults to all-trans (180°).
        For cis-proline, supply a per-residue array with appropriate 0°.

    Returns
    -------
    coords : (N, 4, 3) array
        For each residue i, ``coords[i]`` contains backbone heavy atoms in
        order ``[N, Cα, C, O]`` as xyz in Å.

    Notes
    -----
    Errors in (phi, psi) accumulate along the chain. For long chains
    (N > ~50) end-to-end displacement may be substantial even for small
    per-residue errors. This is intrinsic to torsion-space reconstruction;
    it is not a flaw of the algorithm but a property of the geometry.

    Use as diagnostic visualization (does the model produce a
    physically sensible chain?), not as ground-truth structure.
    """
    phi_deg = np.asarray(phi_deg, dtype=np.float64)
    psi_deg = np.asarray(psi_deg, dtype=np.float64)
    n_res = len(phi_deg)
    if len(psi_deg) != n_res:
        raise ValueError(f"phi_deg and psi_deg length mismatch: {n_res} vs {len(psi_deg)}")
    if n_res < 1:
        raise ValueError("Need at least 1 residue")

    if omega_deg is None:
        omega_deg = np.full(n_res, 180.0)
    elif np.isscalar(omega_deg):
        omega_deg = np.full(n_res, float(omega_deg))
    else:
        omega_deg = np.asarray(omega_deg, dtype=np.float64)
        if len(omega_deg) != n_res:
            raise ValueError(
                f"omega_deg length mismatch: {len(omega_deg)} vs {n_res}"
            )

    # Replace undefined terminal angles
    phi = phi_deg.copy()
    psi = psi_deg.copy()
    phi[0] = DEFAULT_PHI_FIRST
    psi[-1] = DEFAULT_PSI_LAST

    coords = np.zeros((n_res, 4, 3), dtype=np.float64)

    # Place residue 0 in canonical frame
    coords[0, 0] = [0.0, 0.0, 0.0]                           # N₁
    coords[0, 1] = [BL["N-Ca"], 0.0, 0.0]                    # Cα₁
    angle = np.radians(180.0 - BA["N-Ca-C"])
    coords[0, 2] = coords[0, 1] + BL["Ca-C"] * np.array(
        [np.cos(angle), np.sin(angle), 0.0]
    )                                                          # C₁

    if n_res == 1:
        # Single residue: use default psi to place virtual N for O placement
        fake_N_next = _place_next(
            coords[0, 0], coords[0, 1], coords[0, 2],
            BL["C-N"], BA["Ca-C-N"], psi[0],
        )
        coords[0, 3] = _place_oxygen(fake_N_next, coords[0, 1], coords[0, 2])
        return coords

    for i in range(1, n_res):
        N_prev, Ca_prev, C_prev, _ = coords[i - 1]

        # Place N_i: dihedral = psi_{i-1}
        coords[i, 0] = _place_next(
            N_prev, Ca_prev, C_prev,
            BL["C-N"], BA["Ca-C-N"], psi[i - 1],
        )
        # Place Cα_i: dihedral = omega_i (peptide bond, ~180° trans)
        coords[i, 1] = _place_next(
            Ca_prev, C_prev, coords[i, 0],
            BL["N-Ca"], BA["C-N-Ca"], omega_deg[i],
        )
        # Place C_i: dihedral = phi_i
        coords[i, 2] = _place_next(
            C_prev, coords[i, 0], coords[i, 1],
            BL["Ca-C"], BA["N-Ca-C"], phi[i],
        )
        # Place O_{i-1}: now we know N_i, can place carbonyl in peptide plane
        coords[i - 1, 3] = _place_oxygen(coords[i, 0], coords[i - 1, 1], coords[i - 1, 2])

    # Last residue's O — use psi to extrapolate virtual next N
    fake_N_next = _place_next(
        coords[-1, 0], coords[-1, 1], coords[-1, 2],
        BL["C-N"], BA["Ca-C-N"], psi[-1],
    )
    coords[-1, 3] = _place_oxygen(fake_N_next, coords[-1, 1], coords[-1, 2])

    return coords


def trajectory_to_pdb(
    phi_psi_traj: np.ndarray,
    sequence: str,
    out_pdb: str,
    omega_deg: Union[np.ndarray, float, None] = None,
    chain_id: str = "A",
) -> None:
    """Write multi-model PDB from torsion trajectory.

    Parameters
    ----------
    phi_psi_traj : (T, N, 2) array
        Trajectory of backbone torsions: ``T`` frames × ``N`` residues ×
        ``[phi, psi]``. Values may be in radians (max abs < 7) or degrees.
    sequence : str
        One-letter AA string of length N. ``X`` for unknown.
    out_pdb : str
        Output multi-model PDB path.
    omega_deg : array, scalar, or None
        Peptide bond ω. Defaults to 180° (all-trans).
    chain_id : str
        Single-character PDB chain identifier.

    Notes
    -----
    Output contains backbone heavy atoms (N, Cα, C, O) only. No side
    chains, no hydrogens. Suitable for visualizing dynamics in PyMOL,
    VMD, ChimeraX or similar; not for docking or all-atom analysis.
    """
    traj = np.asarray(phi_psi_traj, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(
            f"Expected (T, N, 2) trajectory array, got shape {traj.shape}"
        )

    # Auto-detect radians vs degrees by max absolute value
    if np.abs(traj).max() < 7.0:
        traj_deg = np.degrees(traj)
    else:
        traj_deg = traj

    T, N, _ = traj_deg.shape
    if len(sequence) != N:
        raise ValueError(
            f"sequence length ({len(sequence)}) does not match N residues ({N})"
        )

    atom_names = ["N", "CA", "C", "O"]

    with open(out_pdb, "w", encoding="utf-8") as f:
        f.write(f"REMARK    AlphaDynamics torsion trajectory -> 3D backbone\n")
        f.write(f"REMARK    Sequence: {sequence}\n")
        f.write(f"REMARK    Frames:   {T}\n")
        f.write(f"REMARK    Residues: {N}\n")
        f.write(f"REMARK    Atoms per residue: 4 (N, CA, C, O backbone heavy)\n")
        f.write(f"REMARK    NeRF reconstruction (Parsons 2005)\n")
        f.write(f"REMARK    Standard geometry: Engh & Huber 1991\n")
        for t in range(T):
            f.write(f"MODEL     {t + 1:>4}\n")
            coords = torsions_to_backbone(
                traj_deg[t, :, 0], traj_deg[t, :, 1], omega_deg=omega_deg
            )
            atom_id = 1
            for i, aa in enumerate(sequence):
                aa3 = _AA3.get(aa.upper(), "UNK")
                for j, name in enumerate(atom_names):
                    xyz = coords[i, j]
                    elem = name[0]  # element from atom name
                    f.write(
                        f"ATOM  {atom_id:>5} {name:<4}{aa3:>3} {chain_id:>1}{i + 1:>4}    "
                        f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
                        f"  1.00  0.00          {elem:>2}\n"
                    )
                    atom_id += 1
            f.write("ENDMDL\n")
        f.write("END\n")


def radius_of_gyration(coords: np.ndarray) -> float:
    """Radius of gyration (Rg) of Cα atoms only.

    Parameters
    ----------
    coords : (N, 4, 3) array
        Backbone coordinates as returned by :func:`torsions_to_backbone`.
        Cα is at index 1 along atom axis.

    Returns
    -------
    rg : float
        Rg in Å.
    """
    ca = np.asarray(coords)[:, 1, :]
    com = ca.mean(axis=0)
    return float(np.sqrt(((ca - com) ** 2).sum(axis=1).mean()))


def end_to_end_distance(coords: np.ndarray) -> float:
    """End-to-end distance: ``|Cα_N - Cα_1|``.

    Parameters
    ----------
    coords : (N, 4, 3) array

    Returns
    -------
    distance : float
        End-to-end distance of Cα atoms in Å.
    """
    ca = np.asarray(coords)[:, 1, :]
    return float(np.linalg.norm(ca[-1] - ca[0]))


def trajectory_diagnostics(
    phi_psi_traj: np.ndarray,
    omega_deg: Union[np.ndarray, float, None] = None,
) -> dict:
    """Per-frame Rg and end-to-end distance over a trajectory.

    Parameters
    ----------
    phi_psi_traj : (T, N, 2) array
        Trajectory in radians or degrees.
    omega_deg : array, scalar, or None
        Peptide bond omegas, default 180°.

    Returns
    -------
    diag : dict
        Keys: ``rg`` (T,), ``end_to_end`` (T,), ``rg_mean``, ``rg_std``,
        ``end_to_end_mean``, ``end_to_end_std``.
    """
    traj = np.asarray(phi_psi_traj, dtype=np.float64)
    if np.abs(traj).max() < 7.0:
        traj = np.degrees(traj)

    T = traj.shape[0]
    rg_arr = np.zeros(T)
    e2e_arr = np.zeros(T)
    for t in range(T):
        coords = torsions_to_backbone(traj[t, :, 0], traj[t, :, 1], omega_deg=omega_deg)
        rg_arr[t] = radius_of_gyration(coords)
        e2e_arr[t] = end_to_end_distance(coords)

    return {
        "rg": rg_arr,
        "end_to_end": e2e_arr,
        "rg_mean": float(rg_arr.mean()),
        "rg_std": float(rg_arr.std()),
        "end_to_end_mean": float(e2e_arr.mean()),
        "end_to_end_std": float(e2e_arr.std()),
    }


__all__ = [
    "BL",
    "BA",
    "DEFAULT_PHI_FIRST",
    "DEFAULT_PSI_LAST",
    "torsions_to_backbone",
    "trajectory_to_pdb",
    "radius_of_gyration",
    "end_to_end_distance",
    "trajectory_diagnostics",
]

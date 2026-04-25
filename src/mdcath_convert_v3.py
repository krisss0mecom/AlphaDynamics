"""mdCATH conversion v3 — uses embedded pdbProteinAtoms.

Clean, robust: PDB topology comes from the H5 file itself (CHARMM36m-prepared).
No chain matching, no RCSB fetch, no mismatches.
"""
import argparse
import glob
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import mdtraj as md

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "mdcath_raw"
OUT_DIR = ROOT / "mdcath_real_data" / "mdcath_348K"

TWO_PI = 2 * np.pi
def wrap(a): return (a + np.pi) % TWO_PI - np.pi


def compute_aligned_dihedrals(traj):
    """Return (T, N_common, 2) phi/psi aligned by residue index."""
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)
    top = traj.topology
    phi_res = np.array([top.atom(row[2]).residue.index for row in phi_idx])
    psi_res = np.array([top.atom(row[1]).residue.index for row in psi_idx])
    common = np.intersect1d(phi_res, psi_res)
    if len(common) == 0:
        return None, None
    phi_sel = np.array([np.where(phi_res == r)[0][0] for r in common])
    psi_sel = np.array([np.where(psi_res == r)[0][0] for r in common])
    arr = np.stack([phi[:, phi_sel], psi[:, psi_sel]], axis=-1).astype(np.float32)
    return arr, common


def process(h5_file, out_dir, force=False):
    domain_id = os.path.basename(h5_file).replace("mdcath_dataset_", "").replace(".h5", "")
    out_file = out_dir / f"{domain_id}_dihedrals.npz"
    if out_file.exists() and not force:
        return "SKIP"

    with h5py.File(h5_file, 'r') as h:
        group = h[domain_id]
        if 'pdbProteinAtoms' not in group:
            return f"NO_PDB"
        pdb_bytes = group['pdbProteinAtoms'][()]
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            tf.write(pdb_bytes)
            pdb_path = tf.name

        try:
            top = md.load(pdb_path).topology
        finally:
            os.unlink(pdb_path)

        n_atoms = top.n_atoms
        n_residues = top.n_residues

        # Collect all replicas at 348K
        if '348' not in group:
            return "NO_348K"
        all_coords = []
        for rep_id in sorted(group['348'].keys()):
            coords = group['348'][rep_id]['coords'][:]  # (T, N_atoms, 3) in A
            all_coords.append(coords)
        combined = np.concatenate(all_coords, axis=0)

        if combined.shape[1] != n_atoms:
            return f"ATOM_MISMATCH_{combined.shape[1]}_vs_{n_atoms}"

        traj = md.Trajectory(combined / 10.0, top)  # A -> nm
        arr, residues = compute_aligned_dihedrals(traj)
        if arr is None or arr.shape[1] < 3:
            return "TOO_FEW_DIHEDRALS"

        N_frames = len(arr)
        split = int(0.8 * N_frames)
        diff = wrap(np.diff(arr, axis=0))
        step = np.sqrt((diff ** 2).sum(-1)).mean()
        id_err = np.sqrt((diff ** 2).mean())

        np.savez(out_file, train=arr[:split], val=arr[split:], N=arr.shape[1],
                 domain_id=domain_id, n_residues=n_residues, n_atoms=n_atoms,
                 residue_indices=residues,
                 dihedral_alignment="common_residue_index",
                 source_h5=str(h5_file),
                 mean_step_deg=float(np.degrees(step)),
                 identity_deg=float(np.degrees(id_err)))
        return f"OK N={arr.shape[1]}, {N_frames} frames, step {np.degrees(step):.1f}°"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_dir", default=str(BENCH_DIR))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--force", action="store_true",
                        help="Regenerate output npz files even if they already exist")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for h5_file in sorted(glob.glob(str(bench_dir / "data" / "*.h5"))):
        domain = os.path.basename(h5_file).replace("mdcath_dataset_", "").replace(".h5", "")
        try:
            status = process(h5_file, out_dir, force=args.force)
        except Exception as e:
            status = f"ERROR: {type(e).__name__}: {e}"
        print(f"{domain}: {status}")
        results[domain] = status

    ok = sum(1 for s in results.values() if s.startswith("OK") or s == "SKIP")
    print(f"\n=== {ok}/{len(results)} domains ready ===")


if __name__ == "__main__":
    main()

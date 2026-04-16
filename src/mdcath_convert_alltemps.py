"""mdCATH v5 — convert ALL 5 temperatures with aligned phi/psi indexing.

Output: one npz per (domain, temperature) pair.
Phi and psi aligned by residue index (mdtraj atom_indices → residue.index).
"""
import glob, os, tempfile, math
import h5py
import numpy as np
import mdtraj as md

BENCH_DIR = "/home/krisss0/AlphaDynamics/mdcath_raw"
OUT_DIR = "/home/krisss0/AlphaDynamics/mdcath_real_data/mdcath_alltemps"
os.makedirs(OUT_DIR, exist_ok=True)

TWO_PI = 2 * np.pi
TEMPERATURES = ['320', '348', '379', '413', '450']


def wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


def compute_aligned_dihedrals(traj):
    """Aligned phi/psi by common residue index."""
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)
    top = traj.topology
    # CA atom is index 2 in phi row (C_{i-1}, N_i, CA_i, C_i)
    # CA atom is index 1 in psi row (N_i, CA_i, C_i, N_{i+1})
    phi_res = np.array([top.atom(row[2]).residue.index for row in phi_idx])
    psi_res = np.array([top.atom(row[1]).residue.index for row in psi_idx])
    common = np.intersect1d(phi_res, psi_res)
    if len(common) == 0:
        return None, None
    phi_sel = np.array([np.where(phi_res == r)[0][0] for r in common])
    psi_sel = np.array([np.where(psi_res == r)[0][0] for r in common])
    arr = np.stack([phi[:, phi_sel], psi[:, psi_sel]], axis=-1).astype(np.float32)
    return arr, common


def process(h5_file):
    domain_id = os.path.basename(h5_file).replace("mdcath_dataset_", "").replace(".h5", "")
    with h5py.File(h5_file, 'r') as h:
        group = h[domain_id]
        if 'pdbProteinAtoms' not in group:
            return f"NO_PDB"
        pdb_bytes = group['pdbProteinAtoms'][()]
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            tf.write(pdb_bytes); pdb_path = tf.name
        try:
            top = md.load(pdb_path).topology
        finally:
            os.unlink(pdb_path)

        for T in TEMPERATURES:
            out_file = f"{OUT_DIR}/{domain_id}_T{T}_dihedrals.npz"
            if os.path.exists(out_file):
                print(f"  {domain_id}@{T}K: cached")
                continue
            if T not in group:
                print(f"  {domain_id}@{T}K: NO TEMP")
                continue
            all_coords = []
            for rep_id in sorted(group[T].keys()):
                all_coords.append(group[T][rep_id]['coords'][:])
            combined = np.concatenate(all_coords, axis=0)
            if combined.shape[1] != top.n_atoms:
                print(f"  {domain_id}@{T}K: ATOM MISMATCH")
                continue
            traj = md.Trajectory(combined / 10.0, top)
            arr, residues = compute_aligned_dihedrals(traj)
            if arr is None:
                print(f"  {domain_id}@{T}K: NO COMMON DIHEDRALS")
                continue
            N_frames = len(arr)
            split = int(0.8 * N_frames)
            diff = wrap(np.diff(arr, axis=0))
            step = np.sqrt((diff ** 2).sum(-1)).mean()
            id_err = np.sqrt((diff ** 2).mean())
            np.savez(out_file,
                     train=arr[:split], val=arr[split:],
                     N=arr.shape[1], domain_id=domain_id, temperature=int(T),
                     residue_indices=residues,
                     mean_step_deg=float(np.degrees(step)),
                     identity_deg=float(np.degrees(id_err)))
            print(f"  {domain_id}@{T}K: OK N={arr.shape[1]} frames={N_frames} step={np.degrees(step):.1f}° id={np.degrees(id_err):.1f}°")
    return "DONE"


if __name__ == "__main__":
    files = sorted(glob.glob(f"{BENCH_DIR}/data/*.h5"))
    print(f"Processing {len(files)} domains × 5 temperatures = {len(files) * 5} outputs\n")
    for f in files:
        print(f"\n=== {os.path.basename(f)} ===")
        process(f)
    # Summary
    print("\n=== SUMMARY ===")
    outputs = sorted(glob.glob(f"{OUT_DIR}/*.npz"))
    print(f"Total npz files: {len(outputs)}")
    for o in outputs[:15]:
        d = np.load(o)
        print(f"  {os.path.basename(o)}: N={d['N']}, step={float(d['mean_step_deg']):.1f}°, id={float(d['identity_deg']):.1f}°")

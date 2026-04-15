"""mdCATH conversion v3 — uses embedded pdbProteinAtoms.

Clean, robust: PDB topology comes from the H5 file itself (CHARMM36m-prepared).
No chain matching, no RCSB fetch, no mismatches.
"""
import glob, os, tempfile
import h5py
import numpy as np
import mdtraj as md

BENCH_DIR = "/root/mdcath_bench"
OUT_DIR = "/root/fizyka_bialek_claude/chain/real_data/mdcath"
os.makedirs(OUT_DIR, exist_ok=True)

TWO_PI = 2 * np.pi
def wrap(a): return (a + np.pi) % TWO_PI - np.pi


def process(h5_file):
    domain_id = os.path.basename(h5_file).replace("mdcath_dataset_", "").replace(".h5", "")
    out_file = f"{OUT_DIR}/{domain_id}_dihedrals.npz"
    if os.path.exists(out_file):
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
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)
        n = min(phi.shape[1], psi.shape[1])
        if n < 3:
            return "TOO_FEW_DIHEDRALS"
        arr = np.stack([phi[:, :n], psi[:, :n]], axis=-1).astype(np.float32)

        N_frames = len(arr)
        split = int(0.8 * N_frames)
        diff = wrap(np.diff(arr, axis=0))
        step = np.sqrt((diff ** 2).sum(-1)).mean()
        id_err = np.sqrt((diff ** 2).mean())

        np.savez(out_file, train=arr[:split], val=arr[split:], N=arr.shape[1],
                 domain_id=domain_id, n_residues=n_residues, n_atoms=n_atoms,
                 mean_step_deg=float(np.degrees(step)),
                 identity_deg=float(np.degrees(id_err)))
        return f"OK N={arr.shape[1]}, {N_frames} frames, step {np.degrees(step):.1f}°"


results = {}
for h5_file in sorted(glob.glob(f"{BENCH_DIR}/data/*.h5")):
    domain = os.path.basename(h5_file).replace("mdcath_dataset_", "").replace(".h5", "")
    try:
        status = process(h5_file)
    except Exception as e:
        status = f"ERROR: {type(e).__name__}: {e}"
    print(f"{domain}: {status}")
    results[domain] = status

# Summary
ok = sum(1 for s in results.values() if s.startswith("OK") or s == "SKIP")
print(f"\n=== {ok}/{len(results)} domains ready ===")

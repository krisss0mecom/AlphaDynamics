# Data

Raw MD trajectories are not committed. mdCATH HDF5 files are typically on the
order of hundreds of MB per domain; the full mdCATH corpus is much larger than
this repository should vendor.

## mdCATH (primary benchmark source)

**Dataset:** Mirarchi, A., Giorgino, T. & De Fabritiis, G.
"mdCATH: A Large-Scale MD Dataset for Data-Driven Computational Biophysics."
Sci Data 11, 1299 (2024). DOI: 10.1038/s41597-024-04140-z

**Repository:** `compsciencelab/mdCATH` on Hugging Face

**Protocol:** CHARMM36m force field + TIP3P water, 5 temperatures (320, 348,
379, 413, 450 K), 5 replicas per temperature, ~440 frames per replica.

**Download individual domains:**

```python
from huggingface_hub import hf_hub_download

domain = "1a92A00"  # CATH domain ID
path = hf_hub_download(
    "compsciencelab/mdCATH",
    f"data/mdcath_dataset_{domain}.h5",
    repo_type="dataset",
)
```

Each HDF5 file contains `pdbProteinAtoms` (embedded PDB topology),
`coords`, `forces`, `box`, `dssp`, `rmsd`, `rmsf`, `gyrationRadius` for
each temperature × replica.

**Audited domains used in the v1 manuscript:** 20 aligned N=48 domains and
20 aligned N=98 domains at 348 K, plus matching all-temperature files for the
rollout/free-energy audits. The concrete domain IDs are listed in
`results/mdcath_aligned20_4000step_cpu.md` and
`results/mdcath_aligned20_n100_4000step_gpu.md`.

## File format we produce

After `src/mdcath_convert_v3.py`:

```
mdcath_real_data/mdcath_348K/{domain_id}_dihedrals.npz
  train:       (T_train, N_residues_with_both_phi_psi, 2) float32
  val:         (T_val, N, 2)
  N:           int
  domain_id:   str
  n_residues:  int
  n_atoms:     int
  residue_indices: int array, common residues used for aligned phi/psi pairs
  dihedral_alignment: "common_residue_index"
  source_h5:    source HDF5 path
  mean_step_deg: float
  identity_deg:  float
```

`phi` and `psi` are aligned by residue index. Legacy `.npz` files without
`dihedral_alignment=common_residue_index` should not be used for publication
benchmarks unless explicitly marked as historical.

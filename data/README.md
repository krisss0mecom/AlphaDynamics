# Data

Raw MD trajectories are not committed (200 MB per file × 37 files = 7.4 GB).

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

**50-residue domains used in our benchmark:** 37 domains from the small
end of mdCATH's size distribution. See `src/mdcath_convert_v3.py` and
`src/mdcath_benchmark.py` for the domain list.

## File format we produce

After `src/mdcath_convert_v3.py`:

```
real_data/mdcath/{domain_id}_dihedrals.npz
  train:       (T_train, N_residues_with_both_phi_psi, 2) float32
  val:         (T_val, N, 2)
  N:           int
  domain_id:   str
  n_residues:  int
  n_atoms:     int
  mean_step_deg: float
  identity_deg:  float
```

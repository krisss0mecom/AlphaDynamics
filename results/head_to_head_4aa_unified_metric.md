# Head-to-head AD vs Timewarp on 4AA-large/test — UNIFIED METRIC

Both AlphaDynamics and Microsoft Timewarp 2500-step rollouts evaluated
under the same canonical Ramachandran JSD:

- GT histogram: **held-out val only** (no train leakage)
- bins: 36 per axis
- smoothing: **none** (raw counts)
- per-residue 2D JSD averaged across residues

This corrects the v2 Table 4 inconsistency where AD used smoothed
train+val GT (paper v2 commit fb355be) while Timewarp used raw val GT.

| Peptide | AD JSD | Timewarp JSD | TW / AD |
|---|---:|---:|---:|
| AAAY | **0.1392** | 0.5226 | 3.75× |
| AACE | **0.2012** | 0.2986 | 1.48× |
| AAEW | **0.1545** | 0.5825 | 3.77× |
| **Mean** | **0.1649** | **0.4679** | **2.84×** |

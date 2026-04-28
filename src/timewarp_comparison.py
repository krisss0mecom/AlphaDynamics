"""Utilities for an AlphaDynamics-vs-Timewarp dataset comparison.

This script does not run the Microsoft Timewarp model. It prepares an
apples-to-apples data-side comparison by converting a small, explicit subset of
the public Timewarp Hugging Face dataset into the same aligned phi/psi npz
format used by the AlphaDynamics mdCATH audits.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import mdtraj as md
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "microsoft/timewarp"
REPO_TYPE = "dataset"


def scalar_to_str(value):
    if hasattr(value, "shape") and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def wrap(angles):
    return angles - 2.0 * math.pi * np.round(angles / (2.0 * math.pi))


def domain_id_from_arrays_path(path):
    name = Path(path).name
    return name.removesuffix("-traj-arrays.npz")


def arrays_path(dataset, split, domain):
    return f"{dataset}/{split}/{domain}-traj-arrays.npz"


def pdb_path(dataset, split, domain):
    return f"{dataset}/{split}/{domain}-traj-state0.pdb"


def time_path(dataset, split, domain):
    return f"{dataset}/{split}/{domain}-traj-time.npy"


def list_domains(dataset, split, limit=0):
    prefix = f"{dataset}/{split}/"
    files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    domains = []
    for filename in files:
        if filename.startswith(prefix) and filename.endswith("-traj-arrays.npz"):
            domains.append(domain_id_from_arrays_path(filename))
    domains = sorted(domains)
    if limit > 0:
        domains = domains[:limit]
    return domains


def download_file(filename, cache_dir=None, local_dir=None, dry_run=False):
    if dry_run:
        print(f"[dry-run] would download {filename}")
        return None
    return hf_hub_download(
        REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )


def compute_aligned_dihedrals(pdb_file, arrays_file, max_frames=0, stride=1):
    arrays = np.load(arrays_file)
    positions = arrays["positions"]
    steps = arrays["step"] if "step" in arrays else np.arange(len(positions))
    energies = arrays["energies"] if "energies" in arrays else None

    if stride > 1:
        positions = positions[::stride]
        steps = steps[::stride]
        if energies is not None:
            energies = energies[::stride]
    if max_frames > 0:
        positions = positions[:max_frames]
        steps = steps[:max_frames]
        if energies is not None:
            energies = energies[:max_frames]

    topology = md.load_pdb(pdb_file).topology
    traj = md.Trajectory(positions.astype(np.float32), topology)
    phi_indices, phi = md.compute_phi(traj)
    psi_indices, psi = md.compute_psi(traj)

    phi_residues = np.array([topology.atom(int(row[1])).residue.index for row in phi_indices], dtype=np.int64)
    psi_residues = np.array([topology.atom(int(row[1])).residue.index for row in psi_indices], dtype=np.int64)
    common = np.array(sorted(set(phi_residues.tolist()) & set(psi_residues.tolist())), dtype=np.int64)
    if len(common) == 0:
        raise ValueError(f"No common phi/psi residue indices for {pdb_file}")

    phi_lookup = {int(res): i for i, res in enumerate(phi_residues)}
    psi_lookup = {int(res): i for i, res in enumerate(psi_residues)}
    phi_cols = [phi_lookup[int(res)] for res in common]
    psi_cols = [psi_lookup[int(res)] for res in common]
    torsions = np.stack([phi[:, phi_cols], psi[:, psi_cols]], axis=-1).astype(np.float32)
    return torsions, common, steps, energies


def split_train_val(torsions, train_frac):
    split_idx = int(len(torsions) * train_frac)
    split_idx = max(1, min(split_idx, len(torsions) - 1))
    return torsions[:split_idx], torsions[split_idx:]


def identity_deg(val):
    flat = val.reshape(val.shape[0], -1)
    diff = wrap(flat[:-1] - flat[1:])
    return float(np.degrees(np.sqrt((diff**2).mean(axis=-1)).mean()))


def convert_domain(dataset, split, domain, out_dir, cache_dir, local_dir, max_frames, stride, train_frac, dry_run=False):
    arr_rel = arrays_path(dataset, split, domain)
    pdb_rel = pdb_path(dataset, split, domain)
    if dry_run:
        print(f"[dry-run] convert {dataset}/{split}/{domain}")
        print(f"[dry-run] arrays: {arr_rel}")
        print(f"[dry-run] pdb:    {pdb_rel}")
        return None

    arrays_file = download_file(arr_rel, cache_dir=cache_dir, local_dir=local_dir)
    pdb_file = download_file(pdb_rel, cache_dir=cache_dir, local_dir=local_dir)
    torsions, residue_indices, steps, energies = compute_aligned_dihedrals(
        pdb_file,
        arrays_file,
        max_frames=max_frames,
        stride=stride,
    )
    train, val = split_train_val(torsions, train_frac)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{domain}_dihedrals.npz"
    payload = {
        "domain_id": domain,
        "source_dataset": "microsoft/timewarp",
        "timewarp_dataset": dataset,
        "timewarp_split": split,
        "N": np.array(torsions.shape[1], dtype=np.int64),
        "train": train,
        "val": val,
        "residue_indices": residue_indices,
        "dihedral_alignment": "common_residue_index",
        "identity_deg": np.array(identity_deg(val), dtype=np.float64),
        "source_arrays": arr_rel,
        "source_pdb": pdb_rel,
        "stride": np.array(stride, dtype=np.int64),
        "max_frames": np.array(max_frames, dtype=np.int64),
        "timewarp_steps": steps,
    }
    if energies is not None:
        payload["energies"] = energies
    np.savez_compressed(out_path, **payload)
    return {
        "domain_id": domain,
        "path": str(out_path),
        "N": int(torsions.shape[1]),
        "frames": int(len(torsions)),
        "train": int(len(train)),
        "val": int(len(val)),
        "identity_deg": identity_deg(val),
        "residue_indices": residue_indices.tolist(),
    }


def write_manifest(rows, out_dir, name):
    manifest_path = out_dir / name
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    return manifest_path


def cmd_list(args):
    domains = list_domains(args.dataset, args.split, args.limit)
    print(f"dataset={args.dataset} split={args.split} domains={len(domains)}")
    for domain in domains:
        print(domain)


def cmd_convert(args):
    if args.domains:
        domains = args.domains
    else:
        domains = list_domains(args.dataset, args.split, args.max_domains)
    if args.max_domains > 0:
        domains = domains[:args.max_domains]
    if not domains:
        raise ValueError("No domains selected")

    out_dir = Path(args.out_dir)
    rows = []
    for domain in domains:
        row = convert_domain(
            args.dataset,
            args.split,
            domain,
            out_dir,
            args.cache_dir,
            args.local_dir,
            args.max_frames,
            args.stride,
            args.train_frac,
            dry_run=args.dry_run,
        )
        if row is not None:
            rows.append(row)
            print(
                f"{domain}: N={row['N']} frames={row['frames']} "
                f"train={row['train']} val={row['val']} identity={row['identity_deg']:.1f} deg"
            )

    if args.dry_run:
        return
    manifest_path = write_manifest(rows, out_dir, "timewarp_manifest.json")
    print(f"Wrote {manifest_path}")


def cmd_plan(args):
    rows = {
        "recommended_test": "AlphaDynamics per-system NLL on Timewarp tetrapeptide torsions",
        "dataset": args.dataset,
        "split": args.split,
        "domains": args.domains or list_domains(args.dataset, args.split, args.max_domains),
        "steps_after_convert": [
            "alphadynamics timewarp-comparison convert ...",
            "alphadynamics train --data-dir <converted_out_dir> --out-prefix timewarp_subset_nll --steps 4000 --batch 512 --device auto",
            "alphadynamics report --nll-prefix timewarp_subset_nll ...",
        ],
        "not_claimed": [
            "This does not run the Timewarp checkpoint.",
            "This is a shared-dataset AlphaDynamics audit, not a Cartesian Timewarp-vs-AlphaDynamics sampler comparison.",
        ],
    }
    print(json.dumps(rows, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare AlphaDynamics comparison on the public Timewarp dataset")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["list", "plan"]:
        p = sub.add_parser(name)
        p.add_argument("--dataset", default="4AA-large")
        p.add_argument("--split", default="test")
        p.add_argument("--domains", nargs="*", default=None)
        p.add_argument("--max-domains", type=int, default=3)
        p.add_argument("--limit", type=int, default=20)
        p.set_defaults(func=cmd_list if name == "list" else cmd_plan)

    p = sub.add_parser("convert")
    p.add_argument("--dataset", default="4AA-large")
    p.add_argument("--split", default="test")
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--max-domains", type=int, default=3)
    p.add_argument("--out-dir", default=str(ROOT / "timewarp_real_data" / "4AA-large_test"))
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--local-dir", default=None)
    p.add_argument("--max-frames", type=int, default=2500)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_convert)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

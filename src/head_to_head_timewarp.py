"""
Head-to-head: Microsoft Timewarp (transferable, pretrained on 4AA-big2)
vs AlphaDynamics (per-system, trained on each peptide's train split).

Protocol per peptide ∈ {AAAY, AACE, AAEW} from microsoft/timewarp 4AA-large/test:
  1. Load full Cartesian trajectory (positions+velocities) from HF.
  2. Compute ground-truth phi/psi histogram from validation slice (last 20%).
  3. Timewarp rollout: autoregressive sample 2500 frames from initial state
     (step_width=100000 ≈ 100 ns leap per sample). Convert to phi/psi.
  4. AlphaDynamics rollout: train per-system on train split (80%) phi/psi,
     rollout 2500 steps on val initial conditions, convert to histogram.
  5. JSD per residue and per peptide vs ground truth.
  6. Compare TimewarpJSD vs AlphaDynamicsJSD per peptide and on average.

Outputs JSON + Markdown audit table to AlphaDynamics/results/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# --- PATH + STUBS for training-only dependencies ------------------------------
TIMEWARP_REPO = os.environ.get("TIMEWARP_REPO", "/workspace/timewarp")
sys.path.insert(0, TIMEWARP_REPO)
sys.path.insert(0, str(Path(TIMEWARP_REPO).parent))


def _stub(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_stub("deepspeed", DeepSpeedEngine=type("DeepSpeedEngine", (), {}))
_stub("deepspeed.runtime")
_stub("deepspeed.runtime.dataloader",
       DeepSpeedDataLoader=type("DeepSpeedDataLoader", (), {}))
_stub("deepspeed.runtime.lr_schedules")

for m in ("pymol2", "nglview", "amlt"):
    _stub(m)
# lmdb is a real lightweight package — pip install lmdb (do NOT stub).

_stub("bgflow")
_stub("bgflow.distribution")
_stub("bgflow.distribution.energy")
_bg = _stub("bgflow.distribution.energy.base")
_bg.Energy = type("Energy", (), {})
_bg._evaluate_bridge_energy = lambda *a, **k: None

import torch  # noqa: E402
import mdtraj as md  # noqa: E402

TORCH_LOAD_KW = dict(map_location="cpu", weights_only=False)


def make_omegaconf_struct(yaml_path: Path):
    from timewarp.model_configs import (
        ModelConfig,
        CustomAttentionTransformerNVPConfig,
    )
    from timewarp.modules.layers.custom_attention_encoder import (
        CustomAttentionEncoderLayerConfig,
    )
    from timewarp.modules.model_wrappers.flow import (
        ConditionalFlowDensityConfig,
    )
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    mc = raw["model_config"]
    cnvp = mc["custom_transformer_nvp_config"]
    enc = cnvp["encoder_layer_config"]
    cfd = cnvp["conditional_flow_density"]

    encoder_cfg = CustomAttentionEncoderLayerConfig(
        d_model=enc["d_model"],
        dim_feedforward=enc["dim_feedforward"],
        dropout=enc["dropout"],
        num_heads=enc["num_heads"],
        attention_type=enc["attention_type"],
        lengthscales=enc["lengthscales"],
        max_radius=enc.get("max_radius"),
        normalise_kernel_values=enc.get("normalise_kernel_values", True),
        cheb_order=enc.get("cheb_order"),
        force_asymptotic_zero=enc.get("force_asymptotic_zero"),
    )
    flow_cfg = ConditionalFlowDensityConfig(
        scale_requires_grad=cfd.get("scale_requires_grad", True),
        ignore_conditional_velocity=cfd.get("ignore_conditional_velocity", True),
        use_displacement_as_target=cfd.get("use_displacement_as_target", True),
    )
    cnvp_cfg = CustomAttentionTransformerNVPConfig(
        atom_embedding_dim=cnvp["atom_embedding_dim"],
        latent_mlp_hidden_dims=cnvp["latent_mlp_hidden_dims"],
        num_coupling_layers=cnvp["num_coupling_layers"],
        num_transformer_layers=cnvp["num_transformer_layers"],
        encoder_layer_config=encoder_cfg,
        position_layer_index_mod_2=cnvp.get("position_layer_index_mod_2", 0),
        conditional_flow_density=flow_cfg,
    )
    return ModelConfig(
        model_type=mc["model_type"],
        custom_transformer_nvp_config=cnvp_cfg,
    )


def load_timewarp_model(ckpt_path: Path, config_path: Path, device: str):
    from timewarp.model_constructor import model_constructor

    print(f"[load] reading config from {config_path}", flush=True)
    model_cfg = make_omegaconf_struct(config_path)

    print("[load] building model from config", flush=True)
    module = model_constructor(model_cfg)

    print(f"[load] loading state dict from {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, **TORCH_LOAD_KW)
    sd = ckpt.get("model_state_dict", ckpt.get("module", ckpt)) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in sd.items()}
    module.load_state_dict(sd, strict=False)
    module.eval().to(device)
    print(f"[load] OK; param count: {sum(p.numel() for p in module.parameters())}", flush=True)
    return module


def build_initial_batch(data_dir: Path, peptide: str, device: str,
                        step_width: int = 100000):
    # Skip the lmdb-importing __init__.py — go straight to iterable dataset.
    from timewarp.datasets.iterable_datasets import RawMolDynDataset
    from timewarp.dataloader import moldyn_dense_collate_fn

    ds = RawMolDynDataset(data_dir=str(data_dir), step_width=step_width)
    iterator = ds.make_iterator([peptide])
    dp = next(iter(iterator))
    batch = moldyn_dense_collate_fn([dp])
    return batch


def timewarp_rollout(model, init_batch, n_steps: int, device: str, seed: int = 42):
    """Autoregressive: sample y from x, then feed y back as x."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    atom_types = init_batch.atom_types.to(device)
    adj_list = init_batch.adj_list.to(device)
    edge_batch_idx = init_batch.edge_batch_idx.to(device)
    masked_elements = init_batch.masked_elements.to(device)

    x_coords = init_batch.atom_coords.to(device)
    x_velocs = init_batch.atom_velocs.to(device)

    # B=1 by construction. Atom count V from atom_types shape.
    B, V = atom_types.shape
    coord_traj = np.empty((n_steps, V, 3), dtype=np.float32)
    veloc_traj = np.empty((n_steps, V, 3), dtype=np.float32)

    t0 = time.time()
    for t in range(n_steps):
        with torch.no_grad():
            y_coords, y_velocs = model.conditional_sample(
                atom_types=atom_types,
                x_coords=x_coords,
                x_velocs=x_velocs,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                num_samples=1,
                logger=None,
            )
        # y_*: shape [num_samples, B, V, 3] or [B, V, 3] depending on impl
        yc = y_coords
        yv = y_velocs
        while yc.dim() > 3:
            yc = yc.squeeze(0)
            yv = yv.squeeze(0)
        # Now yc: [B, V, 3] or [V, 3] — normalize to [B, V, 3]
        if yc.dim() == 2:
            yc = yc.unsqueeze(0)
            yv = yv.unsqueeze(0)
        coord_traj[t] = yc[0].cpu().numpy()
        veloc_traj[t] = yv[0].cpu().numpy()
        # Feed back
        x_coords = yc.to(device)
        x_velocs = yv.to(device)
        if (t + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (t + 1) / elapsed
            eta = (n_steps - t - 1) / rate
            print(f"  rollout [{t+1}/{n_steps}] {rate:.2f} steps/s, ETA {eta/60:.1f} min", flush=True)
    return coord_traj, veloc_traj


def positions_to_torsions(positions: np.ndarray, pdb_path: Path):
    """positions: (T, V, 3) [nm]. Returns torsions (T, R, 2) and residue ids."""
    top = md.load_pdb(str(pdb_path)).topology
    traj = md.Trajectory(positions, top)
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)
    phi_res = np.array([top.atom(int(r[1])).residue.index for r in phi_idx])
    psi_res = np.array([top.atom(int(r[1])).residue.index for r in psi_idx])
    common = sorted(set(phi_res.tolist()) & set(psi_res.tolist()))
    phi_lookup = {int(r): i for i, r in enumerate(phi_res)}
    psi_lookup = {int(r): i for i, r in enumerate(psi_res)}
    phi_cols = [phi_lookup[r] for r in common]
    psi_cols = [psi_lookup[r] for r in common]
    out = np.stack([phi[:, phi_cols], psi[:, psi_cols]], axis=-1)
    return out, np.array(common, dtype=np.int64)


def build_2d_hist(torsions: np.ndarray, n_bins: int = 24):
    """torsions: (T, R, 2). Returns (R, n_bins, n_bins) normalised."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    R = torsions.shape[1]
    out = np.zeros((R, n_bins, n_bins), dtype=np.float64)
    for r in range(R):
        h, _, _ = np.histogram2d(torsions[:, r, 0], torsions[:, r, 1], bins=[edges, edges])
        s = h.sum()
        if s > 0:
            out[r] = h / s
    return out, edges


def jsd_per_residue(p, q, eps=1e-12):
    p = p + eps
    q = q + eps
    p = p / p.sum(axis=(1, 2), keepdims=True)
    q = q / q.sum(axis=(1, 2), keepdims=True)
    m = 0.5 * (p + q)
    def _kl(a, b):
        return (a * (np.log(a) - np.log(b))).sum(axis=(1, 2))
    return 0.5 * (_kl(p, m) + _kl(q, m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/timewarp_checkpoints/4aa_best_model.pt")
    ap.add_argument("--config", default="/workspace/timewarp_checkpoints/4aa_config.yaml")
    ap.add_argument("--data_root", default="/workspace/timewarp_data/4AA-large/test")
    ap.add_argument("--peptides", nargs="+", default=["AAAY", "AACE", "AAEW"])
    ap.add_argument("--n_rollout", type=int, default=2500)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--step_width", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default="/workspace/AlphaDynamics/results/timewarp_rollout_4aa.json")
    ap.add_argument("--out_npz_dir", default="/workspace/AlphaDynamics/results/timewarp_rollout_npz")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run: load model, batch 1 peptide, sample 5 frames")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[main] device={device}, peptides={args.peptides}, n_rollout={args.n_rollout}", flush=True)

    model = load_timewarp_model(Path(args.ckpt), Path(args.config), device)
    n_params = sum(p.numel() for p in model.parameters())

    Path(args.out_npz_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    rollout_steps = 5 if args.smoke else args.n_rollout
    results = {
        "experiment": "timewarp_4aa_large_test_rollout",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_params_timewarp": int(n_params),
        "n_rollout": int(rollout_steps),
        "n_bins": int(args.n_bins),
        "step_width": int(args.step_width),
        "ckpt_path": str(args.ckpt),
        "data_root": str(args.data_root),
        "per_peptide": {},
    }

    for pep in args.peptides:
        print(f"\n[peptide={pep}] preparing initial batch", flush=True)
        init_batch = build_initial_batch(Path(args.data_root), pep, device, args.step_width)
        V = init_batch.atom_types.shape[1]
        print(f"[peptide={pep}] V_atoms={V}", flush=True)

        # Ground truth
        arrays = np.load(Path(args.data_root) / f"{pep}-traj-arrays.npz")
        gt_pos = arrays["positions"].astype(np.float32)
        T_total = gt_pos.shape[0]
        gt_split = int(T_total * 0.8)
        gt_test = gt_pos[gt_split:]  # validation slice
        pdb_path = Path(args.data_root) / f"{pep}-traj-state0.pdb"
        gt_torsions, gt_res = positions_to_torsions(gt_test, pdb_path)
        gt_hist, edges = build_2d_hist(gt_torsions, n_bins=args.n_bins)
        print(f"[peptide={pep}] GT torsions {gt_torsions.shape}, residues {len(gt_res)}", flush=True)

        # Timewarp rollout
        print(f"[peptide={pep}] rollout {rollout_steps} frames", flush=True)
        t0 = time.time()
        tw_pos, tw_vel = timewarp_rollout(model, init_batch, rollout_steps, device, seed=42)
        rollout_time = time.time() - t0
        print(f"[peptide={pep}] rollout done in {rollout_time:.1f}s ({rollout_steps/rollout_time:.2f} fr/s)", flush=True)

        # Save raw rollout
        npz_out = Path(args.out_npz_dir) / f"{pep}_timewarp_rollout.npz"
        np.savez_compressed(npz_out, positions=tw_pos, velocities=tw_vel)

        tw_torsions, tw_res = positions_to_torsions(tw_pos, pdb_path)
        tw_hist, _ = build_2d_hist(tw_torsions, n_bins=args.n_bins)
        jsd = jsd_per_residue(tw_hist, gt_hist)
        print(f"[peptide={pep}] mean JSD vs GT = {jsd.mean():.4f}", flush=True)

        results["per_peptide"][pep] = {
            "n_atoms": int(V),
            "n_residues_audit": int(len(tw_res)),
            "residue_indices": tw_res.tolist(),
            "gt_split_idx": int(gt_split),
            "gt_n_frames": int(gt_test.shape[0]),
            "rollout_n_frames": int(rollout_steps),
            "rollout_seconds": float(rollout_time),
            "rollout_throughput_fps": float(rollout_steps / rollout_time),
            "jsd_per_residue": jsd.tolist(),
            "mean_jsd": float(jsd.mean()),
            "median_jsd": float(np.median(jsd)),
            "rollout_npz_path": str(npz_out),
        }

        # Persist intermediate (in case of crash)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)

    # Aggregate
    means = [v["mean_jsd"] for v in results["per_peptide"].values()]
    results["aggregate"] = {
        "peptide_count": len(means),
        "mean_jsd_across_peptides": float(np.mean(means)),
        "median_jsd_across_peptides": float(np.median(means)),
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] aggregate mean JSD = {np.mean(means):.4f}", flush=True)
    print(f"[done] saved {args.out_json}", flush=True)


if __name__ == "__main__":
    main()

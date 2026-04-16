"""Fix 2 v2: Ramachandran free energy comparison — PROPER METRICS.

Replaces flawed |max(G)-max(G)| metric with:
  1. Jensen-Shannon divergence JS(P_model || P_gt) — bounded [0, ln2], symmetric
  2. 2D Wasserstein (Earth Mover) between P_model and P_gt
  3. Per-basin ΔG accuracy — identify top-3 local minima, measure ΔG to deepest
  4. Population fraction accuracy per basin
"""
import sys, os, math, json, glob
sys.path.insert(0, '/home/krisss0/AlphaDynamics/src')

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter, gaussian_filter
from scipy.stats import wasserstein_distance

from chain_model import chain_log_prob
from train_real import ChainPhaseFlowVar

TWO_PI = 2 * math.pi
kT_RT = 0.596  # kcal/mol at 298K
def wrap_np(a): return (a + np.pi) % TWO_PI - np.pi

DATA_DIR = "/home/krisss0/AlphaDynamics/mdcath_real_data/mdcath_alltemps"
RES_DIR = "/home/krisss0/AlphaDynamics/results"
FIG_DIR = "/home/krisss0/AlphaDynamics/paper/figures"
os.makedirs(FIG_DIR, exist_ok=True)

DOMAINS = ['1lwjA03', '1vq8L01', '1kwgA03']
TEMP = 348
N_STEPS_ROLLOUT = 2500
KAPPA_MULT = 30.0
N_BINS = 36


@torch.no_grad()
def rollout(model, seed, n_steps, kappa_mult=30.0):
    state = seed.clone()
    A = state.shape[1]
    traj = torch.zeros(1, n_steps + 1, A, device=state.device)
    traj[:, 0] = state
    for t in range(n_steps):
        log_pi, mu, kappa = model(state)
        k_scaled = kappa * kappa_mult
        pi = log_pi.exp()
        comp = torch.multinomial(pi, 1).squeeze(-1)
        bidx = torch.arange(1, device=state.device)
        nxt = torch.distributions.VonMises(mu[bidx, comp], k_scaled[bidx, comp]).sample()
        state = nxt
        traj[:, t + 1] = state
    return traj.squeeze(0)


def prob_histogram(phi, psi, n_bins=N_BINS):
    """2D probability distribution on torus, smoothed with Gaussian KDE-like filter."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    H, _, _ = np.histogram2d(phi.flatten(), psi.flatten(), bins=[edges, edges])
    # Smooth for stable basin detection (1-bin sigma)
    H = gaussian_filter(H, sigma=1.0, mode='wrap')
    P = H / max(H.sum(), 1e-12)
    return P, edges


def free_energy(P, floor=1e-5):
    """G = -RT ln P, no cap — zero at P_max."""
    P_safe = np.maximum(P, floor)
    G = -kT_RT * np.log(P_safe)
    G -= G.min()
    return G


def jensen_shannon_divergence(P, Q, eps=1e-12):
    """JSD(P || Q) in nats, bounded [0, ln 2]."""
    P_flat = P.flatten() + eps
    Q_flat = Q.flatten() + eps
    P_flat /= P_flat.sum()
    Q_flat /= Q_flat.sum()
    M = 0.5 * (P_flat + Q_flat)
    kl_pm = (P_flat * (np.log(P_flat) - np.log(M))).sum()
    kl_qm = (Q_flat * (np.log(Q_flat) - np.log(M))).sum()
    return 0.5 * (kl_pm + kl_qm)


def wasserstein_2d(P, Q, n_bins=N_BINS):
    """Approximate 2D Wasserstein via row/col marginals (cheap proxy)."""
    P_phi = P.sum(axis=1)
    Q_phi = Q.sum(axis=1)
    P_psi = P.sum(axis=0)
    Q_psi = Q.sum(axis=0)
    centers = np.linspace(-180, 180, n_bins)  # degrees
    w_phi = wasserstein_distance(centers, centers, u_weights=P_phi, v_weights=Q_phi)
    w_psi = wasserstein_distance(centers, centers, u_weights=P_psi, v_weights=Q_psi)
    return 0.5 * (w_phi + w_psi)  # average marginal EMD in degrees


def find_basins(G, n_basins=3, min_separation_bins=5):
    """Find up to n_basins deepest local minima in G; return list of (row, col, G_value).
    min_separation_bins: basins must be at least this far apart."""
    # Local minimum = point where all neighbors are >= it (3x3 minimum filter trick)
    min_filt = minimum_filter(G, size=3, mode='wrap')
    is_min = (G == min_filt)
    # Sort candidates by G
    candidates = [(G[i, j], i, j) for i in range(G.shape[0]) for j in range(G.shape[1]) if is_min[i, j]]
    candidates.sort()
    selected = []
    for g_val, r, c in candidates:
        if len(selected) >= n_basins:
            break
        ok = True
        for _, rr, cc in selected:
            # Torus distance
            dr = min(abs(r - rr), N_BINS - abs(r - rr))
            dc = min(abs(c - cc), N_BINS - abs(c - cc))
            if dr < min_separation_bins and dc < min_separation_bins:
                ok = False; break
        if ok:
            selected.append((g_val, r, c))
    return selected


def basin_populations(P, basins, radius_bins=4):
    """Fraction of probability within `radius_bins` of each basin center (torus)."""
    fractions = []
    for _, r, c in basins:
        frac = 0.0
        for i in range(max(0, r - radius_bins), min(N_BINS, r + radius_bins + 1)):
            for j in range(max(0, c - radius_bins), min(N_BINS, c + radius_bins + 1)):
                # Torus dist check
                dr = min(abs(i - r), N_BINS - abs(i - r))
                dc = min(abs(j - c), N_BINS - abs(j - c))
                if dr * dr + dc * dc <= radius_bins * radius_bins:
                    frac += P[i, j]
        fractions.append(frac)
    return fractions


def train_model(N, train_data, device, steps=4000, batch=256, lr=2e-3, seed=42):
    torch.manual_seed(seed)
    model = ChainPhaseFlowVar(N=N, n_osc=64, n_components=8, hidden=128, t_max=4.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    N_train = len(train_data) - 1
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N_train, (batch,), device=device)
        x, y = train_data[idx], train_data[idx + 1]
        log_pi, mu, kappa = model(x)
        loss = -chain_log_prob(y, log_pi, mu, kappa).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    return model


def plot_G_comparison(ax, phi, psi, title):
    P, _ = prob_histogram(phi, psi)
    G = free_energy(P)
    extent = [-180, 180, -180, 180]
    im = ax.imshow(np.minimum(G.T, 8.0), origin='lower', extent=extent,
                   cmap='viridis', vmin=0, vmax=8, aspect='equal')
    ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)')
    ax.set_title(title, fontsize=9)
    return im


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = {}

    for domain in DOMAINS:
        print(f"\n=== {domain} ===")
        d = np.load(f"{DATA_DIR}/{domain}_T{TEMP}_dihedrals.npz")
        N = int(d['N'])
        train = torch.tensor(d['train'], dtype=torch.float32, device=device).reshape(d['train'].shape[0], -1)
        val_gt = torch.tensor(d['val'], dtype=torch.float32, device=device).reshape(d['val'].shape[0], -1)
        gt_all = np.concatenate([d['train'], d['val']], axis=0)

        print("  Training AlphaDynamics t=4 ...")
        model = train_model(N, train, device, steps=4000)

        print(f"  Rollout {N_STEPS_ROLLOUT} steps (κ×{KAPPA_MULT})...")
        rollout_traj = rollout(model, val_gt[0:1], N_STEPS_ROLLOUT, kappa_mult=KAPPA_MULT).cpu().numpy()
        rollout_np = rollout_traj.reshape(-1, N, 2)

        # Per-residue metrics
        jsd_per_res = []
        emd_per_res = []
        dG_basin_per_res = []
        pop_err_per_res = []
        sample_residues = [0, N // 2, N - 1]
        for r in range(N):
            P_gt, _ = prob_histogram(gt_all[:, r, 0], gt_all[:, r, 1])
            P_model, _ = prob_histogram(rollout_np[:, r, 0], rollout_np[:, r, 1])
            G_gt = free_energy(P_gt)
            G_model = free_energy(P_model)

            jsd = jensen_shannon_divergence(P_model, P_gt)
            emd = wasserstein_2d(P_model, P_gt)
            jsd_per_res.append(jsd)
            emd_per_res.append(emd)

            # Basin-wise ΔG
            basins_gt = find_basins(G_gt, n_basins=3)
            # Measure G_model at the SAME (r, c) positions as GT basins
            dG_errs = []
            for g_gt, row, col in basins_gt:
                g_model_at_gt_basin = G_model[row, col]
                # Both are relative to their own min (=0), compare
                dG_errs.append(abs(g_gt - g_model_at_gt_basin))
            dG_basin_per_res.append(np.mean(dG_errs) if dG_errs else 0.0)

            pops_gt = basin_populations(P_gt, basins_gt)
            pops_model = basin_populations(P_model, basins_gt)
            pop_err_per_res.append(np.mean([abs(a - b) for a, b in zip(pops_gt, pops_model)]))

        # Figure: 2 rows × 3 cols (GT vs model, 3 residues)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
        for col, r in enumerate(sample_residues):
            im1 = plot_G_comparison(axes[0, col], gt_all[:, r, 0], gt_all[:, r, 1],
                                     f"GT residue {r}")
            im2 = plot_G_comparison(axes[1, col], rollout_np[:, r, 0], rollout_np[:, r, 1],
                                     f"AlphaDynamics residue {r}  (JSD={jsd_per_res[r]:.3f})")
        fig.suptitle(f"Ramachandran free energy G(φ,ψ) [kcal/mol, capped display at 8] — {domain} @ {TEMP}K\n"
                     f"Top: ground truth. Bottom: AlphaDynamics 2500-step rollout",
                     fontsize=11)
        # Shared colorbar
        cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.025, pad=0.02)
        cbar.set_label('G (kcal/mol)')
        plt.savefig(f"{FIG_DIR}/ramachandran_{domain}.png", dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved {FIG_DIR}/ramachandran_{domain}.png")

        all_results[domain] = {
            'N': N,
            'jsd_mean': float(np.mean(jsd_per_res)),
            'jsd_max': float(np.max(jsd_per_res)),
            'emd_mean_deg': float(np.mean(emd_per_res)),
            'emd_max_deg': float(np.max(emd_per_res)),
            'dG_basin_mean_kcal': float(np.mean(dG_basin_per_res)),
            'dG_basin_max_kcal': float(np.max(dG_basin_per_res)),
            'pop_err_mean': float(np.mean(pop_err_per_res)),
            'pop_err_max': float(np.max(pop_err_per_res)),
        }
        print(f"  JSD mean={all_results[domain]['jsd_mean']:.3f}  max={all_results[domain]['jsd_max']:.3f}")
        print(f"  EMD mean={all_results[domain]['emd_mean_deg']:.1f}°  max={all_results[domain]['emd_max_deg']:.1f}°")
        print(f"  |ΔG_basin| mean={all_results[domain]['dG_basin_mean_kcal']:.3f} kcal/mol")
        print(f"  Basin pop error mean={all_results[domain]['pop_err_mean']:.3f}")

        del model
        torch.cuda.empty_cache()

    # Save
    with open(f"{RES_DIR}/ramachandran_energy.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    lines = ["# Ramachandran free energy — AlphaDynamics vs ground truth (v2)",
             "",
             "2500-step rollouts. Metrics:",
             "- **JSD**: Jensen-Shannon divergence between P(φ,ψ)_model and P_gt (nats). Range [0, ln2≈0.693]. Lower=better.",
             "- **EMD**: avg marginal Wasserstein distance (°). Measures spatial displacement of density. Lower=better.",
             "- **|ΔG_basin|**: avg error on G at GT-basin centers (kcal/mol). <1 = within thermal kT.",
             "- **Pop err**: avg |P_basin_model - P_basin_gt| over top-3 basins. <0.1 = well-calibrated populations.",
             "",
             "| Domain | N | JSD mean | JSD max | EMD mean (°) | \\|ΔG_basin\\| (kcal) | Pop err |",
             "|---|---|---|---|---|---|---|"]
    for domain, r in all_results.items():
        lines.append(f"| {domain} | {r['N']} | {r['jsd_mean']:.3f} | {r['jsd_max']:.3f} | "
                     f"{r['emd_mean_deg']:.1f} | {r['dG_basin_mean_kcal']:.3f} | {r['pop_err_mean']:.3f} |")
    lines += ["",
              "## Interpretation guidelines",
              "",
              "| Metric | Excellent | Good | Poor |",
              "|---|---|---|---|",
              "| JSD | <0.1 | 0.1–0.3 | >0.4 |",
              "| EMD (°) | <10 | 10–30 | >50 |",
              "| \\|ΔG_basin\\| (kcal) | <0.5 | 0.5–1.5 | >2.0 |",
              "| Pop err | <0.05 | 0.05–0.15 | >0.25 |",
              "",
              "Figures: `paper/figures/ramachandran_{domain}.png`"]
    with open(f"{RES_DIR}/ramachandran_energy.md", 'w') as f:
        f.write('\n'.join(lines))
    print("\n\n" + '\n'.join(lines))


if __name__ == "__main__":
    main()

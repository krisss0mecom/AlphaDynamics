"""Multi-step autoregressive rollout evaluation.

Given a trained chain model, from each val seed frame:
  1. Rollout T=50 frames autoregressively (sample each step, feed back)
  2. Compare the rollout distribution to the ground-truth trajectory distribution

Metrics:
  - Per-step ang_err: at rollout step t, how close is sampled state to true state t?
    (expected to drift — that's fine; we measure the drift rate)
  - Distribution fidelity: histogram of (phi, psi) in 36×36 bins on T^2 for
    a single "reference" residue, KL divergence between rollout pool and true val pool.
"""
import math
import torch

TWO_PI = 2.0 * math.pi


def wrap(a):
    return a - TWO_PI * torch.round(a / TWO_PI)


@torch.no_grad()
def rollout(model, seed_flat, n_steps=50, n_chains=1000, device=None):
    """Start from n_chains seed frames, rollout n_steps ahead, sampling each step.

    seed_flat: (M, A) — at least n_chains frames. Uses first n_chains.
    Returns: rollout (n_chains, n_steps+1, A). Entry [:, 0, :] = seed frames.
    """
    from chain_model import chain_sample
    model.eval()
    ncs = min(n_chains, len(seed_flat))
    state = seed_flat[:ncs].clone()  # (ncs, A)
    A = state.shape[1]
    traj = torch.zeros(ncs, n_steps + 1, A, device=device or state.device)
    traj[:, 0] = state
    for t in range(n_steps):
        log_pi, mu, kappa = model(state)
        nxt = chain_sample(log_pi, mu, kappa, n_samples=1).squeeze(1)  # (ncs, A)
        state = nxt
        traj[:, t + 1] = state
    return traj


@torch.no_grad()
def per_step_err(traj, true_traj):
    """traj, true_traj: (n_chains, T+1, A). Returns (T+1,) mean ang_err per step."""
    diff = wrap(traj - true_traj)
    err = torch.sqrt((diff ** 2).mean(-1))  # (n_chains, T+1)
    return err.mean(0)


@torch.no_grad()
def ramachandran_kl(rollout_flat, truth_flat, residue=0, n_bins=36):
    """Histogram-based KL divergence on single residue (phi, psi) ∈ T^2."""
    # rollout_flat, truth_flat: (*, A). Take residue `residue` (angles 2*r, 2*r+1).
    r_phi = rollout_flat[..., 2 * residue]
    r_psi = rollout_flat[..., 2 * residue + 1]
    t_phi = truth_flat[..., 2 * residue]
    t_psi = truth_flat[..., 2 * residue + 1]

    # Histogram both
    edges = torch.linspace(-math.pi, math.pi, n_bins + 1, device=r_phi.device)
    def hist2d(x, y):
        # clamp into bins
        ix = torch.bucketize(x.flatten(), edges) - 1
        iy = torch.bucketize(y.flatten(), edges) - 1
        ix = ix.clamp(0, n_bins - 1)
        iy = iy.clamp(0, n_bins - 1)
        h = torch.zeros(n_bins, n_bins, device=x.device)
        h.index_put_((ix, iy), torch.ones_like(ix, dtype=torch.float32),
                     accumulate=True)
        return h / h.sum().clamp(min=1e-12)

    p = hist2d(r_phi, r_psi)
    q = hist2d(t_phi, t_psi)
    # KL(p || q), smooth with Laplace
    eps = 1e-6
    kl = (p * ((p + eps).log() - (q + eps).log())).sum().item()
    return kl

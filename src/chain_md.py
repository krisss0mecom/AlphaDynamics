"""Synthetic N-residue Langevin MD on T^{2N}.

Physics:
  - Each residue has (phi_i, psi_i) ∈ T^2. Chain of N residues → T^{2N}.
  - Local Ramachandran potential U_loc(phi, psi): 3 basins (alpha_R, beta, PPII)
    as mixture of von Mises densities fit roughly to Ala2.
  - Nearest-neighbor coupling: V_c = -kappa_c * (cos(phi_i - phi_{i+1}) + cos(psi_i - psi_{i+1})).
    Small kappa → residues mostly independent; large → cooperative chain.
  - Langevin integrator (overdamped): d phi = -dU/dphi * dt + sqrt(2 T dt) * randn.
  - kBT = 1 (unit energy scale).

Stride: simulation runs at dt_sim, saves every N_stride steps. Default
dt_sim=0.005, N_stride=2 -> effective stride 0.01 (quasi-continuous).
Mean per-step rotation ~2-5 deg — model has real information to learn.

Output: chain_data.npz with keys 'train' (T_train, N, 2) and 'val' (T_val, N, 2).
"""
import math
import numpy as np
import argparse
from pathlib import Path

TWO_PI = 2.0 * math.pi


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


class ChainPotential:
    """Potential: sum_i U_loc(phi_i, psi_i) + sum_i V_c(residue_i, residue_{i+1})."""

    def __init__(self, N, kappa_c=1.5):
        self.N = N
        self.kappa_c = kappa_c
        # 3 basins: alpha_R ~ (-60, -45), beta ~ (-120, 130), PPII ~ (-75, 145)
        self.basin_phi = np.array([-1.05, -2.09, -1.31])   # rad
        self.basin_psi = np.array([-0.79,  2.27,  2.53])
        self.basin_kappa = np.array([3.0, 3.0, 3.0])
        self.basin_w = np.array([0.45, 0.35, 0.20])  # relative depth

    def U_loc(self, phi, psi):
        """Local potential per residue. Input (N,), returns scalar U summed."""
        # U = -log(sum_b w_b * exp(kappa_b * (cos(phi-phi_b) + cos(psi-psi_b))))
        d_phi = phi[..., None] - self.basin_phi  # broadcast to (..., 3)
        d_psi = psi[..., None] - self.basin_psi
        s = self.basin_kappa * (np.cos(d_phi) + np.cos(d_psi))
        s_max = s.max(-1, keepdims=True)
        U = -(np.log((self.basin_w * np.exp(s - s_max)).sum(-1)) + s_max[..., 0])
        return U.sum()

    def dU_local(self, phi, psi):
        """Analytic grad of U_loc wrt phi, psi per residue."""
        d_phi = phi[..., None] - self.basin_phi
        d_psi = psi[..., None] - self.basin_psi
        s = self.basin_kappa * (np.cos(d_phi) + np.cos(d_psi))
        s_max = s.max(-1, keepdims=True)
        e = self.basin_w * np.exp(s - s_max)
        Z = e.sum(-1, keepdims=True)
        p = e / Z  # (N, 3)
        # dU/dphi = kappa_b * sin(phi - phi_b) weighted by p
        grad_phi = (p * self.basin_kappa * np.sin(d_phi)).sum(-1)
        grad_psi = (p * self.basin_kappa * np.sin(d_psi)).sum(-1)
        return grad_phi, grad_psi

    def dU_coupling(self, phi, psi):
        """Nearest-neighbor coupling grad. d V_c / d phi_i."""
        # V_c = -kappa_c * sum_{i=0..N-2} [cos(phi_i - phi_{i+1}) + cos(psi_i - psi_{i+1})]
        # dV/dphi_i = kappa_c * [sin(phi_{i-1} - phi_i) - sin(phi_i - phi_{i+1})] * (-1)
        #           = kappa_c * [-sin(phi_{i-1} - phi_i) + sin(phi_i - phi_{i+1})]
        # (checking signs: d/dphi_i [ -cos(phi_{i-1} - phi_i) ] = -sin(phi_{i-1} - phi_i) * (-1) = sin(phi_{i-1}-phi_i)
        # Hmm let me redo:
        # V = -kappa * cos(phi_{i-1} - phi_i); dV/dphi_i = -kappa * sin(phi_{i-1} - phi_i) * (-1) = kappa * sin(phi_{i-1}-phi_i)
        # V = -kappa * cos(phi_i - phi_{i+1}); dV/dphi_i = -kappa * sin(phi_i - phi_{i+1}) * 1 = -kappa*sin(phi_i - phi_{i+1})
        # Total: dV/dphi_i = kappa * [sin(phi_{i-1}-phi_i) - sin(phi_i - phi_{i+1})]
        grad_phi = np.zeros_like(phi)
        grad_psi = np.zeros_like(psi)
        # Left neighbor contribution (for i >= 1)
        grad_phi[1:] += self.kappa_c * np.sin(phi[:-1] - phi[1:])
        grad_psi[1:] += self.kappa_c * np.sin(psi[:-1] - psi[1:])
        # Right neighbor contribution (for i < N-1)
        grad_phi[:-1] -= self.kappa_c * np.sin(phi[:-1] - phi[1:])
        grad_psi[:-1] -= self.kappa_c * np.sin(psi[:-1] - psi[1:])
        return grad_phi, grad_psi

    def grad(self, phi, psi):
        g_phi_l, g_psi_l = self.dU_local(phi, psi)
        g_phi_c, g_psi_c = self.dU_coupling(phi, psi)
        return g_phi_l + g_phi_c, g_psi_l + g_psi_c


def simulate(N, n_frames, dt=0.005, n_stride=2, kT=1.0, kappa_c=1.5,
             seed=0, burn_in=5000):
    """Overdamped Langevin: d phi = -grad U * dt + sqrt(2 T dt) * randn."""
    rng = np.random.default_rng(seed)
    pot = ChainPotential(N, kappa_c=kappa_c)
    phi = rng.uniform(-math.pi, math.pi, N)
    psi = rng.uniform(-math.pi, math.pi, N)
    noise_scale = math.sqrt(2.0 * kT * dt)

    # Burn-in
    for _ in range(burn_in):
        g_phi, g_psi = pot.grad(phi, psi)
        phi = wrap(phi - g_phi * dt + noise_scale * rng.standard_normal(N))
        psi = wrap(psi - g_psi * dt + noise_scale * rng.standard_normal(N))

    # Collect frames (stride n_stride steps between saves)
    out = np.zeros((n_frames, N, 2), dtype=np.float32)
    for f in range(n_frames):
        for _ in range(n_stride):
            g_phi, g_psi = pot.grad(phi, psi)
            phi = wrap(phi - g_phi * dt + noise_scale * rng.standard_normal(N))
            psi = wrap(psi - g_psi * dt + noise_scale * rng.standard_normal(N))
        out[f, :, 0] = phi
        out[f, :, 1] = psi
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=4, help="residues in chain")
    p.add_argument("--n_train", type=int, default=80000)
    p.add_argument("--n_val", type=int, default=20000)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--kT", type=float, default=1.0)
    p.add_argument("--kappa_c", type=float, default=1.5)
    p.add_argument("--out", default="chain_data.npz")
    args = p.parse_args()

    print(f"Simulating chain: N={args.N}, dt={args.dt}, stride={args.stride}, "
          f"kT={args.kT}, kappa_c={args.kappa_c}")
    print(f"Generating train ({args.n_train} frames)...")
    tr = simulate(args.N, args.n_train, dt=args.dt, n_stride=args.stride,
                  kT=args.kT, kappa_c=args.kappa_c, seed=1)
    print(f"Generating val ({args.n_val} frames)...")
    va = simulate(args.N, args.n_val, dt=args.dt, n_stride=args.stride,
                  kT=args.kT, kappa_c=args.kappa_c, seed=2)

    # Step-size statistics (for diagnostic)
    for name, arr in [("train", tr), ("val", va)]:
        phi = arr[:, :, 0]
        psi = arr[:, :, 1]
        dphi = wrap(np.diff(phi, axis=0))
        dpsi = wrap(np.diff(psi, axis=0))
        step = np.sqrt(dphi ** 2 + dpsi ** 2)
        print(f"  {name}: mean |step|={math.degrees(step.mean()):.2f}°, "
              f"p99={math.degrees(np.quantile(step, 0.99)):.2f}°")

    np.savez(args.out, train=tr, val=va,
             N=args.N, dt=args.dt, stride=args.stride,
             kT=args.kT, kappa_c=args.kappa_c)
    print(f"Saved {args.out} — train shape {tr.shape}, val shape {va.shape}")


if __name__ == "__main__":
    main()

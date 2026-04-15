"""V4_Fusion extended to N-residue chain (T^{2N}).

Two models:
  - ChainMLP: pure MLP baseline (sin/cos features of all 2N angles).
  - ChainPhaseFlow: V4_Fusion-style phase ODE encoder + MDN head.

Both output K-component mixture of von Mises on T^{2N} (axis-independent
per residue, per component — 2N * K * (mu_phi, mu_psi, kappa_phi, kappa_psi)
plus K mixture weights).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

TWO_PI = 2.0 * math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0
GOLDEN_ANGLE = TWO_PI / (PHI * PHI)
PRIME_FREQS = [2.11, 1.31, 0.67, 0.31, 0.17]


def log_i0(x):
    return x + torch.log(torch.special.i0e(x) + 1e-30)


def von_mises_logpdf(x, mu, kappa):
    return kappa * torch.cos(x - mu) - math.log(TWO_PI) - log_i0(kappa)


def phyllotaxis_phases(n):
    idx = torch.arange(n, dtype=torch.float32)
    return ((idx * GOLDEN_ANGLE) % TWO_PI) - math.pi


class ChainMDNHead(nn.Module):
    """K-component mixture for chain with 2N angles.

    Per component: 2N * (sin mu, cos mu, logit kappa) + 1 mixture weight.
    Chain output shape: log_pi (B, K), mu (B, K, 2N), kappa (B, K, 2N).
    """
    def __init__(self, in_dim, n_angles, n_components=8, hidden=128,
                 kappa_max=200.0):
        super().__init__()
        self.K = n_components
        self.A = n_angles  # 2N
        self.kappa_max = kappa_max
        # Per component: 1 (pi) + 2A (sin/cos mu) + A (kappa logit) = 1 + 3A
        self.out_per_component = 1 + 3 * n_angles
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_components * self.out_per_component),
        )

    def forward(self, x):
        out = self.net(x).view(-1, self.K, self.out_per_component)
        log_pi = F.log_softmax(out[..., 0], dim=-1)  # (B, K)
        A = self.A
        sin_mu = out[..., 1:1 + A]
        cos_mu = out[..., 1 + A:1 + 2 * A]
        logit_k = out[..., 1 + 2 * A:]
        mu = torch.atan2(sin_mu, cos_mu)  # (B, K, A)
        kappa = self.kappa_max * torch.sigmoid(logit_k) + 0.01  # eps for numerical stability  # (B, K, A)
        return log_pi, mu, kappa


def chain_log_prob(target_angles, log_pi, mu, kappa):
    """target_angles: (B, A). Returns (B,) log p(target | mixture)."""
    # lp per component, per angle (B, K, A)
    lp = von_mises_logpdf(target_angles.unsqueeze(1), mu, kappa)
    lp = lp.sum(-1)  # sum over angles A (independence within component)
    return torch.logsumexp(log_pi + lp, dim=-1)


@torch.no_grad()
def chain_sample(log_pi, mu, kappa, n_samples=10):
    """Sample n_samples (B, n_samples, A) from mixture."""
    B, K, A = mu.shape
    pi = log_pi.exp()  # (B, K)
    comp = torch.multinomial(pi, n_samples, replacement=True)  # (B, n_samples)
    bidx = torch.arange(B, device=pi.device).unsqueeze(1).expand(-1, n_samples)
    mu_sel = mu[bidx, comp]  # (B, n_samples, A)
    k_sel = kappa[bidx, comp]
    samples = torch.distributions.VonMises(mu_sel, k_sel).sample()
    return samples


# ------------ MLP Baseline ------------

class ChainMLP(nn.Module):
    def __init__(self, N, n_components=8, hidden=128):
        super().__init__()
        self.N = N
        self.A = 2 * N
        self.encoder = nn.Sequential(
            nn.Linear(2 * self.A, hidden), nn.GELU(),    # sin, cos of each angle
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = ChainMDNHead(hidden, self.A, n_components, hidden)

    def encode(self, angles):
        # angles: (B, A) — flat angle vector
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.encoder(feats)

    def forward(self, angles):
        return self.head(self.encode(angles))


# ------------ Phase-Flow Encoder for Chain ------------

class ChainPhaseFlowFunc(nn.Module):
    def __init__(self, n_osc):
        super().__init__()
        self.n_osc = n_osc
        freqs = torch.tensor([PRIME_FREQS[i % len(PRIME_FREQS)]
                              for i in range(n_osc)], dtype=torch.float32)
        self.omega = nn.Parameter(freqs)
        self.W = nn.Parameter(torch.randn(n_osc, n_osc) * (0.5 / math.sqrt(n_osc)))
        self.phi_anchor = nn.Parameter(phyllotaxis_phases(n_osc))
        self.log_a = nn.Parameter(torch.tensor(math.log(0.08)))

    def forward(self, t, phi):
        cos_p = torch.cos(phi)
        sin_p = torch.sin(phi)
        A = cos_p * sin_p
        Bj = cos_p * cos_p
        sumA = A @ self.W.T
        sumB = Bj @ self.W.T
        dphi_coupling = sumA * cos_p - sumB * sin_p
        dphi_anchor = self.log_a.exp() * torch.sin(self.phi_anchor - phi)
        return self.omega + dphi_coupling + dphi_anchor


class ChainPhaseEncoder(nn.Module):
    """Lift 2N chain angles to N_osc phases, evolve via RK4 adjoint, readout (sin, cos)."""
    def __init__(self, n_angles, n_osc=64, t_span=(0.0, 1.0)):
        super().__init__()
        self.n_angles = n_angles
        self.n_osc = n_osc
        self.t_span = t_span
        self.func = ChainPhaseFlowFunc(n_osc)

        # Input mixing: each oscillator reads a linear combination of all 2N angles
        # via per-angle weight + offset
        self.W_in = nn.Parameter(torch.randn(n_angles, n_osc) * (0.5 / math.sqrt(n_angles)))
        self.phi_offset = nn.Parameter(torch.randn(n_osc) * math.pi)

    def forward(self, angles):
        # angles: (B, A)
        phi0 = angles @ self.W_in + self.phi_offset  # (B, N_osc)
        t = torch.tensor(self.t_span, device=angles.device, dtype=torch.float32)
        phi_final = odeint(self.func, phi0, t, method="rk4")[-1]
        return torch.cat([torch.sin(phi_final), torch.cos(phi_final)], dim=-1)


class ChainPhaseFlow(nn.Module):
    def __init__(self, N, n_osc=64, n_components=8, hidden=128):
        super().__init__()
        self.N = N
        self.A = 2 * N
        self.encoder = ChainPhaseEncoder(self.A, n_osc=n_osc)
        self.head = ChainMDNHead(2 * n_osc, self.A, n_components, hidden)

    def forward(self, angles):
        return self.head(self.encoder(angles))

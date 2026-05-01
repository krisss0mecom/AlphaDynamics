"""AD-Init — sequence-conditioned initial-torsion ensemble model.

Solves the missing piece for sequence-only AD-Transfer:

    sequence  -->  p(phi, psi | sequence)   per residue, mixture-of-vM
                                            (this module)
                  +
    sample x_0 from p(phi, psi | sequence)
                  +
    AD-Transfer rollout from x_0           (existing alphadynamics.models)
                  =
    sequence-only torsion-dynamics ensemble

Trained on individual MD frames, not transitions:
    p(x_t | sequence)   for any t in the trajectory.

A latent z is added so a single sequence can yield multiple basins
(alpha-helix, beta-sheet, PII, etc.). At inference we draw N_ensemble
samples of z to span the conformational space.

Architecture mirrors the AD-Transfer head so initialised x_0 lies in
the same coordinate space the propagator expects.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


N_AA_WITH_UNKNOWN = 21
TWO_PI = 2 * math.pi


def aa_one_hot(seq: str, device) -> torch.Tensor:
    AA = "ACDEFGHIKLMNPQRSTVWY"
    idx = {a: i for i, a in enumerate(AA)}
    n = len(seq)
    out = torch.zeros(n, N_AA_WITH_UNKNOWN, device=device, dtype=torch.float32)
    for i, a in enumerate(seq):
        out[i, idx.get(a, len(AA))] = 1.0
    return out


def _iv0_torch(kappa: torch.Tensor) -> torch.Tensor:
    """Modified Bessel I_0(kappa) — same approx as AD-Transfer for consistency."""
    small = kappa < 3.75
    t = (kappa / 3.75) ** 2
    series = (1.0 + 3.5156229 * t + 3.0899424 * t**2 + 1.2067492 * t**3
              + 0.2659732 * t**4 + 0.0360768 * t**5 + 0.0045813 * t**6)
    sk = torch.clamp(kappa, min=1e-6)
    asymp = (torch.exp(sk) / torch.sqrt(2 * math.pi * sk)
             * (1.0 + 1.0 / (8 * sk) + 9.0 / (128 * sk**2)))
    return torch.where(small, series, asymp)


class ADInit(nn.Module):
    """Sequence-conditioned per-residue mixture-of-von-Mises initialiser.

    Input per protein P:
      aa_one_hot   : (N, AA_DIM)  one-hot of amino acids
      pos_norm     : (N,)          residue position / max(N-1)
      log_length   : scalar        log(N)/log(N_MAX)
      temp_K       : scalar        log(T/300)
      z (optional) : (B, latent_dim)  conformational basin latent

    Output:
      log_pi : (B, N, K)
      mu     : (B, N, K, 2)   phi/psi means
      kappa  : (B, N, K, 2)   concentrations
    """

    def __init__(self, n_components: int = 8, hidden: int = 128,
                 latent_dim: int = 32, use_temperature: bool = False):
        super().__init__()
        self.K = n_components
        self.latent_dim = latent_dim
        self.use_temperature = use_temperature
        in_dim = N_AA_WITH_UNKNOWN + 1 + 1 + (1 if use_temperature else 0) + latent_dim
        out_dim = n_components * 5  # logit + mu_phi + mu_psi + log_kphi + log_kpsi
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, aa: torch.Tensor, n_residues: int, chain_length: int,
                temp_K: float = None, z: torch.Tensor = None,
                batch_size: int = 1):
        device = aa.device
        # Per-residue features
        idx = torch.arange(n_residues, device=device, dtype=torch.float32)
        pos_norm = (idx / max(n_residues - 1, 1)).view(1, n_residues, 1)
        log_len = torch.tensor(math.log(max(chain_length, 1)) / math.log(300.0),
                                device=device, dtype=torch.float32).view(1, 1, 1)

        feats = [aa.unsqueeze(0).expand(batch_size, n_residues, N_AA_WITH_UNKNOWN),
                  pos_norm.expand(batch_size, n_residues, 1),
                  log_len.expand(batch_size, n_residues, 1)]
        if self.use_temperature:
            t_feat = torch.tensor(math.log(max(temp_K, 1.0) / 300.0),
                                    device=device, dtype=torch.float32).view(1, 1, 1)
            feats.append(t_feat.expand(batch_size, n_residues, 1))
        # Latent z (broadcast across residues)
        if z is None:
            z = torch.zeros(batch_size, self.latent_dim, device=device)
        z_b = z.unsqueeze(1).expand(batch_size, n_residues, self.latent_dim)
        feats.append(z_b)

        x = torch.cat(feats, dim=-1)                    # (B,N,F)
        h = self.encoder(x)
        out = self.head(h).view(batch_size, n_residues, self.K, 5)
        log_pi = F.log_softmax(out[..., 0], dim=-1)
        mu = torch.stack([out[..., 1], out[..., 2]], dim=-1)
        log_kappa = torch.stack([out[..., 3], out[..., 4]], dim=-1)
        kappa = F.softplus(log_kappa) + 0.1
        return log_pi, mu, kappa

    def neg_log_lik(self, aa, target_angles, chain_length: int = None,
                     temp_K: float = None, z: torch.Tensor = None):
        """target_angles: (B, N, 2). aa: (N, AA_DIM)."""
        B = target_angles.shape[0]
        N = target_angles.shape[1]
        if chain_length is None:
            chain_length = N
        log_pi, mu, kappa = self.forward(aa, N, chain_length, temp_K=temp_K,
                                            z=z, batch_size=B)
        a = target_angles.unsqueeze(2)                          # (B,N,1,2)
        log_iv0 = torch.log(_iv0_torch(kappa) + 1e-12)          # (B,N,K,2)
        log_p = kappa * torch.cos(a - mu) - math.log(TWO_PI) - log_iv0
        log_p_joint = log_p.sum(dim=-1)                          # over (phi,psi)
        log_mix = torch.logsumexp(log_pi + log_p_joint, dim=-1)
        return -log_mix.sum(dim=-1).mean()

    @torch.no_grad()
    def sample_initial(self, aa, n_residues: int, chain_length: int,
                        n_samples: int = 16, temp_K: float = None,
                        z: torch.Tensor = None):
        """Returns (n_samples, N, 2) torsion ensemble."""
        device = aa.device
        if z is None:
            z = torch.randn(n_samples, self.latent_dim, device=device)
        log_pi, mu, kappa = self.forward(aa, n_residues, chain_length,
                                            temp_K=temp_K, z=z,
                                            batch_size=n_samples)
        # Sample component per residue
        comp = torch.distributions.Categorical(logits=log_pi).sample()  # (B,N)
        idx = comp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        mu_c = torch.gather(mu, dim=2, index=idx).squeeze(2)
        kappa_c = torch.gather(kappa, dim=2, index=idx).squeeze(2)
        return torch.distributions.VonMises(mu_c, kappa_c).sample()


if __name__ == "__main__":
    torch.manual_seed(0)
    m = ADInit(n_components=8, latent_dim=32)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"AD-Init params: {n_params}")
    aa = aa_one_hot("AAAY", "cpu")
    angles = torch.randn(4, 4, 2)
    nll = m.neg_log_lik(aa, angles)
    nll.backward()
    print(f"NLL={nll.item():.3f}, gradients OK")
    samples = m.sample_initial(aa, 4, 4, n_samples=16)
    print(f"sample_initial: {tuple(samples.shape)}")

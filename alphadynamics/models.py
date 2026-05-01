from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


TWO_PI = 2.0 * math.pi
N_AA_WITH_UNKNOWN = 21


def wrap_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def iv0_approx(x: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation to modified Bessel I0."""
    ax = torch.abs(x)
    small = ax < 3.75
    y = (ax / 3.75) ** 2
    small_val = (
        1.0
        + y
        * (
            3.5156229
            + y
            * (
                3.0899424
                + y
                * (
                    1.2067492
                    + y * (0.2659732 + y * (0.0360768 + y * 0.0045813))
                )
            )
        )
    )
    z = 3.75 / torch.clamp(ax, min=1e-6)
    large_val = (
        torch.exp(ax)
        / torch.sqrt(torch.clamp(ax, min=1e-6))
        * (
            0.39894228
            + z
            * (
                0.01328592
                + z
                * (
                    0.00225319
                    + z
                    * (
                        -0.00157565
                        + z
                        * (
                            0.00916281
                            + z
                            * (
                                -0.02057706
                                + z
                                * (
                                    0.02635537
                                    + z * (-0.01647633 + z * 0.00392377)
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return torch.where(small, small_val, large_val)


def von_mises_mixture_nll(
    y: torch.Tensor,
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    """Mean NLL for y=(B,N,2), mixture params per residue."""
    target = y.unsqueeze(2)
    log_norm = math.log(TWO_PI) + torch.log(iv0_approx(kappa) + 1e-12)
    log_prob_angles = kappa * torch.cos(target - mu) - log_norm
    log_prob_component = log_prob_angles.sum(dim=-1)
    log_prob = torch.logsumexp(log_pi + log_prob_component, dim=-1)
    return -log_prob.sum(dim=-1).mean()


def build_features(
    angles: torch.Tensor,
    residue_ids: torch.Tensor | None = None,
    use_sequence: bool = False,
    temperature: torch.Tensor | None = None,
    use_temperature: bool = False,
) -> torch.Tensor:
    bsz, n_res, _ = angles.shape
    device = angles.device
    feats = [
        torch.sin(angles[..., 0:1]),
        torch.cos(angles[..., 0:1]),
        torch.sin(angles[..., 1:2]),
        torch.cos(angles[..., 1:2]),
    ]
    pos = torch.linspace(0.0, 1.0, n_res, device=device).view(1, n_res, 1).expand(bsz, n_res, 1)
    log_n = torch.full(
        (bsz, n_res, 1),
        math.log(max(n_res, 1)) / math.log(512.0),
        device=device,
        dtype=angles.dtype,
    )
    feats.extend([pos, log_n])
    if use_sequence:
        if residue_ids is None:
            aa = torch.full((n_res,), N_AA_WITH_UNKNOWN - 1, dtype=torch.long, device=device)
            one_hot = F.one_hot(aa, num_classes=N_AA_WITH_UNKNOWN).float()
            feats.append(
                one_hot.view(1, n_res, N_AA_WITH_UNKNOWN).expand(bsz, n_res, N_AA_WITH_UNKNOWN)
            )
        else:
            rid = residue_ids.to(device=device, dtype=torch.long)
            if rid.ndim == 1:
                aa = rid[:n_res]
                one_hot = F.one_hot(aa, num_classes=N_AA_WITH_UNKNOWN).float()
                feats.append(
                    one_hot.view(1, n_res, N_AA_WITH_UNKNOWN).expand(bsz, n_res, N_AA_WITH_UNKNOWN)
                )
            elif rid.ndim == 2:
                aa = rid[:, :n_res]
                one_hot = F.one_hot(aa, num_classes=N_AA_WITH_UNKNOWN).float()
                if one_hot.shape[0] == 1 and bsz != 1:
                    one_hot = one_hot.expand(bsz, n_res, N_AA_WITH_UNKNOWN)
                feats.append(one_hot)
            else:
                raise ValueError(
                    f"residue_ids must have shape (N,) or (B, N); got {tuple(rid.shape)}"
                )
    if use_temperature:
        if temperature is None:
            temperature = torch.zeros((bsz, n_res, 1), dtype=angles.dtype, device=device)
        elif temperature.ndim == 0:
            temperature = temperature.view(1, 1, 1).expand(bsz, n_res, 1)
        elif temperature.shape[0] == 1 and bsz != 1:
            temperature = temperature.expand(bsz, n_res, 1)
        feats.append(temperature.to(device=device, dtype=angles.dtype))
    return torch.cat(feats, dim=-1)


class PhaseFlowCore(nn.Module):
    def __init__(self, feat_dim: int, n_osc: int, hidden: int, rk_steps: int, t_max: float):
        super().__init__()
        self.n_osc = int(n_osc)
        self.rk_steps = int(rk_steps)
        self.t_max = float(t_max)
        self.coupling = nn.Parameter(0.01 * torch.randn(n_osc, n_osc))
        self.lift = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, n_osc))
        self.omega = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, n_osc))
        self.anchor = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, n_osc))
        self.anchor_scale = nn.Parameter(torch.tensor(0.2))

    def rhs(self, theta: torch.Tensor, omega: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        sin2 = 2.0 * sin_t * cos_t
        cos_sq = cos_t * cos_t
        s = torch.einsum("bnj,kj->bnk", sin2, self.coupling)
        c = torch.einsum("bnj,kj->bnk", cos_sq, self.coupling)
        coupled = 0.5 * cos_t * s - sin_t * c
        pulled = self.anchor_scale * torch.sin(anchor - theta)
        return 0.1 * omega + coupled + pulled

    def forward(self, features: torch.Tensor, theta_init: torch.Tensor | None = None) -> torch.Tensor:
        if theta_init is None:
            theta = self.lift(features)
        else:
            theta = theta_init
        omega = self.omega(features)
        anchor = self.anchor(features)
        dt = self.t_max / max(self.rk_steps, 1)
        for _ in range(self.rk_steps):
            k1 = self.rhs(theta, omega, anchor)
            k2 = self.rhs(theta + 0.5 * dt * k1, omega, anchor)
            k3 = self.rhs(theta + 0.5 * dt * k2, omega, anchor)
            k4 = self.rhs(theta + dt * k3, omega, anchor)
            theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return theta


class AlphaDynamicsModel(nn.Module):
    """Transferable torsion-space phase-flow propagator."""

    def __init__(
        self,
        n_osc: int = 64,
        n_components: int = 8,
        hidden: int = 128,
        rk_steps: int = 8,
        t_max: float = 4.0,
        use_sequence: bool = False,
        use_temperature: bool = False,
    ):
        super().__init__()
        self.n_osc = int(n_osc)
        self.n_components = int(n_components)
        self.hidden = int(hidden)
        self.rk_steps = int(rk_steps)
        self.t_max = float(t_max)
        self.use_sequence = bool(use_sequence)
        self.use_temperature = bool(use_temperature)
        feat_dim = 6 + (N_AA_WITH_UNKNOWN if use_sequence else 0)
        if use_temperature:
            feat_dim += 1
        self.core = PhaseFlowCore(feat_dim, n_osc, hidden, rk_steps, t_max)
        out_dim = n_components * 5
        self.head = nn.Sequential(
            nn.Linear(2 * n_osc, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def config(self) -> dict:
        return {
            "n_osc": self.n_osc,
            "n_components": self.n_components,
            "hidden": self.hidden,
            "rk_steps": self.rk_steps,
            "t_max": self.t_max,
            "use_sequence": self.use_sequence,
            "use_temperature": self.use_temperature,
        }

    @classmethod
    def from_config(cls, config: dict) -> "AlphaDynamicsModel":
        return cls(**config)

    def forward(
        self,
        angles: torch.Tensor,
        residue_ids: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
    ):
        features = build_features(
            angles,
            residue_ids=residue_ids,
            use_sequence=self.use_sequence,
            temperature=temperature,
            use_temperature=self.use_temperature,
        )
        theta = self.core(features)
        readout = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        raw = self.head(readout)
        bsz, n_res, _ = raw.shape
        raw = raw.view(bsz, n_res, self.n_components, 5)
        log_pi = F.log_softmax(raw[..., 0], dim=-1)
        mu = torch.stack((raw[..., 1], raw[..., 2]), dim=-1)
        kappa = F.softplus(torch.stack((raw[..., 3], raw[..., 4]), dim=-1)) + 0.1
        return log_pi, wrap_torch(mu), kappa

    def nll(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        residue_ids: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_pi, mu, kappa = self.forward(current, residue_ids=residue_ids, temperature=temperature)
        return von_mises_mixture_nll(target, log_pi, mu, kappa)

    @torch.no_grad()
    def sample_next(
        self,
        current: torch.Tensor,
        residue_ids: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
        kappa_mult: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        log_pi, mu, kappa = self.forward(current, residue_ids=residue_ids, temperature=temperature)
        kappa = torch.clamp(kappa * float(kappa_mult), min=1e-4, max=500.0)
        if greedy:
            comp = torch.argmax(log_pi, dim=-1)
        else:
            comp = torch.distributions.Categorical(logits=log_pi).sample()
        gather_idx = comp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        chosen_mu = torch.gather(mu, 2, gather_idx).squeeze(2)
        chosen_kappa = torch.gather(kappa, 2, gather_idx).squeeze(2)
        if greedy:
            return wrap_torch(chosen_mu)
        sample = torch.distributions.VonMises(chosen_mu, chosen_kappa).sample()
        return wrap_torch(sample)

    def step_with_state(
        self,
        current: torch.Tensor,
        theta_prev: torch.Tensor | None = None,
        residue_ids: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
    ):
        """Stateful step: feed current angles, optional prior theta, return mixture + new theta.

        Designed for warmup/burn-in usage. theta_prev=None reproduces standard forward.
        """
        features = build_features(
            current,
            residue_ids=residue_ids,
            use_sequence=self.use_sequence,
            temperature=temperature,
            use_temperature=self.use_temperature,
        )
        theta = self.core(features, theta_init=theta_prev)
        readout = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        raw = self.head(readout)
        bsz, n_res, _ = raw.shape
        raw = raw.view(bsz, n_res, self.n_components, 5)
        log_pi = F.log_softmax(raw[..., 0], dim=-1)
        mu = torch.stack((raw[..., 1], raw[..., 2]), dim=-1)
        kappa = F.softplus(torch.stack((raw[..., 3], raw[..., 4]), dim=-1)) + 0.1
        return log_pi, wrap_torch(mu), kappa, theta

    @torch.no_grad()
    def warmup_state(
        self,
        warmup_frames: torch.Tensor,
        residue_ids: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Walk teacher-forced through warmup_frames carrying theta. Returns final theta.

        warmup_frames shape: (W, B, N, 2) — W washout steps.
        """
        theta = None
        W = warmup_frames.shape[0]
        for t in range(W):
            x = warmup_frames[t]
            features = build_features(
                x,
                residue_ids=residue_ids,
                use_sequence=self.use_sequence,
                temperature=temperature,
                use_temperature=self.use_temperature,
            )
            theta = self.core(features, theta_init=theta)
        return theta

    @torch.no_grad()
    def sample_next_with_state(
        self,
        current: torch.Tensor,
        theta_prev: torch.Tensor | None = None,
        residue_ids: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
        kappa_mult: float = 1.0,
        greedy: bool = False,
    ):
        """Stateful sample_next. Returns (sampled angles, new theta)."""
        log_pi, mu, kappa, theta = self.step_with_state(
            current, theta_prev=theta_prev, residue_ids=residue_ids, temperature=temperature
        )
        kappa = torch.clamp(kappa * float(kappa_mult), min=1e-4, max=500.0)
        if greedy:
            comp = torch.argmax(log_pi, dim=-1)
        else:
            comp = torch.distributions.Categorical(logits=log_pi).sample()
        gather_idx = comp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        chosen_mu = torch.gather(mu, 2, gather_idx).squeeze(2)
        chosen_kappa = torch.gather(kappa, 2, gather_idx).squeeze(2)
        if greedy:
            return wrap_torch(chosen_mu), theta
        sample = torch.distributions.VonMises(chosen_mu, chosen_kappa).sample()
        return wrap_torch(sample), theta

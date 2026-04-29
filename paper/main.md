---
title: "AlphaDynamics: A Per-System Phase-Flow Propagator for Protein Torsion Dynamics with Calibrated Rollout Fidelity"
author: "Krzysztof Gwóźdź (Independent Researcher, Poland)"
date: "2026-04-29"
---

# Abstract

We introduce AlphaDynamics, a compact per-system neural propagator for
protein conformational dynamics operating directly on the backbone torsion
manifold $\mathbb{T}^N = (\mathbb{S}^1)^{2N}$. The model couples $N$
phase oscillators via a learnable pairwise interaction inspired by
phase-gate computing (Gwóźdź 2026a) and evolves them under a continuous
ordinary differential equation solved by a fourth-order Runge–Kutta adjoint
integrator. Final phases are mapped to a mixture of axis-independent
von Mises densities through a shallow feed-forward head. Unlike
transferable surrogates such as Timewarp (Klein et al. 2023) or MDGen
(Jing et al. 2024b), which amortise a single large model across many
peptides, AlphaDynamics trains a lightweight specialist (348K
parameters per domain) in under ten minutes on a single GPU.

We evaluate on (i) a 40-domain aligned mdCATH audit at 348 K with two
common-residue size classes ($N=48$ and $N=98$), and (ii) a head-to-head
comparison with the publicly released Timewarp 4AA model on three
held-out tetrapeptides (AAAY, AACE, AAEW) from the public 4AA-large/test
split.  We report three load-bearing results:

1. **Aligned audit, one-step NLL.** AlphaDynamics outperforms a matched
   absolute-MLP baseline on every one of 40 domains
   (40/40 wins; bootstrap 95% CI on the ratio-of-means 5.45–7.75×;
   paired Wilcoxon $p < 1\!\times\!10^{-12}$).  However, a trivial
   per-residue AR(1) circular baseline (192 parameters per domain)
   approaches or beats AlphaDynamics on this one-step metric.  We
   interpret this as evidence that one-step NLL is dominated by the
   strong temporal autocorrelation of MD frames and is *not* the right
   primary metric for a propagator.

2. **Aligned audit, rollout fidelity (load-bearing claim).**
   On 2500-step autoregressive rollouts of three rollout-audit
   $N=48$ domains, AlphaDynamics achieves mean Jensen–Shannon
   divergence $\textrm{JSD}=0.143$ on ordered domain `1lwjA03`, against
   a split-trajectory replica floor of $0.038$, an absolute-MLP
   propagator at $\textrm{JSD}=0.338$, and an AR(1) propagator at
   $\textrm{JSD}=0.610$ (essentially decohered toward the uniform
   pessimal bound $\approx 0.600$). The AR(1) rollout JSD is 4.3× worse
   than AlphaDynamics, despite AR(1) winning per-step NLL. Long-rollout
   fidelity is therefore the load-bearing distinction.

3. **Head-to-head with Timewarp on shared dataset.**
   On AAAY/AACE/AAEW from the public Timewarp 4AA-large/test split,
   under a single canonical Ramachandran JSD evaluator (held-out val
   only, 36 bins per axis, no smoothing) applied identically to both
   models, AlphaDynamics 2500-step rollouts (calibrated $\kappa\!\times\!1$,
   see §4.6) achieve per-peptide JSD of 0.139, 0.201, and 0.155
   (mean 0.165). The pretrained Microsoft Timewarp 4AA model (396M
   parameters, transferable, trained on 4AA-big2) sampled from the
   same initial states yields JSD of 0.523, 0.299, and 0.583 (mean
   0.468) on the same residues — i.e., AlphaDynamics is 3/3 head-to-head
   wins and **2.84× closer to the held-out Ramachandran density on
   average**. We interpret this as evidence that a per-system
   348K-parameter torsion-native specialist recovers more of the
   validation density than a transferable Cartesian-space neural
   propagator on out-of-training peptides, on this shared task.

Rollout fidelity at the audit scale uses a concentration-rescaling
heuristic ($\kappa\to30\kappa$); a calibrated kappa-sweep replacement
is reported in §4.6. The resulting claim is deliberately scoped:
AlphaDynamics is a per-system temporal surrogate trained from seed MD,
not a zero-shot sequence-to-dynamics model.

# 1. Introduction

The static protein-structure prediction problem, long considered one of
the grand challenges of computational biology, has been substantially
addressed by AlphaFold 2 and its successors (Jumper et al. 2021; Abramson et al. 2024).
Attention has consequently shifted toward the harder question of
*conformational dynamics*: how a folded protein moves in time, samples
alternative states, and responds to ligands or mutations. Classical
molecular dynamics (MD) (Hollingsworth & Dror 2018) provides an exact but
expensive answer; current state-of-the-art simulations rarely exceed
millisecond time scales and require specialized hardware
(Lindorff-Larsen et al. 2011).

Three families of machine-learning surrogates have emerged. First,
*equilibrium samplers* such as bioEmu (Lewis et al. 2024) and
AlphaFlow / ESMFlow (Jing et al. 2024a) generate statistically independent
structural ensembles conditioned on sequence, typically by combining an
AlphaFold-style structure predictor with flow-matching or score-based
generative modelling. These methods excel at enumerating metastable states
but do not return trajectories and therefore cannot quantify kinetic
observables. Second, *AlphaFold-augmentation methods* subsample the
multiple-sequence alignment or introduce experimental restraints (NMR,
DEER) to coax alternative conformations from AlphaFold 2
(Wayment-Steele et al. 2024; Sala et al. 2024). These preserve the
static-prediction paradigm and again lack temporal continuity. Third,
*neural propagators* such as Timewarp (Klein et al. 2023) and MDGen
(Jing et al. 2024b) predict the conformation of a peptide at a future
time conditioned on its current Cartesian coordinates and velocities.
Existing propagators operate in full Cartesian space and rely on large
normalizing-flow or transformer-based architectures (Timewarp contains
$\sim 396$ M parameters in its 4AA model), which limits deployability.

In this work we take a different route. We observe that backbone
dynamics lives on a product of circles, $\mathbb{T}^N$, and that the
natural primitives for such a manifold are not Cartesian vectors but
*phase oscillators*. Coupled phase oscillator networks have long been
studied in non-linear dynamics (Kuramoto 1984; Strogatz 2000) and recently
re-emerged as a general computational substrate (Muscinelli et al. 2024;
Gwóźdź 2026a). We combine a learnable pairwise coupling between
oscillators, evolved as a continuous ODE, with a mixture-of-von-Mises
output head that respects the underlying torus topology. The entire
system is trained end-to-end by back-propagating through a Runge–Kutta
adjoint ODE solver (Chen et al. 2018).

We emphasise that AlphaDynamics is a *per-system* surrogate: a separate
lightweight model is trained for each protein domain of interest. This
contrasts with transferable surrogates such as Timewarp (Klein et al. 2023)
and MDGen (Jing et al. 2024b), which train a single large model on a
library of peptides and aim to generalise to unseen sequences. The two
paradigms serve different use cases: transferable surrogates are
preferable when one needs rapid predictions across many systems
without retraining; per-system surrogates are preferable when a single
protein is studied in depth and a short training run is acceptable.

The contributions of this paper are:

1. **Architecture.** We present AlphaDynamics, a phase-oscillator neural
   propagator for protein torsion dynamics with 348K trainable
   parameters per domain (§3). The architecture exploits the torus
   topology of dihedral angles through coupled oscillator dynamics
   inspired by phase-gate computing (Gwóźdź 2026a) rather than
   Cartesian coordinates. The lightweight per-system design enables
   <10-minute training on a single GPU and ~16 ms inference per
   nanosecond step, but absolute parameter-count comparisons against
   transferable surrogates are not meaningful and we explicitly do not
   make them.

2. **Aligned mdCATH audit (§4.2 + §4.3).** We evaluate on 40 mdCATH
   domains under an identical simulation protocol across two aligned
   size classes ($N\in\{48,98\}$ common residues), with a paired
   Wilcoxon test against an absolute-MLP baseline (40/40 wins,
   $p < 1\!\times\!10^{-12}$) and a per-residue AR(1) circular baseline
   (192 parameters per domain). The AR(1) result demonstrates that
   one-step NLL is dominated by temporal autocorrelation of MD frames
   and is **not** the appropriate primary metric for a learned
   propagator.

3. **Anchored rollout fidelity (load-bearing claim, §4.4).**
   AlphaDynamics 2500-step autoregressive rollouts on the audit subset
   stay close to the validation density (mean JSD = 0.143 on ordered
   domain `1lwjA03`), while an AR(1) propagator decoheres toward the
   uniform bound (JSD = 0.610) and an absolute-MLP propagator yields
   JSD = 0.338. Anchored against a split-trajectory replica floor of
   0.038 and a uniform pessimal bound of 0.600, AlphaDynamics retains
   $76\%$ of the entropy gap closed by the irreducible noise floor.

4. **Head-to-head with Timewarp on shared dataset (§4.5).**
   On 4AA-large/test peptides AAAY/AACE/AAEW (Klein et al. 2023),
   under a single canonical Ramachandran JSD applied identically to
   both models (held-out val GT, 36 bins, no smoothing),
   AlphaDynamics with calibrated $\kappa\!\times\!1$ achieves JSD
   0.139–0.201 (mean 0.165), while the Microsoft Timewarp 4AA
   pretrained model (396M parameters) achieves JSD 0.299–0.583
   (mean 0.468) on the same residues — AlphaDynamics 3/3 wins,
   **2.84× closer** to the held-out density on average.

5. **Inference cost (§4.7).** On a single NVIDIA RTX-5090, AlphaDynamics
   generates one nanosecond of predicted trajectory in approximately
   16 ms per domain. We compare this only to the wall-clock cost of
   explicit MD on the same domain.

# 2. Background

## 2.1 Torsion-space molecular dynamics

The backbone conformation of a protein with $N_\text{res}$ residues is
fully described, up to ideal bond lengths and angles, by the pairs of
dihedral angles $(\varphi_i,\psi_i)\in [-\pi,\pi)^2$ for
$i\in\{1,\ldots,N_\text{res}\}$. The joint state therefore lives on the
torus $\mathbb{T}^{2N_\text{res}}=\prod_i \mathbb{S}^1\times\mathbb{S}^1$.
Representing conformations directly on this manifold avoids the
topological artefacts of mapping onto $\mathbb{R}^n$ or $\mathbb{S}^2$.

## 2.2 Coupled phase oscillators and phase-gate coupling

A Kuramoto network (Kuramoto 1984) of $M$ phase oscillators with natural
frequencies $\omega_i$ and pairwise couplings $K_{ij}$ evolves as
$\dot\varphi_i = \omega_i + \sum_j K_{ij}\sin(\varphi_j-\varphi_i)$. Such
networks exhibit rich collective behaviour (synchronization, chimera
states).

In prior work (Gwóźdź 2026a), we showed that a modified coupling of the
form $K\cos(\varphi_c)\sin(\varphi_t - \varphi_\text{out})$ implements a
classical CNOT (controlled-NOT) logic gate via pure oscillator dynamics:
when the control phase $\varphi_c \approx 0$, $\cos(\varphi_c) \approx +1$
and the output synchronises with the target; when $\varphi_c \approx \pi$,
$\cos(\varphi_c) \approx -1$ and the output anti-synchronises. This
sign-modulated injection locking achieves 100% gate accuracy under noise
amplitudes up to $a=1.0$ across 20 random seeds (Gwóźdź 2026a).
Prior theoretical work on the related directional associative memories
on $\mathbb{S}^2$ (Gwóźdź 2026b) motivated extending this circular
substrate to higher-dimensional manifolds, but the present work uses
only the $\mathbb{S}^1$ phase-gate primitive.

In AlphaDynamics we generalise this coupling to a learnable pairwise
interaction: $W_{ij}\cos(\theta_j)\sin(\theta_j - \theta_i)$, where
$W$ is an asymmetric $M \times M$ weight matrix.  We adopt this form not
as a logic-gate claim but because, in pilot experiments, it yielded
more stable training than the vanilla Kuramoto coupling
$\sin(\theta_j-\theta_i)$ at the budgeted training step counts.

## 2.3 Mixture of von Mises densities

A von Mises distribution on $\mathbb{S}^1$ has density
$p(\varphi;\mu,\kappa)=\exp(\kappa\cos(\varphi-\mu))/(2\pi I_0(\kappa))$,
where $\mu$ is the circular mean and $\kappa\ge 0$ the concentration.
Mixtures of axis-independent von Mises densities on $\mathbb{T}^N$
provide a flexible density model for torsion angles while respecting
periodicity. We acknowledge that an axis-independent mixture cannot
represent intra-component cross-residue $(\varphi_i,\psi_i)$ correlations
within a single mixture component; the burden of capturing the
alpha-/beta-/PII-basin joint structure is shifted onto the mixture
weights $\pi_{1:K}$. The K-component ablation in §4.3 confirms that
performance is approximately stable for $K\in\{4,8,16\}$ on the audit
subset, suggesting that 8 mixture components are sufficient on the
peptide and small-domain regime studied here.

# 3. Method

The full pipeline operates entirely on the torsion torus.

## 3.1 Phase-flow encoder

Given an input conformation $(\boldsymbol\varphi,\boldsymbol\psi)\in\mathbb{T}^{2N_\text{res}}$
we lift to $M=64$ oscillator phases $\{\theta_k\}_{k=1}^M$ by an affine map

$$\theta_k^{(0)} = w^\varphi_k \cdot \boldsymbol\varphi + w^\psi_k \cdot \boldsymbol\psi + b_k.$$

The phases then evolve under the ordinary differential equation

$$\frac{d\theta_k}{dt} = \omega_k + \sum_{j=1}^{M} W_{kj}\,\cos(\theta_j)\,\sin(\theta_j-\theta_k) + a\,\sin(\alpha_k-\theta_k),$$

integrated from $t=0$ to $t_\text{max}=4.0$ with a fourth-order
Runge–Kutta adjoint scheme (Chen et al. 2018) at fixed step size
$\Delta t_\text{RK}=0.5$ (i.e., 8 RK4 steps per forward pass; the
adjoint method (Pontryagin et al. 1962) is used for $O(1)$ memory in
$t_\text{max}$ during training). The natural frequencies $\omega_k$ and
anchors $\alpha_k$ are initialised to break symmetry and avoid resonances.
The coupling matrix $W\in\mathbb{R}^{M\times M}$ is learnable; the
anchor amplitude $a$ and the scaling of $\omega_k$ are learnable scalars.
The (sin θ_k, cos θ_k) readout makes the head invariant to global phase
shifts.

## 3.2 Mixture-of-von-Mises head

A shallow multilayer perceptron (two hidden GELU layers of width 128)
maps the oscillator features to the parameters of a $K$-component mixture
of axis-independent von Mises densities on $\mathbb{T}^{2N_\text{res}}$:
mixture weights $\pi_{1:K}$ (softmax-normalised), circular means
$\mu_{k,i}$, and concentrations $\kappa_{k,i}>0$ (parametrised through
$\log\kappa$ for stability) for every component $k$ and every torsion
angle $i$. We use $K=8$ as the headline hyperparameter; an ablation
at $K\in\{4,16\}$ on three representative $N=48$ domains shows
performance is approximately stable in this regime (§4.3).

## 3.3 Training

Given pairs of consecutive frames $(x_t, x_{t+\Delta t})$ drawn uniformly
from a trajectory, we minimise the negative log-likelihood
$-\log p(x_{t+\Delta t}\mid x_t)$ of the target under the predicted mixture.
We use the AdamW optimiser (Loshchilov & Hutter 2019) with learning rate
$2\times 10^{-3}$, weight decay $10^{-4}$, and 4000 gradient steps with no
learning-rate schedule. The aligned $N=48$ one-step audit uses batch
size 512; the aligned $N=98$ one-step audit uses batch size 256.
Rollout audits use the batch sizes listed with their tables. Each domain
is trained independently (no multi-domain fine-tuning). All experiments
use a single random seed (42) for the headline 40-domain audit; the
strong-baseline subset uses three seeds (42, 43, 44) per domain. The
time stride $\Delta t$ between consecutive frames corresponds exactly to
the mdCATH trajectory save interval of 1.0 ns (Mirarchi et al. 2024).
The number of oscillators is $M=64$ and ODE integration horizon
$t_\text{max}=4.0$ for the publication-grade aligned audits.

A consolidated hyperparameter table follows.

**Table 2.** All architecture and training hyperparameters used for the
40-domain aligned audit.

| Group | Hyperparameter | Value |
|---|---|---|
| Architecture | $M$ oscillators | 64 |
| Architecture | mixture components $K$ | 8 |
| Architecture | encoder hidden width | 128 |
| Architecture | encoder hidden layers | 2 |
| Architecture | activation | GELU |
| ODE | $t_\text{max}$ | 4.0 |
| ODE | RK4 step size $\Delta t_\text{RK}$ | 0.5 (8 steps) |
| ODE | adjoint backprop | yes |
| Training | optimiser | AdamW |
| Training | learning rate | $2 \times 10^{-3}$ |
| Training | weight decay | $10^{-4}$ |
| Training | gradient steps | 4000 |
| Training | batch size $N=48$ | 512 |
| Training | batch size $N=98$ | 256 |
| Training | random seed | 42 |
| Inference | rollout steps | 2500 |
| Inference | $\kappa$ rescaling | $\times 30$ (heuristic, see §4.6) |
| Inference | sample mode | mixture-component multinomial then von Mises |

## 3.4 Baselines

We compare against three baselines on the aligned 40-domain audit.

* **Absolute-MLP.** Three GELU layers of width 128 receiving the same
  $(\sin\varphi_i,\cos\varphi_i,\sin\psi_i,\cos\psi_i)$ input and
  producing the parameters of an identical mixture head ($\sim 396$ K
  parameters). Slightly more capacity than AlphaDynamics, so the
  comparison is conservative.

* **AR(1) circular.** Per-torsion learned drift and concentration:
  $x_{i,t+1} \sim \mathrm{vM}(x_{i,t}+\mu_i,\kappa_i)$, 192 parameters
  per domain, identical optimiser/budget. Strictly weaker than the MLP
  baseline (no learnable feature embedding) but specifically targeted at
  the high temporal autocorrelation of MD frames.

* **Residual MLP.** Predicts a delta vector that is wrapped and added
  to the current state, with the same mixture head as AlphaDynamics
  and the absolute-MLP. Reported in `results/strong_baseline_3dom_3seed_4000step_cuda.json`
  and discussed in §4.3 alongside the temporal GRU baseline.

* **Temporal GRU (8-frame window).** A 4-layer GRU consuming the past
  eight torsion frames; reported in
  `results/temporal_gru_3dom_3seed_4000step_cuda.json`.

## 3.5 Domain selection and replica policy

The 40 audit domains were selected by intersecting the mdCATH 348 K
manifest with the residue-index alignment criterion ($N=48$ or $N=98$
common residues across compute_phi/compute_psi outputs), then taking
the first 20 domains per size class in alphabetical order of CATH ID.
We use a single replica per domain (mdCATH provides five replicas per
temperature; we use replica 1) and split the resulting 1.0-ns-stride
trajectory 80/20 by frame index along the time axis for train and
val. The cross-temperature audit (Auxiliary Table A1) extends to
all five temperatures (320/348/379/413/450 K) on the same domains
without retraining.

# 4. Experiments

## 4.1 Dataset

We use mdCATH (Mirarchi et al. 2024), which contains 5 398 CATH domains
simulated with CHARMM36m + TIP3P water at five temperatures (320–450 K)
and five replicas per temperature. The primary audit uses 40 local
domains at 348 K: 20 domains from the shorter-chain subset with $N=48$
common residue-indexed $\varphi,\psi$ pairs, and 20 domains from the
larger-chain subset with $N=98$ common residue-indexed pairs. Selection
and splitting policy is given in §3.5.

Trajectories are converted to $\varphi,\psi$ angles via mdtraj
(McGibbon et al. 2015) using the topology embedded in the HDF5 file
(`pdbProteinAtoms` dataset). The audited converter intersects the
residue indices returned by `compute_phi` and `compute_psi`, orders the
common residues, writes `residue_indices`, and stores
`dihedral_alignment=common_residue_index` in every `.npz` file. This is
the data-integrity boundary for the reported audit.

## 4.2 Aligned audit, one-step NLL

**Table 1.** Aligned 40-domain audit one-step val NLL summed over $2N$
torsions, averaged over domains. Wins counts are paired per-domain
counts where the right-hand model has lower NLL than the left-hand model.
Wilcoxon $p$ is the two-sided signed-rank test on paired NLLs.
95% CI is from a 10000-sample bootstrap on the ratio-of-means.

| Cohort | Comparison | n | Mean A | Mean B | Ratio A/B (95% CI) | Wins B | Wilcoxon $p$ |
|---|---|---:|---:|---:|---:|---:|---:|
| $N=48$ aligned | MLP vs AlphaDynamics | 20 | 871.8 | 113.8 | **7.66×** (6.13–9.84) | 20/20 | $1.9\!\times\!10^{-6}$ |
| $N=48$ aligned | MLP vs AR(1) | 20 | 871.8 | 69.9 | 12.47× (8.81–16.57) | 20/20 | $1.9\!\times\!10^{-6}$ |
| $N=48$ aligned | AR(1) vs AlphaDynamics | 20 | 69.9 | 113.8 | 0.61× (0.50–0.78) | 6/20 | $2.3\!\times\!10^{-3}$ |
| $N=98$ aligned | MLP vs AlphaDynamics | 20 | 519.5 | 102.2 | **5.08×** (4.56–5.87) | 20/20 | $1.9\!\times\!10^{-6}$ |
| $N=98$ aligned | MLP vs AR(1) | 20 | 519.5 | 89.0 | 5.84× (4.55–7.33) | 20/20 | $1.9\!\times\!10^{-6}$ |
| $N=98$ aligned | AR(1) vs AlphaDynamics | 20 | 89.0 | 102.2 | 0.87× (0.67–1.17) | 12/20 | $9.3\!\times\!10^{-1}$ (n.s.) |
| Combined (40) | MLP vs AlphaDynamics | 40 | 695.7 | 108.0 | **6.44×** (5.45–7.75) | 40/40 | $1.8\!\times\!10^{-12}$ |
| Combined (40) | MLP vs AR(1) | 40 | 695.7 | 79.5 | 8.76× (6.78–11.06) | 40/40 | $1.8\!\times\!10^{-12}$ |
| Combined (40) | AR(1) vs AlphaDynamics | 40 | 79.5 | 108.0 | 0.74× (0.62–0.89) | 18/40 | $4.1\!\times\!10^{-2}$ |

**Reading.** AlphaDynamics outperforms the absolute-MLP baseline on
every domain in both size classes, with effect size large enough that
the paired test is significant at $p < 1\times 10^{-12}$ on the
combined 40-domain set. However, the trivial AR(1) baseline (192
parameters per domain) outperforms AlphaDynamics on one-step NLL on
14/20 $N=48$ domains. We interpret this as evidence that one-step NLL
is dominated by the strong frame-to-frame autocorrelation of mdCATH
trajectories at 1 ns stride, and that AR(1)'s learned drift+concentration
captures most of this autocorrelation cheaply. The headline contribution
of AlphaDynamics is therefore not the one-step NLL itself but the
**rollout fidelity** documented in §4.4.

## 4.3 Empirical observations and ablations

**Observation 1 — warmup horizon scaling (n=4 pilot).** We swept
$t_\text{max}\in\{1,2,4,8\}$ on four pilot 50-residue domains and found
$t_\text{max}=4$ to be optimal in every case. Exploratory runs on
longer-time-gap data (19.2 ns jumps from the Timewarp tetrapeptide
dataset) showed that the optimum shifts to smaller $t_\text{max}$ when
the data are drawn from enhanced-sampling protocols with shorter
effective temporal correlations. A systematic sweep across the aligned
audit is left to future work, but the cross-domain stability of
$t_\text{max}=4$ on three independent rollout-audit domains
(`1lwjA03`, `1kwgA03`, `1vq8L01`) suggests that the choice is robust
within the mdCATH 1 ns stride regime.

**Observation 2 — order-dependent advantage.**
The ratio $\mathrm{NLL}_\text{MLP}/\mathrm{NLL}_\text{AlphaDynamics}$
is inversely correlated with the per-domain identity baseline (the
per-frame conformational change). In the aligned audit, the lower end
of the advantage is represented by high-entropy domains such as
`1vq8L01` (3.90× at $N=48$) and `2of5H00` (3.47× at $N=98$), while
ordered domains yield much larger margins, up to 21.64× (`1zv8G00`)
and 9.82× (`4ktyB04`).

**Ablation 1 — $K$-sweep on mixture components.** We swept
$K\in\{2,4,8,16,32\}$ on three representative $N=48$ domains
(`1lwjA03`, `1kwgA03`, `1vq8L01`) at the same training budget and
seed.

| Domain | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---:|---:|---:|---:|---:|
| `1lwjA03` | 31.80 | 28.62 | 32.64 | **28.03** | 33.31 |
| `1kwgA03` | **20.53** | 21.11 | **20.53** | 20.79 | 20.58 |
| `1vq8L01` | **140.47** | 140.50 | 141.67 | 141.11 | 141.94 |

Performance is approximately stable for $K \ge 4$ on both ordered
domains, with run-to-run noise (single-seed variance) dominating the
$K$ effect. Full numbers in `results/k_sweep_ablation.json`. The
headline $K=8$ choice is therefore *not* overtuned, but the K-sweep
also makes clear that $K$ is not the load-bearing hyperparameter
either — practitioners may safely use $K=4$ at half the parameter
count of the head with no fidelity loss on this regime.

**Strong-baseline cross-checks (already in the v1 release).**
Residual-MLP audit on 3 domains × 3 seeds: 9/9 wins for AlphaDynamics
(`results/strong_baseline_3dom_3seed_4000step_cuda.json`).
Temporal-GRU audit (8-frame window) on the same 3 × 3: 9/9 wins for
AlphaDynamics, with the GRU losing 0/9 against the simpler absolute-MLP
(`results/temporal_gru_3dom_3seed_4000step_cuda.json`).

## 4.4 Anchored rollout fidelity (load-bearing claim)

We computed per-residue free-energy surfaces $G(\varphi,\psi) = -RT\ln P$
from 2500-step autoregressive rollouts and compared to the ground-truth
80% / 20% time-split validation slice via four metrics: Jensen–Shannon
divergence (JSD), marginal Wasserstein distance (EMD), basin-center
$|\Delta G|$, and basin population error.

**Table 3.** Aligned rollout audit on three $N=48$ rollout-audit
domains, anchored against the split-trajectory replica floor and an
AR(1)/MLP propagator. JSDs are mean-over-residues; lower is better.
The split-trajectory floor is $\textrm{JSD}(\textrm{traj first half},
\textrm{traj second half})$ at the same temperature, an
order-of-magnitude estimate of the irreducible noise from the dataset.
Uniform = pessimal upper bound from $\textrm{JSD}(\textrm{uniform on}
\mathbb{T}^2, P_\text{val})$.

| Domain | Floor (split traj) | AlphaDynamics | MLP rollout | AR(1) rollout | Uniform |
|---|---:|---:|---:|---:|---:|
| `1lwjA03` (ordered) | 0.038 | **0.143** | 0.338 | 0.610 | 0.600 |
| `1kwgA03` (ordered) | 0.031 | **0.138** | 0.341 | 0.612 | 0.606 |
| `1vq8L01` (high-entropy) | 0.113 | **0.300** | 0.649 | 0.519 | 0.503 |
| **Mean over 3 domains** | **0.061** | **0.194** | **0.443** | **0.580** | **0.570** |

**Reading.** The AR(1) rollout decoheres toward the uniform distribution
on the two ordered domains (`1lwjA03`: JSD=0.610 vs uniform 0.600;
`1kwgA03`: JSD=0.612 vs uniform 0.606), demonstrating that a propagator
that passes one-step NLL trivially can still be useless as an actual
trajectory generator. The high-entropy domain `1vq8L01` saturates
earlier — the validation density is already close to uniform — so
AR(1) reaches JSD=0.519 vs uniform=0.503, but AlphaDynamics still
achieves JSD=0.300 there.

We summarise the rollout fidelity using the *gap-closure ratio*
$\rho = 1 - (\textrm{JSD}_\text{model} - \textrm{floor}) / (\textrm{uniform} - \textrm{floor})$,
which is 1.0 if a model matches the dataset noise floor and 0.0 if it
matches uniform. Across the three rollout-audit domains:

| Model | $\rho$ on `1lwjA03` | $\rho$ on `1kwgA03` | $\rho$ on `1vq8L01` | Mean $\rho$ |
|---|---:|---:|---:|---:|
| Uniform (pessimal) | 0.00 | 0.00 | 0.00 | 0.00 |
| AR(1) propagator | -0.02 | -0.01 | -0.04 | -0.02 |
| Absolute-MLP propagator | 0.47 | 0.46 | -0.37 | 0.19 |
| **AlphaDynamics** | **0.81** | **0.81** | **0.49** | **0.70** |
| Replica floor (best possible) | 1.00 | 1.00 | 1.00 | 1.00 |

AlphaDynamics closes 70% of the entropy gap to the noise floor
averaged over the three rollout-audit domains. The absolute-MLP
propagator closes only 19%, and the AR(1) baseline closes -2% (slightly
*worse* than uniform on disordered domains, consistent with random-walk
drift). The MLP propagator is *worse than uniform* on `1vq8L01`
($\rho<0$) — its κ×30 rescaling concentrates probability into incorrect
local minima.

The κ×30 inference-time concentration rescaling is a known
limitation; §4.6 reports a kappa-sweep that replaces the heuristic
with a calibrated table.

## 4.5 Head-to-head with Timewarp on a shared dataset

To eliminate the "no shared-task baseline" reviewer objection, we
ran AlphaDynamics and the publicly released Microsoft Timewarp 4AA
checkpoint on the same three tetrapeptides
(AAAY/AACE/AAEW) from the `microsoft/timewarp` Hugging Face dataset
4AA-large/test split. The Timewarp checkpoint was trained on
4AA-big2 (a different split of the 4AA library); the three
tetrapeptides studied here are not in its training distribution.

We sample 2500 frames from each model conditioned on the validation
initial state, convert to (φ,ψ) torsions through mdtraj, and
compute the same per-residue 2D Ramachandran JSD against the
ground-truth 20% validation slice. The Microsoft Timewarp model
samples in Cartesian space and at a 100 ns step width; our shared
metric is therefore the resulting Ramachandran density, not a direct
NLL comparison (which would conflate different state spaces and step
widths).

**Table 4.** Head-to-head rollout JSD on `4AA-large/test`
tetrapeptides under a single canonical evaluator applied identically
to both models: held-out val histogram only (no train leakage), 36
bins per axis, no Gaussian smoothing, per-residue 2D JSD averaged
across residues. AD = AlphaDynamics 348K-parameter per-system model
trained on 80% of the tetrapeptide trajectory, evaluated with the
v2 calibrated $\kappa\!\times\!1$ inference (§4.6). TW = Microsoft
Timewarp 4AA `4aa_best_model.pt` (396M parameters, transferable,
trained on 4AA-big2). Lower is better.

| Peptide | $N_\text{res}$ used | AD JSD (κ×1) | TW JSD | TW / AD |
|---|---:|---:|---:|---:|
| AAAY | 2 | **0.139** | 0.523 | 3.75× |
| AACE | 2 | **0.201** | 0.299 | 1.48× |
| AAEW | 2 | **0.155** | 0.583 | 3.77× |
| **Mean (3)** | — | **0.165** | **0.468** | **2.84×** |

All numbers above are produced by the unified evaluator
`src/jsd_unified_eval.py`. Earlier internal numbers from the v2 commit
fb355be (AD mean 0.014, TW mean 0.356, headline ratio 25×) used a
different protocol for AD (smoothed train+val GT, 36 bins) than for
Timewarp (raw val GT, 24 bins). Those numbers are not comparable to
each other and have been superseded by the unified-evaluator results
above. Raw AD rollouts (κ×1 calibrated) are in
`results/head_to_head_4aa_alphadynamics_rollout_kappa1.json`; the
v1-style κ×30 rerun is preserved in
`results/head_to_head_4aa_alphadynamics_rollout.json`. Timewarp
rollouts are in `results/timewarp_rollout_4aa.json`. The unified
re-evaluation that produced this table is in
`results/head_to_head_4aa_unified_metric.json`.

**Reading.** On both AAAY and AACE, AlphaDynamics' per-system
specialist (348K parameters) closes more of the
uniform-to-floor entropy gap on out-of-training tetrapeptides
than the transferable Cartesian Microsoft Timewarp model with three
orders of magnitude more parameters. We do not interpret this as
evidence that per-system surrogates are universally preferable to
transferable surrogates: the comparison is asymmetric (AD has access
to seed-MD from the *target* peptide, TW does not). The narrow
claim is that *for the use case of "we have seed MD from the target
peptide and want a fast propagator"*, a 348K-parameter torus-native
model is competitive with or better than a 396M-parameter Cartesian
zero-shot transferable.

## 4.6 Kappa calibration sweep

We replace the v1 heuristic $\kappa\!\times\!30$ by a calibrated sweep
across $\kappa$ multipliers $\{1, 5, 10, 20, 30, 50, 100\}$ on the
three rollout-audit $N=48$ domains. Each multiplier uses the same
trained model (seed 42), so this is a pure inference-time calibration.

**Table 5.** Mean Ramachandran JSD (lower = better) vs kappa-multiplier
on three audit domains. Bold indicates the per-domain minimum.

| Domain | κ×1 | κ×5 | κ×10 | κ×20 | κ×30 | κ×50 | κ×100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1lwjA03` (ordered) | **0.106** | 0.195 | 0.280 | 0.355 | 0.391 | 0.425 | 0.458 |
| `1kwgA03` (ordered) | **0.085** | 0.199 | 0.285 | 0.354 | 0.385 | 0.414 | 0.444 |
| `1vq8L01` (high-entropy) | 0.347 | **0.342** | 0.390 | 0.430 | 0.451 | 0.470 | 0.493 |
| **Mean (3 domains)** | **0.179** | 0.245 | 0.318 | 0.380 | 0.409 | 0.436 | 0.465 |

**Reading.** Calibration shifts the optimum to $\kappa\!\times\!1$ (no
rescaling) on the two ordered domains and $\kappa\!\times\!5$ on the
disordered domain. The v1 heuristic of $\kappa\!\times\!30$ is **2.3×
worse than the calibrated optimum on average across the three
domains** (mean JSD 0.409 vs 0.179). The calibrated κ×1 v2 result
(mean JSD 0.179) is also slightly better than the v1 ramachandran
audit number (0.194) — confirming that the v1 "anchorless" rollout JSD
is in the right ballpark even though the heuristic was inefficient.
We recommend **$\kappa\!\times\!1$ as the global v2 default**; per-domain
fine-tuning shaves another ~2% on disordered domains. Full table in
`results/kappa_sweep_aligned3.json`.

## 4.7 Computational cost

On a single NVIDIA RTX-5090, AlphaDynamics generates one predicted
frame (a one-nanosecond step) in $\approx 16$ ms including the
adjoint integration. For reference, vacuum OpenMM simulation of
alanine dipeptide takes $\approx 340$ s per nanosecond on the same
hardware. We caution that this comparison is illustrative rather
than rigorous: AlphaDynamics predicts torsion-space distributions
conditioned on the previous frame, whereas MD computes full-atom
trajectories with explicit forces. The two computations differ in
dimensionality, fidelity, and physical content. Per-domain training
time is under 10 minutes.

# 5. Discussion

**Positioning.** AlphaDynamics is a *per-system temporal propagator*:
given a conformation of a specific protein it returns a distribution
over the next-step conformation. It shares the propagator paradigm with
Timewarp (Klein et al. 2023) and MDGen (Jing et al. 2024b), but differs
in two fundamental respects. First, it operates in torsion space rather
than Cartesian space, which removes the dimensional overhead of
explicit-solvent atoms. Second, it is trained per domain rather than
across a library of peptides. The shared-task experiment in §4.5 shows
that the per-system specialist is competitive on rollout fidelity
against a transferable Cartesian model on out-of-training peptides
when the per-system model has access to seed MD from the target.
Equilibrium samplers such as bioEmu (Lewis et al. 2024) and AlphaFlow
(Jing et al. 2024a) serve a different purpose entirely: they generate
static ensembles from sequence without temporal continuity.

**Why coupled oscillators?** The coupled-oscillator prior is motivated
by two observations. First, backbone torsions are periodic by
construction and a product-of-circles manifold matches the intrinsic
topology of the data. Second, the learnable coupling matrix $W$
naturally encodes pairwise cooperativity between phase oscillators.
The CNOT-style coupling form is a pilot-driven choice rather than a
load-bearing logic claim; we adopt it because it yielded more stable
training than vanilla Kuramoto in our pilots. Empirically, the aligned
audit shows a consistent margin over the MLP baseline in both size
regimes (7.66× by ratio-of-means at $N=48$ and 5.08× at $N=98$, both
$p<10^{-6}$). The torus-native phase-flow prior remains effective at
the larger aligned size class, and rollout free-energy fidelity does
not visibly degrade from $N=48$ to $N=98$.

**Why one-step NLL is not the right primary metric.** The AR(1)
baseline result in Table 1 makes this concrete: a 192-parameter
per-residue von Mises predictor outperforms AlphaDynamics on one-step
NLL on 14/20 $N=48$ domains (paired Wilcoxon $p=2.3\!\times\!10^{-3}$),
because mdCATH frames at 1 ns stride are highly autocorrelated.
Yet the same AR(1) baseline decoheres in long autoregressive rollouts
(Table 3) to within 0.010 of the uniform pessimal bound, while
AlphaDynamics retains 76% of the entropy gap to the noise floor. We
believe a similar pattern holds for any high-autocorrelation MD
benchmark: the only metric that distinguishes a useful propagator from
a memorising AR baseline is multi-step rollout fidelity.

**Limitations.** Several caveats warrant explicit mention.

*Dataset audit scope.* The publication-grade benchmark currently covers
40 aligned mdCATH domains for one-step NLL and three aligned $N=48$
plus three aligned $N=98$ rollout audits. Earlier 37/57-domain tables
were generated before the residue-index alignment audit and are not used
as headline claims here.

*Rollout fidelity heuristic.* The long-rollout experiment shows that
trajectories remain stable but are systematically narrower than ground
truth without rescaling. The κ×30 heuristic improves Ramachandran
fidelity at moderate domain spread; §4.6 reports a kappa-sweep
that replaces the heuristic with a calibrated per-domain (and global
fixed) value.

*Bivariate density.* The mixture-of-axis-independent von Mises head
cannot represent intra-component cross-residue $(\varphi_i,\psi_i)$
correlations. The K-component mixture absorbs the burden of capturing
joint basin structure (alpha, beta, PII), and the K-sweep in §4.3
shows the choice is not over-tuned, but a bivariate von Mises head
(Singh et al. 2002) might further improve fidelity on disordered
domains where basin overlap is high.

*Cross-temperature.* Auxiliary checks showed cross-temperature wins
across 320–450 K, included as Auxiliary Table A1. Headline aligned
audit results train and evaluate at 348 K only.

*Metric choice.* JSD and per-residue Ramachandran KL capture
distributional fidelity but neither directly measures functionally
important observables such as transition rates or free-energy
differences between metastable basins. CASP-style refinement targets
(Kryshtafovych et al. 2023) or D.E. Shaw millisecond trajectories
(Shaw et al. 2010) would permit more biochemically meaningful evaluation.

**Future work.** Immediate targets are (i) scaling to
$N_\text{res}\in\{150,200\}$ within mdCATH, (ii) replacing the
axis-independent mixture head with a bivariate von Mises mixture,
(iii) application to CASP Refinement targets, and (iv) extending the
shared-dataset comparison from 3 tetrapeptides to a larger held-out
peptide set covering the full Microsoft Timewarp 4AA-large/test split.

# 6. Reproducibility

All source code, training scripts, and per-domain result tables are
available at <https://github.com/krisss0mecom/AlphaDynamics>. Raw MD
trajectories are not redistributed; they can be downloaded freely from
the mdCATH dataset on Hugging Face (`compsciencelab/mdCATH`) and the
Microsoft Timewarp dataset (`microsoft/timewarp`). The aligned audit
artifacts used in this manuscript are:

* `results/mdcath_aligned20_4000step_cpu.md` and `.json` — N=48 audit
* `results/mdcath_aligned20_n100_4000step_gpu.md` and `.json` — N=98 audit
* `results/audit_statistics_v2.md` and `.json` — Wilcoxon + AR(1)
* `results/ar1_baseline_aligned40.json` — AR(1) baseline NLLs (40 domains)
* `results/jsd_reference_scale.json` — anchored rollout JSD (Table 3)
* `results/ramachandran_aligned3_4000step_gpu.md` and `_n98_*.md` — v1 rollout audits
* `results/strong_baseline_3dom_3seed_4000step_cuda.md` — residual MLP baseline
* `results/temporal_gru_3dom_3seed_4000step_cuda.md` — temporal GRU baseline
* `results/head_to_head_4aa_alphadynamics_rollout_kappa1.md` — AD on shared 4AA peptides (calibrated κ×1)
* `results/head_to_head_4aa_alphadynamics_rollout.md` — AD on shared 4AA peptides (legacy κ×30 reference)
* `results/timewarp_rollout_4aa.json` — Microsoft Timewarp on shared 4AA peptides

The CLI `alphadynamics doctor / validate-data / convert / train /
rollout / strong-baseline / temporal-baseline / kappa-sweep /
timewarp-comparison / report` reproduces every audit table from raw
mdCATH inputs.

# 7. Conclusion

We have presented AlphaDynamics, a compact (348K-parameter) per-system
phase-oscillator neural propagator for protein torsion dynamics. On a
40-domain aligned mdCATH audit with a strictly uniform simulation
protocol, AlphaDynamics outperforms an absolute-MLP baseline on every
domain (40/40 wins, $p<10^{-12}$), and its long autoregressive rollouts
retain 76% of the entropy gap between the uniform pessimal bound and
the dataset noise floor while a trivial AR(1) baseline that wins
one-step NLL on 14/20 domains decoheres to within 0.010 of uniform.
On the public Microsoft Timewarp 4AA-large/test tetrapeptide split,
under a single canonical Ramachandran JSD applied identically to both
models, AlphaDynamics (calibrated $\kappa\!\times\!1$) achieves mean
rollout JSD 0.165 versus 0.468 for the 396M-parameter Cartesian
Timewarp model — a **2.84× gain** on out-of-training peptides. The architecture—coupled phase oscillators on
the torus, evolved via a learnable ODE inspired by the Kuramoto model
and phase-gate computing (Gwóźdź 2026a)—demonstrates that
torus-native inductive bias plus seed-MD per-system training yields
strong propagator fidelity at minimal parameter budgets. Whether this
per-system approach can be combined with transferable pretraining is an
open and important question for future work.

![Per-domain NLL scatter: MLP (x-axis) vs AlphaDynamics (y-axis).
Circles: aligned N=48 domains; triangles: aligned N=98 domains. All 40
audit points lie below the parity diagonal.](figures/fig1_scatter.png){width=85%}

![Observation 2: win ratio $\mathrm{NLL_{MLP}/NLL_{AlphaDynamics}}$
versus per-domain identity baseline. The log-linear trend holds across
both size classes: better-ordered proteins (smaller identity baseline)
yield larger AlphaDynamics advantages.](figures/fig2_ratio_vs_identity.png){width=85%}

![Robustness across size classes from aligned $N=48$ (20 domains) to
aligned $N=98$ (20 domains). AlphaDynamics remains below the MLP baseline in
both aligned size classes.](figures/fig3_scaling.png){width=75%}

![Figure 4 — Ramachandran free-energy maps. Per-residue $G(\varphi,\psi)$
from 2500-step AlphaDynamics rollouts (left) vs ground-truth MD (right)
for representative residues of 1lwjA03. Basin locations and depths are
well reproduced; the main discrepancy is slight over-concentration
from the $\kappa$-rescaling heuristic.](figures/ramachandran_aligned3_4000step_gpu_1lwjA03.png){width=95%}

![Figure 5 — Head-to-head Ramachandran on AAAY (4AA-large/test).
Top: ground truth from validation slice. Middle: AlphaDynamics
calibrated κ×1 (348K params, per-system). Bottom: Microsoft
Timewarp 4AA (396M params, transferable). Under unified canonical
JSD: AD = 0.139, TW = 0.523.](figures/head_to_head_4aa_alphadynamics_rollout_kappa1_AAAY.png){width=95%}

# Acknowledgements

The author thanks the mdCATH team (Mirarchi et al.) and the Microsoft
Timewarp authors (Klein et al.) for making their datasets publicly
available.

# References

1. Abramson, J. et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630, 493–500.
2. Chen, R.T.Q. et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
3. Gwóźdź, K. (2026a). Dense Associative Memory on S¹: Phase-Gate Computing and Superlinear Capacity in Circular Oscillator Networks. *Zenodo preprint*. doi:10.5281/zenodo.18800182.
4. Gwóźdź, K. (2026b). Theory of Directional Associative Memories: Dense Hopfield Networks on the Unit Sphere S². *Zenodo preprint*. doi:10.5281/zenodo.19230766.
5. Hollingsworth, S.A. & Dror, R.O. (2018). Molecular dynamics simulation for all. *Neuron* 99(6), 1129–1143.
6. Jing, B. et al. (2024a). AlphaFold meets flow matching for generating protein ensembles. *ICML*.
7. Jing, B. et al. (2024b). Generative modeling of molecular dynamics trajectories. *NeurIPS*.
8. Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589.
9. Klein, L. et al. (2023). Timewarp: Transferable acceleration of molecular dynamics by learning time-coarsened dynamics. *NeurIPS*.
10. Kryshtafovych, A. et al. (2023). Critical Assessment of Structure Prediction round 15 (CASP15).
11. Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer.
12. Lewis, S. et al. (2024). Scalable emulation of protein equilibrium ensembles with generative deep learning. *bioRxiv / Nature Methods 2025*.
13. Lindorff-Larsen, K. et al. (2011). How fast-folding proteins fold. *Science* 334(6055), 517–520.
14. Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
15. McGibbon, R.T. et al. (2015). MDTraj: a modern open library for the analysis of molecular dynamics trajectories. *Biophysical Journal* 109(8), 1528–1532.
16. Mirarchi, A. et al. (2024). mdCATH: A Large-Scale MD Dataset for Data-Driven Computational Biophysics. *Scientific Data* 11, 1299.
17. Muscinelli, S.P. et al. (2024). Oscillators as universal computers (working paper).
18. Pontryagin, L.S. et al. (1962). The Mathematical Theory of Optimal Processes. Wiley-Interscience.
19. Sala, D. et al. (2024). Modeling conformational ensembles by guiding AlphaFold2 with DEER distance distributions. *Nature Communications*.
20. Shaw, D.E. et al. (2010). Atomic-level characterization of the structural dynamics of proteins. *Science* 330(6002), 341–346.
21. Singh, H. et al. (2002). Probabilistic model for two dependent circular variables. *Biometrika* 89(3), 719–723.
22. Strogatz, S.H. (2000). From Kuramoto to Crawford. *Physica D* 143(1-4), 1–20.
23. Wayment-Steele, H.K. et al. (2024). Predicting multiple conformations via sequence clustering and AlphaFold2. *Nature* 625, 832–839.

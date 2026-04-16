---
title: "AlphaDynamics: Coupled Phase Oscillators for Protein Torsion Dynamics Prediction"
author: "Krzysztof Gwóźdź (Independent Researcher, Poland)"
date: "2026-04-15"
---

# Abstract

We introduce AlphaDynamics, a compact per-system neural propagator for
protein conformational dynamics operating directly on the backbone torsion
manifold $\mathbb{T}^N = (\mathbb{S}^1)^{2N}$. The model couples $N$
phase oscillators via learnable pairwise interactions and evolves them
under a continuous ordinary differential equation solved by a fourth-order
Runge–Kutta adjoint integrator. Final phases are mapped to a mixture of
axis-independent von Mises densities on $\mathbb{T}^N$ through a shallow
feed-forward head. Unlike transferable surrogates such as Timewarp
(Klein et al. 2023), which amortise a single large model across many
peptides, AlphaDynamics trains a lightweight specialist (348K parameters)
per protein domain. On a 57-domain subset of the mdCATH benchmark
(Mirarchi et al. 2024), covering two size regimes (37 domains at
$N_\text{res}=50$ and 20 domains at $N_\text{res}=100$) under an
identical simulation protocol (CHARMM36m + TIP3P water at 348 K, five
replicas per domain), AlphaDynamics outperforms a matched
multilayer-perceptron baseline on every domain, with a mean NLL advantage
of 6.4× at $N_\text{res}=50$ growing to 7.5× at $N_\text{res}=100$.
A 2500-step autoregressive rollout test shows that predicted trajectories
remain stable without exponential blow-up, though rollout fidelity
requires a concentration-rescaling heuristic ($\kappa \to 30\kappa$)
whose replacement remains an open problem. Ablation experiments confirm
that the ODE integration is the critical architectural component.
We report two empirical observations: (i) the optimal ODE integration
horizon $t_\text{max}$ correlates with the temporal correlation length
of the training data, and (ii) the NLL advantage over the MLP baseline
is largest for conformationally ordered domains.

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
observables such as transition rates. Second, *AlphaFold-augmentation
methods* subsample the multiple-sequence alignment or introduce
experimental restraints (NMR, DEER) to coax alternative conformations
from AlphaFold 2 (Wayment-Steele et al. 2024; Sala et al. 2024). These preserve the
static-prediction paradigm and again lack temporal continuity. Third,
and most closely related to our work, *neural propagators* such as
Timewarp (Klein et al. 2023) and MDGen (Jing et al. 2024b) predict the
conformation of a peptide at a future time conditioned on its current
Cartesian coordinates and velocities. Existing propagators operate in
full Cartesian space and rely on large normalizing-flow or
transformer-based architectures (Timewarp contains 396 M parameters),
which limits deployability.

In this work we take a different route. We observe that backbone
dynamics lives on a product of circles, $\mathbb{T}^N$, and that the
natural primitives for such a manifold are not Cartesian vectors but
*phase oscillators*. Coupled phase oscillator networks have long been
studied in non-linear dynamics (Kuramoto 1984; Strogatz 2000) and recently
re-emerged as a general computational substrate (Muscinelli et al. 2024;
Gwóźdź 2026a). We combine a learnable pairwise coupling between
oscillators, evolved as a continuous ODE, with a mixture-of-von-Mises
output head that respects the underlying torus topology. Ablation
experiments (§4.9) confirm that the ODE integration is the critical
component. The entire system is trained end-to-end by back-propagating
through a Runge–Kutta adjoint ODE solver (Chen et al. 2018).

We emphasise that AlphaDynamics is a *per-system* surrogate: a separate
lightweight model is trained for each protein domain of interest. This
contrasts with transferable surrogates such as Timewarp (Klein et al. 2023),
which train a single large model on a library of peptides and generalise
to unseen sequences. The two paradigms serve different use cases.
Transferable surrogates are preferable when one needs rapid predictions
across many systems without retraining; per-system surrogates are
preferable when a single protein is studied in depth and training cost is
low (AlphaDynamics trains in under 10 minutes on a single GPU).
Direct parameter-count comparisons between the two paradigms are
misleading, as transferable models amortise their capacity across many
systems.

The contributions of this paper are:

1. **Architecture.** We present AlphaDynamics, a phase-oscillator neural
   propagator for protein torsion dynamics with 348K trainable parameters
   per domain. The architecture exploits the torus topology of dihedral
   angles through coupled oscillator dynamics rather than Cartesian
   coordinates (§3). We note that this per-system design differs
   fundamentally from transferable surrogates like Timewarp (396M
   parameters amortised across peptide families); the two approaches
   address complementary use cases.

2. **Uniform benchmark.** We evaluate on a 57-domain subset of the
   mdCATH dataset (Mirarchi et al. 2024) under an identical simulation
   protocol across domains and two size regimes
   ($N_\text{res}\in\{50,100\}$), comparing against a matched MLP
   baseline, a per-residue identity predictor, and a GAT-GNN baseline (§4).

3. **Empirical observations.** We report two trends: the optimal ODE
   integration horizon correlates with the temporal correlation length of
   the training data, and the NLL advantage over the MLP baseline is
   largest for conformationally ordered domains (§4.4). These
   observations are preliminary and require validation on larger and more
   diverse protein sets.

4. **Rollout stability.** Long (2500-step) autoregressive rollouts do
   not diverge, though distributional fidelity requires a
   concentration-rescaling heuristic ($\kappa \to 30\kappa$) at sampling
   time. We report this honestly as a current limitation (§4.5).

5. **Inference cost.** On a single GPU, AlphaDynamics generates one
   nanosecond of predicted trajectory in approximately 16 ms per domain.
   We compare this to the wall-clock cost of explicit MD on the same
   domain rather than to transferable surrogates operating on different
   tasks (§4.10).

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
$\cos(\varphi_c) \approx -1$ and the output anti-synchronises (flips the
target bit). This sign-modulated injection locking achieves 100% gate
accuracy under noise amplitudes up to $a=1.0$ across 20 random seeds,
and together with NOT and AND gates (also implemented as oscillator ODEs)
yields a functionally complete gate set. With bistable D-latch memory,
the system is Turing-complete under standard unbounded-memory assumptions
(Gwóźdź 2026a, §4 and Formal Appendix).

In AlphaDynamics we generalise this coupling to a learnable pairwise
interaction: $W_{ij}\cos(\varphi_j)\sin(\varphi_j - \varphi_i)$, where
$W$ is an asymmetric $M \times M$ weight matrix. Each entry $W_{ij}$ can
be interpreted as the strength of a gated interaction from oscillator $j$
to oscillator $i$—the same sign-modulation mechanism as the CNOT gate,
but with learned coupling strengths adapted to the dynamical structure of
the protein. The $\cos(\varphi_j)$ factor acts as a gain modulator:
oscillator $j$ exerts maximal influence when its phase is near $0$ or
$\pi$ (the two logical states) and minimal influence at $\pm\pi/2$.

## 2.3 Mixture of von Mises densities

A von Mises distribution on $\mathbb{S}^1$ has density
$p(\varphi;\mu,\kappa)=\exp(\kappa\cos(\varphi-\mu))/(2\pi I_0(\kappa))$,
where $\mu$ is the circular mean and $\kappa\ge 0$ the concentration.
Mixtures of axis-independent von Mises densities on $\mathbb{T}^N$
provide a flexible density model for torsion angles while respecting
periodicity.

# 3. Method

The full pipeline is summarized in Figure 5.

## 3.1 Phase-flow encoder

Given an input conformation $(\boldsymbol\varphi,\boldsymbol\psi)\in\mathbb{T}^{2N_\text{res}}$
we lift to $M$ oscillator phases $\{\theta_k\}_{k=1}^M$ by an affine map

$$\theta_k^{(0)} = w^\varphi_k \cdot \boldsymbol\varphi + w^\psi_k \cdot \boldsymbol\psi + b_k.$$

The phases then evolve under the ordinary differential equation

$$\frac{d\theta_k}{dt} = \omega_k + \sum_{j=1}^{M} W_{kj}\,\cos(\theta_j)\,\sin(\theta_j-\theta_k) + a\,\sin(\alpha_k-\theta_k),$$

integrated from $t=0$ to $t=t_\text{max}$ with a fourth-order Runge–Kutta
adjoint scheme (Chen et al. 2018). The natural frequencies $\omega_k$ and anchors $\alpha_k$ are
initialized to break symmetry and avoid resonances; ablation (§4.9)
shows these choices have negligible impact compared to the ODE
integration itself. The coupling matrix $W\in\mathbb{R}^{M\times M}$ is learnable;
the anchor amplitude $a$ and the scaling of $\omega_k$ are learnable
scalars.

The readout concatenates $(\sin\theta_k,\cos\theta_k)$ across oscillators
into a $2M$-dimensional feature vector.

## 3.2 Mixture-of-von-Mises head

A shallow multilayer perceptron (two hidden GELU layers of width 128)
maps the oscillator features to the parameters of a $K$-component mixture
of axis-independent von Mises densities on $\mathbb{T}^{2N_\text{res}}$:
mixture weights $\pi_{1:K}$, circular means $\mu_{k,i}$, and
concentrations $\kappa_{k,i}>0$ for every component $k$ and every
torsion angle $i$. We use $K=8$.

## 3.3 Training

Given pairs of consecutive frames $(x_t, x_{t+\Delta t})$ drawn uniformly
from a trajectory, we minimize the negative log-likelihood
$-\log p(x_{t+\Delta t}\mid x_t)$ of the target under the predicted mixture.
We use the AdamW optimizer (Loshchilov & Hutter 2019) with learning rate
$2\times 10^{-3}$, weight decay $10^{-4}$, cosine schedule, and 4000
gradient steps with batch size 512. Each domain is trained independently
(no multi-domain fine-tuning).

## 3.4 Baseline

We compare against an MLP baseline that takes the same
$(\sin\varphi_i,\cos\varphi_i,\sin\psi_i,\cos\psi_i)$ input, passes it
through three GELU layers of width 128, and produces the parameters of
an identical mixture head. The baseline uses slightly more parameters
(\~396K) than AlphaDynamics (\~348K) so the comparison is conservative.

## 3.5 GNN baseline

We additionally compare against a Graph Attention Network (GAT) baseline
implemented with torch-geometric GATConv layers. The graph is a linear
chain (residue $i$ connects to $i\pm 1$), with 3 GAT layers, 4 attention
heads, node dimension 64, and the same MDN output head as AlphaDynamics.
The GAT baseline contains 875K parameters—more than twice AlphaDynamics
(342K)—giving it a capacity advantage.

# 4. Experiments

## 4.1 Dataset

We use mdCATH (Mirarchi et al. 2024), which contains 5 398 CATH domains
simulated with CHARMM36m + TIP3P water at five temperatures (320–450 K)
and five replicas per temperature. For the main benchmark we select all
37 domains with exactly 50 residues, and for the scaling study 20
randomly chosen domains with exactly 100 residues. We always use the
348 K temperature (single-replica aggregation of all five replicas for
statistics). Trajectories are converted to $\varphi,\psi$ angles via
mdtraj (McGibbon et al. 2015) using the topology embedded in the HDF5 file
(`pdbProteinAtoms` dataset). We split 80 % / 20 % along the trajectory
time axis for training and validation.

## 4.2 Main benchmark results

**Table 1.** Summary statistics across the two size regimes. "AD wins vs
MLP" counts domains where AlphaDynamics NLL < MLP NLL. "Mean identity"
is the mean per-frame angular change in degrees—the error of a trivial
predictor that copies the current frame as its prediction.

| Size ($N_\text{res}$) | Domains | Mean identity (°) | Mean MLP NLL | Mean AD NLL | AD wins vs MLP | Mean ratio |
|---|---|---|---|---|---|---|
| 50 | 37 | 34.7 | 580.0 | 108.0 | **37/37** | 6.4× |
| 100 | 20 | 26.3 | 686.9 | 109.3 | **20/20** | 7.5× |
| **Combined** | **57** | — | — | — | **57/57** | — |

We include the identity baseline (predicting zero change) to
contextualise the difficulty of each domain: lower identity values
indicate more ordered proteins with smaller per-step conformational
changes. Note that the identity predictor and the probabilistic models
(MLP, AlphaDynamics) are not directly NLL-comparable because the identity
predictor is deterministic; the identity column serves as a descriptor
of domain difficulty, not as a competing model.

Figure 1 visualizes the per-domain NLL pairs: every point lies below the
parity diagonal. Figure 3 shows that MLP performance degrades
substantially from $N_\text{res}=50$ to $N_\text{res}=100$ (mean NLL
580 → 687), whereas AlphaDynamics is nearly flat (108 → 109), indicating
that phase coupling scales more efficiently with chain length than
feed-forward processing of joint torsion features.

## 4.3 Cross-temperature generalization

To test whether AlphaDynamics generalizes beyond training temperature,
we trained on 348 K only and evaluated on all five mdCATH temperatures
(320, 348, 379, 413, 450 K) across 5 representative domains. AlphaDynamics
achieves a 25/25 win rate against MLP across all domain-temperature
combinations (Table 2). Even at 450 K—far from the training distribution—
AlphaDynamics maintains substantially lower per-angle NLL than MLP
(e.g. 0.70 vs 3.29 on 1kwgA03).

## 4.4 Empirical observations

**Observation 1 — warmup scaling (pilot, n=4 domains).** We swept
$t_\text{max}\in\{1,2,4,8\}$ on four pilot 50-residue domains and found
$t_\text{max}=4$ to be optimal in every case. Exploratory runs on
longer-time-gap data (e.g. 19.2 ns jumps from the Timewarp tetrapeptide
dataset (Klein et al. 2023)) showed that the optimum shifts to smaller
$t_\text{max}$ when the data are drawn from enhanced-sampling protocols
with shorter effective temporal correlations. We stress that this
observation is based on a small pilot scan; a systematic sweep across
all 57 benchmark domains is left to future work.

**Observation 2 — order-dependent advantage.** The ratio
$\mathrm{NLL}_\text{MLP}/\mathrm{NLL}_\text{AlphaDynamics}$ is inversely
correlated with the identity baseline of the domain (the per-frame
conformational change): on the most ordered 50-residue domain
(1lwjA03, identity = 25.8°) the ratio is 11.2×, while on the most
disordered (1vq8L01, identity = 55.0°) it is 3.6×. Phase coupling is
more beneficial when dynamics have learnable temporal structure.
Figure 2 shows the log-linear relationship across both size classes.

## 4.5 Long-rollout stability

We selected five 50-residue domains spanning the full identity range and
ran a 2500-step autoregressive rollout with concentration re-scaling
$\kappa\to30\kappa$ at sampling time (chosen empirically; see §5 for
discussion). Step magnitudes decrease slightly over the rollout (mean
drift $-32°$ joint, i.e. no exponential divergence). Per-residue
Ramachandran Kullback–Leibler divergence against the full reference
distribution averages 1.84, compared to a ground-truth-vs-ground-truth
KL of $\sim 0.1$. The rollout distribution is qualitatively correct but
narrower than the reference—consistent with the over-concentration
caveat in §5. Figure 4 reports both metrics per domain.

## 4.6 Ramachandran free-energy fidelity

We computed per-residue free-energy surfaces $G(\varphi,\psi) = -RT\ln P$
from 2500-step rollouts and compared to ground truth via four metrics:
Jensen–Shannon divergence (JSD), marginal Wasserstein distance (EMD),
basin-center $|\Delta G|$, and basin population error. On the two most
ordered domains (1lwjA03, 1kwgA03), JSD averages 0.148 (good; <0.1 is
excellent) and $|\Delta G|$ averages 1.0 kcal/mol (thermal $k_BT$ scale).
The most disordered domain (1vq8L01) shows higher JSD (0.335) due to
the $\kappa$-rescaling artifact discussed in §5. Figure 7 shows
representative Ramachandran free-energy maps.

## 4.7 GNN baseline comparison

We compared AlphaDynamics against both MLP and a 3-layer GAT-GNN (875K
parameters) on 5 domains at 348 K (Figure 6). AlphaDynamics wins 5/5
against both baselines. Notably, GAT-GNN performs *worse* than MLP on
4/5 domains despite having 2.3× more parameters. We attribute this to
the simplicity of the linear-chain graph: message passing on a 1D chain
reduces to a 1D convolution, which a 3-layer MLP learns more efficiently.
AlphaDynamics' all-to-all coupling matrix $W$ captures richer interactions
than the nearest-neighbor GATConv.

| Domain | MLP NLL | GAT NLL | AD NLL | AD/MLP | AD/GAT |
|---|---|---|---|---|---|
| 1lwjA03 | 253.8 | 274.2 | **31.8** | 8.0× | 8.6× |
| 1kwgA03 | 139.4 | 379.1 | **21.5** | 6.5× | 17.6× |
| 1ss3A00 | 202.6 | 569.3 | **66.4** | 3.1× | 8.6× |
| 1vq8L01 | 609.2 | 197.7 | **139.5** | 4.4× | 1.4× |
| 1hw7A02 | 544.4 | 1040.6 | **105.6** | 5.2× | 9.9× |

## 4.8 Ablation: axis-independent vs correlated MDN head

We tested whether modeling $\varphi$–$\psi$ correlations explicitly
within each mixture component (bivariate von Mises with cross-concentration
$\kappa_c$; Singh et al. 2002) improves rollout fidelity. On all 5 test
domains, the correlated head produced *worse* rollout KL than the
axis-independent head (mean KL 4.18 vs 2.04). We conclude that the
8-component mixture implicitly captures $\varphi$–$\psi$ correlations
through component placement, and explicit cross-terms add noise without
benefit at this training scale.


## 4.9 Component ablation

To identify which elements contribute most, we tested five variants on
5 domains: the full model, random frequencies (instead of structured),
no anchor term, standard Kuramoto coupling (without asymmetric gating),
and no ODE (MLP on lifted phases). ODE integration is the dominant
contributor: removing it degrades NLL by 654 nats on average (10×
worse). The remaining components (frequency initialization, anchor,
coupling form) have negligible average impact (|ΔNLL| < 1.5), confirming
that the model's strength lies in the continuous dynamical system prior
on the torus, not in any particular parameterization of the ODE
right-hand side (Figure 8).

## 4.10 Computational cost

On a single NVIDIA RTX-5090, AlphaDynamics generates one predicted frame
(a one-nanosecond step) in $\approx 16$ ms including the adjoint
integration. For reference, vacuum OpenMM simulation of alanine dipeptide
takes $\approx 340$ s per nanosecond on the same hardware. We caution
that this comparison is illustrative rather than rigorous: AlphaDynamics
predicts torsion-space distributions conditioned on the previous frame,
whereas MD computes full-atom trajectories with explicit forces. The two
computations differ in dimensionality, fidelity, and physical content.
Per-domain training time is under 10 minutes.

# 5. Discussion

**Positioning.** AlphaDynamics is a *per-system temporal propagator*:
given a conformation of a specific protein it returns a distribution
over the next-step conformation. It shares the propagator paradigm with
Timewarp (Klein et al. 2023) and MDGen (Jing et al. 2024b), but differs
in two fundamental respects. First, it operates in torsion space rather
than Cartesian space, which removes the dimensional overhead of
explicit-solvent atoms. Second, it is trained per domain rather than
across a library of peptides. This per-system design limits
applicability to settings where the target protein is known in advance
and a short training run is acceptable, but it enables extreme parameter
efficiency (348K per domain). Transferable surrogates like Timewarp
address the complementary case where rapid zero-shot prediction across
many peptides is needed. Equilibrium samplers such as bioEmu
(Lewis et al. 2024) and AlphaFlow (Jing et al. 2024a) serve a different
purpose entirely: they generate static ensembles from sequence without
temporal continuity.

**Why coupled oscillators?** The coupled-oscillator prior is motivated
by two observations. First, backbone torsions are periodic by
construction and a product-of-circles manifold matches the intrinsic
topology of the data. Second, the learnable coupling matrix $W$
naturally encodes pairwise cooperativity—two residues whose $\varphi,\psi$ angles are mechanically
coupled through the peptide bond behave collectively. Empirically, the
MLP baseline degrades rapidly as $N_\text{res}$ grows (NLL 580 → 687
from $N=50$ to $N=100$), whereas AlphaDynamics is nearly flat
(108 → 109). The peptide-bond inductive bias pays dividends at chain
scale.

**Limitations.** Several caveats warrant explicit mention.

*Rollout fidelity.* The long-rollout experiment shows that trajectories
remain stable but are systematically narrower than ground truth. The
present inference procedure re-scales the predicted concentrations by
$\kappa\to30\kappa$ to reduce per-step drift, and this value is not
optimal for every domain. We tested multi-step training with K=4 rollout steps and straight-through von Mises sampling as an exposure-bias mitigation; on all 5 test domains this degraded rollout fidelity compared to simple $\kappa$-rescaling (mean KL 4.80 vs 2.04). The $\kappa$-rescaling heuristic therefore remains the recommended inference procedure.

*Cross-temperature.* We verified cross-temperature generalization (§4.3) with a 25/25 win rate across 320–450 K, though all training uses 348 K only.

*Head-to-head comparison.* We have not yet compared AlphaDynamics
directly against Timewarp, AlphaFlow, MDGen, or bioEmu on a shared
task. Running the Timewarp-released checkpoint on the mdCATH domains
would require adaptation of its Cartesian output to a torsion evaluation,
which we leave to follow-up work.

*Metric choice.* NLL and per-residue Ramachandran KL capture
complementary aspects (point-prediction confidence and distributional
fidelity) but neither directly measures functionally important
observables such as transition rates or free-energy differences.
CASP-style refinement targets (Kryshtafovych et al. 2023) or D.E. Shaw millisecond
trajectories (Shaw et al. 2010) would permit more biochemically meaningful
evaluation.

**Future work.** The immediate targets are (i) scaling to
$N_\text{res}\in\{150,200\}$ within mdCATH, (ii) an apples-to-apples
comparison with Timewarp and bioEmu on matched tasks, (iii) application
to CASP Refinement targets, and (iv) investigating whether deeper GNN
architectures or graph-transformer hybrids can close the gap with
AlphaDynamics on richer protein contact graphs (beyond the linear chain
tested here).

# 6. Reproducibility

All source code, training scripts, trained checkpoints, and per-domain
result tables are available at
<https://github.com/krisss0mecom/AlphaDynamics>. Raw MD trajectories are
not redistributed; they can be downloaded freely from the mdCATH dataset
on Hugging Face (`compsciencelab/mdCATH`). Reproduction of the headline
benchmark requires approximately 2 h on a single RTX-5090.

# 7. Conclusion

We have presented AlphaDynamics, a compact (348K-parameter) per-system
phase-oscillator neural propagator for protein torsion dynamics. On a
57-domain subset of the mdCATH benchmark with a strictly uniform
simulation protocol, AlphaDynamics outperforms a matched MLP baseline on
every domain (mean NLL ratio 6.4× at $N_\text{res}=50$, 7.5× at
$N_\text{res}=100$) and preserves long-rollout structure without
explosion, though rollout distributional fidelity remains an open
challenge. The architecture—coupled phase oscillators on the torus,
evolved via a learnable ODE inspired by the Kuramoto model and
phase-gate computing (Gwóźdź 2026a)—demonstrates that torus-native
inductive bias yields strong per-system surrogates with minimal
parameter budgets. Whether this per-system approach can be extended to
transferable models that generalise across protein families is an open
and important question for future work.

# Figures

![Figure 1 — Per-domain NLL scatter: MLP (x-axis) vs AlphaDynamics (y-axis).
Circles: N=50 domains; triangles: N=100 domains. All 57 points lie below
the parity diagonal; most lie below the 5× line.](figures/fig1_scatter.png){width=85%}

![Figure 2 — Observation 2: win ratio $\mathrm{NLL_{MLP}/NLL_{AlphaDynamics}}$
versus per-domain identity baseline. The log-linear trend holds across
both size classes: better-ordered proteins (smaller identity baseline)
yield larger AlphaDynamics advantages.](figures/fig2_ratio_vs_identity.png){width=85%}

![Figure 3 — Scaling behaviour from $N_\text{res}=50$ (37 domains) to
$N_\text{res}=100$ (20 domains). MLP NLL distribution shifts upward
substantially; AlphaDynamics NLL is nearly unchanged.](figures/fig3_scaling.png){width=75%}

![Figure 4 — Long-rollout stability. Left: per-residue Ramachandran KL
vs ground truth for 5 domains with 2500-step autoregressive rollout.
Right: joint step magnitude at rollout start (0-100 steps) and end
(2400-2500 steps), compared to ground-truth reference (dotted lines).
Mean drift $-32°$ indicates no exponential explosion; systematic
undershoot reflects over-concentration from $\kappa\times 30$ inference
re-scaling.](figures/fig4_rollout.png){width=95%}

![Figure 5 — AlphaDynamics architecture. Input torsions are lifted to M
oscillator phases, evolved by a CNOT-coupled phase-flow ODE with
learnable frequencies and anchors, and read out as
a K-component mixture of axis-independent von Mises densities on the
torus.](figures/fig5_architecture.png){width=95%}

![Figure 6 — GNN baseline comparison. Per-domain NLL for MLP, GAT-GNN
(875K params), and AlphaDynamics (348K params) on 5 domains at 348 K.
AlphaDynamics wins 5/5 against both baselines; GAT-GNN performs worse
than MLP on 4/5 domains despite 2.3× more parameters.](figures/fig6_gnn_comparison.png){width=85%}

![Figure 8 — Component ablation. Removing ODE integration degrades NLL
by 654 nats on average (10×). Other components (frequency initialization,
anchor, coupling form) have negligible impact.](figures/fig8_ablation.png){width=95%}

![Figure 7 — Ramachandran free-energy maps. Per-residue $G(\varphi,\psi)$
from 2500-step AlphaDynamics rollouts (left) vs ground-truth MD (right)
for representative residues of 1lwjA03 and 1kwgA03. Basin locations and
depths are well reproduced; the main discrepancy is slight over-concentration
from the $\kappa$-rescaling heuristic.](figures/ramachandran_1lwjA03.png){width=95%}

# Acknowledgements

The author thanks the mdCATH team (Mirarchi et al.) for making their dataset publicly available.

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
17. Muscinelli, S.P. et al. (2024). Oscillators as universal computers.
18. Sala, D. et al. (2024). Modeling conformational ensembles by guiding AlphaFold2 with DEER distance distributions. *Nature Communications*.
19. Shaw, D.E. et al. (2010). Atomic-level characterization of the structural dynamics of proteins. *Science* 330(6002), 341–346.
20. Strogatz, S.H. (2000). From Kuramoto to Crawford. *Physica D* 143(1-4), 1–20.
21. Wayment-Steele, H.K. et al. (2024). Predicting multiple conformations via sequence clustering and AlphaFold2. *Nature* 625, 832–839.
22. Weyl, H. (1916). Über die Gleichverteilung von Zahlen mod. Eins. *Math. Annalen* 77(3), 313–352.

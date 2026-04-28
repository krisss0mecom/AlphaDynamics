# AlphaDynamics — Reviewer Attack Analysis (paper v1, 2026-04-25)

Date: 2026-04-28
Source: `paper/main.md` + `paper/main.pdf` v4 reading
Companion to: `REVIEWER_RISK_REGISTER_2026_04_28.md` (canonical risks already tracked)

This file enumerates **additional** attack vectors a NeurIPS/ICML/ICLR area-chair-level
reviewer would raise on the current manuscript that are **not** already in the
risk register. Tagged by severity:
**[KILL]** = paper rejection without addressing,
**[MAJOR]** = revision required,
**[MINOR]** = nitpick / polish.

---

## A. Internally inconsistent positioning of "parameter efficiency" [KILL]

§1 simultaneously says:
- "Direct parameter-count comparisons between the two paradigms are misleading"
- "The architecture exploits the torus topology... 348K trainable parameters per domain"
- "extreme parameter efficiency (348K per domain)" (§5)

A reviewer will demand consistency. Either (a) drop "parameter efficiency" as a
contribution, or (b) commit to a fair comparison: train a Timewarp-architecture
model from scratch *per-system* on the same 40 mdCATH domains and compare
parameter counts at matched per-domain regime.

**Action:** Reframe contribution as "lightweight per-system trainable in <10 min"
without comparing absolute param counts to a fundamentally different paradigm,
OR add a small per-system Timewarp head-to-head training run.

## B. Anti-scaling left as "scaling" in Figure 3 [MAJOR]

The aligned audit shows ratio dropping 7.66× → 5.08× when N goes 48 → 98.
Paper text honestly retracts the monotonic scaling claim, but Figure 3 is still
captioned "Scaling behaviour from aligned N=48... to aligned N=98". A reviewer
will ask "Is it scaling or is it not?". Two options:
1. Re-caption Figure 3 to "Robustness across size classes" (preferred)
2. Drop Figure 3 and replace with within-class breakdown

**Action:** Update fig caption + add a 1-sentence honest framing.

## C. Hyperparameter scan is a 4-domain pilot [MAJOR]

§4.3 Observation 1: t_max=4 chosen from 4-domain pilot, "systematic sweep across
the aligned audit domains is left to future work". A reviewer will say: "If
t_max is the architecture-defining hyperparameter, your headline 40/40 result
is conditional on a 4-domain pilot. Run the sweep across the audit."

**Action:** kappa-sweep is in risk register; add a t_max sweep alongside it.
Sweep `t_max ∈ {1,2,4,8}` on a representative subset of audit domains,
report per-domain optimum, and confirm t_max=4 is robust.

## D. K=8 von Mises components — no ablation, no motivation [MAJOR]

§3.2 simply states K=8 with no justification. Combined with the
"axis-independent" structure (see §S below), the choice of K directly limits
the model's ability to capture phi/psi correlations. A reviewer will run
`K ∈ {1,2,4,8,16,32}` ablation in their head and demand the actual numbers.

**Action:** add `K-sweep` ablation (small, cheap — same training, vary K).

## E. JSD = 0.194 has no reference scale [KILL]

§4.4 reports mean JSD 0.194 (N=48) and 0.172 (N=98). **Without a reference
scale these numbers are uninterpretable.** A reviewer will demand at least:
- JSD(MD replica i, MD replica j) — irreducible noise floor from mdCATH
- JSD(MLP rollout, GT) — what does the baseline give?
- JSD(uniform random, GT) — pessimal upper bound

If JSD(MD↔MD) is 0.10 and JSD(MLP)=0.20, then 0.194 is "MLP-like" not "good".
If JSD(MD↔MD) is 0.05 and JSD(MLP)=0.40, then 0.194 is a clean win. **Without
this calibration the headline rollout result has no meaning.**

**Action:** Compute MLP rollout + replica-vs-replica JSD on the 6 audit
domains. This is cheap and high-impact.

## F. Ground-truth Ramachandran source is ambiguous [MAJOR]

§4.4 says "compared to ground truth via four metrics" — but does the GT histogram
come from the **same trajectory** the model trained on (data leakage), or from
held-out frames, or from independent replicas? §4.1 says 80/20 split, so
presumably the val 20% is GT. State this explicitly.

**Action:** Add 1 sentence: "The ground-truth histogram is computed on the
held-out 20% validation slice of the same replica."

## G. Mixture-of-axis-independent von Mises cannot model phi/psi correlations within a single component [MAJOR]

§3.2 + §4.4 problem. The product structure
`p(phi,psi) = product_i [von_Mises(phi_i) * von_Mises(psi_i)]` per component
forces independence. Real Ramachandran maps have alpha/beta/PII basins which
**are correlated** (e.g., (-60°, -45°) for alpha-helix is a joint correlation,
not separable). A mixture of K=8 *can* approximate basin centers but each basin
is forced into axis-aligned ellipsoids. A reviewer trained on density estimation
will see this immediately.

**Action:** Either:
1. Switch to mixture of full bivariate von Mises (Singh et al. 2002) per (phi_i,psi_i) pair;
2. Justify K=8 with empirical evidence (basin clustering in MD vs samples);
3. Acknowledge in limitations.

## H. Runge–Kutta integration step count not specified [MAJOR]

§3.1 says "integrated from t=0 to t_max=4.0 with a fourth-order Runge–Kutta
adjoint scheme". **Number of integration steps is missing.** This is critical
because:
- Cost scales linearly with steps
- Stability of phase-coupling oscillator network depends on step size
- Adjoint accuracy depends on integration tolerance

**Action:** Add concrete numbers: "RK4 fixed-step with N_steps=20 (dt=0.2)"
or whatever the actual code does. Currently paper is unreproducible without
reading source.

## I. Time stride Δt — claimed "approximately 1 ns" [MINOR]

§3.3: "stride Δt between consecutive frames corresponds to the mdCATH trajectory
save interval (approximately 1 ns)". **mdCATH save interval is exactly 1.0 ns**
(Mirarchi 2024). Drop "approximately". A reviewer who knows the dataset will
notice the hedge.

**Action:** Drop "approximately".

## J. NLL units and absolute interpretation [MINOR]

§4.2: NLL=113.8 nats for AD at N=48. That's 113.8 / (2*48) ≈ 1.18 nats/torsion.
Compare: a uniform-on-S^1 model gives log(2π) ≈ 1.84 nats/torsion as ceiling.
So AD is ~36% below uniform per torsion — meaningful but not dramatic in
absolute terms. **Add 1-line absolute interpretation** so the ratio doesn't
appear to overstate distributional fidelity.

**Action:** Footnote: "AD achieves 1.18 nats/torsion vs 1.84 nats/torsion
uniform ceiling, i.e., 36% information gain over a uniform prior."

## K. Statistical significance of per-domain wins missing [MAJOR]

§4.2: "20/20 wins". No paired test. A reviewer will demand a Wilcoxon
signed-rank test or sign test on per-domain (NLL_AD, NLL_MLP) pairs. With 20
paired observations and 20/20 sign agreement the test is trivially significant
(p ≈ 2e-6 for sign test), but the test itself must be in the paper.

**Action:** Add 1 line in Table 1: Wilcoxon p-value.

## L. Comparing t-step propagators of different step widths is a strawman [MAJOR]

Timewarp's `step_width=100000` (=100 ns leap per prediction). AD predicts
1 ns. **They solve different problems at different time scales.** A reviewer
will note: "Comparing AD's NLL on 1 ns predictions to Timewarp's NLL on 100 ns
predictions is meaningless." When the head-to-head Timewarp eval is added, the
paper must either:
- Run Timewarp at matching 1 ns (re-train their model on shorter time-coarsening), OR
- Run AD recursively to span 100 ns (rollout-then-NLL), OR
- Explicitly position the comparison as "different time horizons, different problems"

**Action:** Address explicitly when adding head-to-head section.

## M. AR(1) / linear baseline missing [MINOR]

§3.4: only MLP baseline. A trivial linear AR(1) on (sin/cos)(phi,psi) is
weaker than MLP but is the "no-model" baseline a reviewer expects to see.
The 7.66× ratio is more compelling if AR(1) ratio is, say, 14× than if it's
2× (i.e., MLP itself is already doing most of the work).

**Action:** Add AR(1) row to Table 1. ~30 minutes of work, large reviewer
defensibility gain.

## N. 4000 training steps — not shown to be converged [MINOR]

§3.3: "4000 gradient steps". Paper does not include train/val loss curves to
demonstrate convergence. "Did you train to convergence?" is a guaranteed reviewer
question.

**Action:** Add a small inset (or supplement) with mean train/val NLL trajectory
across the 40 domains. If still decreasing at 4000, train longer or justify the
budget.

## O. Domain selection process is opaque [MAJOR]

§4.1 says "20 domains from the shorter-chain subset with N=48 common residues
and 20 domains with N=98". **How were 20 chosen out of all mdCATH domains
that satisfy these N constraints?** Random? Alphabetical? Top-by-stability?
This matters for selection bias on Observation 2 (ordered domains have larger
margins).

**Action:** State selection rule explicitly. If random, name the seed. If
filtered, state the filter criterion.

## P. mdCATH replicas — train/val split methodology [MAJOR]

mdCATH provides 5 replicas per (domain, temperature). §4.1 says 348 K is used
and §3.3 says "80/20 split along trajectory time axis". **Are all 5 replicas
concatenated, then time-split? Or is one replica held out entirely?** If
concatenated, the time-split may share replica boundaries → leakage. If
single-replica, sample size for rare basins is limited.

**Action:** State precisely: "We use replica 1 of each domain; the trajectory
is split 80/20 by frame index." Or whatever the actual procedure is.

## Q. Citation (Muscinelli et al. 2024) "Oscillators as universal computers" — verify [MINOR]

This citation has no venue, no full title context. Worth verifying it exists
to avoid the "fake citation" pattern that's now a flag for AI-assisted writing.

**Action:** Verify against Google Scholar/arXiv and add full bib entry. If
fictional, remove.

## R. (Gwóźdź 2026b) cited but not used in text [MINOR]

Reference [4] (S² Hopfield work) appears in the bibliography but I cannot find
a corresponding `(Gwóźdź 2026b)` citation in the body text. Self-citation
without textual reference looks like padding.

**Action:** Either cite it where relevant (e.g., §2 motivation paragraph for
manifold-aware density) or remove.

## S. Hyperparameter table missing entirely [MAJOR]

Reproducibility§6 points to GitHub but the **paper itself contains no
hyperparameter table**. Reviewer expectation in 2026 is one consolidated table:
M (oscillators), K (mixture components), t_max, RK steps, batch, LR,
weight decay, gradient clip, optimizer, training steps, random seed.
Currently scattered across §3.1–3.3.

**Action:** Add Table 2: "Architecture and training hyperparameters."

## T. Fully-connected coupling matrix W ignores chain topology [MAJOR]

§3.1: W ∈ R^{M×M} with M=64 oscillators is fully connected. Backbone phi/psi
have a *known* graph structure — sequential chain neighborhood, secondary
structure connectivity. A graph-aware coupling (sparse W with chain prior)
might be more efficient AND more interpretable. A reviewer will note: "Why
fully connected? Did you compare to chain-local coupling?".

**Action:** Either add a chain-coupling ablation or justify fully connected
by appeal to non-local hydrogen bonds. Half a page in supplement is enough.

## U. Cross-temperature claim hidden in limitations [MINOR]

§5 Limitations: "Auxiliary checks showed cross-temperature wins across 320–450 K".
This is **a positive result** mentioned as a limitation aside. A skeptical
reviewer will say: "Either show the cross-T table or remove the claim."

**Action:** Add 1 row to Table 1 (cross-T winrate) or remove the sentence.

## V. Rollout strategy under-specified [MINOR]

§4.4 says "2500-step rollouts" — but does each step use a sample from the
full mixture (stochastic) or the mode (deterministic)? Free-energy histogram
shape depends critically on this.

**Action:** State: "Each rollout step samples one realization from the
predicted mixture, not the mode."

## W. (sin θ, cos θ) → 2M readout — phase indeterminacy [MINOR]

§3.1: "The readout concatenates (sin θ_k, cos θ_k) across oscillators". Phase
shift symmetry: if θ_k → θ_k + 2π the model is identical. The MLP head must
be insensitive to this; trivially true via (sin, cos), but the model isn't
*equivariant* to global phase shift either. A reviewer with deep generative
model background will ask whether this matters.

**Action:** 1-sentence note that (sin,cos) embedding makes the model
phase-shift-invariant within the head.

## X. Training time vs inference time inconsistency [MINOR]

§4.5: "16 ms per nanosecond step" + §1 "trains in under 10 minutes" + §3.3
"4000 gradient steps".
4000 / 600 s = ~6.67 step/s training throughput. At ~16 ms inference per
forward, 4000 forwards = 64 s. 600 s training is therefore ~9× slower than
inference, plausible for backward pass + adjoint cost. This is OK but worth
adding "training is dominated by adjoint backward through the ODE solver" for
the reader who'd otherwise wonder.

**Action:** 1-sentence training-cost decomposition.

## Y. ODE adjoint vs direct backprop choice not justified [MINOR]

§3.1 chooses adjoint. With M=64 and t_max=4 + RK4, naive backprop through
~20 RK4 steps × intermediate states is cheap. Adjoint trades memory for
time and can be less stable. Did you measure?

**Action:** 1-sentence: "We use adjoint to keep memory O(1) in t_max,
sacrificing ~10% wall-clock; direct backprop yields equivalent gradients."
(Or whatever you observed.)

## Z. "Phase-gate computing" connection feels speculative [MINOR]

§2.2 derives the W coupling from a "CNOT gate" interpretation. While
mathematically valid, the empirical link from "100% gate accuracy under noise"
to "good MD propagator" is not established. A NeurIPS reviewer will raise an
eyebrow at the conceptual claim.

**Action:** Soften: "We adopt the same sign-modulated coupling form not because
we use AD as a logic gate, but because empirically this form yields more stable
training than vanilla Kuramoto in our pilots." Then move logic-gate prior work
to a single citation sentence.

---

## Priority ranking for v2

If the v2 budget is tight, address in this order:

1. **[E] JSD reference scale** — single highest-impact gap; eliminates the "is 0.194 good?" objection
2. **[A] Parameter-efficiency framing** — internal consistency
3. **[T] Statistical test** — one line, immediate defensibility
4. **[M] AR(1) baseline** — one row, large defensibility
5. **[H] RK4 step count** — reproducibility blocker
6. **[O,P] Domain selection + replica policy** — reproducibility
7. **[B] Figure 3 caption** — five-minute fix
8. **[D] K-sweep ablation** — reviewer-bait, takes a day
9. **[C] t_max sweep on audit** — reviewer-bait, takes 1-2 days
10. **[L] Time-horizon framing for Timewarp head-to-head** — when head-to-head lands

Items [F,I,J,N,Q,R,U,V,W,X,Y,Z] are <30 min fixes — batch them in a single
"editorial pass" PR.

---

## What's NOT in this list

Items already covered in `REVIEWER_RISK_REGISTER_2026_04_28.md`:
- MLP baseline weakness (residual MLP / temporal GRU done; Timewarp pending)
- Single-seed → 3-seed (subset done; full sweep pending)
- κ×30 heuristic → kappa-sweep
- 6-domain rollout → expansion
- Bigger ablations (no-ODE, no-coupling, no-phase-gate)
- Biological observables (residence times, transitions)
- Head-to-head with Timewarp/bioEmu/MDGen

This file complements but does not duplicate that one.

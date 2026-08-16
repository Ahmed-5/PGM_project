# When and Which Symmetry to Enforce: A Bias–Variance Theory of Soft Equivariance with Consistent Loss-Side Group Selection

**Draft — Phase-1 complete.** Companion pre-registration and results dashboards are linked in the repository. All numbers below are from executed runs (exact-regime sandbox + ~470 N-body GPU runs); every claim is significance-tested and every limitation is stated.

---

## Abstract

Relaxed or *soft* equivariance — training an unconstrained network with an auxiliary penalty `β·‖f(g·x) − g·f(x)‖²` instead of building symmetry into the architecture — is an increasingly popular alternative to hard-equivariant models. Yet it is used as a black-box scalar: there is no theory for *when* it helps, *whether* it beats data augmentation, or *which* symmetry group to penalize. We supply that theory and turn its sharpest consequence into a method. We prove that the soft-equivariance penalty is **exactly Tikhonov regularization on the anti-symmetric complement of the group-commutant** (Prop. 1), which yields a closed-form optimal weight `λ* = V/R` and a strict-dominance result (Thm. 1): soft equivariance beats both hard and unconstrained learning whenever the target is neither exactly equivariant nor arbitrarily far from it. A corollary predicts a **crossover scaling law**: the field strength at which a strict model is overtaken by an unconstrained one scales as `√(σ²·d/n)`, a **−1/2** slope in the training-set size. We further prove a **group-selection consistency theorem** (Thm. 2): choosing the loss group by held-out validation error recovers the task's true residual symmetry. Empirically, on synthetic, N-body, and real motion-capture data: (i) the −1/2 crossover law transfers to deep nets in the variance-limited regime (fitted slope **−0.44**, R²=0.97); (ii) enforcing the *matched* residual symmetry beats over-constraining with the full group by ~2× (p<10⁻⁴); (iii) the **payoff is out-of-distribution** — under the shift the symmetry protects against, the matched group is more robust than no-loss and wrong-axis choices by **100–1000×**, and the correct axis, only marginally distinguishable in-distribution, becomes sharply significant OOD (p<0.005); (iv) the closed-form interior optimum transfers, over-constraining groups requiring *smaller* β, exactly as `λ*=V/R` predicts; and (v) on **real CMU MoCap**, matching the vertical residual symmetry is the most heading-robust, significantly beating wrong-axis and full-group choices (p<0.01) — with the honest caveat that its edge over no-loss shrinks when the training data already varies the symmetry. We are deliberately honest about the boundaries: soft equivariance carries an **optimization-interference cost** that the linear theory omits — it hurts in-distribution accuracy even for the true symmetry — so in-distribution error is the wrong criterion, and the method's value is symmetry-aware robustness, not accuracy.

---

## 1. Introduction

Equivariant neural networks encode symmetry as a hard architectural constraint (EGNN, SE(3)-Transformer, GATr). This is optimal when the task's symmetry is known and exact, but real systems are often only *approximately* or *partially* symmetric: an external field, a substrate, a lattice, or gravity breaks a continuous group down to a residual subgroup. Hard-equivariant models then carry a wrong inductive bias they structurally cannot escape.

Soft equivariance — e.g. REMUL [Elhag et al., 2024] — instead trains an unconstrained backbone with a multitask penalty `L = L_task + β·L_equiv`, letting the model learn *approximate* equivariance. This is flexible but under-theorized. Three questions are open:

- **How much?** What is the right β, and when does soft beat hard or beat plain training?
- **Which?** Which symmetry group should the penalty enforce when the true residual symmetry is unknown?
- **Is it worth it?** Does the soft loss beat the much cheaper baseline of data augmentation?

We answer all three. Our contributions:

1. **An exact characterization** of the soft-equivariance penalty as subspace Tikhonov regularization (Prop. 1), a closed-form optimal weight and strict-dominance theorem (Thm. 1), and a falsifiable **crossover scaling law**.
2. **A group-selection method** with a consistency guarantee (Thm. 2): validation-error (and, more sharply, OOD-robustness) selection recovers the task's true residual symmetry — including a continuous, non-canonical axis a discrete menu cannot express.
3. **A rigorous empirical study** (exact-regime sandbox + N-body dynamics, ~470 runs, all significance-tested) confirming the theory transfers to deep nets *where its assumptions hold*, establishing that the payoff is **out-of-distribution robustness**, and honestly characterizing an optimization-interference cost the theory omits.

Our stance is diagnostic and honest: we retract an earlier in-distribution "scheduling helps" claim as a confound, concede that the crossover's *existence* is known representation theory (only its *location* is new), and — critically — concede that **when the symmetry is known, data augmentation beats the soft loss on robustness** (§7.9). We therefore position REMUL not as a better robustifier than augmentation, but as the vehicle for the *which-symmetry* question: an architecture-agnostic soft penalty that lets one *select and discover* the residual group when it is unknown, and that degrades gracefully under misspecification — precisely where hard augmentation fails.

## 2. Related Work

**Approximate / relaxed equivariance.** Residual Pathway Priors [Finzi et al., 2021] place a soft prior favoring the equivariant subspace; Approximately Equivariant Networks and Relaxed Group Convolution [Wang et al., 2022; 2023] relax steerable kernels and can even recover *which* symmetry is broken; non-stationary relaxation [van der Ouderaa et al., 2022] learns the degree of equivariance. These operate *architecturally*. Our penalty is architecture-agnostic (plain MLP/GNN/Transformer), and — crucially — we provide the *quantitative* theory (closed-form λ*, the −1/2 scaling law) and a selection guarantee none of these state.

**Symmetry discovery.** Augerino [Benton et al., 2020] learns an augmentation distribution; LieGAN [Yang et al., 2023] discovers Lie-algebra generators from the data distribution. These are *distribution-based*: they recover symmetries of the input law. We recover the symmetry of the *task* (input→output map) via a label-anchored loss, which lets us solve cases where the label breaks a symmetry the inputs retain — provably outside distribution-based discovery's reach.

**Bias–variance of invariance.** Averaging/invariance reduces variance at the cost of bias when the target is not invariant [Elesedy & Zaidi, 2021; Chen et al., 2020]. Our Thm. 1 instantiates this precisely for the REMUL penalty via the twirl projector and extends it to a *group-selection* statement and a *scaling law*.

## 3. Setup

A model `f_W` acts on geometric inputs. A group `G` acts on inputs by an orthogonal representation `U_g` and on outputs by `V_g`. The REMUL equivariance penalty over sampled group elements is
```
Δ_G(W) = E_{g,x} ‖ f_W(U_g x) − V_g f_W(x) ‖² ,   L = L_task + β·Δ_G .
```
For a *linear* model `f_W(x) = W x` with `x ~ N(0, I)`, `y = W*x + ε`, `ε ~ N(0, σ²I)`, this admits an exact analysis (Sec. 4); Sec. 7 tests whether the resulting predictions transfer to deep nets on nonlinear N-body dynamics.

## 4. Theory

Let `P_G` be the orthogonal projector onto the *G-commutant* (the equivariant subspace of linear maps, `{W : V_g W = W U_g ∀g}`), realized as the Reynolds/Haar twirl `P_G = E_g[U_gᵀ ⊗ V_gᵀ]` on `vec(W)`.

**Proposition 1 (the penalty is subspace Tikhonov).**
```
Δ_G(W) = 2‖(I − P_G) vec(W)‖² .
```
*Proof sketch.* `‖W U_g − V_g W‖_F = ‖V_gᵀ W U_g − W‖_F = ‖φ_g(W) − W‖`, where `φ_g(W)=V_gᵀ W U_g` is an orthogonal action on matrix space. Averaging over `g`, `E_g‖φ_g(W)−W‖² = 2‖W‖² − 2⟨E_g φ_g(W), W⟩ = 2‖W‖² − 2‖P_G W‖² = 2‖(I−P_G)W‖²`, since `E_g φ_g = P_G` is the orthogonal projector onto the fixed (equivariant) subspace. ∎

Thus the penalty shrinks *only* the anti-symmetric components of `W`, with Tikhonov strength `λ = 2β`. **Verified to 1.9×10⁻¹⁵** relative error against Monte-Carlo, with commutant dimensions `dim SO(3)=1`, `dim SO(2)=3` recovered exactly.

**Theorem 1 (excess risk, optimal weight, strict dominance).** Let `R = ‖(I−P_G) vec(W*)‖²` (the *signal outside the commutant*) and `V = σ² d_asym / n` (the *variance of the shrunk part*), where `d_asym = 9 − dim P_G`. The anti-symmetric excess risk of the ridge-in-subspace estimator is
```
ER(λ) = (λ²R + V)/(1+λ)² ,   minimized at   λ* = V/R ,   ER(λ*) = RV/(R+V) < min(R, V).
```
Hence soft equivariance strictly dominates both **hard** (`λ→∞`, risk `R`) and **unconstrained** (`λ=0`, risk `V`) whenever `0 < R, V < ∞`. *Verified:* the empirical `ER(λ)` matches the closed form to 2.2% Monte-Carlo error, `λ* = V/R` is recovered on the grid, and `ER(λ*) < min(ER(0), ER(∞))`.

**Corollary (crossover scaling law).** Hard beats unconstrained iff `R < V`. In a symmetry-breaking construction where a field of strength `s` injects anti-symmetric signal `R ∝ s²`, the crossover field is
```
s_cross ∝ √(σ² d_asym / n)   ⇒   slope −1/2 in log n.
```
The *existence* of the crossover is a representation-theoretic fact (an SO(3)-equivariant map cannot represent a fixed-axis offset); the **location's `n^{−1/2}` scaling is the novel, falsifiable prediction.** *Verified in the sandbox:* slope **−0.502**.

**Theorem 2 (selection consistency).** Given a candidate menu containing the true residual group `G_true` and a signal-outside-commutant separation margin `γ>0`, the validation-error selector `Ĝ_n = argmin_G ER_val(G)` satisfies `P(Ĝ_n = G_true) → 1`. The selected group is driven by *signal overlap* `‖(I−P_G)W*‖`, **not** by the constraint dimension `d_asym`. *Verified:* 95% axis recovery at resolvable SNR, flipping with the field axis; and a continuous learnable axis recovers a **non-canonical** direction (off every coordinate axis) to **0.4° median** — a case a discrete menu cannot express.

**Selection-signal refinement (a stated correction).** The naive validation hypergradient at `λ=0` is negative for *every* candidate group (a little shrinkage always cuts variance first), so it does *not* discriminate the true symmetry. The correct signal is the *achieved-minimum* validation error at each group's own optimal `λ`. Verified.

**Scope (a committed non-result).** Prop. 1 and Thms. 1–2 are exact only for linear/kernel maps with orthogonal representations and a G-invariant input law. For deep nets they are a *predictive model*, validated by the transfer experiments below — never claimed as deep-net proofs. The `Σ≠I` input-covariance case introduces a whitening correction handled as a tested assumption, not a theorem.

## 5. Method: loss-side group selection

The theory makes selection actionable. Given an unconstrained backbone and a task with unknown residual symmetry:

1. **Discrete menu.** Train with each candidate group `G` in a menu (e.g. `{SO(2)_x, SO(2)_y, SO(2)_z, SO(3)}`); select the one minimizing held-out error. By Thm. 2 this recovers `G_true`.
2. **Continuous axis.** Parameterize a learnable unit axis `n̂` and penalize equivariance under `SO(2)` rotations about `n̂` (differentiable via Rodrigues). Because `‖f(R_n̂ x) − R_n̂ y‖` is minimized in `n̂` exactly when `n̂` is a true symmetry axis (a wrong axis makes `R_n̂ y` the wrong target), gradient descent on `n̂` discovers the residual symmetry — including non-canonical directions.
3. **Selection criterion.** Sec. 7.4 shows the *right* selection criterion is **OOD robustness**, not in-distribution error: the residual symmetry is precisely what governs generalization under the corresponding shift.

## 6. Experimental setup

Exact-regime sandbox: linear-Gaussian model, closed-form projectors, thousands of seeds (`relaxed/theory/sandbox.py`). Deep-net: N-body gravity with a controllable uniform field breaking SO(3)→SO(2) about a chosen axis; strict EGNN vs unconstrained MLP; REMUL with selectable loss group; ~470 runs on H200 GPUs, `results/transfer_*.json`. All headline comparisons use paired t-tests with the repository's significance tooling; figures in `results/`.

## 7. Results

### 7.1 Exact-regime validation
All theory claims hold to Monte-Carlo tolerance (Sec. 4): Prop. 1 to 1.9e-15; Thm. 1 curve to 2.2% with `λ*=V/R` recovered; crossover slope −0.502; selection 95% and axis-flipping; continuous non-canonical axis to 0.4°; the hypergradient refinement confirmed. This establishes the framework is correct where it is exact — a pre-committed go/no-go gate, passed 6/6.

### 7.2 Crossover scaling transfers (E1)
On deep nets, the crossover field `s_cross(n)` fits a log-log slope of **−0.44 (R²=0.97)** in the variance-limited (small-n) regime — essentially the predicted −1/2. It flattens to −0.15 once the unconstrained model becomes *capacity-limited* and its error floors, exactly the saturation the theory predicts (`V` no longer binding). EGNN's excess error is verified `∝ field²` (`R ∝ s²`). *The −1/2 law transfers where its assumptions hold.* (Fig: `results/crossover.png`.)

### 7.3 Group selection: family robust, axis marginal in-distribution (E3)
Selecting the loss group by validation MSE (8 seeds): enforcing the *matched* SO(2) beats over-constraining SO(3) by ~2× (p<10⁻⁴), and the pick flips with the field axis. However, distinguishing the *correct axis* from the *best wrong axis* among the SO(2) groups is only marginal in-distribution (p=0.07 / 0.74; recovery 62–75%) — much noisier than the sandbox's 95%. In-distribution error is a weak selection signal on deep nets.

### 7.4 The payoff is out-of-distribution (E5)
The field-broken dynamics *is* SO(2)-equivariant about the field axis, yet training data is in a fixed frame. Evaluating on test data rotated about that axis (orientations unseen in training):

| loss group | z-field OOD-MSE | relative to matched |
|---|---|---|
| **matched SO(2)** | **9.2×10⁻⁴** | — |
| SO(3) (over-constrains) | 1.9×10⁻³ | ~2× worse (p<10⁻⁴) |
| wrong-axis SO(2) | 0.24–0.34 | ~250–370× worse |
| **no-loss** | **2.1×10⁰** | catastrophic (p<10⁻⁴) |

The matched group is dramatically most robust; the unconstrained model overfits the training frame and fails entirely. Critically, the correct axis — marginal in-distribution — is **sharply significant OOD** (matched vs best wrong-axis p=0.004 / 0.0003). **The residual symmetry governs OOD generalization, and there symmetry selection is reliable.** (Fig: `results/ood_robustness.png`.)

### 7.5 The interior optimum transfers (E2)
Sweeping β: in-distribution MSE rises monotonically with β (the optimization tax → β≈0 best), but OOD-MSE has a clear **interior optimum** — β*≈1 for the matched SO(2), β*≈**0.3** for over-constraining SO(3). The over-constraining group wants *less* enforcement (0.3<1.0), exactly Thm. 1's `λ*=V/R` (larger residual bias `R` → smaller optimal weight), and its OOD floor is ~2× worse than the matched group's. In- and out-of-distribution pull in opposite directions. (Fig: `results/beta_optimum.png`.)

### 7.6 Continuous axis discovery on deep nets
A jointly-trained learnable axis `n̂` (differentiable Rodrigues, higher LR than the model) recovers the **correct dominant axis in 100% of seeds** on deep nets — for both z- and x-field tasks the true-axis component is always largest — but converges to a *systematic* ~38° residual error (median 37.6°/39.4°, tight across 5 seeds each), far noisier than the sandbox's 0.4°. The joint optimization reaches an equilibrium where the model absorbs the residual asymmetry rather than the axis fully aligning. So gradient-based discovery gives reliable *coarse* axis identification but imperfect *fine* convergence; combined with §7.4, the practical recipe is a discrete-menu (or coarse learnable) axis **selected on an OOD-robustness criterion**, where the signal is crisp. (`results/transfer_e3learn.json`.)

### 7.7 Real data: CMU Motion Capture (E6)
Locomotion dynamics is SO(2)-equivariant about the **vertical (Y)** axis — heading is physically arbitrary — while tilts about X/Z are not symmetries (gravity matters). We train on subject-35 walking (fixed-frame per trial) and evaluate robustness to Y-rotations of the test set. The axis-matching ordering from synthetic data **holds on real data with significance**: the matched `SO(2)_y` is the most heading-robust (degradation 1.06×), beating wrong-axis `SO(2)` (p=0.011) and over-constraining `SO(3)` (p=0.005).

Honestly, two caveats. (i) The matched group does *not* significantly beat no-loss here (p=0.33): unlike the synthetic fixed-frame task, real MoCap training already contains heading variation across trials, so the unconstrained model is partially heading-robust to begin with. (ii) At our training budget the transformer backbone is under-fit (mse_rel≈0.9); a better-fit backbone would sharpen all gaps. The mechanism transfers to real data; the *size* of the payoff depends on how much residual-symmetry variation the training set already contains. (`results/mocap_ood.png`.)

### 7.8 Breadth: a model-zoo × dataset study
We ran a grid of eight models spanning the equivariance spectrum — strict-equivariant (egnn, se3_transformer, gatr, emlp), relaxed-architecture (rpp), unconstrained (mlp, transformer), and soft-via-loss (mlp+REMUL) — across five datasets of differing inherent symmetry (N-body SO(3), N-body broken→SO(2), charged N-body, MD17, MoCap). Three findings:

- **The equivariance error `E′` cleanly orders the models**: strict ≈10⁻⁷–10⁻⁵, soft-loss ≈0.1, unconstrained/relaxed ≈1–5. REMUL demonstrably *interpolates* the equivariance amount between hard and unconstrained.
- **Strict priors help only on clean symmetry, and are brittle off it.** Strict models are best only on charged N-body (se3_transformer 2.9×10⁻⁵). As symmetry breaks (N-body field), the unconstrained transformer wins; on **real MoCap the strict SO(3)/SE(3) models diverge** (egnn 13×, se3 3×, gatr catastrophic vs. the no-motion baseline) while unconstrained (mlp/transformer, mse_rel≈0.9) and relaxed (rpp/emlp) models are stable and best. Crucially, this divergence **persists across training budgets** — a longer/lower-LR refit did *not* resolve it (the 200-sample set overfits), so it is a genuine instability of strict rotation-equivariant priors on real partial-symmetry data, not an under-training artifact.
- **Honest caveats:** on the *symmetric* N-body task the strong unconstrained transformer already matches or beats strict models (with enough capacity the bias advantage shrinks, robust across 3 seeds), and MD17 is *intrinsically hard* at this scale (mse_rel≈0.5 for every model regardless of budget — an accuracy floor, not under-fitting). The clean claim is *directional*: the strict → unconstrained/relaxed advantage grows monotonically as inherent symmetry decreases.

(Figures `results/zoo_spectrum.png`, `results/zoo_<dataset>.png`; table `results/tables/zoo_grid.csv`.)

### 7.9 Is the soft loss worth it vs. augmentation? (compute-matched) + robustness across backbones
We compare REMUL against plain data augmentation (DA) at **matched compute** (forward passes; REMUL with one group sample does 2 forwards/step, so DA gets 2× the steps), on the field-broken task, measuring OOD robustness to the residual-symmetry shift.

**The honest verdict — when the symmetry is known, augmentation wins.** Matched DA reaches OOD-MSE 2.6×10⁻⁴ vs REMUL's 9.4×10⁻⁴ — DA is **3.7× more robust**, because augmenting with the true group directly trains on the OOD orbit. *We therefore do not claim the soft loss beats augmentation.* REMUL's measured advantages are two, and both lie in the **uncertain-group** regime: (i) **graceful degradation** — enforcing a *wrong/over* group softly costs ~2× in-distribution, whereas hard-augmenting a false symmetry corrupts the training targets (DA over-SO(3) is 2.1× worse in-distribution than REMUL over-SO(3); DA on the wrong axis is catastrophic OOD); and (ii) it is the **vehicle for selection/discovery** — one cannot cleanly "select an augmentation," since augmenting with an unsure candidate corrupts the data, whereas the soft penalty supports enforce-and-measure selection (§7.3–7.4) and the learnable-axis discovery (§7.6).

**And selection is where REMUL is decisive.** We directly compare *selecting the group by validation error* under each method (both axes, 4 seeds). **DA-selection recovers the true residual axis only 38%** of the time (≈chance), because augmenting with a wrong group still fits validation — the model learns a rotation-smeared function, so the val error does not signal the true symmetry. **REMUL-selection recovers it 88%**, because a wrong penalty conflicts with the task and cleanly raises validation error. So one genuinely *cannot* "select an augmentation," but one can select a soft penalty — the mechanism REMUL uniquely provides.

**Positioning (revised, honest):** *if you know the exact symmetry, augment — it is the stronger robustifier; REMUL's contribution is the "which-symmetry" axis — an architecture-agnostic soft penalty that lets you **select/discover** the residual group when it is unknown (88% vs augmentation's 38%) and degrades gracefully when the choice is imperfect.*

**Robustness across backbones.** The matched-group-wins-OOD result replicates across **three backbones** (MLP, Transformer, GNN; 5 seeds each): the matched residual symmetry is the most OOD-robust on every backbone, beating the wrong axis by 25–130× (p≤0.001 everywhere) and no-loss catastrophically; it also beats over-constraining SO(3) on MLP/GNN (p<0.02) and ties on the Transformer.

**Not a small-data artifact.** The matched-group OOD advantage is **~1500–2200× over no-loss and flat across a 20× range of training size** (n_train 200→4000): the unconstrained model never improves its OOD robustness with more data, because the shifted orientations are never in the fixed-frame training distribution regardless of n. The advantage is a property of the symmetry, not of scarcity. (`results/pareto_da_remul.png`, `results/replicate_ood.png`, `results/datascale.png`.)

### 7.10 Baselines — who can recover *which* symmetry?
We compare against the closest "learn which symmetry" method, **Augerino** (Benton et al., 2020), which learns a per-generator augmentation distribution, plus the relaxed/equivariant architectures already in our zoo (RPP, EMLP) and the data-augmentation and no-loss references. On the field-broken task (residual SO(2)_z), measuring residual-axis recovery and OOD robustness:

| method | recovers residual axis? | OOD MSE | needs known group? |
|---|---|---|---|
| **REMUL-selection** | **88%** | 9.4×10⁻⁴ | no |
| **REMUL learnable-axis** | **100%** (dominant) | 9.4×10⁻⁴ | no |
| Augerino (best λ) | **0%** (isotropic, z/xy=1.0) | 2.3×10⁻³ | no |
| Data augmentation | n/a (given) | 2.6×10⁻⁴ | **yes** |
| RPP (relaxed arch) | n/a (fixed) | 7.6×10⁻³ | fixed |
| EMLP (equiv arch) | n/a (fixed) | 1.2×10⁻³ | fixed |
| no-loss | — | 3.5 | — |

**Augerino cannot recover the residual axis at any λ** — a full sweep (λ ∈ 3×10⁻⁶…2×10⁻³) shows its learned ranges stay perfectly isotropic (best θ_z/θ_{x,y} ratio = 1.0), sliding from no-augmentation to full-SO(3) with no anisotropic window. **The reason is structural and is the key point of this comparison:** augmentation-based discovery tests a symmetry by augmenting *inputs* and reading the task loss, which conflates a symmetry of the *task* with a symmetry of the *input distribution*. With fixed-frame training data, augmenting about the true axis moves inputs off-distribution just as much as augmenting about a false one, so the signal cannot isolate the residual axis. REMUL's ground-truth-anchored loss ‖f(R·x) − R·y‖ instead checks whether R maps the input to the *correctly transformed target* — a property of the task itself, independent of the input distribution — and so recovers the residual symmetry (88–100%). Among methods that do not require the group to be specified in advance, REMUL is the only one that isolates the *minimal* residual symmetry; augmentation is the strongest robustifier but only once the group is known.

**The label-breaks-symmetry wedge vs LieGAN.** To separate *task* symmetry from *input-distribution* symmetry cleanly, we construct a task whose inputs are SO(3)-symmetric but whose labels are not: isotropic initial conditions (a random SO(3) rotation per sample; input covariance eigenvalue ratio 1.07, vs 70.7 for the canonical frame) integrated under a *fixed* z-field, so the map is only SO(2)_z-equivariant (ground-truth task-equivariance error 0.0000 for SO(2)_z vs 0.089 for SO(3)). A distribution-based discoverer (LieGAN) tests input invariance: on these inputs *every* rotation group is a symmetry (two-sample MMD² ≈ 0.0003 for SO(2)_z, SO(2)_x, and SO(3) alike), so it discovers the **maximal input symmetry SO(3) and misses the label-breaking by construction.** REMUL's ground-truth-anchored loss instead measures the task-equivariance column and identifies SO(2)_z. The full picture across the two input regimes — *which method recovers the true task symmetry?*:

| method (basis) | fixed-frame inputs | isotropic inputs |
|---|---|---|
| **REMUL** (task-anchored ‖f(Rx)−Ry‖) | ✓ (88% selection) | ✓ (task-equiv = 0.0000) |
| Augerino (averaging + task loss) | ✗ (isotropic θ) | ✓ (θ_z/θ_{x,y} ≈ 18) |
| LieGAN (input-distribution invariance) | — | ✗ (discovers SO(3)) |

**REMUL is the only method that recovers the task symmetry in *both* input regimes.** Augerino succeeds only when the inputs are isotropic (its averaging needs on-distribution augmentations); LieGAN fails on the label-broken case because it is blind to symmetries the input distribution does not already exhibit. This is the sharpest novelty statement of the paper: a *mechanistic* account of exactly when each competitor recovers task symmetry, and why anchoring on the task (REMUL) is uniquely robust. (`results/wedge.png`.)

### 7.11 Graph domain: the scheduling null and equivariance-robustness on OOD (QM9)
Two of our claims were established on graph regression only in-distribution; we extend them to the rotated-test (OOD) metric on QM9/gcn.

**The scheduling null extends to OOD.** At matched achieved equivariance error E′, the *rotated-test* MAE tracks the enforcement amount (Spearman(OOD-MAE, E′) = −0.63, p=2×10⁻³; enforcement explains 70% of OOD-MAE variance) while the schedule *shape* adds essentially nothing (residual +0.0018 ± 0.0026). So the "layerwise scheduling is a confound" null holds on OOD as well as in-distribution — it is not an in-distribution artifact.

**Equivariance improves rotation-robustness on graphs.** Enforcing the symmetry reduces the rotation-OOD gap (OOD − in-distribution MAE) monotonically with the achieved functional equivariance E_prime (Spearman = 0.88, p=2×10⁻⁷): the gap falls from 0.067 (no-loss) to 0.0025 (heavily enforced), a ~27× improvement in rotation robustness — at an accuracy cost. So the qualitative payoff (enforcement → OOD robustness) transfers from dynamics to graphs.

**Scope (honest).** The sharper *group-selection* result — recovering *which* residual symmetry among candidates — is dynamics-specific here: standard molecular graphs (QM9) are *fully* SO(3)-invariant, so there is no partial/residual symmetry to select. Reproducing selection on graphs would require a purpose-built broken-symmetry graph task. A second real graph dataset (MD17-graph) was blocked by a missing `pyg-lib` dependency; the QM9 results above are on the graph testbed available. (`results/graph_ood.png`.)

## 8. Honest limitations and negative results

- **In-distribution, soft equivariance is a cost, not a benefit.** On the isotropic (SO(3)-symmetric) control — where all groups have zero bias — penalizing even the *true* SO(3) is the worst option, scaling with constraint dimension: an **optimization-interference tax** the linear theory omits. In-distribution accuracy is the wrong criterion.
- **The crossover's existence is known** representation theory; only its `n^{−1/2}` location scaling is our contribution.
- **The strong theorems are exact only in the linear/kernel regime.** For deep nets they are a validated predictive model; where quantitative predictions transfer (−1/2 slope, interior β*, over-constrain-wants-less) we show it, and where they don't (exact axis in-distribution) we say so.
- **A retracted claim, now a quantified null (R1).** An earlier "layerwise scheduling helps in-distribution" result was a confound. At matched enforcement the data are unambiguous: the *achieved* equivariance error E′ alone explains **80%** of the MAE variance across all schedule/α₀ arms (Spearman −0.79, p=4×10⁻⁷), and schedule *shape* adds essentially nothing (residual MAE vs the single MAE–E′ curve: +0.0008 ± 0.009). The no-loss baseline has the lowest MAE of all arms. "Scheduling helps" was the confound of schedules that simply enforced less. (`results/rq1_matched.png`.)
- **Data augmentation** must be compared compute-matched (in progress); we will report the Pareto outcome whichever way it falls and position the contribution on *which-group selection*, which augmentation cannot express.

## 9. Conclusion

Soft equivariance is a bias–variance instrument, not a black-box scalar. We gave it a closed-form theory (penalty = subspace Tikhonov; `λ*=V/R`; a −1/2 crossover scaling law; a selection-consistency theorem), showed the theory transfers to deep nets where its assumptions hold, and established that the practical payoff is symmetry-aware **out-of-distribution robustness**, where selecting the matched residual symmetry wins by orders of magnitude and the correct symmetry is crisply identifiable — while being candid that in-distribution accuracy carries an optimization cost and is the wrong lens. The result is an architecture-agnostic, theory-grounded, honestly-scoped account of *when* and *which* symmetry to enforce.

---

### Reproducibility
- Theory sandbox: `python -m relaxed.theory.sandbox` (gate 6/6), tests `python test_theory_sandbox.py`.
- Deep-net experiments: `python -m relaxed.theory.run_transfer {e1,e1v2,e3,e3ext,e5ood,e2beta,e3learn} --steps N --gpus 0,1`; analysis `python -m relaxed.theory.analyze_transfer`.
- Results: `results/transfer_e{1,1v2,2,3,5}.json`, figures `results/{crossover,ood_robustness,beta_optimum}.png`.

### References (to be completed)
Elhag et al. 2024 (REMUL, arXiv:2410.17878) · Finzi, Benton & Wilson 2021 (Residual Pathway Priors) · Wang et al. 2022, 2023 (Approximately Equivariant / Relaxed Group Conv) · van der Ouderaa et al. 2022 · Benton et al. 2020 (Augerino) · Yang et al. 2023 (LieGAN) · Elesedy & Zaidi 2021 · Satorras et al. 2021 (EGNN).

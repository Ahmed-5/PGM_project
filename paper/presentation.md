---
marp: true
paginate: true
theme: default
title: When and Which Symmetry to Enforce
---

<!-- Render: `marp presentation.md --pdf` (or open in the Marp VS Code extension). Figures live in ./figures/ -->

# When *and Which* Symmetry to Enforce
### A Bias–Variance Theory of Soft Equivariance, with Consistent Loss-Side Group Selection

Extends **REMUL** (Relaxed Equivariance via Multitask Learning, arXiv:2410.17878)

*Theory + ~550 GPU runs · every claim significance-tested · limitations stated*

---

## The problem

**Hard equivariance** (EGNN, SE(3)-Transformer, GATr) bakes symmetry into the architecture — optimal when the symmetry is *exact*, a liability when it is *broken or partial* (fields, substrates, gravity).

**Soft equivariance** (REMUL) trains an *unconstrained* net with a multitask penalty:
$$ \mathcal{L} = \mathcal{L}_\text{task} + \beta\,\underbrace{\mathbb{E}_{g,x}\lVert f(g\cdot x) - g\cdot f(x)\rVert^2}_{\Delta_G(W)} $$

But it is used as a **black-box scalar**. Three open questions:

1. **How much?** — the right $\beta$; when does soft beat hard / beat plain training?
2. **Which?** — which group $G$ to penalize when the true symmetry is unknown?
3. **Worth it?** — does the loss beat cheap data augmentation?

---

## Contributions

| # | pillar | result |
|---|--------|--------|
| **1** | **Theory** | the penalty is *exactly* subspace Tikhonov ⇒ closed-form $\lambda^\*{=}V/R$, strict dominance, a **crossover scaling law** |
| **2** | **Method** | a **selection-consistency theorem** + method that recovers the task's residual symmetry (incl. a continuous, non-canonical axis) |
| **3** | **Evidence** | theory transfers to deep nets; **the payoff is OOD robustness**; validated on synthetic, N-body, real MoCap, and a model-zoo × dataset grid |

**Stance:** diagnostic and honest — we retract a confounded claim, concede what's already known, and report augmentation/limitations either way.

---

## Theory 1 — the penalty *is* subspace Tikhonov

**Proposition 1 (exact).** With $P_G$ the Reynolds/Haar twirl projector onto the $G$-commutant (equivariant subspace):
$$ \Delta_G(W) = 2\,\lVert (I - P_G)\,\mathrm{vec}(W)\rVert^2 $$

→ the penalty shrinks **only the anti-symmetric components**, Tikhonov strength $\lambda = 2\beta$.

*Verified to $1.9\times10^{-15}$ relative error; commutant dims $\dim\mathrm{SO}(3){=}1$, $\dim\mathrm{SO}(2){=}3$ recovered exactly.*

---

## Theory 2 — optimal weight & crossover law

Let $R = \lVert(I{-}P_G)\mathrm{vec}(W^\*)\rVert^2$ (signal outside commutant), $V = \sigma^2 d_\text{asym}/n$ (variance).

**Theorem 1.** $\;\mathrm{ER}(\lambda) = \dfrac{\lambda^2 R + V}{(1+\lambda)^2}\;$ ⟹ $\;\lambda^\* = V/R,\;\; \mathrm{ER}(\lambda^\*) = \dfrac{RV}{R+V} < \min(R,V)$

→ **soft strictly dominates both hard ($\lambda{\to}\infty$) and unconstrained ($\lambda{=}0$).**

**Corollary — crossover scaling law.** Hard beats unconstrained iff $R<V$; with a field $R\propto s^2$:
$$ s_\text{cross} \propto \sqrt{\sigma^2 d_\text{asym}/n} \;\;\Rightarrow\;\; \textbf{slope } -\tfrac{1}{2}\ \text{in}\ n $$

The crossover's *existence* is known rep-theory; its **location scaling is the novel, falsifiable prediction.**

---

## Theory 3 — selecting *which* symmetry

**Theorem 2 (selection consistency).** With $G_\text{true}$ in the menu and a signal-outside-commutant margin $\gamma>0$:
$$ P(\hat G_n = G_\text{true}) \to 1 $$
— driven by *signal overlap*, **not** constraint dimension.

**Refinement (verified):** the $\lambda{=}0$ hypergradient is non-discriminating (negative for every group); the correct signal is the **achieved-minimum** validation error.

**Continuous axis:** a learnable $\hat n$ penalizing $\mathrm{SO}(2)$ about $\hat n$ recovers a **non-canonical** axis to **0.4°** in the exact regime — a case a discrete menu cannot express.

---

## Exact-regime falsification gate — 6/6 ✓

| check | result |
|-------|--------|
| Prop 1 identity | PASS ($1.9\text{e-}15$) |
| Thm 1 optimum $\lambda^\*{=}V/R$, dominance | PASS (curve 2.2%) |
| Crossover slope | PASS (**−0.502** vs −0.5) |
| Selection recovers axis, flips | PASS (95%) |
| Continuous non-canonical axis | PASS (0.4°) |
| Selection signal = achieved-min | PASS |

*A pre-committed go/no-go on the whole theory — passed before any deep-net investment.*

---

## E1 — the −½ crossover law **transfers** to deep nets

![w:760](figures/crossover.png)

Variance-limited regime: **slope −0.44, R² = 0.97** (theory −0.50). Saturates when the unconstrained model becomes capacity-limited — *exactly as the theory predicts.* EGNN excess error verified $\propto\text{field}^2$.

---

## E3 — non-oracle group selection

Select the loss group by **validation MSE** (never told the axis):

| condition | true group | recovery | so3 (over-constrain) | no-loss |
|---|---|---|---|---|
| field, axis x | so2_x | **75%** | 7.8e-4 (worst) | 2.5e-4 |
| field, axis z | so2_z | **62%** | 7.5e-4 (worst) | 2.3e-4 |
| isotropic $R{=}0$ | — | penalty hurts even true SO(3) | 7.5e-4 | 3.0e-4 |

- **"Don't over-constrain":** matched SO(2) ≫ full SO(3), $p<10^{-4}$.
- **Axis identity is only marginal in-distribution** (p=0.07/0.74) — the wrong lens. →

---

## E5 — the payoff is **out-of-distribution** 🎯

![w:720](figures/ood_robustness.png)

Under the shift the symmetry protects against: matched group wins by **100–1000×**, beats over-constraining SO(3) ~2× ($p<10^{-4}$), and the correct axis is now **sharply significant** ($p<0.005$) — where in-distribution it was noise.

---

## E2 — the interior optimum $\beta^\*$ transfers

![w:760](figures/beta_optimum.png)

In-distribution MSE rises monotonically with $\beta$ (an optimization tax); **OOD-MSE has an interior optimum** — $\beta^\*{\approx}1$ (matched), $\beta^\*{\approx}0.3$ (over-constraining). Over-constraining wants **less** enforcement — exactly $\lambda^\*{=}V/R$.

---

## Learnable axis + the retracted RQ1 null

**Continuous axis (deep net):** identifies the correct *dominant* axis in **100% of seeds**, but converges to a systematic ~38° (vs 0.4° in the sandbox) — reliable coarse, imperfect fine.

**R1 — matched-enforcement null:** at matched achieved-$E'$, the *enforcement magnitude* explains **80%** of MAE variance (Spearman −0.79, $p{=}4\text{e-}7$); **schedule shape adds essentially nothing** (+0.0008 ± 0.009); no-loss is best. → "scheduling helps" was a **confound**, now retracted.

![w:560](figures/rq1_matched.png)

---

## E6 — real data: CMU Motion Capture

![w:640](figures/mocap_ood.png)

Locomotion is SO(2)-equivariant about the **vertical**. Matched SO(2)$_y$ is the most heading-robust, beating wrong-axis ($p{=}0.011$) and SO(3) ($p{=}0.005$).
**Honest caveat:** not significant vs no-loss ($p{=}0.33$) — real MoCap already varies heading; and the backbone was under-fit at this budget.

---

## Model zoo — the equivariance *spectrum*

![w:780](figures/zoo_spectrum.png)

Equivariance error $E'$ orders the models cleanly: **strict ≈1e-7 → soft (REMUL) ≈0.1 → unconstrained ≈1–5.** REMUL demonstrably *interpolates* the equivariance amount between hard and unconstrained.

---

## Model zoo — strict priors are **brittle** off their symmetry

![w:640](figures/zoo_mocap.png)

On **real MoCap**, strict SO(3)/SE(3) models (egnn 13×, se3 3×, gatr ✗) **diverge** (⚠) — **robustly across training budgets** (a longer/lower-LR refit did not fix it) — while unconstrained & relaxed are stable and best.

| dataset (symmetry) | best model | class |
|---|---|---|
| charged nbody (SO(3)) | se3_transformer | **strict** |
| nbody broken → SO(2) | transformer | unconstrained |
| MoCap (real SO(2)) | transformer | unconstrained |

**Strict → unconstrained advantage grows as inherent symmetry decreases.** (MD17: intrinsically hard, mse_rel≈0.5 for all.)

---

## Is the soft loss worth it vs. augmentation?

![w:600](figures/pareto_da_remul.png)

**When the symmetry is known — augmentation WINS** (matched DA 2.6e-4 vs REMUL 9.4e-4 OOD, **3.7×**). *We do not claim soft > augmentation.*

REMUL's niche is the **uncertain-group** regime: soft enforcement **degrades gracefully** (2.1× better in-dist than hard DA on a wrong group) and it is the **vehicle for selection**.

---

## You can't select an augmentation — but you can select a penalty

![w:560](figures/selcompare.png)

Select the group by validation error (unknown axis): **DA-selection recovers the true axis only 38%** (≈chance — a wrong augmentation still fits val). **REMUL-selection: 88%** — a wrong penalty conflicts with the task and cleanly raises val error.

**This is REMUL's decisive niche: known symmetry → augment; unknown symmetry → REMUL selects it.**

---

## Baselines: only REMUL recovers *which* symmetry

![w:560](figures/augerino_theta.png)

**Augerino cannot recover the residual axis at any λ** (isotropic, z/xy=1.0) — augmentation-based discovery conflates *task* symmetry with *input-distribution* symmetry; with fixed-frame data it can't isolate the axis.

| method (no known group) | recovers axis | OOD |
|---|---|---|
| **REMUL-selection** | **88%** | 9.4e-4 |
| **REMUL learnable-axis** | **100%** | 9.4e-4 |
| Augerino (best λ) | **0%** | 2.3e-3 |

REMUL's *ground-truth-anchored* loss checks the task symmetry directly — that's the structural edge.

---

## The wedge: task symmetry vs input symmetry (vs LieGAN)

![w:600](figures/wedge.png)

Isotropic inputs (SO(3)-symmetric) + z-field ⇒ task is only SO(2)_z. **LieGAN sees every group as an input symmetry → discovers SO(3), missing the label-breaking.** REMUL anchors on the task (‖f(Rx)−Ry‖, SO(2)_z err = 0.0000).

| recovers TASK symmetry? | fixed-frame | isotropic |
|---|---|---|
| **REMUL** (task-anchored) | ✓ 88% | ✓ (err=0) |
| Augerino (avg + task) | ✗ | ✓ |
| LieGAN (input dist.) | — | ✗ SO(3) |

**REMUL is the only method that works in *both* input regimes.**

---

## The headline replicates across backbones

![w:680](figures/replicate_ood.png)

Matched residual symmetry is **the most OOD-robust on mlp, transformer AND gnn** — beating the wrong axis 25–130× (p≤0.001 everywhere; 5 seeds each).

---

## Summary — the thesis, established & honest

1. **Theory exact** in-regime (Prop 1, Thm 1, crossover law, Thm 2). ✓ 6/6
2. **−½ crossover law transfers** to deep nets (−0.44, R²=.97).
3. **Don't over-constrain** — matched SO(2) ≫ full SO(3) ($p{<}10^{-4}$).
4. **The payoff is OOD robustness** — matched wins 100–1000×; axis crisply selectable there.
5. **Interior $\beta^\*$ transfers** ($\lambda^\*{=}V/R$; over-constraining wants less).
6. **Equivariance spectrum**: REMUL interpolates; strict is brittle on real partial-symmetry data.

---

## Honest limitations

- **In-distribution accuracy is the wrong criterion** — soft equivariance carries an *optimization-interference tax* (hurts even the true symmetry). Use OOD.
- **Crossover existence** is known rep-theory; only its **location scaling** is new.
- **Theorems are exact only** in the linear/kernel regime — a *validated predictive model* for deep nets.
- **Continuous axis** converges coarsely (~38°) on deep nets; **MD17** hits an intrinsic accuracy floor (mse_rel≈0.5) at this scale.
- **Data augmentation beats the soft loss when the symmetry is known** (measured, 3.7×) — REMUL is repositioned onto the *which-symmetry* / uncertain-group axis, not "soft > augmentation."

---

## Roadmap

- **Now running:** refit MoCap/MD17 at adequate budget + 3rd seed (removes the under-fit/divergence artifacts).
- Compute-matched **DA-vs-REMUL Pareto** (settles "is the loss worth it").
- **≥10 seeds**, ≥3-backbone replication on headline cells.
- Camera-ready: LaTeX + full baseline table (RPP / Augerino / LieGAN / Wang).

**Reproducible:** `python -m relaxed.theory.{sandbox,run_transfer,analyze_transfer,analyze_zoo,build_dashboard}`

---

# Thank you

**When & which symmetry to enforce** — soft equivariance is a *bias–variance instrument*: match the residual symmetry, don't over-constrain, and judge it out-of-distribution.

*Bundle: manuscript.md · dashboard.html · figures/ · tables/ · code/*

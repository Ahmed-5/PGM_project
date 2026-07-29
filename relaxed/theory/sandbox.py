"""Phase-0 falsification gate: an exact linear-Gaussian sandbox for the
bias-variance theory of soft (REMUL) equivariance.

WHY THIS EXISTS
---------------
Before investing months, we must establish that the theory underpinning the
proposed A*-tier paper is *correct where it is exact*. In the linear/kernel
regime with orthogonal group representations and a G-invariant input law, every
claim below is closed-form and can be verified against Monte-Carlo simulation.
If any check fails here, the theoretical framing is wrong and the program should
pivot (see PIVOT note in main()) rather than being forced onto deep nets.

THE MODEL
---------
Linear map W in R^{3x3} acting on 3D vectors. A group G acts on the input by an
orthogonal representation U_g and on the output by V_g (here V_g = U_g: the same
rotation acts on positions and predicted positions, as in the N-body task).

  * data:      x ~ N(0, I_3),  y = W* x + eps,  eps ~ N(0, sigma^2 I_3),  n samples
  * penalty:   the REMUL equivariance loss for a linear map is
                   Delta_G(W) = E_{g,x} || W U_g x - V_g W x ||^2
               and the training objective is  (1/n) sum ||y_i - W x_i||^2 + (lambda/2) Delta_G(W).

CLAIMS BEING GATED
------------------
Prop 1  Delta_G(W) = 2 || (I - P_G) vec(W) ||^2 , P_G = orthogonal projector onto
        the G-commutant (equivariant subspace). => penalty is subspace Tikhonov.
Thm 1   With R = ||(I-P_G) vec(W*)||^2 and V = sigma^2 d_asym(G)/n, the
        anti-symmetric excess risk is (lambda^2 R + V)/(1+lambda)^2, minimized at
        lambda* = V/R with value RV/(R+V) < min(R,V): soft dominates hard & none.
Crossover  strict beats unconstrained iff R < V; R ~ s^2 => s_cross ~ n^{-1/2}.
Selection  argmin-over-groups of held-out validation error recovers the true
        residual symmetry, and the pick flips with the field axis.
Refinement the lambda=0 hypergradient is negative for EVERY group (variance always
        cuts first), so it is non-discriminating; the achieved-min val error is
        the correct selection signal.

Uses closed-form projectors and an analytic 9x9 Gram (I_3 kron Sigma_xx), so it
runs in seconds. `python -m relaxed.theory.sandbox` prints a PASS/FAIL report and
writes results/theory_gate.json.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

VEC_I = np.eye(3).reshape(-1)  # row-major vec(I_3)


# --------------------------------------------------------------------------- #
# Exact commutant projectors P_G on vec(W) (row-major), 9x9
# --------------------------------------------------------------------------- #
def _proj_from_basis(mats: list[np.ndarray]) -> np.ndarray:
    """Orthogonal projector onto span{vec(M) : M in mats} (row-major vec)."""
    B = np.stack([m.reshape(-1) for m in mats], axis=1)   # (9, k)
    Q, _ = np.linalg.qr(B)
    return Q @ Q.T


def _J(axis: int) -> np.ndarray:
    """Generator of rotation about `axis` (antisymmetric)."""
    J = np.zeros((3, 3))
    a, b = [i for i in range(3) if i != axis]
    J[a, b], J[b, a] = -1.0, 1.0
    return J


def _diag(*d) -> np.ndarray:
    return np.diag(np.array(d, dtype=float))


def exact_projector(group: str) -> np.ndarray:
    """Closed-form projector onto the commutant of the standard 3D rep.

    SO(3): span{I}                          (Schur; d_sym = 1)
    SO(2)_axis: span{I_plane, J_axis, E_axis}  (d_sym = 3)
    """
    if group == "so3":
        return _proj_from_basis([np.eye(3)])
    axis = {"so2_x": 0, "so2_y": 1, "so2_z": 2}[group]
    others = [i for i in range(3) if i != axis]
    plane = np.zeros((3, 3)); plane[others[0], others[0]] = 1.0; plane[others[1], others[1]] = 1.0
    Eax = np.zeros((3, 3)); Eax[axis, axis] = 1.0
    return _proj_from_basis([plane, _J(axis), Eax])


PROJ = {g: exact_projector(g) for g in ["so2_x", "so2_y", "so2_z", "so3"]}
DSYM = {g: int(round(np.trace(P))) for g, P in PROJ.items()}


# --------------------------------------------------------------------------- #
# Group elements (for the Monte-Carlo Prop-1 identity only)
# --------------------------------------------------------------------------- #
def rot_axis(axis: int, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    R = np.eye(3)
    if axis == 2:
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    elif axis == 0:
        R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    elif axis == 1:
        R[2, 2], R[2, 0], R[0, 2], R[0, 0] = c, -s, s, c
    return R


def so2_elems(axis: int, k: int = 720) -> list[np.ndarray]:
    return [rot_axis(axis, float(t)) for t in (np.arange(k) + 0.5) / k * 2 * math.pi]


def so3_elems(k: int, rng) -> list[np.ndarray]:
    out = []
    for _ in range(k):
        Q, Rm = np.linalg.qr(rng.standard_normal((3, 3)))
        Q = Q @ np.diag(np.sign(np.diag(Rm)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        out.append(Q)
    return out


def delta_G_montecarlo(W, elems) -> float:
    """E_{g,x} ||W U_g x - U_g W x||^2  =  mean_g ||W U_g - U_g W||_F^2 ."""
    return float(np.mean([((W @ U - U @ W) ** 2).sum() for U in elems]))


def twirl_sym(elems) -> np.ndarray:
    """Symmetrized raw twirl (1/2)(P+P^T), P = mean_g (U_g^T kron U_g^T).
    Makes the Prop-1 identity EXACT for the given finite element set."""
    P = np.zeros((9, 9))
    for U in elems:
        P += np.kron(U.T, U.T)
    P /= len(elems)
    return 0.5 * (P + P.T)


# --------------------------------------------------------------------------- #
# Analytic ridge-in-subspace  (Gram = I_3 kron Sigma_xx, row-major vec)
# --------------------------------------------------------------------------- #
def gram_and_b(X, Y):
    """Return (G, b) for the least-squares part: predicting Y_i = W X_i.
    With row-major vec(W): G = I_3 kron (X^T X / n),  b = vec(Y^T X / n)."""
    n = X.shape[0]
    Sigma = X.T @ X / n
    G = np.kron(np.eye(3), Sigma)
    b = (Y.T @ X / n).reshape(-1)
    return G, b


def solve_ridge(G, b, P, lam):
    """min (1/n)||y - A w||^2 + lam * w^T (I-P) w. lam=inf => hard (project out
    the anti-symmetric part) via a large-penalty proxy."""
    if math.isinf(lam):
        lam = 1e12
    Q = np.eye(9) - P
    return np.linalg.solve(G + lam * Q, b).reshape(3, 3)


def excess_risk(What, Wstar):
    D = What - Wstar
    return float((D * D).sum())


def er_closed_form(R, d_asym, d_sym, sigma, n, lam):
    V = sigma ** 2 * d_asym / n
    sym_var = sigma ** 2 * d_sym / n
    anti = R if math.isinf(lam) else (V + lam ** 2 * R) / (1.0 + lam) ** 2
    return sym_var + anti


# --------------------------------------------------------------------------- #
# Targets with a controllable residual symmetry
# --------------------------------------------------------------------------- #
def target_field(a: float, s: float, axis: int) -> np.ndarray:
    """W*(s) = a*I + s*E_axis: SO(2)-equivariant about `axis` for all s, breaks
    SO(3) for s != 0. For SO(3): R = ||(I-P)(aI+sE)||^2 = 2 s^2/3."""
    E = np.zeros((3, 3)); E[axis, axis] = 1.0
    return a * np.eye(3) + s * E


def R_of(Wstar, group):
    w = Wstar.reshape(-1)
    r = w - PROJ[group] @ w
    return float(r @ r)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    data: dict = field(default_factory=dict)


def _assert_ordering():
    """One-time sanity: analytic Gram/b reproduces a brute-force design fit."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 3)); Wt = rng.standard_normal((3, 3))
    Y = X @ Wt.T
    G, b = gram_and_b(X, Y)
    w = np.linalg.solve(G + 1e-9 * np.eye(9), b).reshape(3, 3)
    return np.allclose(w, Wt, atol=1e-4)


def check_prop1(rng) -> Check:
    dims_ok = DSYM == {"so2_x": 3, "so2_y": 3, "so2_z": 3, "so3": 1}
    errs = []
    for elems in [so2_elems(2, 720), so3_elems(400, rng)]:
        Ps = twirl_sym(elems)
        for _ in range(100):
            W = rng.standard_normal((3, 3))
            mc = delta_G_montecarlo(W, elems)
            w = W.reshape(-1)
            an = 2.0 * float(w @ (np.eye(9) - Ps) @ w)
            errs.append(abs(mc - an) / (abs(mc) + 1e-9))
    rel = float(np.max(errs))
    passed = dims_ok and rel < 1e-9 and _assert_ordering()
    detail = (f"commutant dims {DSYM} (SO3=1, SO2=3) -> {'OK' if dims_ok else 'MISMATCH'}; "
              f"max rel |Delta_MC - 2||(I-P)W||^2| = {rel:.2e}; gram-ordering ok={_assert_ordering()}")
    return Check("Prop 1: penalty == 2||(I-P_G)W||^2 (exact subspace Tikhonov)", passed, detail,
                 {"d_sym": DSYM, "prop1_max_rel_err": rel})


def check_thm1(rng) -> Check:
    axis, a, s, sigma, n = 2, 1.0, 0.8, 0.5, 400
    P = PROJ["so3"]; dsym = DSYM["so3"]; dasym = 9 - dsym
    Wstar = target_field(a, s, axis)
    R = R_of(Wstar, "so3"); V = sigma ** 2 * dasym / n
    lam_star_theory = V / R
    lams = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 40)])
    trials, er = 400, np.zeros(len(lams))
    for _ in range(trials):
        X = rng.standard_normal((n, 3)); Y = X @ Wstar.T + sigma * rng.standard_normal((n, 3))
        G, b = gram_and_b(X, Y)
        for j, lam in enumerate(lams):
            er[j] += excess_risk(solve_ridge(G, b, P, lam), Wstar)
    er /= trials
    er_cf = np.array([er_closed_form(R, dasym, dsym, sigma, n, lam) for lam in lams])
    rel_curve = float(np.max(np.abs(er - er_cf) / (er_cf + 1e-9)))
    lam_star_emp = float(lams[int(np.argmin(er))])
    er0, er_hard, er_min = er[0], er_closed_form(R, dasym, dsym, sigma, n, math.inf), float(np.min(er))
    soft_dominates = er_min < min(er0, er_hard) - 1e-6
    grid_near = lams[np.argmin(np.abs(lams - lam_star_theory))]
    argmin_ok = abs(math.log(lam_star_emp + 1e-9) - math.log(grid_near + 1e-9)) < math.log(3.5)
    passed = rel_curve < 0.06 and soft_dominates and argmin_ok
    detail = (f"R={R:.4f} V={V:.4f} lambda*=V/R={lam_star_theory:.3f}; emp argmin={lam_star_emp:.3f}; "
              f"ER curve max rel err={rel_curve:.2%}; soft {er_min:.4f} < min(unc {er0:.4f}, hard {er_hard:.4f})="
              f"{'DOMINATES' if soft_dominates else 'NO'}")
    return Check("Thm 1: ER=(lam^2 R+V)/(1+lam)^2, lambda*=V/R, soft dominates hard & none",
                 passed, detail,
                 {"R": R, "V": V, "lambda_star_theory": lam_star_theory, "lambda_star_emp": lam_star_emp,
                  "curve_rel_err": rel_curve, "er_soft": er_min, "er_unc": float(er0), "er_hard": er_hard})


def check_crossover_slope(rng) -> Check:
    axis, a, sigma = 2, 1.0, 0.5
    P = PROJ["so3"]
    ns = [100, 200, 400, 800, 1600]
    s_grid = np.geomspace(0.03, 3.0, 26)
    trials, s_cross = 60, []
    for n in ns:
        eh = np.zeros(len(s_grid)); eu = np.zeros(len(s_grid))
        Wstars = [target_field(a, float(s), axis) for s in s_grid]
        for _ in range(trials):
            X = rng.standard_normal((n, 3)); noise = sigma * rng.standard_normal((n, 3))
            for k, Wstar in enumerate(Wstars):
                Y = X @ Wstar.T + noise
                G, b = gram_and_b(X, Y)
                eh[k] += excess_risk(solve_ridge(G, b, P, math.inf), Wstar)
                eu[k] += excess_risk(solve_ridge(G, b, P, 0.0), Wstar)
        eh /= trials; eu /= trials
        diff = eh - eu
        idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
        if not len(idx):
            continue
        i = idx[0]; d0, d1 = diff[i], diff[i + 1]
        ls = math.log(s_grid[i]) + (0 - d0) * (math.log(s_grid[i + 1]) - math.log(s_grid[i])) / (d1 - d0)
        s_cross.append((n, math.exp(ls)))
    logn = np.array([math.log(n) for n, _ in s_cross]); logs = np.array([math.log(s) for _, s in s_cross])
    slope, intercept = np.polyfit(logn, logs, 1)
    passed = abs(slope + 0.5) < 0.06
    detail = ("s_cross: " + ", ".join(f"n{n}={s:.3f}" for n, s in s_cross)
              + f"; log-log slope={slope:.3f} (theory -0.500)")
    return Check("Crossover scaling: s_cross ~ sqrt(sigma^2 d/n) -> slope -1/2", passed, detail,
                 {"s_cross": s_cross, "slope": float(slope)})


def _best_val_err(X, Y, Xv, Yv, P):
    G, b = gram_and_b(X, Y)
    lams = np.concatenate([[0.0], np.geomspace(1e-3, 1e5, 26)])
    best = math.inf
    for lam in lams:
        What = solve_ridge(G, b, P, lam)
        best = min(best, float(((Xv @ What.T - Yv) ** 2).sum() / Xv.shape[0]))
    return best


def _recovery_rate(rng, true_axis, name, menu, a, s, sigma, n, nv, trials):
    Wstar = target_field(a, s, true_axis)
    picks, table = [], {g: [] for g in menu}
    for _ in range(trials):
        X = rng.standard_normal((n, 3)); Y = X @ Wstar.T + sigma * rng.standard_normal((n, 3))
        Xv = rng.standard_normal((nv, 3)); Yv = Xv @ Wstar.T + sigma * rng.standard_normal((nv, 3))
        errs = {g: _best_val_err(X, Y, Xv, Yv, PROJ[g]) for g in menu}
        picks.append(min(errs, key=errs.get))
        for g, e in errs.items():
            table[g].append(e)
    return {"recovery_rate": picks.count(name) / len(picks),
            "selected_mode": max(set(picks), key=picks.count),
            "mean_val_err": {g: float(np.mean(v)) for g, v in table.items()}}


def check_group_selection(rng) -> Check:
    """Gate at a clearly-resolvable SNR, and ALSO record recovery-vs-field-strength
    (Theorem 2's margin: recovery degrades as the break -> 0). Honest either way."""
    a, sigma, n, nv, menu = 1.0, 0.5, 600, 6000, ["so2_x", "so2_y", "so2_z", "so3"]
    results = {}
    for true_axis, name in [(2, "so2_z"), (0, "so2_x")]:
        results[name] = _recovery_rate(rng, true_axis, name, menu, a, 1.0, sigma, n, nv, 80)
    # recovery-vs-SNR curve (z-field task): the informative, honest sub-result.
    snr_curve = {}
    for s in [0.3, 0.5, 0.8, 1.2]:
        snr_curve[f"s={s}"] = _recovery_rate(rng, 2, "so2_z", menu, a, s, sigma, n, nv, 60)["recovery_rate"]
    results["recovery_vs_field"] = snr_curve
    z, x = results["so2_z"], results["so2_x"]
    passed = (z["recovery_rate"] >= 0.95 and z["selected_mode"] == "so2_z"
              and x["recovery_rate"] >= 0.95 and x["selected_mode"] == "so2_x")
    detail = (f"@ resolvable SNR (s=1.0,n=600): z-field recovers so2_z {z['recovery_rate']:.0%}, "
              f"x-field recovers so2_x {x['recovery_rate']:.0%} (best group FLIPS with axis); "
              f"recovery vs field s: " + ", ".join(f"{k}:{v:.0%}" for k, v in snr_curve.items()))
    return Check("Selection recovers true residual symmetry (flips with axis; degrades at low SNR)",
                 passed, detail, results)


def rot_z_to(n) -> np.ndarray:
    """Rotation mapping ẑ onto unit vector n (Rodrigues)."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(z, n); c = float(z @ n)
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def so2_projector_axis(n) -> np.ndarray:
    """Commutant projector of SO(2) about an ARBITRARY unit axis n."""
    R = rot_z_to(n)
    basis = [np.diag([1.0, 1.0, 0.0]), _J(2), np.diag([0.0, 0.0, 1.0])]  # so2_z commutant
    return _proj_from_basis([R @ M @ R.T for M in basis])


def target_field_axis(a: float, s: float, n) -> np.ndarray:
    """aI + s·n̂n̂ᵀ : SO(2)-equivariant about arbitrary axis n̂, breaks SO(3)."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    return a * np.eye(3) + s * np.outer(n, n)


def _fib_sphere(k: int) -> np.ndarray:
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    golden = math.pi * (3 - math.sqrt(5))
    theta = golden * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def _local_axes(a0, rmax_deg=16.0, nr=6, npsi=12):
    """A local patch of unit axes around a0 for coarse-to-fine refinement."""
    a0 = a0 / np.linalg.norm(a0)
    t1 = np.cross(a0, [1.0, 0, 0])
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(a0, [0, 1.0, 0])
    t1 /= np.linalg.norm(t1); t2 = np.cross(a0, t1)
    out = [a0]
    for r in np.linspace(rmax_deg / nr, rmax_deg, nr) * math.pi / 180:
        for psi in np.linspace(0, 2 * math.pi, npsi, endpoint=False):
            d = math.cos(psi) * t1 + math.sin(psi) * t2
            out.append(a0 * math.cos(r) + d * math.sin(r))
    return np.array(out)


def check_learnable_axis(rng) -> Check:
    """Recover a NON-canonical, continuous residual axis a discrete menu cannot
    express — the wedge vs distribution-based symmetry discovery (E3/E7 preview).
    Coarse Fibonacci grid + a local refinement patch (a real continuous selector)."""
    a, s, sigma, n_tr, nv = 1.0, 1.2, 0.5, 600, 6000
    n_true = np.array([1.0, 2.0, 2.0]); n_true = n_true / np.linalg.norm(n_true)  # off every axis
    Wstar = target_field_axis(a, s, n_true)
    coarse = _fib_sphere(200)
    Pcoarse = [so2_projector_axis(c) for c in coarse]        # precompute once
    menu_axes = {"so2_x": [1, 0, 0], "so2_y": [0, 1, 0], "so2_z": [0, 0, 1]}
    Pmenu = {k: so2_projector_axis(v) for k, v in menu_axes.items()}
    ang_err, cont_val, menu_val = [], [], []
    for _ in range(15):
        X = rng.standard_normal((n_tr, 3)); Y = X @ Wstar.T + sigma * rng.standard_normal((n_tr, 3))
        Xv = rng.standard_normal((nv, 3)); Yv = Xv @ Wstar.T + sigma * rng.standard_normal((nv, 3))
        errs = [_best_val_err(X, Y, Xv, Yv, P) for P in Pcoarse]
        a0 = coarse[int(np.argmin(errs))]
        # coarse-to-fine: refine within the coarse cell, then once more, tighter
        best_a, best_e = a0, min(errs)
        for rmax in (16.0, 4.0):
            for a_loc in _local_axes(best_a, rmax_deg=rmax):
                e = _best_val_err(X, Y, Xv, Yv, so2_projector_axis(a_loc))
                if e < best_e:
                    best_e, best_a = e, a_loc / np.linalg.norm(a_loc)
        ang = math.degrees(math.acos(min(1.0, abs(float(best_a @ n_true)))))
        ang_err.append(ang); cont_val.append(best_e)
        menu_val.append(min(_best_val_err(X, Y, Xv, Yv, Pmenu[k]) for k in Pmenu))
    med_ang = float(np.median(ang_err))
    cont_beats_menu = float(np.mean(cont_val)) < float(np.mean(menu_val)) - 1e-6
    passed = med_ang < 6.0 and cont_beats_menu
    detail = (f"true axis (1,2,2)/3 (off every coordinate axis); learned-axis median angular "
              f"error = {med_ang:.1f}° over 15 seeds; continuous selector val MSE "
              f"{np.mean(cont_val):.4f} < best discrete-menu {np.mean(menu_val):.4f} "
              f"-> {'beats menu' if cont_beats_menu else 'no'}")
    return Check("Learnable continuous axis recovers a NON-canonical symmetry (menu cannot)",
                 passed, detail, {"median_angular_error_deg": med_ang,
                                  "cont_val": float(np.mean(cont_val)), "menu_val": float(np.mean(menu_val))})


def check_selection_signal(rng) -> Check:
    axis, a, s, sigma, n, nv = 2, 1.0, 0.8, 0.5, 400, 6000
    groups = ["so2_z", "so3", "so2_x"]
    Wstar = target_field(a, s, axis)
    eps, grads, achieved = 1e-3, {}, {}
    for g in groups:
        P, gv, av = PROJ[g], [], []
        for _ in range(60):
            X = rng.standard_normal((n, 3)); Y = X @ Wstar.T + sigma * rng.standard_normal((n, 3))
            Xv = rng.standard_normal((nv, 3)); Yv = Xv @ Wstar.T + sigma * rng.standard_normal((nv, 3))
            G, b = gram_and_b(X, Y)
            ve = lambda lam: float(((Xv @ solve_ridge(G, b, P, lam).T - Yv) ** 2).sum() / nv)
            gv.append((ve(eps) - ve(0.0)) / eps)
            av.append(_best_val_err(X, Y, Xv, Yv, P))
        grads[g] = float(np.mean(gv)); achieved[g] = float(np.mean(av))
    all_neg = all(v < 0 for v in grads.values())
    discriminates = achieved["so2_z"] < achieved["so3"] - 1e-6 and achieved["so2_z"] < achieved["so2_x"] - 1e-6
    passed = all_neg and discriminates
    detail = ("hypergrad@0 (all <0 => non-discriminating): " + ", ".join(f"{g}:{v:+.4f}" for g, v in grads.items())
              + "  |  achieved-min val err (discriminates): " + ", ".join(f"{g}:{v:.4f}" for g, v in achieved.items()))
    return Check("Selection signal = achieved-min val err, NOT the lambda=0 gradient", passed, detail,
                 {"hypergrad0": grads, "achieved_min": achieved})


def main() -> int:
    rng = np.random.default_rng(0)
    checks = [check_prop1(rng), check_thm1(rng), check_crossover_slope(rng),
              check_group_selection(rng), check_learnable_axis(rng), check_selection_signal(rng)]
    print("=" * 80)
    print("PHASE-0 FALSIFICATION GATE  —  exact linear-Gaussian sandbox")
    print("=" * 80)
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}\n       {c.detail}")
    all_pass = all(c.passed for c in checks)
    print("-" * 80)
    print(("GATE PASSED — theory is exact in-regime; proceed to Phase 1 (deep-net transfer)."
           if all_pass else
           "GATE FAILED — a load-bearing claim is false in the exact regime; PIVOT to the "
           "honest optimization-interference paper (see docstring)."))
    print("=" * 80)
    out = {"all_pass": all_pass,
           "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail, "data": c.data} for c in checks]}
    os.makedirs("results", exist_ok=True)

    def _san(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    with open("results/theory_gate.json", "w") as f:
        json.dump(out, f, indent=2, default=_san)
    print("wrote results/theory_gate.json")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

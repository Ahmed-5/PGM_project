"""Fast guards for the Phase-0 theory sandbox (relaxed/theory/sandbox.py).

Locks in the exact-regime identities so a future edit can't silently break the
bias-variance theory the research program rests on. Follows the repo convention:
`python test_theory_sandbox.py` exits non-zero on failure.
"""
import math

import numpy as np

from relaxed.theory import sandbox as sb


def test_commutant_dims():
    assert sb.DSYM == {"so2_x": 3, "so2_y": 3, "so2_z": 3, "so3": 1}, sb.DSYM


def test_prop1_exact_identity():
    """Delta_G(W) == 2||(I-P_G)W||^2 to machine precision (Prop 1)."""
    rng = np.random.default_rng(3)
    for elems in [sb.so2_elems(2, 360), sb.so3_elems(300, rng)]:
        Ps = sb.twirl_sym(elems)
        for _ in range(50):
            W = rng.standard_normal((3, 3))
            mc = sb.delta_G_montecarlo(W, elems)
            w = W.reshape(-1)
            an = 2.0 * float(w @ (np.eye(9) - Ps) @ w)
            assert abs(mc - an) <= 1e-9 * (abs(mc) + 1e-12), (mc, an)


def test_gram_ordering():
    assert sb._assert_ordering()


def test_thm1_optimum_and_dominance():
    """lambda* = V/R recovered; soft strictly dominates hard and unconstrained."""
    rng = np.random.default_rng(4)
    axis, a, s, sigma, n = 2, 1.0, 0.8, 0.5, 400
    P, dsym = sb.PROJ["so3"], sb.DSYM["so3"]
    dasym = 9 - dsym
    Wstar = sb.target_field(a, s, axis)
    R = sb.R_of(Wstar, "so3")
    V = sigma ** 2 * dasym / n
    lam_star = V / R
    lams = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 40)])
    er = np.zeros(len(lams))
    trials = 300
    for _ in range(trials):
        X = rng.standard_normal((n, 3))
        Y = X @ Wstar.T + sigma * rng.standard_normal((n, 3))
        G, b = sb.gram_and_b(X, Y)
        for j, lam in enumerate(lams):
            er[j] += sb.excess_risk(sb.solve_ridge(G, b, P, lam), Wstar)
    er /= trials
    lam_emp = float(lams[int(np.argmin(er))])
    # empirical argmin within a factor of ~4 of V/R
    assert abs(math.log(lam_emp + 1e-9) - math.log(lam_star + 1e-9)) < math.log(4), (lam_emp, lam_star)
    er_hard = sb.er_closed_form(R, dasym, dsym, sigma, n, math.inf)
    assert er.min() < min(er[0], er_hard) - 1e-6, (er.min(), er[0], er_hard)


def test_r_of_so3_matches_analytic():
    """R for W* = aI + s E_zz under SO(3) equals 2 s^2 / 3 exactly."""
    for s in [0.3, 0.7, 1.5]:
        W = sb.target_field(1.0, s, 2)
        assert abs(sb.R_of(W, "so3") - 2 * s ** 2 / 3) < 1e-9


def test_arbitrary_axis_projector():
    """so2_projector_axis matches the canonical projector on z, has rank 3, and the
    target respects its own axis (R=0) but not a wrong axis (R>0)."""
    Pz = sb.so2_projector_axis([0, 0, 1])
    assert np.allclose(Pz, sb.PROJ["so2_z"], atol=1e-9)
    n = np.array([1.0, 2.0, 2.0]); n /= np.linalg.norm(n)
    Pn = sb.so2_projector_axis(n)
    assert abs(np.trace(Pn) - 3) < 1e-6
    W = sb.target_field_axis(1.0, 0.9, n)
    w = W.reshape(-1)
    R_true = float((w - Pn @ w) @ (w - Pn @ w))           # own axis -> ~0
    R_wrong = float((w - sb.PROJ["so2_x"] @ w) @ (w - sb.PROJ["so2_x"] @ w))
    assert R_true < 1e-9 and R_wrong > 1e-3, (R_true, R_wrong)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)

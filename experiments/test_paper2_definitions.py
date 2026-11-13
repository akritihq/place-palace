"""Numeric tests of the pleb code against the definitions and theorems of the PALACE paper.

Written 2026-11-08 after the same exercise on COMPLEX found
three definition/code mismatches that months of accuracy sweeps could not see (accuracy is
invariant to all of them).  A FAIL below is a place where the implemented object is not the
one the paper defines; the message gives the measured size of the gap.

    python experiments/test_paper2_definitions.py          (~2-4 min; local caches only)

T1  embed == 2^{-3/2} w phi and rho_nu == 2^{-3/2} tau w_min/4  (code internally consistent;
    the paper's eq:sum_embed / eq:nu_rho_formula carry no 2^{-3/2}: documentation gap)
T2  d_B1_batch == eq:db_single (brute force)
T3  Lemma lip_coord: |phi(x)-phi(y)| <= d_B(x,y)
T4  eq:config_def radius cap r_k <= d_B(p_k, Delta) on the paper's run configurations
T5  eq:upper_distortion ||Phi(A)-Phi(B)|| <= N_max d_B(A,B): holds under the cap, and a
    constructive violation without it
T6  FPS runs in d_B (eq:fps_rule): code's sum-via-diagonal vs the max of eq:db_single
T7  lebesgue_number == eq:lebesgue (brute force); def:sep_radius admissibility of run configs;
    is_admissible vs def:sep_radius
T8  Thm nu_rho on real cross-class pairs (exact d_B via gudhi, exact Def nu_coherence_pII)
    and the audit script's approximations against it
T9  certificate radii in grid_common / recompute_cert_firing_quarter == the paper's formulas
"""
from __future__ import annotations

import math, sys, pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

import gudhi
from embedding.nonuniform import (NonUniformEmbedding, hard_coords, farthest_point_sampling,
                                  init_nonuniform_from_data)
from utils.bottleneck import d_B1_batch

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def db_pts(p, q):
    """eq:db_single, scalar."""
    direct = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
    via = max((p[1] - p[0]) / 2, (q[1] - q[0]) / 2)
    return min(direct, via)


def auto_L(diags):
    return 1.1 * max(float(d[:, 1].max()) for d in diags if len(d))


def load_mutag():
    from exp_noninterference_audit import load_combined_diagrams, filter_topN, N_MAX
    from utils.datasets import load_tu_dataset
    D = filter_topN(load_combined_diagrams("MUTAG", ["degree", "hks_t10"]), N_MAX)
    _, y = load_tu_dataset("MUTAG"); y = np.asarray(y)[:len(D)]
    return D, y


def load_orbit():
    for name in ("orbit5k_alpha_H1_top50.pkl", "orbit5k_diagrams_dim1_top50.pkl"):   # local / cluster cache names
        f = ROOT / "data/cache" / name
        if f.exists():
            D = pickle.load(open(f, "rb")); break
    else:
        raise SystemExit("no Orbit5k top-50 cache in data/cache")
    y = np.repeat(np.arange(5), len(D) // 5)
    return [np.asarray(d, float) for d in D], y


def fit(D, y, K, alpha, seed, enforce_cap=True):
    byc = [[D[i] for i in range(len(D)) if y[i] == c] for c in np.unique(y)]
    return init_nonuniform_from_data(byc, K=K, L=auto_L(D), seed=seed, alpha=alpha, enforce_cap=enforce_cap)


def test_T1_T2_T3(emb, D, rng):
    worst = 0.0
    for i in rng.choice(len(D), min(200, len(D)), replace=False):
        ref = 2 ** -1.5 * emb.weights * hard_coords(emb.positions, emb.radii, emb._clip(D[i]))
        worst = max(worst, float(np.abs(emb.embed(D[i]) - ref).max()))
    tau = 0.1
    r = emb.rho_nu(tau); mask = emb.radii >= tau / 4
    ref = 2 ** -1.5 * tau * emb.weights[mask].min() / 4
    check("T1 embed == 2^{-3/2} w phi ; rho_nu == 2^{-3/2} tau w_min/4 (code consistent; paper has no 2^{-3/2})",
          worst < 1e-12 and abs(r - ref) < 1e-15, f"max diff {worst:.1e}; rho_nu {r:.3e} vs {ref:.3e}")
    P = rng.uniform(0, 1, (60, 2)); P[:, 1] = P[:, 0] + rng.uniform(0, 1, 60)
    X = rng.uniform(0, 1, (80, 2)); X[:, 1] = X[:, 0] + rng.exponential(0.2, 80)
    B = d_B1_batch(P, X)
    ref = np.array([[db_pts(p, x) for x in X] for p in P])
    check("T2 d_B1_batch == eq:db_single", np.abs(B - ref).max() < 1e-12, f"max diff {np.abs(B - ref).max():.1e}")
    worst = 0.0
    for _ in range(3000):
        p = P[rng.integers(60)]; r_ = rng.uniform(0.01, 2.0)
        x = X[rng.integers(80)]; yv = X[rng.integers(80)]
        if rng.random() < 0.3:                       # near-diagonal y
            yv = np.array([x[0], x[0] + 1e-6])
        d = abs(max(r_ - db_pts(p, x), 0) - max(r_ - db_pts(p, yv), 0))
        worst = max(worst, d - db_pts(x, yv))
    check("T3 Lemma lip_coord |phi(x)-phi(y)| <= d_B(x,y)", worst <= 1e-12, f"max excess {worst:.1e}")


def cap_violation(emb):
    pers = (emb.positions[:, 1] - emb.positions[:, 0]) / 2
    return float((emb.radii > pers + 1e-12).mean()), pers


def test_T4_T5(configs, rng, legacy):
    for name, emb, D in configs:
        frac, pers = cap_violation(emb)
        check(f"T4 eq:config_def cap r_k <= d_B(p_k,Delta) [{name}]", frac == 0.0,
              f"{100*frac:.1f}% of {emb.K} landmarks violate; max r/pers {float((emb.radii/np.maximum(pers,1e-12)).max()):.1f}x"
              f"  (legacy enforce_cap=False: {100*cap_violation(legacy[name])[0]:.1f}% violate)")
    # T5a: with the cap enforced the Lipschitz bound holds (code scale: 2^{-3/2} N_max d_B)
    name, emb, D = configs[0]
    capped = NonUniformEmbedding(emb.positions, np.minimum(emb.radii, np.maximum((emb.positions[:, 1] - emb.positions[:, 0]) / 2, 0)),
                                 emb.weights, emb.L)
    N = max(len(d) for d in D)
    worst = 0.0
    for _ in range(300):
        i, j = rng.integers(len(D), size=2)
        A, B = capped._clip(D[i]), capped._clip(D[j])
        if len(A) == 0 or len(B) == 0:
            continue
        d = gudhi.bottleneck_distance(A, B)
        if d <= 0:
            continue
        worst = max(worst, np.linalg.norm(capped.embed(A) - capped.embed(B)) / (2 ** -1.5 * N * d))
    check(f"T5a eq:upper_distortion under the cap [{name}]", worst <= 1 + 1e-9, f"max ||dPhi||/(N_max d_B) = {worst:.3f}")
    # T5b: the LEGACY uncapped configuration: one point eps from the diagonal under a violating
    # landmark breaks the bound; kept as the record of why enforce_cap defaults to True
    for name, emb, D in [(nm, legacy[nm], D_) for nm, _, D_ in configs]:
        pers = (emb.positions[:, 1] - emb.positions[:, 0]) / 2
        viol = np.where(emb.radii > pers + 1e-12)[0]
        if len(viol) == 0:
            print(f"       (T5b skipped for {name}: no violating landmark)"); continue
        k = viol[np.argmax(emb.radii[viol] - pers[viol])]
        A = emb._clip(D[int(rng.integers(len(D)))])
        eps = 1e-4
        b = emb.positions[k, 0]
        A2 = np.vstack([A, [[b, b + 2 * eps]]])          # d_B(A, A2) = eps (match the new point to Delta)
        d = gudhi.bottleneck_distance(A, A2)
        ratio = np.linalg.norm(emb.embed(A) - emb.embed(A2)) / (2 ** -1.5 * N * max(d, 1e-300))
        check(f"T5b(info) eq:upper_distortion FAILS without the cap, as expected [{name}]", ratio > 1,
              f"adding one point {eps:g} from the diagonal: ||dPhi|| / (N_max d_B) = {ratio:.1f}x  (landmark r={emb.radii[k]:.3f}, pers={pers[k]:.3f})")


def fps_dB(pts, K, seed):
    """eq:fps_rule in the metric of eq:db_single, same seed/start as the code."""
    rng = np.random.default_rng(seed)
    M = len(pts)
    if M <= K:
        return pts.copy()
    sel = [rng.integers(M)]; mind = np.full(M, np.inf)
    for _ in range(K - 1):
        mind = np.minimum(mind, d_B1_batch(pts[sel[-1]][None], pts)[0])
        sel.append(int(np.argmax(mind)))
    return pts[sel]


def test_T6(D, y, K_per_class=100, seed=0):
    pts = np.vstack([d for d, c in zip(D, y) if c == y[0] and len(d)])
    a = farthest_point_sampling(pts, K_per_class, seed=seed)
    b = fps_dB(pts, K_per_class, seed)
    same = sum(1 for p in a if (np.abs(b - p).max(1) < 1e-12).any())
    cov = lambda P: float(d_B1_batch(P, pts).min(0).max())
    check("T6 farthest_point_sampling uses the metric of eq:db_single (eq:fps_rule)", same == len(a),
          f"{len(a)-same}/{len(a)} selections differ from d_B-FPS; d_B covering radius code {cov(a):.4f} vs d_B-FPS {cov(b):.4f}")


def test_T7(configs, rng):
    name, emb, D = configs[0]
    sub = NonUniformEmbedding(emb.positions[:12], emb.radii[:12], emb.weights[:12] / np.linalg.norm(emb.weights[:12]), emb.L)
    pts_list = [emb._clip(D[i]) for i in rng.choice(len(D), 30, replace=False)]
    lam = sub.lebesgue_number(pts_list)
    allp = np.vstack([p for p in pts_list if len(p)])
    ref = min(max(max(r - db_pts(p, x), 0) for p, r in zip(sub.positions, sub.radii)) for x in allp)
    ref = ref if ref > 0 else 0.0
    check("T7a lebesgue_number == eq:lebesgue (brute force)", abs(lam - ref) < 1e-12, f"{lam:.5f} vs {ref:.5f}")
    mism = []
    for name, emb, D in configs:
        pts_list = [emb._clip(d) for d in D]
        lam = emb.lebesgue_number(pts_list)
        pers_all = np.concatenate([(p[:, 1] - p[:, 0]) / 2 for p in pts_list if len(p)])
        for tname, tau in (("tau=median half-persistence (code clip, protocol table)", float(np.median(pers_all))),
                           ("tau=median radius (Algorithm box, audit)", float(np.median(emb.radii)))):
            i_ok = lam >= tau / 4
            ii = float((emb.radii <= (tau + lam) / 2).mean())
            code_adm = emb.is_admissible(tau)
            print(f"       {name} | {tname}: tau={tau:.4f} lambda_0={lam:.4f}  (i) lambda_0>=tau/4: {i_ok}  "
                  f"(ii) r_k<=(tau+lambda_0)/2 holds for {100*ii:.0f}% of landmarks  | code is_admissible={code_adm}")
            mism.append(code_adm != (i_ok and ii == 1.0))
    check("T7b is_admissible implements def:sep_radius", not any(mism),
          f"disagrees with (i)+(ii) in {sum(mism)}/{len(mism)} (config, tau) cases: it checks min r>=gamma, max r<=4tau, rho_nu>gamma instead")


def test_T8(emb, D, y, tau, n_pairs=600, seed=0):
    from audit_nu_coherence_pii import db_pair, check_nu_coherence
    rng = np.random.default_rng(seed)
    cls = np.unique(y)
    stats = dict(q_exact=0, q_audit=0, q_audit_not_exact=0, coh_exact=0, coh_audit=0, floor_viol=0, min_ratio=np.inf)
    rho = emb.rho_nu(tau)
    tried = 0
    while stats["q_exact"] < n_pairs and tried < 20 * n_pairs:
        tried += 1
        ca, cb = rng.choice(len(cls), 2, replace=False)
        i = rng.choice(np.where(y == cls[ca])[0]); j = rng.choice(np.where(y == cls[cb])[0])
        A, B = emb._clip(D[i]), emb._clip(D[j])
        if len(A) == 0 or len(B) == 0:
            continue
        d_exact = gudhi.bottleneck_distance(A, B); d_audit = db_pair(A, B)
        if d_audit >= tau:
            stats["q_audit"] += 1
            if d_exact < tau: stats["q_audit_not_exact"] += 1
            stats["coh_audit"] += check_nu_coherence(A, B, emb, tau)
        if d_exact < tau:
            continue
        stats["q_exact"] += 1
        act = emb.radii >= tau / 4
        dphi = np.abs(hard_coords(emb.positions, emb.radii, A) - hard_coords(emb.positions, emb.radii, B))
        coh = act.any() and dphi[act].max() >= tau / 4            # Def nu_coherence_pII, weight cancelled
        if coh:
            stats["coh_exact"] += 1
            ratio = np.linalg.norm(emb.embed(A) - emb.embed(B)) / rho
            stats["min_ratio"] = min(stats["min_ratio"], ratio)
            stats["floor_viol"] += ratio < 1 - 1e-9
    s = stats
    check("T8a Thm nu_rho: ||dPhi|| >= rho_nu on every exactly-coherent pair", s["floor_viol"] == 0 and s["coh_exact"] > 0,
          f"{s['coh_exact']}/{s['q_exact']} coherent (exact Def 2.4, gudhi d_B); violations {s['floor_viol']}; min ratio {s['min_ratio']:.2f}")
    check("T8b audit script's db_pair qualifies only pairs with true d_B >= tau", s["q_audit_not_exact"] == 0,
          f"{s['q_audit_not_exact']}/{s['q_audit']} audit-qualified pairs have gudhi d_B < tau")
    print(f"       coherence rate: exact {100*s['coh_exact']/max(s['q_exact'],1):.1f}%  vs audit-script method "
          f"(approx d_B, paper-scale floor vs code-scale embed) {100*s['coh_audit']/max(s['q_audit'],1):.1f}%")


def test_T9():
    from grid_common import compute_r_m, compute_r_m_gauss, NC_ALPHA
    import recompute_cert_firing_quarter as q
    from scipy.stats import chi2
    y = np.array([0] * 30 + [1] * 50); k, m, R, sig, ell = 2, 30, 3.0, 0.7, 120
    L = math.log(2 * k / NC_ALPHA)
    check("T9a compute_r_m == 2 R sqrt(2 log(2k/delta)/m_min)   (paper radius (i), L3217; R_max in place of R-bar)",
          abs(compute_r_m(R, y, k) - 2 * R * math.sqrt(2 * L / m)) < 1e-12)
    check("T9a' quarter script r_pinelis == paper radius (i)",
          abs(q.r_pinelis(R, m, k) - 2 * R * math.sqrt(2 * L / m)) < 1e-12)
    check("T9b compute_r_m_gauss == sqrt(sigma_op chi2_{ell,1-delta/k}/m)   (radius (ii), plug-in sigma_op)",
          abs(compute_r_m_gauss(sig, m, k, ell) - math.sqrt(sig * chi2.ppf(1 - NC_ALPHA / k, ell) / m)) < 1e-12)
    check("T9c quarter script r_vP == sqrt(2 sigma L/m) + 4 R L/(3m)   (paper (iii) line 3236 with tr -> sigma_op proxy)",
          abs(q.r_var_pinelis(sig, R, m, k) - (math.sqrt(2 * sig * L / m) + 4 * R * L / (3 * m))) < 1e-12)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Dm, ym = load_mutag(); Do, yo = load_orbit()
    cfgs = [("MUTAG deg+hks10 K=200 a=1.75 (audit cfg)", fit(Dm, ym, 200, 1.75, 42), Dm),
            ("MUTAG deg+hks10 K=200 a=0.75 (Mah table cfg)", fit(Dm, ym, 200, 0.75, 42), Dm),
            ("Orbit5k alpha_H1 K=200 a=4.0 (headline cfg)", fit(Do, yo, 200, 4.0, 42), Do)]
    legacy = {"MUTAG deg+hks10 K=200 a=1.75 (audit cfg)": fit(Dm, ym, 200, 1.75, 42, enforce_cap=False),
              "MUTAG deg+hks10 K=200 a=0.75 (Mah table cfg)": fit(Dm, ym, 200, 0.75, 42, enforce_cap=False),
              "Orbit5k alpha_H1 K=200 a=4.0 (headline cfg)": fit(Do, yo, 200, 4.0, 42, enforce_cap=False)}
    print(f"landmarks built: {[c[1].K for c in cfgs]}  (K=200 requested; code builds (K//n_classes)*n_classes)")
    test_T1_T2_T3(cfgs[0][1], Dm, rng)
    test_T4_T5(cfgs, rng, legacy)
    test_T6(Dm, ym)
    test_T7(cfgs, rng)
    emb = cfgs[0][1]
    test_T8(emb, Dm, ym, tau=float(emb.tau))
    test_T9()
    print(f"\n{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILED: ' + '; '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)

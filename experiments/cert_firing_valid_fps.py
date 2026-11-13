"""Certificate firing on the PALACE (FPS, K=200) embedding, in a form the theorems cover.

Paper II's Table certificate_firing was computed on PLACE's uniform-grid embedding with plug-in
quantities (sample covariance for Sigma_c, training max norm for the a.s. bound R-bar), and its
"3 of 11" does not survive the paper's own plug-in correction (notes/paper-II-code-review-
2026-11-08.md, item 3).  This script computes, on the FPS embedding the paper advertises:

  VALID (every quantity controlled, configuration independent of the certified sample)
    - stratified half F of each training fold fits landmarks/radii/tau (init_nonuniform_from_data,
      cap enforced) and a pilot mean mu~_c; the other half A gives the centroids mu^_c;
    - a.s. coordinate bound c_k = w_k 2^{-3/2} N_max r_k and the tighter landmark-geometry bounds
      of COMPLEX's pointwise_valid.Geometry (norm bound B, range of <u, Phi>);
    - V-bar_c: empirical-Bernstein upper bound on E||Phi - mu~_c||^2 (Maurer-Pontil);
    - r_c: Pinelis radius solving 2 exp(-m r^2 / (2 (V-bar + 2 B r/3))) = delta_r / C;
    - rule-level firing  max_c r_c < Delta^/4   (Paper I Prop 3.2's honest empirical condition);
    - per-point agreement certificate (Thm certified (a) with the F/A decomposition of
      pointwise_valid: G_{c'}(x) > 2(h_c^ + h_c' + t_c^ + t_c') + r_c'^2).
  PLUG-IN, same split, for comparison (the deployed forms of radii (i)-(iii)):
    - r_pin = 2 R_max sqrt(2L/m), r_vP(op) = sqrt(2 sigma_op L/m) + 4 R_max L/(3m),
      r_vP(tr) with tr(Sigma^) in place of sigma_op, r_gauss = sqrt(sigma_op chi2_{ell,1-d/k}/m).
  Confidence: delta = 0.05 split as delta_V = delta_r = 0.0125 (shared), delta_h = 0.025 (per point).

    python experiments/cert_firing_valid_fps.py --dataset MUTAG            (seeds 0-4, 10 folds)
    python experiments/cert_firing_valid_fps.py --dataset Orbit5k --alpha 4.0
Outputs results/paper_II/cert_firing_valid_fps_<ds>.csv (one row per seed x fold) and a summary.
"""
from __future__ import annotations

import argparse, csv, math, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from embedding.nonuniform import init_nonuniform_from_data

DELTA = 0.05
D_V, D_R, D_H = 0.0125, 0.0125, 0.025
N_MAX = 50                                   # grid_common.TOP_K / filter_topN
# the filtration the paper's firing table used per data set (results/paper_II/cert_firing_vp_selected.csv)
FILT = {"MUTAG": "betw+hks1", "COX2": "betw+hks1", "DHFR": "closeness", "PTC": "betw+hks1",
        "NCI1": "betw+hks1", "PROTEINS": "ricci+hks10", "DD": "clustering", "IMDB-B": "jaccard+hks10",
        "IMDB-M": "deg+hks10", "NCI109": "hks2+hks25"}
CHEM = {"MUTAG", "COX2", "DHFR", "PTC", "NCI1", "NCI109"}


# ----------------------------------------------------------------------------- data
def load(ds):
    if ds == "Orbit5k":
        for name in ("orbit5k_diagrams_dim1_top50.pkl", "orbit5k_alpha_H1_top50.pkl"):
            f = ROOT / "data/cache" / name
            if f.exists():
                D = pickle.load(open(f, "rb")); break
        else:
            raise SystemExit("no Orbit5k cache")
        return [np.asarray(d, float) for d in D], np.repeat(np.arange(5), len(D) // 5)
    from utils.datasets import load_dataset
    from grid_common import build_graph_diagrams
    import pool_chemical, pool_social
    pool = (pool_chemical if ds in CHEM else pool_social).POOL
    cfg = [c for c in pool if c["name"] == FILT[ds]]
    assert cfg, FILT[ds]
    graphs, y = load_dataset(ds)
    D = build_graph_diagrams(dataset_name=ds, graphs=graphs, pool=cfg, extended=True, union_h0=True)[FILT[ds]]
    return [np.asarray(d, float) for d in D], np.asarray(y)


# ----------------------------------------------------------------------------- bounds
def eb_dev(s2, n, b, L):
    """Maurer-Pontil empirical Bernstein: sqrt(2 s^2 L/n) + 7 b L/(3(n-1)), s^2 unbiased, L = log(2/d)."""
    return np.sqrt(2.0 * s2 * L / n) + 7.0 * b * L / (3.0 * (n - 1))


class Geometry:
    """a.s. bounds from the landmark geometry (single PALACE block); see COMPLEX pointwise_valid."""

    def __init__(self, emb):
        pos, r = emb.positions, emb.radii
        s = np.abs(emb.weights) * emb.norm
        dinf = np.abs(pos[:, None, :] - pos[None, :, :]).max(2)
        self.adj = (dinf <= r[:, None] + r[None, :]).astype(float)
        exc = np.maximum(r - (pos[:, 1] - pos[:, 0]) / 2.0, 0.0)          # 0 under the cap
        self.sr, self.se, self.N = s * r, s * exc, N_MAX
        self.c = s * N_MAX * r
        self.B = N_MAX * (math.sqrt(float((self.adj @ (self.sr ** 2)).max())) + float(np.linalg.norm(self.se)))

    def range_width(self, U):
        w = np.zeros(len(U))
        for V in (np.maximum(U, 0.0), np.maximum(-U, 0.0)):
            w += self.N * ((V * self.sr) @ self.adj).max(1) + self.N * (V * self.se).sum(1)
        return w


# ----------------------------------------------------------------------------- one fold
def fold_record(XF, yF, XA, yA, Xte, yte, geo, ell):
    classes = np.unique(yA); C = len(classes); T = len(Xte)
    box_viol = float(max(XA.max(0).__truediv__(np.maximum(geo.c, 1e-300)).max(),
                         (Xte / np.maximum(geo.c, 1e-300)).max()))
    norm_viol = float(max(np.linalg.norm(XA, axis=1).max(), np.linalg.norm(Xte, axis=1).max()) / geo.B)
    Dbox = min(float(np.linalg.norm(geo.c)), 2.0 * geo.B)
    L_V, L_r = math.log(2 * C / D_V), math.log(2 * C / D_R)
    mu_hat, mu_til, r, tco, ns = [], [], [], [], []
    for cl in classes:
        Ac, Fc = XA[yA == cl], XF[yF == cl]
        n = len(Ac); mh, mt = Ac.mean(0), Fc.mean(0)
        W = ((Ac - mt) ** 2).sum(1)
        bW = min(float((np.maximum(mt, geo.c - mt) ** 2).sum()), (geo.B + float(np.linalg.norm(mt))) ** 2)
        Vbar = min(float(W.mean() + eb_dev(W.var(ddof=1), n, bW, L_V)), float((geo.c ** 2).sum() / 4.0))
        q = 2.0 * L_r * Dbox / 3.0
        rc = (q + math.sqrt(q * q + 8.0 * n * L_r * Vbar)) / (2.0 * n)
        mu_hat.append(mh); mu_til.append(mt); r.append(rc); ns.append(n)
        tco.append((float(np.linalg.norm(mt - mh)) + rc) * rc)
    mu_hat = np.stack(mu_hat); r = np.array(r); tco = np.array(tco); ns = np.array(ns)
    D2 = ((Xte[:, None, :] - mu_hat[None]) ** 2).sum(2); ci = D2.argmin(1); pred = classes[ci]
    idx = np.arange(T); G = D2 - D2[idx, ci][:, None]
    # per-point valid certificate
    L_h = math.log(4 * C / D_H)
    h = np.zeros((T, C))
    for k, cl in enumerate(classes):
        U = Xte - mu_til[k]; Z = XA[yA == cl] @ U.T
        bu = np.minimum(np.abs(U) @ geo.c, geo.range_width(U))
        h[:, k] = eb_dev(Z.var(0, ddof=1), ns[k], bu, L_h)
    need = 2 * (h[idx, ci][:, None] + h + tco[ci][:, None] + tco[None, :]) + (r ** 2)[None, :]
    ok = G > need; ok[idx, ci] = True; cert_pt = ok.all(1)
    # rule-level quantities on the A centroids
    dm = np.linalg.norm(mu_hat[:, None] - mu_hat[None], axis=2); dhat = float(dm[np.triu_indices(C, 1)].min())
    m = int(ns.min()); L = math.log(2 * C / DELTA)
    Rmax = float(np.linalg.norm(XA, axis=1).max())
    sig_op = tr = 0.0
    for cl in classes:
        Y = XA[yA == cl] - XA[yA == cl].mean(0); n = len(Y)
        Gm = Y @ Y.T / max(n - 1, 1)
        sig_op = max(sig_op, float(np.linalg.eigvalsh(Gm).max())); tr = max(tr, float(np.trace(Gm)))
    rad = {"valid": float(r.max()),
           "pin_plugin": 2 * Rmax * math.sqrt(2 * L / m),
           "pin_asR": 2 * geo.B * math.sqrt(2 * L / m),
           "vP_op_plugin": math.sqrt(2 * sig_op * L / m) + 4 * Rmax * L / (3 * m),
           "vP_tr_plugin": math.sqrt(2 * tr * L / m) + 4 * Rmax * L / (3 * m),
           "vP_tr_asR": math.sqrt(2 * tr * L / m) + 4 * geo.B * L / (3 * m),
           "gauss_plugin": math.sqrt(sig_op * chi2.ppf(1 - DELTA / C, ell) / m)}
    rec = dict(T=T, C=C, m_min=m, n_F=int(len(XF)), ell=ell, delta_hat=dhat, R_max=Rmax, R_as=geo.B,
               sigma_op=sig_op, trace=tr, box_viol=box_viol, norm_viol=norm_viol,
               nc_acc_A=float((pred == yte).mean()),
               valid_pt_frac=float(cert_pt.mean()),
               valid_pt_acc=float((pred[cert_pt] == yte[cert_pt]).mean()) if cert_pt.any() else float("nan"))
    for k, v in rad.items():
        rec[f"r_{k}"] = v; rec[f"fire_{k}"] = int(v < dhat / 4)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["Orbit5k"] + sorted(FILT))
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=None, help="radius factor; default 4.0 Orbit5k, 0.75 graphs")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--fit_frac", type=float, default=0.5)
    a = ap.parse_args()
    alpha = a.alpha if a.alpha is not None else (4.0 if a.dataset == "Orbit5k" else 0.75)
    t0 = time.time()
    D, y = load(a.dataset)
    L = 1.1 * max(float(d[:, 1].max()) for d in D if len(d))
    print(f"{a.dataset}: {len(D)} diagrams, filt {FILT.get(a.dataset, 'alpha H0+H1')}, K={a.K} alpha={alpha} "
          f"fit_frac={a.fit_frac} ({time.time()-t0:.0f}s)", flush=True)
    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        skf = StratifiedKFold(10, shuffle=True, random_state=42 + seed)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            if a.folds != "all" and f not in {int(x) for x in a.folds.split(",")}:
                continue
            Fi, Ai = train_test_split(tr, train_size=a.fit_frac, stratify=y[tr], random_state=42 + seed)
            byc = [[D[i] for i in Fi if y[i] == c] for c in np.unique(y)]
            emb = init_nonuniform_from_data(byc, K=a.K, L=L, seed=42 + seed, alpha=alpha, enforce_cap=True)
            geo = Geometry(emb)
            E = lambda idx: np.stack([emb.embed(D[i]) for i in idx])
            rec = fold_record(E(Fi), y[Fi], E(Ai), y[Ai], E(te), y[te], geo, emb.K)
            rec.update(dataset=a.dataset, seed=seed, fold=f, K=emb.K, alpha=alpha, tau=emb.tau)
            rows.append(rec)
            if f == 0:
                print(f"  seed {seed} fold 0: m_min={rec['m_min']} Delta^={rec['delta_hat']:.4f} r_valid={rec['r_valid']:.4f} "
                      f"r_vP(op,plugin)={rec['r_vP_op_plugin']:.4f} box/norm viol {rec['box_viol']:.2f}/{rec['norm_viol']:.2f} "
                      f"NC {100*rec['nc_acc_A']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    assert max(r["box_viol"] for r in rows) <= 1 + 1e-9 and max(r["norm_viol"] for r in rows) <= 1 + 1e-9, "a.s. bound violated"
    out = ROOT / "results/paper_II" / f"cert_firing_valid_fps_{a.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    keys = [k for k in rows[0] if k.startswith("fire_")]
    print(f"\n{a.dataset}: {len(rows)} seed-folds; firing = r < Delta^/4 on the held-out-half centroids")
    for k in keys:
        print(f"  {k[5:]:<14} fires {100*np.mean([r[k] for r in rows]):5.1f}% of folds   median r/(Delta^/4) = "
              f"{np.median([r['r_'+k[5:]] / (r['delta_hat']/4) for r in rows]):.2f}")
    print(f"  per-point VALID certificate: {100*np.mean([r['valid_pt_frac'] for r in rows]):.1f}% of test points, "
          f"NC acc on them {100*np.nanmean([r['valid_pt_acc'] for r in rows]):.1f}; NC acc {100*np.mean([r['nc_acc_A'] for r in rows]):.1f}")
    print(f"wrote {out}\nCERT-FIRING-VALID DONE")


if __name__ == "__main__":
    main()

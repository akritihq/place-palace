#!/usr/bin/env python3
"""
Experiment: PALACE certified two-sample test vs landscape two-sample test
on synthetic diagrams with controlled bottleneck-distance effect size.

Hypothesis (Paper IV motivating result):
    PALACE's adaptive FPS landmarks concentrate in regions of high
    diagram-data density.  Landscape MMD aggregates over the full
    scale axis and dilutes a localized class signal.  At small
    effect sizes δ in d_B, PALACE certified MMD has higher power
    than landscape MMD.

Setup:
    Each diagram has ~20 i.i.d.\ noise points uniform in [0,1]² with
    d > b (truncated to the half-plane), plus one signal point.

    Class A signal: (b₀, d₀)              = (0.30, 0.50)
    Class B signal: (b₀, d₀ + δ)          (death shifted by δ)

    Gaussian jitter σ_jitter = 0.02 on the signal point per draw.

    For each δ ∈ {0, 0.025, 0.05, 0.10, 0.20, 0.40}:
        Generate n diagrams from each class.
        Run PALACE certified MMD test (level α=0.05).
        Run landscape MMD test (level α=0.05).
        Both calibrated via permutation null with 200 shuffles.
        Repeat over R=50 simulation seeds; record rejection rate.

Methods:
    PALACE: K=50 landmarks via class-aware FPS on training, additive
            WLK-Gaussian gram with σ=0.02, MMD² = α^T G α.
    Landscape: gudhi.representations.Landscape with k=2, resolution=100,
               sample_range=(0, 1.5).  Mean landscape per class, L²
               distance as the test statistic.
    Both: permutation null with 200 shuffles for calibration.

Output: results/paper_IV/landscape_vs_palace_power.csv
"""
import sys; sys.path.insert(0, '.')
import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path
from scipy.spatial.distance import cdist

from gudhi.representations import Landscape
from persim import PersistenceImager
from embedding.nonuniform import init_nonuniform_from_data


OUT_DIR = Path('results/paper_IV')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────── synthetic diagram generation ─────────────────

def generate_diagram(rng, signal_pos, n_noise=20, sigma_jitter=0.02,
                     L=1.0):
    """One diagram = n_noise random above-diagonal points + 1 signal."""
    # Noise: uniform in unit square with d > b
    pts = rng.uniform(0, L, (n_noise * 3, 2))
    pts = pts[pts[:, 1] > pts[:, 0]][:n_noise]
    while len(pts) < n_noise:
        more = rng.uniform(0, L, (n_noise, 2))
        more = more[more[:, 1] > more[:, 0]]
        pts = np.vstack([pts, more])[:n_noise]
    # Signal
    sig = np.asarray(signal_pos, dtype=float) \
          + rng.normal(0, sigma_jitter, 2)
    if sig[1] <= sig[0]:
        sig[1] = sig[0] + 0.01
    return np.vstack([pts, sig[None, :]])


def generate_dataset(rng, n_per_class, signal_A, signal_B,
                     n_noise=20, sigma_jitter=0.02, L=1.0):
    """Returns (diagrams, labels) with labels in {0, 1}."""
    dgms = []
    labels = []
    for _ in range(n_per_class):
        dgms.append(generate_diagram(rng, signal_A, n_noise, sigma_jitter, L))
        labels.append(0)
    for _ in range(n_per_class):
        dgms.append(generate_diagram(rng, signal_B, n_noise, sigma_jitter, L))
        labels.append(1)
    return dgms, np.array(labels)


# ───────────────────── PALACE MMD test ──────────────────────────────

def palace_mmd_stat(dgms, labels, K=50, sigma=0.02, L=1.0, seed=0):
    """Embed diagrams via PALACE, compute MMD² between class means."""
    n_classes = 2
    dbc = [[dgms[i] for i in range(len(dgms)) if labels[i] == c]
           for c in range(n_classes)]
    if any(len(dc) == 0 for dc in dbc):
        return float('nan')
    try:
        emb = init_nonuniform_from_data(dbc, K=K, L=L, n_diagram=1,
                                        seed=seed, alpha=0.75)
    except ValueError:
        return float('nan')
    X = emb.embed_dataset(dgms)
    n, K_dim = X.shape
    if K_dim == 0:
        return float('nan')
    # Additive Gaussian gram
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    G = np.zeros((n, n))
    for k in range(K_dim):
        d = X[:, k:k+1] - X[:, k:k+1].T
        G += np.exp(-d * d * inv2s2)
    idx_A = np.where(labels == 0)[0]
    idx_B = np.where(labels == 1)[0]
    nA, nB = len(idx_A), len(idx_B)
    mean_AA = G[np.ix_(idx_A, idx_A)].mean()
    mean_BB = G[np.ix_(idx_B, idx_B)].mean()
    mean_AB = G[np.ix_(idx_A, idx_B)].mean()
    mmd2 = mean_AA - 2 * mean_AB + mean_BB
    return float(mmd2)


# ───────────────────── Landscape MMD test ───────────────────────────

def landscape_mmd_stat(dgms, labels, num_landscapes=2, resolution=100,
                       sample_range=(0.0, 1.5)):
    """Mean landscape per class, MMD² = ||λ_A - λ_B||² in L²."""
    ls = Landscape(num_landscapes=num_landscapes,
                   resolution=resolution,
                   sample_range=list(sample_range))
    dgms_clean = []
    for d in dgms:
        d = np.asarray(d, dtype=float)
        d = d[np.all(np.isfinite(d), axis=1)]
        if len(d) == 0:
            d = np.zeros((1, 2))
        dgms_clean.append(d)
    feats = ls.fit_transform(dgms_clean)
    feats = np.asarray(feats)
    idx_A = np.where(labels == 0)[0]
    idx_B = np.where(labels == 1)[0]
    mean_A = feats[idx_A].mean(axis=0)
    mean_B = feats[idx_B].mean(axis=0)
    return float(np.sum((mean_A - mean_B) ** 2))


def pi_mmd_stat(dgms, labels, pixel_size=0.05, sigma=0.05,
                birth_range=(0.0, 1.0), pers_range=(0.0, 1.0)):
    """Mean persistence-image per class, MMD² = ||PI_A - PI_B||² in L².

    Uses persim.PersistenceImager with the standard linear ramp weight
    function on the (birth, persistence) plane.
    """
    pimgr = PersistenceImager(
        pixel_size=pixel_size,
        birth_range=birth_range,
        pers_range=pers_range,
        kernel_params={'sigma': [[sigma, 0.0], [0.0, sigma]]},
    )
    # PI expects (birth, persistence) pairs, not (birth, death).
    dgms_bp = []
    for d in dgms:
        d = np.asarray(d, dtype=float)
        d = d[np.all(np.isfinite(d), axis=1)]
        if len(d) == 0:
            dgms_bp.append(np.zeros((1, 2)))
        else:
            bp = np.column_stack([d[:, 0], d[:, 1] - d[:, 0]])
            dgms_bp.append(bp)
    pimgr.fit(dgms_bp, skew=False)
    feats = np.array([pimgr.transform(d, skew=False).flatten()
                      for d in dgms_bp])
    idx_A = np.where(labels == 0)[0]
    idx_B = np.where(labels == 1)[0]
    mean_A = feats[idx_A].mean(axis=0)
    mean_B = feats[idx_B].mean(axis=0)
    return float(np.sum((mean_A - mean_B) ** 2))


# ───────────────────── Permutation calibration ──────────────────────

def permutation_pvalue(stat_fn, dgms, labels, n_perm=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    obs = stat_fn(dgms, labels)
    if not np.isfinite(obs):
        return float('nan')
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(labels)
        null[i] = stat_fn(dgms, perm)
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return float('nan')
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


# ───────────────────── Main ─────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-per-class', type=int, default=30)
    p.add_argument('--R', type=int, default=50,
                   help='Simulation replications per (δ, method) cell')
    p.add_argument('--n-perm', type=int, default=200)
    p.add_argument('--deltas', type=float, nargs='+',
                   default=[0.0, 0.025, 0.05, 0.10, 0.20, 0.40])
    p.add_argument('--alpha', type=float, default=0.05)
    args = p.parse_args()

    signal_A = (0.30, 0.50)
    rows = []
    t0 = time.time()
    n_jobs = len(args.deltas) * args.R
    done = 0
    for delta in args.deltas:
        signal_B = (0.30, 0.50 + delta)
        for rep in range(args.R):
            rng = np.random.default_rng(1000 * rep + int(1e6 * delta))
            dgms, labels = generate_dataset(
                rng, args.n_per_class, signal_A, signal_B,
            )
            # PALACE
            p_palace = permutation_pvalue(
                lambda d, l: palace_mmd_stat(d, l, K=50, sigma=0.02,
                                              seed=rep),
                dgms, labels, n_perm=args.n_perm,
                rng=np.random.default_rng(7 * rep + 17),
            )
            # Landscape
            p_ls = permutation_pvalue(
                landscape_mmd_stat, dgms, labels,
                n_perm=args.n_perm,
                rng=np.random.default_rng(11 * rep + 23),
            )
            # Persistence image
            p_pi = permutation_pvalue(
                pi_mmd_stat, dgms, labels,
                n_perm=args.n_perm,
                rng=np.random.default_rng(13 * rep + 29),
            )
            rows.append({
                'delta': delta, 'rep': rep,
                'n_per_class': args.n_per_class,
                'p_palace': p_palace, 'p_landscape': p_ls,
                'p_pi': p_pi,
                'reject_palace': int(p_palace < args.alpha)
                                 if np.isfinite(p_palace) else 0,
                'reject_landscape': int(p_ls < args.alpha)
                                    if np.isfinite(p_ls) else 0,
                'reject_pi': int(p_pi < args.alpha)
                             if np.isfinite(p_pi) else 0,
            })
            done += 1
            if done % max(1, n_jobs // 20) == 0:
                el = time.time() - t0
                rate = done / el
                eta = (n_jobs - done) / rate if rate > 0 else float('inf')
                print(f'  {done}/{n_jobs}  '
                      f'({rate:.2f} reps/s, ETA {eta/60:.1f} min)',
                      flush=True)

    df = pd.DataFrame(rows)
    out = OUT_DIR / f'landscape_vs_palace_power_n{args.n_per_class}.csv'
    df.to_csv(out, index=False)
    print(f'\nWrote {out} ({len(df)} rows, {time.time()-t0:.0f}s)',
          flush=True)

    # Summary
    print('\n-- Power summary (rejection rate at α=0.05) --')
    print(f'{"δ":>7}  {"PALACE":>8}  {"Landscape":>10}  '
          f'{"PI":>8}  {"P-L (pp)":>10}  {"P-PI (pp)":>10}')
    for delta in args.deltas:
        sub = df[df['delta'] == delta]
        p_p  = sub['reject_palace'].mean() * 100
        p_l  = sub['reject_landscape'].mean() * 100
        p_pi = sub['reject_pi'].mean() * 100
        print(f'{delta:>7.3f}  {p_p:>7.1f}%  {p_l:>9.1f}%  '
              f'{p_pi:>7.1f}%  {p_p - p_l:>+8.1f}  {p_p - p_pi:>+8.1f}')


if __name__ == '__main__':
    main()

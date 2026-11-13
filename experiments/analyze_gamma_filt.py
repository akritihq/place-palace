"""
Multi-dataset search for the right filtration ranker in Paper II.

For each (seed, fold, filt) in the dataset's filtration pool we compute
a battery of candidate ranking statistics (all sharing the same WLK
gram, so marginal cost is negligible) and the fold's inner-CV accuracy:
    (i)   γ̂                 : kernel margin (½ min unbiased MMD_u)
    (ii)  γ̂/√K              : dimension-normalized (current §4.4 stat)
    (iii) sigma_pool         : pooled within-class kernel variance
    (iv)  fisher_kernel      : MMD² / (σ²_c + σ²_c')  -- kernel Fisher
    (v)   wlk_acc            : inner 5-fold CV accuracy on the same gram

All quantities share the non-uniform (FPS) embedding with K=200
landmarks, equal weights w_k=K^{-1/2}, and fixed bandwidth σ=10^{-3}.

Parallelism: forks worker processes that inherit pre-computed diagrams,
so per-job cost is embed + gram + SVM only.

Usage:
    python experiments/analyze_gamma_filt.py --dataset MUTAG \
        --seeds 0 1 2 3 4 --workers 8
"""
from __future__ import annotations
import sys; sys.path.insert(0, '.')

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from utils.datasets import load_dataset
from experiments.grid_common import build_graph_diagrams
from experiments import pool_chemical, pool_social
from embedding.nonuniform import init_nonuniform_from_data, WeightedLandmarkKernel


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'results' / 'filtration_grid' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LANDMARKS_DEFAULT = 200
SIGMA_DEFAULT       = 1e-3
N_INNER_CV          = 5
C_GRID              = [0.01, 0.1, 1.0, 10.0, 100.0]

CSV_FIELDS = ['dataset', 'seed', 'fold', 'filt', 'n_train', 'K',
              'gamma_hat', 'gamma_over_sqrt_K',
              'sigma_pool', 'fisher_kernel', 'fisher_kernel_pooled',
              'rho_mahalanobis',
              'tau_hat', 'rho_nu_hat',
              'wlk_acc',
              't_embed_s', 't_gram_s', 't_stats_s', 't_tau_s', 't_svm_s']

# Cross-class bottleneck subsampling for τ̂.
TAU_N_PAIRS = 50
TAU_QUANTILE = 0.10


# ----- Shared worker state (set by initializer) --------------------------
_WSTATE = {}

def _worker_init(dgms_per_filt, labels, L, n_classes, dataset, K_landmarks, sigma):
    _WSTATE['dgms_per_filt'] = dgms_per_filt
    _WSTATE['labels']        = labels
    _WSTATE['L']             = L
    _WSTATE['n_classes']     = n_classes
    _WSTATE['dataset']       = dataset
    _WSTATE['K_landmarks']   = K_landmarks
    _WSTATE['sigma']         = sigma


# ----- Core computations -------------------------------------------------

def empirical_gamma(G, labels, n_classes):
    """γ̂ = ½ min_{c≠c'} ||μ̂_c − μ̂_{c'}||_H via unbiased MMD² on gram G.

    For additive WLK with G[i,i]=K, the biased within-class mean inflates
    class self-products; the unbiased MMD² excludes diagonals.
    """
    diag_G = np.diag(G)
    classes = [c for c in range(n_classes) if (labels == c).any()]
    min_mmd = np.inf
    for i, c in enumerate(classes):
        idx_c = np.where(labels == c)[0]
        n_c = len(idx_c)
        if n_c < 2:
            continue
        Kcc_sum = G[np.ix_(idx_c, idx_c)].sum() - diag_G[idx_c].sum()
        Kcc_u   = Kcc_sum / (n_c * (n_c - 1))
        for c2 in classes[i+1:]:
            idx_c2 = np.where(labels == c2)[0]
            n_c2 = len(idx_c2)
            if n_c2 < 2:
                continue
            Kc2c2_sum = G[np.ix_(idx_c2, idx_c2)].sum() - diag_G[idx_c2].sum()
            Kc2c2_u   = Kc2c2_sum / (n_c2 * (n_c2 - 1))
            Kcc2      = G[np.ix_(idx_c, idx_c2)].mean()
            mmd2      = Kcc_u - 2 * Kcc2 + Kc2c2_u
            mmd       = np.sqrt(max(mmd2, 0.0))
            if mmd < min_mmd:
                min_mmd = mmd
    return float(0.5 * min_mmd) if np.isfinite(min_mmd) else float('nan')


def class_kernel_variance(G, idx_c):
    """σ²_c = (1/n_c) Σ_{i∈c} k(A_i,A_i) − (1/n_c²) Σ_{i,j∈c} k(A_i,A_j).

    Equal to the within-class kernel variance (trace of class-c feature
    covariance operator)."""
    n_c = len(idx_c)
    if n_c < 2:
        return float('nan')
    diag_mean = np.diag(G)[idx_c].mean()
    block_mean = G[np.ix_(idx_c, idx_c)].mean()
    return float(max(diag_mean - block_mean, 0.0))


def pooled_kernel_variance(G, labels, n_classes):
    """Weighted average of per-class σ²_c."""
    total_n = 0
    weighted = 0.0
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        n_c = len(idx_c)
        if n_c < 2:
            continue
        s_c = class_kernel_variance(G, idx_c)
        if np.isnan(s_c):
            continue
        weighted += n_c * s_c
        total_n += n_c
    return float(weighted / total_n) if total_n > 0 else float('nan')


def kernel_mahalanobis(G, labels, n_classes, shrinkage=0.1):
    """Minimum pairwise kernel-Mahalanobis margin between class means.

    ρ_Mah(c,c') = √[ (m_c - m_{c'})^T (N + λI)^{-1} (m_c - m_{c'}) ]

    where:
      m_c[i]  = (1/n_c) Σ_{j∈c} G[i,j]           kernelized class-c mean
      N       = Σ_c K_c (I - (1/n_c) 1 1^T) K_c^T  pooled within-class
                                                    scatter (Mika 1999,
                                                    'Fisher Discriminant
                                                    Analysis with Kernels')
      λ       = shrinkage · tr(N)/n                 Ledoit-Wolf-style
                                                    regularization

    This is the LDA Bayes-margin form of the kernel Fisher discriminant
    ratio in the RKHS H_{k_LC}, computed entirely from the gram matrix.
    Returns the minimum across class pairs."""
    n = G.shape[0]
    classes = [c for c in range(n_classes) if (labels == c).sum() >= 2]
    if len(classes) < 2:
        return float('nan')

    # Pooled within-class scatter N (n × n PSD)
    N = np.zeros((n, n))
    m = {}
    for c in classes:
        idx_c = np.where(labels == c)[0]
        n_c = len(idx_c)
        K_c = G[:, idx_c]                           # n × n_c
        sum_Kc = K_c.sum(axis=1, keepdims=True)     # n × 1
        N += K_c @ K_c.T - (sum_Kc @ sum_Kc.T) / n_c
        m[c] = (sum_Kc.flatten()) / n_c

    # Ledoit-Wolf-style shrinkage on the pooled scatter
    trN = float(np.trace(N))
    if trN <= 0 or not np.isfinite(trN):
        return float('nan')
    lambda_reg = shrinkage * trN / n
    N_reg = N + lambda_reg * np.eye(n)

    from scipy.linalg import cho_factor, cho_solve
    try:
        c_and_low = cho_factor(N_reg, lower=True)
    except np.linalg.LinAlgError:
        return float('nan')

    best = float('inf')
    for i, c in enumerate(classes):
        for c2 in classes[i+1:]:
            delta = m[c] - m[c2]
            sol = cho_solve(c_and_low, delta)
            rho2 = float(delta @ sol)
            if rho2 < best:
                best = rho2

    if not np.isfinite(best) or best < 0:
        return float('nan')
    return float(np.sqrt(best))


def kernel_fisher_ratio(G, labels, n_classes):
    """Minimum pairwise Fisher ratio in two flavors:

      perpair: MMD²(c,c') / (σ²_c + σ²_{c'})      Welch-style, per-pair denom
      pooled : MMD²(c,c') / (2 σ²_pool)           single pooled denom across pairs

    Returns (best_perpair, best_pooled). Pooled denom uses the existing
    sample-weighted pooled_kernel_variance to match kernel-FDA's S_W trace.
    Spearman-vs-accuracy is monotonic-invariant, so the factor of 2 is
    cosmetic; we keep it for scale comparability with per-pair."""
    classes = [c for c in range(n_classes) if (labels == c).sum() >= 2]
    if len(classes) < 2:
        return float('nan'), float('nan')
    sigma_pool = pooled_kernel_variance(G, labels, n_classes)
    denom_pool = 2.0 * sigma_pool if (sigma_pool > 0 and np.isfinite(sigma_pool)) else 0.0
    best_pp = float('inf')
    best_pl = float('inf')
    for i, c in enumerate(classes):
        idx_c = np.where(labels == c)[0]
        sigma_c = class_kernel_variance(G, idx_c)
        mean_cc = G[np.ix_(idx_c, idx_c)].mean()
        for c2 in classes[i+1:]:
            idx_c2 = np.where(labels == c2)[0]
            sigma_c2 = class_kernel_variance(G, idx_c2)
            mean_c2c2 = G[np.ix_(idx_c2, idx_c2)].mean()
            mean_cc2  = G[np.ix_(idx_c, idx_c2)].mean()
            mmd2 = max(mean_cc - 2 * mean_cc2 + mean_c2c2, 0.0)
            denom_pp = sigma_c + sigma_c2
            if denom_pp > 0:
                r = mmd2 / denom_pp
                if r < best_pp:
                    best_pp = r
            if denom_pool > 0:
                r = mmd2 / denom_pool
                if r < best_pl:
                    best_pl = r
    bp = float(best_pp) if np.isfinite(best_pp) else float('nan')
    bl = float(best_pl) if np.isfinite(best_pl) else float('nan')
    return bp, bl


def estimate_tau(tr_dgms, tr_labels, n_classes, n_pairs, quantile, rng):
    """Estimate τ̂ = q-th quantile of cross-class bottleneck distances
    on a subsample of n_pairs cross-class pairs.

    Returns NaN if no cross-class pair can be sampled."""
    import gudhi
    classes = [c for c in range(n_classes) if (tr_labels == c).any()]
    if len(classes) < 2:
        return float('nan')
    # Sample cross-class pairs (a, b) with a from class c, b from class c'≠c
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        c1, c2 = rng.choice(classes, size=2, replace=False)
        idx_c1 = np.where(tr_labels == c1)[0]
        idx_c2 = np.where(tr_labels == c2)[0]
        if len(idx_c1) == 0 or len(idx_c2) == 0:
            attempts += 1
            continue
        i = rng.choice(idx_c1)
        j = rng.choice(idx_c2)
        pairs.append((i, j))
        attempts += 1
    if not pairs:
        return float('nan')
    dists = []
    for (i, j) in pairs:
        a = tr_dgms[i]
        b = tr_dgms[j]
        # gudhi expects list of lists / arrays
        a_list = a.tolist() if hasattr(a, 'tolist') and len(a) > 0 else []
        b_list = b.tolist() if hasattr(b, 'tolist') and len(b) > 0 else []
        try:
            d = gudhi.bottleneck_distance(a_list, b_list)
            dists.append(d)
        except Exception:
            continue
    if not dists:
        return float('nan')
    return float(np.quantile(dists, quantile))


def inner_cv_accuracy(G, labels, rng_seed):
    """5-fold CV accuracy on precomputed gram; best of the C grid."""
    skf = StratifiedKFold(n_splits=N_INNER_CV, shuffle=True, random_state=rng_seed)
    accs = []
    for tr, te in skf.split(np.zeros(len(labels)), labels):
        G_tr = G[np.ix_(tr, tr)]
        G_te = G[np.ix_(te, tr)]
        best = 0.0
        for C in C_GRID:
            svm = SVC(kernel='precomputed', C=C)
            svm.fit(G_tr, labels[tr])
            acc = svm.score(G_te, labels[te])
            if acc > best:
                best = acc
        accs.append(best)
    return float(np.mean(accs))


def process_one(job):
    """Process a single (seed, fold, filt) job."""
    seed, fold, filt = job
    dgms_per_filt = _WSTATE['dgms_per_filt']
    labels        = _WSTATE['labels']
    L             = _WSTATE['L']
    n_classes     = _WSTATE['n_classes']
    dataset       = _WSTATE['dataset']

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + seed)
    splits = list(skf.split(np.zeros_like(labels), labels))
    tr_idx, _ = splits[fold]

    K_landmarks = _WSTATE.get('K_landmarks', K_LANDMARKS_DEFAULT)
    sigma       = _WSTATE.get('sigma', SIGMA_DEFAULT)

    dgms = dgms_per_filt.get(filt, None)
    if dgms is None:
        return None

    tr_dgms   = [dgms[i] for i in tr_idx]
    tr_labels = labels[tr_idx]
    dbc       = [[tr_dgms[i] for i in range(len(tr_dgms))
                   if tr_labels[i] == c]
                  for c in range(n_classes)]
    # skip if any class is empty
    if any(len(dc) == 0 for dc in dbc):
        return {
            'dataset': dataset, 'seed': seed, 'fold': fold, 'filt': filt,
            'n_train': len(tr_idx), 'K': K_landmarks,
            'gamma_hat': float('nan'), 'gamma_over_sqrt_K': float('nan'),
            'wlk_acc': float('nan'),
            't_embed_s': 0.0, 't_gram_s': 0.0, 't_svm_s': 0.0,
        }

    try:
        t0 = time.time()
        emb = init_nonuniform_from_data(dbc, K=K_landmarks, L=L,
                                        n_diagram=1, seed=42 + seed)
        X_tr = emb.embed_dataset(tr_dgms)
        t_embed = time.time() - t0
        K_actual = X_tr.shape[1]

        t0 = time.time()
        wlk = WeightedLandmarkKernel(omega=np.ones(K_actual), sigma=sigma)
        G = wlk.gram_matrix(X_tr)
        t_gram = time.time() - t0

        t0 = time.time()
        gamma_hat = empirical_gamma(G, tr_labels, n_classes)
        sigma_pool = pooled_kernel_variance(G, tr_labels, n_classes)
        fisher, fisher_pool = kernel_fisher_ratio(G, tr_labels, n_classes)
        rho_mah = kernel_mahalanobis(G, tr_labels, n_classes, shrinkage=0.1)
        t_stats = time.time() - t0

        # τ̂ = q-th quantile of cross-class bottleneck distances on n_pairs
        # subsampled training pairs; ρ_ν = c_n · τ̂ / √K (equal-weight,
        # using c_n = 2^{-n-5/2} with n=1 for single-point landmarks).
        t0 = time.time()
        tau_rng = np.random.default_rng(42 + seed * 1000 + fold)
        tau_hat = estimate_tau(tr_dgms, tr_labels, n_classes,
                                n_pairs=TAU_N_PAIRS,
                                quantile=TAU_QUANTILE, rng=tau_rng)
        c_n = 2.0 ** (-1 - 5.0 / 2.0)
        rho_nu_hat = (c_n * tau_hat / np.sqrt(K_actual)
                       if (not np.isnan(tau_hat) and K_actual > 0)
                       else float('nan'))
        t_tau = time.time() - t0

        t0 = time.time()
        wlk_acc = inner_cv_accuracy(G, tr_labels, rng_seed=42 + seed)
        t_svm = time.time() - t0

        return {
            'dataset': dataset, 'seed': seed, 'fold': fold, 'filt': filt,
            'n_train': len(tr_idx), 'K': K_actual,
            'gamma_hat': gamma_hat,
            'gamma_over_sqrt_K': gamma_hat / np.sqrt(K_actual) if K_actual > 0 else float('nan'),
            'sigma_pool': sigma_pool,
            'fisher_kernel': fisher,
            'fisher_kernel_pooled': fisher_pool,
            'rho_mahalanobis': rho_mah,
            'tau_hat': tau_hat,
            'rho_nu_hat': rho_nu_hat,
            'wlk_acc': wlk_acc,
            't_embed_s': t_embed, 't_gram_s': t_gram,
            't_stats_s': t_stats, 't_tau_s': t_tau, 't_svm_s': t_svm,
        }
    except Exception as e:
        return {
            'dataset': dataset, 'seed': seed, 'fold': fold, 'filt': filt,
            'n_train': len(tr_idx), 'K': K_landmarks,
            'gamma_hat': float('nan'), 'gamma_over_sqrt_K': float('nan'),
            'sigma_pool': float('nan'), 'fisher_kernel': float('nan'),
            'fisher_kernel_pooled': float('nan'),
            'tau_hat': float('nan'), 'rho_nu_hat': float('nan'),
            'wlk_acc': float('nan'),
            't_embed_s': 0.0, 't_gram_s': 0.0, 't_stats_s': 0.0,
            't_tau_s': 0.0, 't_svm_s': 0.0,
            '_error': str(e),
        }


# ----- Dataset registry --------------------------------------------------

_N_CLASSES = {'MUTAG': 2, 'COX2': 2, 'DHFR': 2, 'PTC': 2,
              'NCI1': 2, 'NCI109': 2, 'PROTEINS': 2, 'DD': 2,
              'IMDB-B': 2, 'IMDB-M': 3}
_POOL_MODULE = {
    'MUTAG': pool_chemical, 'COX2': pool_chemical, 'DHFR': pool_chemical,
    'PTC': pool_chemical,   'NCI1': pool_chemical, 'NCI109': pool_chemical,
    'PROTEINS': pool_social, 'DD': pool_social,
    'IMDB-B': pool_social,   'IMDB-M': pool_social,
}


def _auto_L(all_diagrams):
    """L = 1.1 × max death value across training diagrams."""
    maxes = [d[:, 1].max() for d in all_diagrams if len(d) > 0]
    return (max(maxes) * 1.1) if maxes else 1.0


def _load_done(out_path):
    if not out_path.exists():
        return set()
    try:
        df = pd.read_csv(out_path)
    except Exception:
        return set()
    if not {'seed', 'fold', 'filt'} <= set(df.columns):
        return set()
    return {(int(r.seed), int(r.fold), str(r.filt)) for r in df.itertuples()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(_N_CLASSES))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--folds', type=int, nargs='+', default=None)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--K', type=int, default=K_LANDMARKS_DEFAULT,
                   help='Landmark budget (default 200).')
    p.add_argument('--sigma', type=float, default=SIGMA_DEFAULT,
                   help='WLK bandwidth (default 1e-3).')
    p.add_argument('--no-restrict', action='store_true',
                   help='Use full pool (not just filtrations in the cluster CSV).')
    p.add_argument('--out-dir', default=str(OUT_DIR))
    p.add_argument('--out-suffix', default='',
                   help='Suffix appended to output filename, e.g., "_K500".')
    args = p.parse_args()

    folds   = args.folds if args.folds is not None else list(range(10))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'gamma_filt_{args.dataset.lower()}{args.out_suffix}.csv'
    print(f'  config: K={args.K}, σ={args.sigma:.1e}, output={out_path.name}', flush=True)

    pool_module = _POOL_MODULE[args.dataset]
    n_classes   = _N_CLASSES[args.dataset]

    print(f'Loading {args.dataset}...', flush=True)
    graphs, labels = load_dataset(args.dataset)
    labels = np.array(labels)
    print(f'  n={len(labels)}, {n_classes} classes', flush=True)

    pool = pool_module.POOL
    if not args.no_restrict:
        cluster_csv = ROOT / 'results' / 'filtration_grid' / f'{args.dataset.lower()}_records.csv'
        if cluster_csv.exists():
            existing = pd.read_csv(cluster_csv)
            cluster_filts = set(existing['filt'].unique())
            pool = [cfg for cfg in pool if cfg['name'] in cluster_filts]
            print(f'  restricted to cluster pool: {len(pool)} filtrations', flush=True)
    else:
        print(f'  full pool: {len(pool)} filtrations', flush=True)

    print(f'Computing diagrams (cached where possible)...', flush=True)
    t0 = time.time()
    dgms_per_filt = build_graph_diagrams(
        dataset_name=args.dataset, graphs=graphs, pool=pool,
        extended=True, union_h0=True,
    )
    print(f'  diagrams done in {time.time()-t0:.1f}s', flush=True)

    # L = max death across all diagrams, all filtrations
    flat_all = [d for dgms in dgms_per_filt.values() for d in dgms]
    L = _auto_L(flat_all)
    print(f'  L = {L:.3f}', flush=True)

    done = _load_done(out_path)
    if done:
        print(f'Resuming: {len(done)} rows already in {out_path.name}', flush=True)

    jobs = []
    for seed in args.seeds:
        for fold in folds:
            for cfg in pool:
                if (seed, fold, cfg['name']) in done:
                    continue
                jobs.append((seed, fold, cfg['name']))

    print(f'\n=== {len(jobs)} jobs, {args.workers} workers ===', flush=True)
    if not jobs:
        print('Nothing to do.', flush=True)
    else:
        t_sweep = time.time()
        ctx = mp.get_context('fork')
        write_header = not out_path.exists()
        with ctx.Pool(args.workers, initializer=_worker_init,
                      initargs=(dgms_per_filt, labels, L, n_classes,
                                args.dataset, args.K, args.sigma)) as pool_mp:
            with open(out_path, 'a') as f:
                if write_header:
                    f.write(','.join(CSV_FIELDS) + '\n')
                rows_written = 0
                for result in pool_mp.imap_unordered(process_one, jobs, chunksize=1):
                    if result is None:
                        continue
                    f.write(','.join(str(result.get(k, '')) for k in CSV_FIELDS) + '\n')
                    f.flush()
                    rows_written += 1
                    if rows_written % max(1, len(jobs) // 20) == 0:
                        elapsed = time.time() - t_sweep
                        rate = rows_written / elapsed
                        eta = (len(jobs) - rows_written) / rate if rate > 0 else float('inf')
                        print(f'  {rows_written}/{len(jobs)} done '
                              f'({rate:.1f} rows/s, ETA {eta/60:.1f} min)', flush=True)
        print(f'\nTotal sweep wall-clock: {time.time()-t_sweep:.1f}s', flush=True)

    # ---- Summary ----
    if not out_path.exists():
        return
    df = pd.read_csv(out_path)
    df = df[df['dataset'] == args.dataset]
    agg_cols = {
        'gamma_hat':          ('gamma_hat', 'mean'),
        'gamma_over_sqrt_K':  ('gamma_over_sqrt_K', 'mean'),
        'fisher_kernel':      ('fisher_kernel', 'mean'),
        'sigma_pool':         ('sigma_pool', 'mean'),
        'tau_hat':            ('tau_hat', 'mean'),
        'rho_nu_hat':         ('rho_nu_hat', 'mean'),
        'acc':                ('wlk_acc', 'mean'),
        'n_obs':              ('wlk_acc', 'count'),
    }
    avg = df.groupby('filt').agg(**agg_cols).reset_index()
    avg = avg[avg['n_obs'] > 0].dropna(subset=['acc'])
    if len(avg) < 3:
        return

    print(f'\n=== {args.dataset} summary '
          f'({df["seed"].nunique()} seeds × {df["fold"].nunique()} folds × {len(avg)} filtrations) ===')
    print(f'{"Statistic":<22}  ρ        τ        p       ')
    for name, col in [
        ('γ̂ (raw)',          'gamma_hat'),
        ('γ̂/√K',             'gamma_over_sqrt_K'),
        ('Fisher (kernel)',   'fisher_kernel'),
        ('τ̂ (cross-class)',  'tau_hat'),
        ('ρ_ν̂ = c_n τ̂/√K',   'rho_nu_hat'),
    ]:
        sub = avg.dropna(subset=[col])
        if len(sub) < 3:
            continue
        rho, pval = spearmanr(sub[col], sub['acc'])
        tau, _    = kendalltau(sub[col], sub['acc'])
        print(f'{name:<22}  {rho:+.3f}   {tau:+.3f}   {pval:.3f}')

    best = avg.sort_values('acc', ascending=False).iloc[0]
    print(f"\nAccuracy winner: {best['filt']} @ {best['acc']*100:.1f}%")
    for name, col in [
        ('γ̂/√K',             'gamma_over_sqrt_K'),
        ('Fisher (kernel)',   'fisher_kernel'),
        ('τ̂',                'tau_hat'),
        ('ρ_ν̂',               'rho_nu_hat'),
    ]:
        sub = avg.dropna(subset=[col])
        if best['filt'] not in set(sub['filt']):
            continue
        order = sub.sort_values(col, ascending=False).reset_index(drop=True)
        rank = order[order['filt'] == best['filt']].index[0] + 1
        print(f"  rank by {name}: #{rank}/{len(sub)}")


if __name__ == '__main__':
    main()

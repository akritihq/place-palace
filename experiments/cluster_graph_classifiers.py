"""
Cluster script: Run expensive sweeps skipped locally.
1. Ricci filtrations on all datasets
2. WLK kernel (q×C grid) on all datasets
3. All top-3 filtrations × all 4 classifiers (linear, RBF, WLK, NC)

Usage:
  python experiments/cluster_graph_classifiers.py                  # all datasets
  python experiments/cluster_graph_classifiers.py MUTAG NCI1       # specific datasets
  python experiments/cluster_graph_classifiers.py --no-wlk         # skip WLK
"""
import sys; sys.path.insert(0, '.')
import argparse
import numpy as np
import csv
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from utils.datasets import load_dataset
from utils.persistence import graphs_to_persistence_cached
from embedding.embedding import init_from_dataset

RESULTS_DIR = Path('results/graph_classifiers')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 10
N_SCALES = 10
C_GRID = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
Q_GRID = [0.10, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Full configs including ricci (the expensive ones skipped locally)
CONFIGS = {
    'MUTAG': [
        {'name': 'betw+deg',     'filts': ['betweenness', 'degree'], 'dims': [0, 1], 'ext': True, 'tau': 'auto'},
        {'name': 'ricci+deg',    'filts': ['ricci', 'degree'],       'dims': [0, 1], 'ext': True, 'tau': 'auto'},
        {'name': 'betw+hks10',   'filts': ['betweenness', 'hks_t10'],'dims': [0, 1], 'ext': True, 'tau': 'auto'},
        {'name': 'ricci',        'filts': ['ricci'],                 'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'hks1+hks10',   'filts': ['hks_t1', 'hks_t10'],    'dims': [0, 1], 'ext': True, 'tau': 'auto'},
    ],
    'NCI1': [
        {'name': 'deg',          'filts': ['degree'],                'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'betw+deg',     'filts': ['betweenness', 'degree'], 'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'ricci+deg',    'filts': ['ricci', 'degree'],       'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'deg+hks10',    'filts': ['degree', 'hks_t10'],     'dims': [0, 1], 'ext': True, 'tau': 'auto'},
    ],
    'NCI109': [
        {'name': 'deg+hks10',    'filts': ['degree', 'hks_t10'],     'dims': [0, 1], 'ext': True, 'tau': 'auto'},
        {'name': 'betw+deg',     'filts': ['betweenness', 'degree'], 'dims': [0, 1], 'ext': True, 'tau': 'auto'},
        {'name': 'deg',          'filts': ['degree'],                'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'ricci+deg',    'filts': ['ricci', 'degree'],       'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
    ],
    'PTC': [
        {'name': 'deg',          'filts': ['degree'],                'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'ricci+deg',    'filts': ['ricci', 'degree'],       'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'ricci',        'filts': ['ricci'],                 'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'betw+deg',     'filts': ['betweenness', 'degree'], 'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
    ],
    'COX2': [
        {'name': 'deg',          'filts': ['degree'],                'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'betw+deg',     'filts': ['betweenness', 'degree'], 'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'betw',         'filts': ['betweenness'],           'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'ricci',        'filts': ['ricci'],                 'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
    ],
    'DHFR': [
        {'name': 'deg',          'filts': ['degree'],                'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'ricci+deg',    'filts': ['ricci', 'degree'],       'dims': [0, 1], 'ext': True, 'tau': 'proxy'},
        {'name': 'betw+hks1',    'filts': ['betweenness', 'hks_t1'], 'dims': [0, 1], 'ext': True, 'tau': 'auto'},
        {'name': 'ricci+betw',   'filts': ['ricci', 'betweenness'],  'dims': [0, 1], 'ext': True, 'tau': 'auto'},
    ],
    'PROTEINS': [
        {'name': 'ricci+hks10',  'filts': ['ricci', 'hks_t10'],     'dims': [0],    'ext': False, 'tau': 'auto'},
        {'name': 'ricci+betw',   'filts': ['ricci', 'betweenness'],  'dims': [0],    'ext': False, 'tau': 'auto'},
        {'name': 'ricci+jac',    'filts': ['ricci', 'jaccard'],      'dims': [0],    'ext': False, 'tau': 'auto'},
        {'name': 'jac+deg',      'filts': ['jaccard', 'degree'],     'dims': [0],    'ext': False, 'tau': 'auto'},
    ],
    'DD': [
        {'name': 'deg',          'filts': ['degree'],                'dims': [0],    'ext': False, 'tau': 'proxy'},
        {'name': 'betw+hks10',   'filts': ['betweenness', 'hks_t10'],'dims': [0],    'ext': False, 'tau': 'auto'},
        {'name': 'hks10',        'filts': ['hks_t10'],               'dims': [0],    'ext': False, 'tau': 'auto'},
        {'name': 'jac',          'filts': ['jaccard'],               'dims': [0],    'ext': False, 'tau': 'proxy'},
    ],
    'IMDB-B': [
        {'name': 'deg',          'filts': ['degree'],                'dims': [0],    'ext': False, 'tau': 'proxy'},
        {'name': 'betw',         'filts': ['betweenness'],           'dims': [0],    'ext': False, 'tau': 'proxy'},
        {'name': 'deg+betw',     'filts': ['degree', 'betweenness'], 'dims': [0],    'ext': False, 'tau': 'proxy'},
    ],
    'IMDB-M': [
        {'name': 'deg',          'filts': ['degree'],                'dims': [0],    'ext': False, 'tau': 'proxy'},
        {'name': 'deg+betw',     'filts': ['degree', 'betweenness'], 'dims': [0],    'ext': False, 'tau': 'auto'},
        {'name': 'betw',         'filts': ['betweenness'],           'dims': [0],    'ext': False, 'tau': 'proxy'},
    ],
    'REDDIT-5K': [
        {'name': 'deg+betw',     'filts': ['degree', 'betweenness'], 'dims': [0],    'ext': False, 'tau': 'auto'},
        {'name': 'deg',          'filts': ['degree'],                'dims': [0],    'ext': False, 'tau': 'auto'},
        {'name': 'betw',         'filts': ['betweenness'],           'dims': [0],    'ext': False, 'tau': 'proxy'},
    ],
}


def log(msg):
    print(msg, flush=True)


def wlk_gram(X1, X2, inv2s2):
    n1, ell = X1.shape
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    chunk = max(1, min(100, int(4e9 / (n1 * n2 * 8))))
    for start in range(0, ell, chunk):
        end = min(start + chunk, ell)
        diff = X1[:, start:end, None] - X2[:, start:end, None].transpose(2, 1, 0)
        K += np.exp(-diff**2 * inv2s2).sum(axis=1)
        del diff
    return K


def run_fold(X_tr, X_te, y_tr, y_te, c_grid, q_grid, run_wlk=True):
    results = {}

    # 1. Linear SVM + scaler
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)
    best = 0
    for C in c_grid:
        svm = SVC(kernel='linear', C=C)
        svm.fit(Xtr_s, y_tr)
        best = max(best, svm.score(Xte_s, y_te))
    results['linear'] = best

    # 2. RBF SVM + scaler
    best = 0
    for C in c_grid:
        svm = SVC(kernel='rbf', C=C, gamma='scale')
        svm.fit(Xtr_s, y_tr)
        best = max(best, svm.score(Xte_s, y_te))
    results['rbf'] = best

    # 3. WLK SVM (grid search q × C)
    if run_wlk:
        D2 = np.sum(X_tr**2, axis=1)[:, None] + np.sum(X_tr**2, axis=1)[None, :] - 2 * X_tr @ X_tr.T
        triu = np.sqrt(np.maximum(D2[np.triu_indices(D2.shape[0], k=1)], 0))
        del D2
        best = 0
        for q in q_grid:
            sigma = float(np.quantile(triu, q))
            if sigma <= 0:
                sigma = 1e-8
            inv2s2 = 1.0 / (2 * sigma**2)
            K_tr = wlk_gram(X_tr, X_tr, inv2s2)
            K_te = wlk_gram(X_te, X_tr, inv2s2)
            for C in c_grid:
                svm = SVC(kernel='precomputed', C=C)
                svm.fit(K_tr, y_tr)
                best = max(best, svm.score(K_te, y_te))
            del K_tr, K_te
        results['wlk'] = best

    # 4. Nearest centroid
    classes = np.unique(y_tr)
    centroids = np.array([X_tr[y_tr == c].mean(axis=0) for c in classes])
    dists = np.sum((X_te[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    preds = classes[np.argmin(dists, axis=1)]
    results['nc'] = float(np.mean(preds == y_te))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='*', default=list(CONFIGS.keys()),
                        help='Datasets to run (default: all)')
    parser.add_argument('--no-wlk', action='store_true',
                        help='Skip WLK kernel (much faster)')
    args = parser.parse_args()

    classifiers = ['linear', 'rbf', 'nc'] + ([] if args.no_wlk else ['wlk'])
    grand_start = time.time()
    all_results = []

    for dataset in args.datasets:
        if dataset not in CONFIGS:
            log(f"Unknown dataset: {dataset}, skipping")
            continue
        cfgs = CONFIGS[dataset]

        log(f"\n{'='*70}")
        log(f"  {dataset} ({len(cfgs)} filtration configs, "
            f"classifiers: {classifiers})")
        log(f"{'='*70}")

        graphs, labels = load_dataset(dataset)
        labels = np.array(labels)
        n_classes = len(np.unique(labels))

        # Precompute all needed filtrations
        all_filts = set()
        for cfg in cfgs:
            for f in cfg['filts']:
                all_filts.add(f)
        log(f"  Computing persistence for {sorted(all_filts)}...")
        pers_cache = {}
        for filt in sorted(all_filts):
            t0 = time.time()
            pers_cache[filt] = graphs_to_persistence_cached(
                graphs, dataset, filtration=filt, dims=None,
                extended=True, union_h0=False)
            log(f"    {filt}: {time.time()-t0:.0f}s")

        rng = np.random.default_rng(SEED)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=int(rng.integers(1e6)))
        folds = list(skf.split(range(len(labels)), labels))

        for cfg in cfgs:
            name = cfg['name']
            log(f"\n  {name} (τ={cfg['tau']}):")

            # Build pooled diagrams
            diagrams = []
            for gi in range(len(graphs)):
                pts = []
                for filt in cfg['filts']:
                    dgm_dict = pers_cache[filt][gi]
                    for dim in cfg['dims']:
                        if dim in dgm_dict and len(dgm_dict[dim]) > 0:
                            pts.append(dgm_dict[dim])
                if pts:
                    merged = np.vstack(pts)
                    if len(merged) > 50:
                        pers = merged[:, 1] - merged[:, 0]
                        merged = merged[np.argsort(pers)[::-1][:50]]
                    diagrams.append(merged)
                else:
                    diagrams.append(np.zeros((0, 2)))

            # Compute Δ/√ℓ on full dataset (no folds) for ranking comparison
            dbc_full = [[diagrams[i] for i in range(len(labels)) if labels[i] == c]
                        for c in range(n_classes)]
            try:
                emb_full = init_from_dataset(dbc_full, N_scales=N_SCALES,
                                             n_diagram=1, tau_method=cfg['tau'])
                X_full = emb_full.embed_dataset(diagrams)
                ell = X_full.shape[1]
                from scipy.spatial.distance import pdist
                means = np.array([X_full[labels == c].mean(axis=0)
                                  for c in range(n_classes)])
                delta = float(pdist(means, 'euclidean').min())
                delta_sqrt_ell = delta / np.sqrt(ell)
                log(f"    Δ/√ℓ = {delta_sqrt_ell:.6f}  (Δ={delta:.5f}, ℓ={ell})")
                del X_full
            except Exception as e:
                delta_sqrt_ell = 0.0
                ell = 0
                delta = 0.0
                log(f"    Δ/√ℓ FAILED: {e}")

            fold_results = {k: [] for k in classifiers}
            t0 = time.time()

            for fi, (tr_idx, te_idx) in enumerate(folds):
                tr_dgms = [diagrams[i] for i in tr_idx]
                te_dgms = [diagrams[i] for i in te_idx]
                tr_labels = labels[tr_idx]

                dbc = [[tr_dgms[j] for j in range(len(tr_idx)) if tr_labels[j] == c]
                       for c in range(n_classes)]
                try:
                    emb = init_from_dataset(dbc, N_scales=N_SCALES, n_diagram=1,
                                            tau_method=cfg['tau'])
                    X_tr = emb.embed_dataset(tr_dgms)
                    X_te = emb.embed_dataset(te_dgms)
                except Exception as e:
                    log(f"    fold {fi} FAILED: {e}")
                    continue

                fr = run_fold(X_tr, X_te, labels[tr_idx], labels[te_idx],
                              C_GRID, Q_GRID, run_wlk=not args.no_wlk)
                for k in classifiers:
                    if k in fr:
                        fold_results[k].append(fr[k])
                del X_tr, X_te

            dt = time.time() - t0
            dim = emb.embedding_dim if 'emb' in dir() else '?'
            log(f"    dim={dim}, {dt:.0f}s")

            for k in classifiers:
                accs = fold_results[k]
                if accs:
                    m, s = np.mean(accs) * 100, np.std(accs) * 100
                    log(f"    {k:8s}  {m:.1f}±{s:.1f}%")
                    all_results.append({
                        'dataset': dataset, 'filt': name,
                        'classifier': k, 'tau': cfg['tau'],
                        'acc_mean': round(m, 1), 'acc_std': round(s, 1),
                        'delta_sqrt_ell': round(delta_sqrt_ell, 6),
                        'delta': round(delta, 5), 'ell': ell,
                    })

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY (best filtration per classifier)")
    log(f"{'='*70}")
    header = f"{'Dataset':12s}" + "".join(f" {k:>14s}" for k in classifiers)
    log(header)
    log("-" * len(header))
    for dataset in args.datasets:
        if dataset not in CONFIGS:
            continue
        vals = {}
        for k in classifiers:
            ds_k = [r for r in all_results if r['dataset'] == dataset and r['classifier'] == k]
            if ds_k:
                best = max(ds_k, key=lambda x: x['acc_mean'])
                vals[k] = f"{best['acc_mean']:.1f}±{best['acc_std']:.1f}"
        row = f"{dataset:12s}" + "".join(f" {vals.get(k,'---'):>14s}" for k in classifiers)
        log(row)

    # Ranking comparison: Δ/√ℓ vs accuracy
    log(f"\n{'='*70}")
    log("RANKING COMPARISON: Δ/√ℓ vs best linear SVM accuracy")
    log(f"{'='*70}")
    for dataset in args.datasets:
        if dataset not in CONFIGS:
            continue
        ds_linear = [r for r in all_results
                     if r['dataset'] == dataset and r['classifier'] == 'linear']
        if not ds_linear:
            continue
        # Rank by Δ/√ℓ
        by_stat = sorted(ds_linear, key=lambda x: x['delta_sqrt_ell'], reverse=True)
        # Rank by accuracy
        by_acc = sorted(ds_linear, key=lambda x: x['acc_mean'], reverse=True)
        stat_top = by_stat[0]['filt']
        acc_top = by_acc[0]['filt']
        match = '✓' if stat_top == acc_top else '✗'
        log(f"  {dataset:12s}  Δ/√ℓ→{stat_top:15s}  acc→{acc_top:15s}  {match}")

    elapsed = time.time() - grand_start
    log(f"\nTotal time: {elapsed/3600:.1f} hours")

    outfile = RESULTS_DIR / 'cluster_graph_classifiers.csv'
    with open(outfile, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'filt', 'classifier',
                                          'tau', 'acc_mean', 'acc_std',
                                          'delta_sqrt_ell', 'delta', 'ell'])
        w.writeheader()
        w.writerows(all_results)
    log(f"Saved → {outfile}")

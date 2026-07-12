"""
Pre-Paper-V sanity check: empirical pull for multiparameter persistence.

Constructs a synthetic 4-class point-cloud task in the JOINT
(size, density-pattern) plane, where each axis alone is a 50%
2-class problem and the 4-class oracle is 100%.  A single-parameter
filtration can therefore only see one axis; the question is how
much joint information PALACE recovers via concatenated
single-parameter embeddings.

Class layout (2x2 in (size, arc-pattern) space; same low noise):
  0: small ring (r=0.5), uniform angular points
  1: small ring (r=0.5), points clustered in 2 opposing arcs
  2: large ring (r=1.0), uniform angular points
  3: large ring (r=1.0), points clustered in 2 opposing arcs

Alpha persistence sees H_1 ring SIZE (small vs large) but is
blind to the angular distribution: a uniform ring and an
arcs-only ring produce the same H_1 cycle.

DTM-density sees the angular density pattern (clustered vs
uniform) but the radial location is shared between the two
density patterns at each ring size, so density alone is
ambiguous about which ring size produced it.

A true bifiltration on (alpha-radius, density-DTM) could
disambiguate all four classes by carrying both axes jointly.

Reproduction: `python experiments/exp_pv_joint_feature.py`
"""
import sys; sys.path.insert(0, '.')
import csv
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from utils.persistence import pointcloud_to_persistence, _density_persistence
from embedding.nonuniform import init_nonuniform_from_data


# ---- task spec ----
N_PER_CLASS = 60
N_PTS       = 80
SIGMA       = 0.05   # low noise: ring is preserved for alpha
SEED        = 42
N_FOLDS     = 5
K_LANDMARKS = 32

# (radius, arc_pattern). arc_pattern: 'uniform' or 'two_arcs'.
CLASSES = [
    (0.5, 'uniform'),
    (0.5, 'two_arcs'),
    (1.0, 'uniform'),
    (1.0, 'two_arcs'),
]


def sample_ring(n: int, radius: float, sigma: float, pattern: str,
                rng: np.random.Generator) -> np.ndarray:
    """n points on a circle, with angular pattern + Gaussian noise."""
    if pattern == 'uniform':
        angles = rng.uniform(0, 2 * np.pi, n)
    elif pattern == 'two_arcs':
        # Half points in arc near angle 0, half near pi; each arc spans pi/3.
        n_each = n // 2
        a1 = rng.uniform(-np.pi / 6, np.pi / 6, n_each)
        a2 = rng.uniform(np.pi - np.pi / 6, np.pi + np.pi / 6, n - n_each)
        angles = np.concatenate([a1, a2])
        rng.shuffle(angles)
    else:
        raise ValueError(pattern)
    pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    pts += rng.normal(0, sigma, pts.shape)
    return pts


def build_dataset(rng):
    pts_per_cloud, labels = [], []
    for cls_idx, (radius, pattern) in enumerate(CLASSES):
        for _ in range(N_PER_CLASS):
            pts = sample_ring(N_PTS, radius, SIGMA, pattern, rng)
            pts_per_cloud.append(pts)
            labels.append(cls_idx)
    return pts_per_cloud, np.array(labels)


def diagrams(pts_per_cloud, mode):
    out = []
    for pts in pts_per_cloud:
        if mode == 'alpha':
            d = pointcloud_to_persistence(pts, max_dim=1, method='alpha')
        elif mode == 'density':
            d = _density_persistence(pts, max_dim=1, k=10)
        elif mode == 'concat':
            d_a = pointcloud_to_persistence(pts, max_dim=1, method='alpha')
            d_d = _density_persistence(pts, max_dim=1, k=10)
            d = [np.vstack([d_a[i], d_d[i]]) if (len(d_a[i]) or len(d_d[i]))
                 else np.zeros((0, 2)) for i in range(2)]
        else:
            raise ValueError(mode)
        flat = np.vstack([d[0] if len(d[0]) else np.zeros((0, 2)),
                          d[1] if len(d[1]) else np.zeros((0, 2))])
        if len(flat) > 30:
            pers = flat[:, 1] - flat[:, 0]
            idx = np.argsort(-pers)[:30]
            flat = flat[idx]
        out.append(flat)
    return out


def run_one(pts_per_cloud, y, mode, seed=SEED):
    diags = diagrams(pts_per_cloud, mode)
    L = max((d[:, 1].max() for d in diags if len(d) > 0), default=1.0) * 1.1
    diagrams_by_class = [[d for d, lab in zip(diags, y) if lab == c]
                         for c in sorted(set(y))]
    model = init_nonuniform_from_data(diagrams_by_class, K=K_LANDMARKS, L=L,
                                       seed=seed)
    X = np.stack([model.embed(d) for d in diags])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    accs = []
    from sklearn.metrics.pairwise import euclidean_distances
    for tr, te in skf.split(X, y):
        sigma = max(float(np.median(euclidean_distances(X[tr], X[tr]))), 1e-3)
        Gtr = np.exp(-euclidean_distances(X[tr], X[tr])**2 / (2 * sigma**2))
        Gte = np.exp(-euclidean_distances(X[te], X[tr])**2 / (2 * sigma**2))
        clf = SVC(kernel='precomputed', C=1.0).fit(Gtr, y[tr])
        accs.append((clf.predict(Gte) == y[te]).mean())
    return np.array(accs)


def main():
    rng = np.random.default_rng(SEED)
    pts_per_cloud, y = build_dataset(rng)
    print(f"Dataset: {len(pts_per_cloud)} clouds, "
          f"{len(set(y))} classes ({N_PER_CLASS} per class), "
          f"{N_PTS} pts/cloud, sigma={SIGMA}", flush=True)
    print(f"Chance: {1.0/len(set(y))*100:.0f}%   Oracle: 100%\n", flush=True)

    rows = []
    print("4-class task (joint size + arc-pattern):", flush=True)
    for mode in ['alpha', 'density', 'concat']:
        accs = run_one(pts_per_cloud, y, mode)
        mean, sd = accs.mean()*100, accs.std()*100
        print(f"  {mode:10s}: {mean:5.1f} ± {sd:4.1f}  per-fold: "
              f"{[f'{a*100:.0f}' for a in accs]}", flush=True)
        rows.append({'task': '4class', 'mode': mode,
                     'mean_pct': mean, 'std_pct': sd})

    # Per-axis 2-class diagnostic
    print("\nPer-axis 2-class diagnostic (PALACE on each filtration alone):",
          flush=True)
    for axis_name, fold_fn in [
        ('size',    lambda lab: (lab == 0) | (lab == 1)),  # small (T) vs large (F)
        ('density', lambda lab: (lab == 0) | (lab == 2)),  # uniform (T) vs arcs (F)
    ]:
        y_bin = fold_fn(y).astype(int)
        for mode in ['alpha', 'density']:
            accs = run_one(pts_per_cloud, y_bin, mode)
            print(f"  {axis_name:8s} via {mode:8s}: "
                  f"{accs.mean()*100:5.1f}% (chance: 50%)", flush=True)
            rows.append({'task': f'2class_{axis_name}', 'mode': mode,
                         'mean_pct': accs.mean()*100,
                         'std_pct': accs.std()*100})

    out = Path('results/paper_V/exp_pv_joint_feature.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}", flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Spectral clustering of a run's per-loop triplane codes.

Reads the `<run>_triplane.npz` written at the end of a fit, builds the dense
cosine affinity W = (1 + cos) / 2, and runs spectral clustering with k-means on
its eigenvectors.  k is chosen by the cosine silhouette unless `--k` is given.
With `--per-axis` the x, y and z plane families are clustered separately, so a
cluster never mixes loops from different slicing axes; `--k` then applies to
each family and "auto" chooses one k per family.

    python utils/spectral_code_clusters.py log/axis_frz100_horse/*_triplane.npz
    python utils/spectral_code_clusters.py <npz> --k 12
    python utils/spectral_code_clusters.py <npz> --per-axis
    python utils/spectral_code_clusters.py <npz> --k 4 --per-axis --out labels.csv

The CSV uses the same columns as `code_clusters.py`, so `utils/cluster_loops_viz.py`
reads it directly:
    python utils/cluster_loops_viz.py --run log/<run> \
        --clusters <run>/<stem>_code_clusters_spectral[_axis].csv \
        --out <run>/<run>_cluster_loops_spectral[_axis]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


def spectral_labels(unit, k, seed):
    affinity = (1.0 + unit @ unit.T) / 2.0
    return SpectralClustering(
        n_clusters=k, affinity='precomputed', assign_labels='kmeans',
        random_state=seed).fit_predict(affinity)


def order_by_size(labels):
    order = sorted(np.unique(labels), key=lambda c: -(labels == c).sum())
    remap = {c: i for i, c in enumerate(order)}
    return np.array([remap[c] for c in labels])


def cluster_group(unit, k, kmax, seed):
    """Cluster one family and return (labels, k, silhouette)."""
    n = len(unit)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0, float('nan')
    if n <= 2:
        labels = np.arange(n, dtype=np.int64)
        return order_by_size(labels), n, float('nan')
    if k == 'auto':
        scores = {}
        for kk in range(2, min(kmax, n - 1) + 1):
            lab = spectral_labels(unit, kk, seed)
            scores[kk] = silhouette_score(unit, lab, metric='cosine')
        kk = max(scores, key=scores.get)
        labels = spectral_labels(unit, kk, seed)
        return order_by_size(labels), kk, scores[kk]
    kk = int(k)
    if not 1 <= kk <= n:
        raise SystemExit(f'k = {kk} does not fit a family of {n} loops')
    labels = (np.zeros(n, dtype=np.int64) if kk == 1
              else spectral_labels(unit, kk, seed))
    score = (silhouette_score(unit, labels, metric='cosine')
             if 1 < kk < n else float('nan'))
    return order_by_size(labels), kk, score


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('npz', help='<run>_triplane.npz')
    ap.add_argument('--k', default='auto',
                    help='number of clusters or "auto"; with --per-axis, per family')
    ap.add_argument('--kmax', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--per-axis', action='store_true',
                    help='cluster x, y and z loops separately instead of all together')
    ap.add_argument('--out', default=None, help='output CSV stem')
    args = ap.parse_args()

    data = np.load(args.npz)
    feats = data['features'].astype(np.float64)
    keys = data['keys'].astype(np.float64)
    xyz = data['queries'].astype(np.float64)
    n = len(feats)
    unit = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    axis = np.array(['xyz'[int(k[0])] for k in keys])
    off, slot = keys[:, 1], keys[:, 2].astype(int)

    lines = [f'npz: {args.npz}', f'loops: {n}', f'codes: {feats.shape[1]}-D']
    if args.per_axis:
        labels = np.full(n, -1, dtype=np.int64)
        next_label = 0
        parts = []
        for a in 'xyz':
            sel = np.flatnonzero(axis == a)
            if not len(sel):
                continue
            lab, k_a, score_a = cluster_group(unit[sel], args.k, args.kmax,
                                              args.seed)
            labels[sel] = lab + next_label
            next_label += k_a
            parts.append((a, len(sel), k_a, score_a))
        k = next_label
        valid = [(na, s) for _, na, _, s in parts if np.isfinite(s)]
        sil = (sum(na * s for na, s in valid) / sum(na for na, _ in valid)
               if valid else float('nan'))
        print(f'{Path(args.npz).name}: {n} loops, codes {feats.shape[1]}-D, '
              f'dense affinity (1+cos)/2, clustered per axis')
        for a, na, k_a, score_a in parts:
            score = 'n/a' if not np.isfinite(score_a) else f'{score_a:.3f}'
            print(f'  axis {a}: {na} loops, k = {k_a}, silhouette {score}')
        print(f'  {k} clusters total, within-axis silhouette {sil:.3f}')
        lines += [f'method: spectral per axis, dense affinity (1+cos)/2, kmeans',
                  f'k: {k}', f'silhouette within axis: {sil:.4f}']
        lines += [f'axis {a}: {na} loops, k = {k_a}, silhouette '
                  + ('n/a' if not np.isfinite(s) else f'{s:.4f}')
                  for a, na, k_a, s in parts]
    else:
        scores = {}
        for kk in range(2, min(args.kmax, n - 1) + 1):
            lab = spectral_labels(unit, kk, args.seed)
            scores[kk] = silhouette_score(unit, lab, metric='cosine')
        k = max(scores, key=scores.get) if args.k == 'auto' else int(args.k)
        labels = spectral_labels(unit, k, args.seed)
        if k not in scores:
            scores[k] = silhouette_score(unit, labels, metric='cosine')
        labels = order_by_size(labels)
        print(f'{Path(args.npz).name}: {n} loops, codes {feats.shape[1]}-D, '
              f'dense affinity (1+cos)/2')
        print(f'  k = {k}, silhouette {scores[k]:.3f}')
        lines += [f'method: spectral, dense affinity (1+cos)/2, kmeans',
                  f'k: {k}', f'silhouette: {scores[k]:.4f}']

    for c in range(k):
        sel = labels == c
        mix = {a: int((axis[sel] == a).sum()) for a in 'xyz' if (axis[sel] == a).any()}
        intra = (unit[sel] @ unit[sel].T)[np.triu_indices(sel.sum(), 1)].mean() \
            if sel.sum() > 1 else np.nan
        inter = unit[sel] @ unit[~sel].T
        desc = []
        for a in 'xyz':
            a_sel = sel & (axis == a)
            if a_sel.any():
                desc.append(f'{a}: {a_sel.sum()} loops on '
                            f'{len(np.unique(off[a_sel]))} planes '
                            f'[{off[a_sel].min():+.3f}, {off[a_sel].max():+.3f}]')
        text = (f'  c{c}: {sel.sum():3d} loops ('
                f'{" ".join(f"{a}{v}" for a, v in mix.items())})'
                f'  intra cos {intra:+.3f}  inter {inter.mean():+.3f}'
                + ('  ' + '; '.join(desc) if desc else ''))
        print(text)
        lines.append(text.strip())

    suffix = '_code_clusters_spectral_axis' if args.per_axis else \
        '_code_clusters_spectral'
    out = Path(args.out) if args.out else (
        Path(args.npz).parent /
        f'{Path(args.npz).stem.removesuffix("_triplane")}{suffix}.csv')
    with open(out, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['axis', 'offset', 'slot', 'x', 'y', 'z', 'cluster'])
        for i in range(n):
            writer.writerow([axis[i], f'{off[i]:.4f}', slot[i], f'{xyz[i, 0]:.4f}',
                             f'{xyz[i, 1]:.4f}', f'{xyz[i, 2]:.4f}', labels[i]])
    out.with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print(f'wrote {out}')
    print(f'wrote {out.with_suffix(".txt")}')


if __name__ == '__main__':
    main()

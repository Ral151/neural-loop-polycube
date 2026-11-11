#!/usr/bin/env python3
"""PCA snapshots of the cube warm-up's triplane.

Each cell of each of the three planes carries a 16-D feature. All cells are
stacked, centred, and the top three principal directions are painted into
R, G and B, so every plane becomes a 32 x 32 colour snapshot in one shared
feature basis. The unit cube's 12 reference loop centroids are marked as fixed
places on the grid.

    python utils/triplane_pca.py warmup/cube_warmup_xavier.pt
    python utils/triplane_pca.py warmup/cube_warmup_random_ranges.pt \\
        warmup/cube_warmup_axis.pt

The checkpoint's frame is the unit cube, so cell centres are on [-0.5, 0.5].
The colour scale is symmetric per component, with zero fixed at 0.5, and
clipped at the 99th percentile of |score| (the maximum when that is zero, as
in a grid that is almost entirely untouched).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLANES = (('XY', 0, 1), ('XZ', 0, 2), ('YZ', 1, 2))
AXES = 'xyz'


def initial_grid(ckpt, kind):
    """The grid before training, from the checkpoint's seed and shape.

    The triplane is the first module constructed, so its initialiser is the
    first consumer of the seeded RNG; that makes the reconstruction independent
    of the rest of the network. Only the two initialisers the warm-ups used are
    represented.
    """
    shape = tuple(ckpt['state_dict']['triplane.grid'].shape)
    if kind == 'zero':
        return np.zeros(shape)
    torch.manual_seed(ckpt['seed'])
    grid = torch.empty(shape)
    nn.init.xavier_uniform_(grid)
    return grid.numpy().astype(np.float64)


def load_grid(path, delta=False, init='auto'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('kind') != 'cube_warmup':
        raise SystemExit(f'{path} is not a cube warm-up checkpoint')
    grid = ckpt['state_dict']['triplane.grid'].numpy().astype(np.float64)
    if not delta:
        return grid, ckpt

    kinds = ('zero', 'xavier') if init == 'auto' else (init,)
    found = []
    for kind in kinds:
        change = grid - initial_grid(ckpt, kind)
        touched = np.abs(change).sum(1) > 0
        untouched = ~touched
        ## A wrong initialiser can mark every cell as touched; with no untouched
        ## cell there is nothing to check, so it is not evidence
        outside = max((np.abs(change[p][:, untouched[p]]).max()
                       for p in range(change.shape[0]) if untouched[p].any()),
                      default=np.inf)
        if outside <= 1e-6:
            found.append((int(touched.sum()), kind, change))
    if not found:
        raise SystemExit(f'{path}: no initialiser in {kinds} leaves the untouched '
                         f'cells unchanged; cannot form a delta')
    _, kind, change = min(found, key=lambda item: item[0])
    print(f'  delta: initial grid was {kind!r} (seed {ckpt["seed"]})')
    return change, ckpt


def pca(x, k=3):
    mean = x.mean(0)
    centred = x - mean
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:k].copy()
    ## PCA sign is arbitrary; fix it so the same checkpoint always paints the same
    ## colours
    for i in range(k):
        j = np.argmax(np.abs(comps[i]))
        if comps[i, j] < 0:
            comps[i] *= -1
    ev = s ** 2
    ev = ev / ev.sum()
    return mean, comps, centred @ comps.T, ev


def component_rgb(z):
    scale = np.percentile(np.abs(z), 99.0)
    if scale <= 0:
        scale = np.abs(z).max()
    if scale <= 0:
        scale = 1.0
    return np.clip(0.5 + 0.5 * z / scale, 0.0, 1.0)


def cube_queries(ckpt):
    planes = int((ckpt.get('warmup') or {}).get('planes', 4))
    half = float(ckpt['triplane']['half'][0])
    offsets = np.linspace(-0.8 * half, 0.8 * half, planes)
    return np.stack([d * np.eye(3)[a] for a in range(3) for d in offsets])


def render(checkpoints, out, delta=False, init='auto'):
    grids = [load_grid(p, delta, init) for p in checkpoints]
    rows = len(grids)
    fig = plt.figure(figsize=(4.0 * 3 + 2.6, 3.9 * rows + 0.9))
    gs = fig.add_gridspec(rows, 4, width_ratios=(1, 1, 1, 0.78),
                          wspace=0.32, hspace=0.42)
    stats = []

    for r, ((grid, ckpt), path) in enumerate(zip(grids, checkpoints)):
        nplanes, channels, res, _ = grid.shape
        cells = grid.reshape(nplanes * res * res, channels)
        mean, comps, scores, ev = pca(cells)
        painted = component_rgb(scores).reshape(nplanes, res, res, 3)
        queries = cube_queries(ckpt)
        nonzero = (np.abs(grid).sum(1) > 0).reshape(nplanes, -1).sum(1)

        for c, (name, a, b) in enumerate(PLANES):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(painted[c], origin='lower',
                      extent=(-0.5, 0.5, -0.5, 0.5), interpolation='nearest')
            ax.scatter(queries[:, a], queries[:, b], s=26, facecolors='none',
                       edgecolors='white', linewidths=0.9)
            ax.set_xticks((-0.4, 0.0, 0.4), ('-0.4', '0', '0.4'))
            ax.set_yticks((-0.4, 0.0, 0.4), ('-0.4', '0', '0.4'))
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel(f'{AXES[a]}  (PC colour)')
            ax.set_ylabel(AXES[b])
            ax.set_title(f'{name} plane', fontsize=10)
            ax.tick_params(labelsize=7)

        axv = fig.add_subplot(gs[r, 3])
        top = min(10, len(ev))
        axv.bar(np.arange(1, top + 1), ev[:top] * 100, color='#4c72b0')
        axv.set_xlabel('principal component', fontsize=8)
        axv.set_ylabel('explained variance (%)', fontsize=8)
        axv.set_xticks(np.arange(1, top + 1))
        axv.tick_params(labelsize=7)
        label = 'changed' if delta else 'non-zero'
        axv.set_title(f'{Path(path).stem}\n'
                      f'{label} cells/plane '
                      f'{",".join(str(int(n)) for n in nonzero)} of {res * res}; '
                      f'top-3 {ev[:3].sum() * 100:.1f}%', fontsize=8)
        axv.grid(alpha=0.25, axis='y')
        axv.set_ylim(0, max(ev[:top].max() * 100 * 1.15, 1))
        stats.append(dict(path=path, ev=ev, nonzero=nonzero,
                          components=comps, mean=mean, scores=scores))

        print(f'{path}: grid {grid.shape}, channels {channels}, '
              f'{res}x{res} cells per plane'
              f'{" (change from initialisation)" if delta else ""}')
        print(f'  {label} cells per plane: {nonzero.tolist()} / {res * res}')
        print(f'  explained variance top-5: '
              f'{np.round(ev[:5] * 100, 2).tolist()} %')
        print(f'  cumulative top-3: {ev[:3].sum() * 100:.2f} %')
        for i in range(3):
            j = np.argmax(np.abs(comps[i]))
            print(f'  PC{i + 1}: loading peak channel {j} '
                  f'({comps[i, j]:+.3f}), |score| p99 '
                  f'{np.percentile(np.abs(scores[:, i]), 99):.4f}')

    what = ('change from initialisation' if delta else 'cell features')
    fig.suptitle(f'Cube warm-up triplane {what}: RGB = PC1/PC2/PC3 of the 16-D '
                 'cell vectors (shared basis, zero at 0.5); white circles = the '
                 'unit cube\'s 12 reference loop centroids', fontsize=11)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'wrote {out}')

    np.savez(Path(out).with_suffix('.npz'),
             **{Path(s['path']).stem: s['scores'] for s in stats})
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoints', nargs='*',
                        default=['warmup/cube_warmup_random_ranges.pt'],
                        help='cube warm-up checkpoints (default: '
                             'warmup/cube_warmup_random_ranges.pt)')
    parser.add_argument('--out', default='log/triplane_pca/triplane_pca.png',
                        help='output PNG')
    parser.add_argument('--delta', action='store_true',
                        help='paint the change from the initial grid instead of '
                             'the grid itself')
    parser.add_argument('--init', choices=('auto', 'zero', 'xavier'), default='auto',
                        help='initialiser to subtract in --delta mode; auto tries '
                             'both and keeps the one that leaves untouched cells '
                             'unchanged')
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(args.checkpoints, out, args.delta, args.init)


if __name__ == '__main__':
    main()

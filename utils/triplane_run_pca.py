#!/usr/bin/env python3
"""PCA snapshots of a trained run's triplane field.

The run's `*_model.pt` carries the learned grid, the world box it is mapped
onto, and the loops' query points.  Cells are stacked, centred, and the top
three principal directions are painted into R, G and B; the three planes share
one basis.  Loop query points are projected onto each plane so the cells each
loop family actually reads are visible.  `--delta` adds a second row with the
change from the cube warm-up; it is off by default because a sparse warm-up
leaves the trained field almost unchanged in its delta.

    python plc_utils/triplane_run_pca.py log/axis_seam10_horse/*_model.pt
    python plc_utils/triplane_run_pca.py log/axis_seam10_horse/*_model.pt --delta

The grid is parameterised on the unit cube and mapped onto the run's box, so
the panels are drawn in world coordinates with that box's extents.  Colours are
symmetric per component with zero at 0.5, clipped at the 99th percentile of
|score|.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _paths  # noqa: E402
from triplane_pca import AXES, PLANES, component_rgb, pca

AXIS_COLOURS = {'x': '#1f77b4', 'y': '#d62728', 'z': '#2ca02c'}


def load_run(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    tri = ckpt['triplane']
    grid = sd['triplane.grid'].numpy().astype(np.float64)
    centre = np.asarray(sd.get('triplane.box_centre', tri['centre']))
    half = np.asarray(sd.get('triplane.box_half', tri['half']))
    ## plane_normal is the loop's slice normal, so its dominant component names
    ## the family the loop belongs to
    normals = np.asarray(ckpt['plane_normal'])
    axis_of = np.abs(normals).argmax(1)
    return dict(grid=grid, centre=centre, half=half,
                queries=np.asarray(ckpt['queries']),
                axis_of=axis_of)


def load_warmup(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return ckpt['state_dict']['triplane.grid'].numpy().astype(np.float64)


def paint(ax, painted, centre, half, a, b, loops, axis_of, title):
    lo = centre - half
    hi = centre + half
    ax.imshow(painted, origin='lower',
              extent=(lo[a], hi[a], lo[b], hi[b]), interpolation='nearest',
              aspect='equal')
    for c, axis in enumerate('xyz'):
        sel = axis_of == c
        if not sel.any():
            continue
        ax.scatter(loops[sel, a], loops[sel, b], s=9, alpha=0.75,
                   facecolors='none', edgecolors=AXIS_COLOURS[axis],
                   linewidths=0.7)
    ax.set_xlabel(AXES[a])
    ax.set_ylabel(AXES[b])
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def explained(ax, ev, label, note):
    top = min(10, len(ev))
    ax.bar(np.arange(1, top + 1), ev[:top] * 100, color='#4c72b0')
    ax.set_xlabel('principal component', fontsize=8)
    ax.set_ylabel('explained variance (%)', fontsize=8)
    ax.set_xticks(np.arange(1, top + 1))
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, axis='y')
    ax.set_ylim(0, max(ev[:top].max() * 100 * 1.15, 1))
    ax.set_title(f'{label}\ntop-3 {ev[:3].sum() * 100:.1f}%  ·  {note}',
                 fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('checkpoint', help='a run\'s *_model.pt')
    ap.add_argument('--warmup', default='warmup/cube_warmup_random_ranges.pt',
                    help='the warm-up the fit started from')
    ap.add_argument('--delta', action='store_true',
                    help='add a row with the change from the warm-up')
    ap.add_argument('--out', default=None, help='output PNG')
    args = ap.parse_args()

    run = load_run(args.checkpoint)
    stem = Path(args.checkpoint).stem
    out = Path(args.out or f'log/triplane_pca/{stem}_triplane_pca.png')
    out.parent.mkdir(parents=True, exist_ok=True)

    delta = None
    if Path(args.warmup).is_file():
        w = load_warmup(args.warmup)
        if w.shape == run['grid'].shape:
            ratio = np.linalg.norm(w) / np.linalg.norm(run['grid'])
            corr = np.corrcoef(w.ravel(), run['grid'].ravel())[0, 1]
            print(f'  warm-up: {ratio * 100:.1f}% of the field norm, '
                  f'corr {corr:.3f}')
            if args.delta:
                delta = run['grid'] - w
        else:
            print(f'  warm-up grid {w.shape} does not match '
                  f'{run["grid"].shape}; no delta')

    rows = [(run['grid'], 'field', 'absolute value')]
    if delta is not None:
        rows.append((delta, 'change from warm-up', 'change from initialisation'))

    fig = plt.figure(figsize=(4.0 * 3 + 2.6, 3.9 * len(rows) + 0.9))
    gs = fig.add_gridspec(len(rows), 4, width_ratios=(1, 1, 1, 0.78),
                          wspace=0.32, hspace=0.42)
    lo, hi = run['centre'] - run['half'], run['centre'] + run['half']
    print(f'{stem}: grid {run["grid"].shape}')
    print(f'  box lo {np.round(lo, 4).tolist()}  hi {np.round(hi, 4).tolist()}')
    print(f'  {len(run["queries"])} loop queries, '
          f'{np.bincount(run["axis_of"], minlength=3).tolist()} per x/y/z family')

    for r, (grid, label, note) in enumerate(rows):
        nplanes, channels, res, _ = grid.shape
        cells = grid.reshape(nplanes * res * res, channels)
        mean, comps, scores, ev = pca(cells)
        painted = component_rgb(scores).reshape(nplanes, res, res, 3)
        touched = (np.abs(grid).sum(1) > 0).reshape(nplanes, -1).sum(1)

        for c, (name, a, b) in enumerate(PLANES):
            ax = fig.add_subplot(gs[r, c])
            paint(ax, painted[c], run['centre'], run['half'], a, b,
                  run['queries'], run['axis_of'], f'{name} plane · {label}')

        axv = fig.add_subplot(gs[r, 3])
        explained(axv, ev, label,
                  f'{",".join(str(int(n)) for n in touched)}/{res * res} '
                  f'cells per plane')

        kind = 'changed' if r else 'non-zero'
        print(f'  [{label}] {kind} cells/plane {touched.tolist()} / {res * res}')
        print(f'    explained variance top-5: {np.round(ev[:5] * 100, 2).tolist()} %')
        print(f'    PC peaks: ' + ', '.join(
            f'PC{i + 1} ch{np.argmax(np.abs(comps[i]))} '
            f'({comps[i, np.argmax(np.abs(comps[i]))]:+.3f})'
            for i in range(3)))

    handles = [Line2D([], [], marker='o', ls='', markerfacecolor='none',
                      markeredgecolor=AXIS_COLOURS[a], label=f'{a}-slice loops')
               for a in 'xyz']
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'{stem} — triplane grid: RGB = PC1/PC2/PC3 of the 16-D cell '
                 f'vectors (one basis per row, zero at 0.5); circles = loop '
                 f'query points', fontsize=11)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
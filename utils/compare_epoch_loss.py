"""Overlay the loss curves of several runs, one panel per shape.

Runs are recognised by their folder name, `<anything>_<arm>_<shape>`, with the
arms this project compares (`fixed`, `random`, `random_anneal_floored`) and the
four comparison shapes.  The top row is the total objective each run's own
`run.txt` prints; the bottom row is its fit term, which is directly comparable
across arms even when a run leaves tv out of the loss.

  python utils/compare_epoch_loss.py 'log/20261108_*_fixed_*' \
      'log/20261108_*_random_*' --out log/epoch_loss_compare.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_epoch_loss import parse, positive

ARMS = ('fixed', 'random', 'random_new', 'random_new_eta4',
        'random_anneal_floored')
SHAPES = ('bunny', 'armadillo', 'horse', 'laurent_hand')
LABEL = {'fixed': 'fixed unit cube',
         'random': 'random 0.1-0.45, constant lr',
         'random_new': 'random 0.1-0.45, cosine lr, eta 1e-5',
         'random_new_eta4': 'random 0.1-0.45, cosine lr, eta 1e-4',
         'random_anneal_floored': 'random + anneal floor'}
COLOR = {'fixed': '0.25', 'random': 'tab:blue', 'random_new': 'tab:green',
         'random_new_eta4': 'tab:red',
         'random_anneal_floored': 'tab:orange'}


def classify(path):
    for shape in SHAPES:
        if path.name.endswith('_' + shape):
            head = path.name[:-len(shape) - 1]
            for arm in ARMS:
                if head.endswith('_' + arm):
                    return arm, shape
    raise ValueError(f'cannot read an arm and shape from {path.name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', type=Path,
                    help='run folders, e.g. log/20261108_*_fixed_*')
    ap.add_argument('--out', type=Path,
                    default=Path('log/epoch_loss_compare.png'))
    ap.add_argument('--freeze', type=int, default=100,
                    help='epochs f was frozen (0 to hide the line)')
    args = ap.parse_args()

    runs = {}
    for path in args.runs:
        family, shape = classify(path)
        try:
            d = parse((path / 'run.txt').read_text())
        except SystemExit as exc:
            print(f'skip {path.name}: {exc}', file=sys.stderr)
            continue
        runs.setdefault(shape, {})[family] = d

    shapes = [s for s in SHAPES if s in runs]
    fig, axs = plt.subplots(2, len(shapes), figsize=(4.3 * len(shapes), 7.8),
                            dpi=140, squeeze=False,
                            sharex=True, sharey='row')
    finals = {}
    for k, shape in enumerate(shapes):
        for row, (metric, ylabel) in enumerate(
                (('loss', 'total loss'), ('fit', 'fit term'))):
            ax = axs[row, k]
            for arm in ARMS:
                if arm not in runs[shape]:
                    continue
                d = runs[shape][arm]
                ax.plot(d['epoch'], positive(d[metric]), color=COLOR[arm],
                        lw=1.1, label=LABEL[arm])
                if row == 0:
                    finals[(shape, arm)] = float(d['loss'][-1])
            if args.freeze:
                ax.axvline(args.freeze - 0.5, color='0.75', ls='--', lw=0.9)
            ax.set_yscale('log')
            ax.grid(alpha=0.25)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(shape)
            else:
                ax.set_xlabel('epoch')
    axs[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('All settings on the same fits'
                 + (f' (dashed: f frozen through epoch {args.freeze - 1})'
                    if args.freeze else '')
                 + '\ntop: total loss; bottom: fit term. The two cosine arms '
                 'differ only in eta_min (1e-5 vs 1e-4); the constant-lr '
                 'runs predate the scheduler.',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f'wrote {args.out}')

    print(f'\n{"shape":14s} {"arm":22s} {"final loss":>10s} {"final fit":>10s}')
    for shape in shapes:
        for arm in ARMS:
            if arm in runs[shape]:
                d = runs[shape][arm]
                print(f'{shape:14s} {arm:22s} {d["loss"][-1]:10.6f} '
                      f'{d["fit"][-1]:10.6f}')


if __name__ == '__main__':
    main()

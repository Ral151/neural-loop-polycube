"""Every raw loss term the runs print, by shape and arm.

Rows are shapes and columns are the printed values themselves: total loss, fit,
normal align (mean 1 - cos), triplane tv, and the seam hold (distillation).
None of the non-fit columns is multiplied by its weight here, so the columns do
not add up to the loss: the run weights align, tv and distill by an annealed
`--normal`, `--triplane-tv` and `--distill` when it forms the loss.  The tv
column is absent for an arm trained with `--triplane-tv 0`.

  python plc_utils/compare_epoch_terms.py 'log/20261108_*_fixed_*' \
      'log/20261108_*_random_*' --out log/epoch_terms_compare.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from compare_epoch_loss import ARMS, COLOR, LABEL, SHAPES, classify
from plot_epoch_loss import parse, positive

TERMS = (
    ('loss', 'loss'),
    ('fit', 'fit'),
    ('align', 'align (1 - cos)'),
    ('tv', 'tv (triplane)'),
    ('distill', 'seam hold (distill)'),
)
ARM_ABBR = {'fixed': 'fixed\nconst. lr',
            'random': 'random\nconst. lr',
            'random_new': 'random\ncosine lr',
            'random_anneal_floored': 'random\nnormal+tv off'}


def final_figure(runs, shapes, out):
    """The last epoch only, as bars: each run's printed value, no weight."""
    arms = [a for a in ARMS if any(a in runs[s] for s in shapes)]
    fig, axs = plt.subplots(len(shapes), len(TERMS),
                            figsize=(3.3 * len(TERMS), 2.8 * len(shapes)),
                            dpi=140, squeeze=False, sharey='col')
    for col, (key, _) in enumerate(TERMS):
        seen = [runs[s][a][key][-1] for s in shapes for a in arms
                if a in runs[s] and np.isfinite(runs[s][a][key][-1])
                and runs[s][a][key][-1] > 0]
        lo = min(seen) / 4 if seen else 1e-6
        hi = max(seen) * 8 if seen else 1.0
        for row, shape in enumerate(shapes):
            ax = axs[row, col]
            for i, arm in enumerate(arms):
                if arm not in runs[shape]:
                    continue
                v = runs[shape][arm][key][-1]
                if not np.isfinite(v) or v <= 0:
                    ax.text(i, 0.06, 'off' if not np.isfinite(v) else '0',
                            transform=ax.get_xaxis_transform(), ha='center',
                            va='bottom', fontsize=7, color='0.4')
                    continue
                ax.bar(i, v, 0.72, color=COLOR[arm])
                ax.text(i, v * 1.2, f'{v:.2g}', ha='center', va='bottom',
                        fontsize=6.5, rotation=90)
            ax.set_yscale('log')
            ax.set_ylim(lo, hi)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([ARM_ABBR.get(a, a) for a in arms],
                               fontsize=6.5)
            ax.grid(alpha=0.25, axis='y')
            if row == 0:
                ax.set_title(TERMS[col][1], fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{shape}\nraw value')
    handles = [Patch(color=COLOR[a], label=LABEL[a]) for a in arms]
    fig.legend(handles=handles, loc='lower center', ncol=len(arms),
               frameon=False, fontsize=9)
    last = int(max(runs[s][a]['epoch'][-1] for s in shapes for a in arms
                   if a in runs[s]))
    fig.suptitle(f'Final epoch ({last}): every term exactly as the run '
                 'printed it, no weight applied'
                 '\nThe loss weights are absent here: align w is ~0 by the '
                 'end, tv is weighted by 0.01 where on, and the seam hold by '
                 '10. "off" means that run did not train the term.',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', type=Path,
                    help='run folders, e.g. log/20261108_*_fixed_*')
    ap.add_argument('--out', type=Path,
                    help='default: log/epoch_terms_compare.png, or '
                         'log/final_terms_compare.png with --final')
    ap.add_argument('--final', action='store_true',
                    help='draw only the last epoch, as raw bars')
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
    out = args.out or Path('log/final_terms_compare.png' if args.final
                           else 'log/epoch_terms_compare.png')
    if args.final:
        final_figure(runs, shapes, out)
    else:
        fig, axs = plt.subplots(len(shapes), len(TERMS),
                                figsize=(4.0 * len(TERMS), 2.9 * len(shapes)),
                                dpi=140, squeeze=False, sharex=True,
                                sharey='col')
        for row, shape in enumerate(shapes):
            for col, (key, _) in enumerate(TERMS):
                ax = axs[row, col]
                for arm in ARMS:
                    if arm not in runs[shape]:
                        continue
                    d = runs[shape][arm]
                    ax.plot(d['epoch'], positive(d[key]), color=COLOR[arm],
                            lw=1.0, label=LABEL[arm])
                if args.freeze:
                    ax.axvline(args.freeze - 0.5, color='0.8', ls='--', lw=0.8)
                ax.set_yscale('log')
                ax.grid(alpha=0.25)
                if row == 0:
                    ax.set_title(TERMS[col][1], fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'{shape}\nraw value')
                if row == len(shapes) - 1:
                    ax.set_xlabel('epoch')
        axs[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle('Every term in the fit loss, raw as printed, by shape and '
                     'setting'
                     + (f' (dashed: f frozen through epoch {args.freeze - 1})'
                        if args.freeze else '')
                     + '\nThe columns do not add up to the loss: align is '
                       'weighted by a cosine anneal from --normal to 0, tv by '
                       '0.01, and the seam hold by 10.',
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f'wrote {out}')

    print('\nFinal epoch, raw values as the run printed them '
          '(not multiplied by a weight)')
    print(f'{"shape":14s} {"arm":22s} {"loss":>9s} {"fit":>9s} '
          f'{"align":>9s} {"tv":>9s} {"distill":>9s}')
    for shape in shapes:
        for arm in ARMS:
            if arm not in runs[shape]:
                continue
            d = runs[shape][arm]

            def last(x):
                return f'{x[-1]:9.6f}' if np.isfinite(x[-1]) else '      off'

            print(f'{shape:14s} {arm:22s} {last(d["loss"])} {last(d["fit"])} '
                  f'{last(d["align"])} {last(d["tv"])} {last(d["distill"])}')


if __name__ == '__main__':
    main()

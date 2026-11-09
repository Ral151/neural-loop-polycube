"""Plot the loss terms a fit prints per epoch.

The trainer writes one line per epoch with the total objective and each term
beside it; this reads those lines back from the run's own record and draws them
together, so the composition of the loss is visible next to its size.  It does
not re-fit or read a checkpoint.

  python plc_utils/plot_epoch_loss.py log/20261108_211335_fixed_laurent_hand
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LINE = re.compile(r'^\s*(\d+) epoch: loss: ([0-9.eE+\-]+) \((.*?)\)'
                  r'( \[f frozen\])?\s*$')
NUM = r'[0-9.eE+\-]+'


def parse(text):
    """One array per printed term, NaN where the run left the term out."""
    rows = []
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        epoch, loss, inside, frozen = m.groups()
        row = dict(epoch=int(epoch), loss=float(loss),
                   frozen=bool(frozen))
        for name, pat, group in (
                ('fit', rf'fit ({NUM})', 1),
                ('match', rf'match ({NUM})% \(moved ({NUM})%\)', 1),
                ('moved', rf'match ({NUM})% \(moved ({NUM})%\)', 2),
                ('align', rf'align ({NUM})', 1),
                ('pair_cos', rf'pair cos ({NUM})', 1),
                ('tv', rf'tv ({NUM})', 1),
                ('distill', rf'distill ({NUM})', 1)):
            hit = re.search(pat, inside)
            row[name] = float(hit.group(group)) if hit else np.nan
        rows.append(row)
    if not rows:
        raise SystemExit('no "epoch: loss:" lines in the log')
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def positive(x):
    """Log axes need the zeros the run prints while a term is still off."""
    return np.where(np.asarray(x) > 0, x, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path, help='folder holding run.txt')
    ap.add_argument('--out', type=Path,
                    help='default: <run>/epoch_loss_terms.png')
    ap.add_argument('--tv-weight', type=float, default=0.01,
                    help='--triplane-tv weight of the run (default 0.01)')
    ap.add_argument('--title')
    args = ap.parse_args()

    text = (args.run / 'run.txt').read_text()
    d = parse(text)
    out = args.out or args.run / 'epoch_loss_terms.png'
    e = d['epoch']

    distill_w = re.search(rf'distill ({NUM})', text)
    distill_w = float(distill_w.group(1)) if distill_w else 0.0
    frozen = re.search(r'frozen for (\d+) epochs', text)
    freeze = int(frozen.group(1)) if frozen else 0

    fig, axs = plt.subplots(2, 3, figsize=(16, 8.4), dpi=140)
    fig.suptitle(args.title or f'{args.run.name}: loss terms by epoch',
                 fontsize=12)

    ax = axs[0, 0]
    ax.plot(e, positive(d['loss']), color='0.2', lw=1.2, label='loss')
    ax.plot(e, positive(d['fit']), color='tab:blue', lw=1.0, label='fit')
    ax.set_yscale('log')
    ax.set_ylabel('squared distance')
    ax.set_title('total objective and its fit term')
    ax.legend(frameon=False)

    ax = axs[0, 1]
    ax.plot(e, positive(args.tv_weight * d['tv']), color='tab:green',
            lw=1.0, label=f'tv x {args.tv_weight:g}')
    ax.plot(e, positive(distill_w * d['distill']), color='tab:red',
            lw=1.0, label=f'distill x {distill_w:g}')
    ax.set_yscale('log')
    ax.set_ylabel('contribution to loss')
    ax.set_title('the other terms, at the weight the loss gave them')
    ax.legend(frameon=False)

    ax = axs[0, 2]
    ax.plot(e, positive(d['tv']), color='tab:green', lw=1.0, label='tv')
    ax.plot(e, positive(d['distill']), color='tab:red', lw=1.0,
            label='distill')
    ax.set_yscale('log')
    ax.set_ylabel('unweighted value')
    ax.set_title('smoothness and seam terms, unweighted')
    ax.legend(frameon=False)

    ax = axs[1, 0]
    ax.plot(e, d['align'], color='tab:orange', lw=1.0, label='align (1 - cos)')
    ax.set_ylabel('mean 1 - cos')
    ax.set_title('direction term, unweighted')
    ax.legend(frameon=False, loc='upper right')

    ax = axs[1, 1]
    ax.plot(e, d['match'], color='tab:blue', lw=1.0, label='match')
    ax.plot(e, d['moved'], color='tab:purple', lw=1.0, label='moved')
    ax.set_ylabel('% of predicted points')
    ax.set_title('normal pairing')
    ax.legend(frameon=False)

    ax = axs[1, 2]
    ax.plot(e, d['pair_cos'], color='tab:brown', lw=1.0)
    ax.set_ylabel('mean cosine distance')
    ax.set_title("neighbouring loops' code separation")

    for ax in axs.ravel():
        if freeze:
            ax.axvline(freeze - 0.5, color='0.75', ls='--', lw=0.9)
        ax.grid(alpha=0.25)
        ax.set_xlabel('epoch')

    fig.text(0.5, 0.005,
             f'dashed line: f frozen for the first {freeze} epochs'
             if freeze else '', ha='center', fontsize=9, color='0.4')
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out)
    print(f'wrote {out}')

    for name in ('loss', 'fit', 'align', 'pair_cos', 'tv', 'distill'):
        print(f'  {name:9s} {np.nanmin(d[name]):.6f} .. {np.nanmax(d[name]):.6f}'
              f'  final {d[name][-1]:.6f}')


if __name__ == '__main__':
    main()

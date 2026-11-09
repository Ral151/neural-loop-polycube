#!/usr/bin/env python3
"""Detail the per-axis loss curves of a replayed fit and the epoch its loops turn.

The epoch lines in a run's log are means over every loop, so a whole plane family
running the other way round is invisible in them.  A fit writes, beside the
checkpoint, one row per epoch and axis (`_epoch_terms.csv`) and every loop's
winding at every epoch (`_loop_winding.npz`); this reads those and draws

  a  fit error by axis, log scale, whole run
  b  direction term (1 - cos) by axis, whole run, with the term's weight
  c  the same term over the first 120 epochs, where the turn happens
  d  loops wound clockwise over the same window
  e  the winding of every y loop over the run
  f  foldover, the angle a loop spends per turn; a retrace reads above 1

Usage: python plc_utils/plot_y_orientation.py [run_dir] [stem]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else 'log/axis_seam10_horse_repro')
STEM = sys.argv[2] if len(sys.argv) > 2 else 'horse_points_cubic_lam3'
AXES = ('x', 'y', 'z')
COLOUR = {'x': '#1f77b4', 'y': '#d62728', 'z': '#2ca02c'}
ZOOM = 120

rows = list(csv.DictReader(open(RUN / f'{STEM}_epoch_terms.csv')))
series = {a: {} for a in AXES}
for row in rows:
    a = row['axis']
    for key in ('epoch', 'loops', 'reversed', 'fit', 'align', 'winding', 'fold',
                'loss'):
        if row[key] != '':
            series[a].setdefault(key, []).append(float(row[key]))

winding = np.load(RUN / f'{STEM}_loop_winding.npz')
wind = winding['winding']
loop_axis = winding['axis']
epochs = winding['epoch']
y_loops = loop_axis == 1
rev = wind[:, y_loops] < 0
first = np.where(rev.any(0), rev.argmax(0) + epochs[0], 0)
half = rev.sum(1) >= (0.5 * rev.shape[1])
flip_lo = int(first[first > 0].min()) if (first > 0).any() else None
flip_hi = int(np.median(first[first > 0])) if (first > 0).any() else None

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for a in AXES:
    lw = 2.3 if a == 'y' else 1.1
    axes[0, 0].plot(series[a]['epoch'], series[a]['fit'], color=COLOUR[a], lw=lw,
                    label=a)
    axes[0, 1].plot(series[a]['epoch'], series[a]['align'], color=COLOUR[a], lw=lw,
                    label=a)
    axes[0, 2].plot(series[a]['epoch'], series[a]['align'], color=COLOUR[a], lw=lw,
                    label=a)
    axes[1, 0].plot(series[a]['epoch'], series[a]['reversed'], color=COLOUR[a],
                    lw=lw, label=a, drawstyle='steps-post')
    axes[1, 2].plot(series[a]['epoch'], series[a]['fold'], color=COLOUR[a], lw=lw,
                    label=a)

axes[0, 0].set_yscale('log')
axes[0, 0].set_ylabel('fit (mean squared distance)')
axes[0, 0].set_title('a  fit error by axis, whole run')
axes[0, 1].set_ylabel('1 - cos(predicted normal, target normal)')
axes[0, 1].set_title('b  direction term by axis, whole run')
axes[0, 2].set_ylabel('1 - cos')
axes[0, 2].set_title(f'c  direction term, epochs 0-{ZOOM}')
axes[1, 0].set_ylabel('loops wound clockwise')
axes[1, 0].set_title(f'd  reversed loops, epochs 0-{ZOOM}')
axes[1, 2].set_ylabel('angle spent / angle of one turn')
axes[1, 2].set_title('f  foldover by axis (1 = one clean turn)')
axes[1, 2].set_ylim(0.9, 2.2)

if flip_lo is not None:
    for ax in (axes[0, 2], axes[1, 0]):
        ax.axvspan(flip_lo, flip_hi, color='#d62728', alpha=0.10, lw=0)
        ax.text(flip_hi + 3, ax.get_ylim()[1] * 0.92,
                f'y turns\n{flip_lo}-{flip_hi}', color='#d62728', fontsize=8,
                va='top')
axes[1, 0].axhline(0.5 * rev.shape[1], color='0.6', lw=0.8, ls=':')
axes[1, 0].text(2, 0.5 * rev.shape[1] + 0.6, 'half the y loops',
                color='0.5', fontsize=8)

for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 2]):
    ax.set_xlabel('epoch')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc='best')
axes[0, 2].set_xlim(0, ZOOM)
axes[1, 0].set_xlim(0, ZOOM)
axes[0, 1].set_xlim(0, epochs[-1])

im = axes[1, 1].imshow(wind[:, y_loops].T, aspect='auto', interpolation='nearest',
                       extent=[epochs[0], epochs[-1], y_loops.sum(), 0],
                       cmap='RdBu_r', vmin=-1.2, vmax=1.2)
fig.colorbar(im, ax=axes[1, 1], fraction=0.03, pad=0.02,
             label='net winding about +y')
axes[1, 1].set_ylabel('y loop (target index)')
axes[1, 1].set_xlabel('epoch')
axes[1, 1].set_title('e  winding of every y loop over the run  (blue = clockwise)')

final = {a: (int(series[a]['reversed'][-1]), int(series[a]['loops'][-1]))
         for a in AXES}
fig.suptitle(f'{STEM}: ' + ', '.join(f'{a} {v[0]}/{v[1]} clockwise' for a, v in final.items())
             + f'   |   y first turns at epoch {flip_lo}, half by {flip_hi}',
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = RUN / 'y_orientation_terms.png'
fig.savefig(out, dpi=140)
print(out)

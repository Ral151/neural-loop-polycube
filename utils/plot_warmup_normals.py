#!/usr/bin/env python3
"""Figure for the warm-up normal probe, over every seed that was run.

Reads the CSVs written by warmup_normal_probe.py (seeds 0-3 at 29 planes and
4x32) and finetune_normal_probe.py and draws four panels:

  a  loops reversed per seed at 30 and 300 warm-up steps
  b  loops reversed per axis, pooled over seeds
  c  how far a query can move from a loop centroid before its sign flips
  d  reversed loops through fine-tuning, 30-step versus 300-step warm-up

Seed 1 has no perturbed/random-query CSV (it was run with --perturb 0), so
panel c pools seeds 0, 2 and 3.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('log')
SEEDS = (0, 1, 2, 3)
WARM = {s: ROOT / f'warmup_normals_s29_seed{s}_4x32' / 'warmup_normals.csv'
        for s in SEEDS}
PERTURB = {s: ROOT / f'warmup_normals_s29_seed{s}_4x32' / 'perturbed_queries.csv'
           for s in (0, 2, 3)}
RANDOM = {s: ROOT / f'warmup_normals_s29_seed{s}_4x32' / 'random_queries.csv'
          for s in (0, 2, 3)}
FT30 = ROOT / 'finetune_normals_seed1' / 'epochs.csv'
FT300 = ROOT / 'finetune_normals_seed1_p300' / 'epochs.csv'
OUT = ROOT / 'warmup_normals_s29_seed1_4x32' / 'warmup_normals.png'

FIG, AX = plt.subplots(2, 2, figsize=(13, 9))
COLOURS = {'x': '#1f77b4', 'y': '#2ca02c', 'z': '#d62728'}
STEP_COLOUR = {'30': '#ff7f0e', '300': '#9467bd'}


def rows(path):
    return list(csv.DictReader(open(path)))


warm = {}
for s, path in WARM.items():
    warm[s] = rows(path)
    for r in warm[s]:
        r['seed'] = s

n_loops = sum(1 for r in warm[0] if r['steps'] == '30')

# a  reversed loops per seed
width = 0.36
x = np.arange(len(SEEDS))
for k, steps in enumerate(('30', '300')):
    counts = [sum(r['reversed'] == 'True' for r in warm[s]
                  if r['steps'] == steps) for s in SEEDS]
    bars = AX[0, 0].bar(x + (k - 0.5) * width, counts, width,
                        color=STEP_COLOUR[steps],
                        label=f'{steps} steps')
    AX[0, 0].bar_label(bars, fontsize=9)
AX[0, 0].axhline(n_loops, color='k', lw=0.6, ls=':', alpha=0.6)
AX[0, 0].text(len(SEEDS) - 0.45, n_loops * 0.94, f'all {n_loops} loops',
              fontsize=8, ha='right', alpha=0.7)
medians = {s: {st: np.median([float(r['cos']) for r in warm[s]
                              if r['steps'] == st])
               for st in ('30', '300')} for s in SEEDS}
AX[0, 0].set_xticks(x, [f'seed {s}\n{medians[s]["30"]:+.2f} → '
                        f'{medians[s]["300"]:+.2f}' for s in SEEDS])
AX[0, 0].set_ylabel('loops reversed after the warm-up')
AX[0, 0].set_title('a  reversed loops per seed (29 planes, 4×32); '
                   'x tick: median cos 30 → 300')
AX[0, 0].legend(fontsize=8)
AX[0, 0].grid(alpha=0.25, axis='y')
AX[0, 0].set_ylim(0, n_loops * 1.12)

# b  reversed per axis, pooled over seeds
for k, steps in enumerate(('30', '300')):
    frac, err, xs = [], [], []
    for j, axis in enumerate('xyz'):
        vals = [sum(r['reversed'] == 'True' for r in warm[s]
                    if r['steps'] == steps and r['axis'] == axis)
                for s in SEEDS]
        totals = [sum(1 for r in warm[s]
                      if r['steps'] == steps and r['axis'] == axis) for s in SEEDS]
        p = np.sum(vals) / np.sum(totals)
        frac.append(p * 100)
        err.append(100 * np.sqrt(max(p * (1 - p), 1e-9) / np.sum(totals)))
        xs.append(j + (k - 0.5) * width)
    AX[0, 1].bar(xs, frac, width, yerr=err, capsize=3,
                 color=[COLOURS[a] for a in 'xyz'],
                 alpha=1.0 if steps == '30' else 0.55,
                 hatch=None if steps == '30' else '//',
                 label=f'{steps} steps (pooled)')
for j, axis in enumerate('xyz'):
    total = sum(1 for s in SEEDS for r in warm[s]
                if r['steps'] == '30' and r['axis'] == axis)
    print(f'{axis}: {total} loop slots over 4 seeds')
AX[0, 1].set_xticks(np.arange(3), [f'{a} axis' for a in 'xyz'])
AX[0, 1].set_ylabel('% reversed, pooled over 4 seeds')
AX[0, 1].set_title('b  which axis fails depends on the seed')
AX[0, 1].legend(fontsize=8)
AX[0, 1].grid(alpha=0.25, axis='y')

# c  displacement stability, pooled over seeds 0, 2, 3
pert = {s: rows(PERTURB[s]) for s in PERTURB}
rand = {s: rows(RANDOM[s]) for s in RANDOM}
for steps in ('30', '300'):
    centres, frac = [], []
    for pct in (1, 3, 6, 10):
        sub = [r for s in pert for r in pert[s]
               if r['steps'] == steps and float(r['scale']) == pct / 100]
        centres.append(pct)
        frac.append(np.mean([r['same_sign'] == 'True' for r in sub]) * 100)
    AX[1, 0].plot(centres, frac, 'o-', color=STEP_COLOUR[steps],
                  label=f'{steps} steps')
    same = [r['same_sign'] == 'True' for s in rand
            for r in rand[s] if r['steps'] == steps]
    AX[1, 0].axhline(np.mean(same) * 100, color=STEP_COLOUR[steps], ls='--',
                     lw=1, alpha=0.8,
                     label=f'{steps} steps, uniform bbox query')
AX[1, 0].axhline(50, color='k', ls=':', lw=0.8)
AX[1, 0].set_xlabel('query displacement from loop centroid (% of extent)')
AX[1, 0].set_ylabel('% keeping the circulation sign')
AX[1, 0].set_ylim(0, 105)
AX[1, 0].set_title('c  triplane orientation away from a loop centroid '
                   '(seeds 0/2/3)')
AX[1, 0].legend(fontsize=7.5, loc='lower left')
AX[1, 0].grid(alpha=0.25)

# d  fine-tuning trace
for path, label, colour in ((FT30, '30-step warm-up', '#ff7f0e'),
                            (FT300, '300-step warm-up', '#9467bd')):
    if path.exists():
        rr = [r for r in rows(path) if r['axis'] == 'all']
        e = [int(r['epoch']) for r in rr]
        rev = [int(r['reversed']) for r in rr]
        AX[1, 1].plot(e, rev, 'o-', color=colour, label=label)
AX[1, 1].set_xlabel('fine-tuning epoch')
AX[1, 1].set_ylabel('loops reversed')
AX[1, 1].set_title('d  a reversed warm-up stays reversed (seed 1)')
AX[1, 1].legend(fontsize=8)
AX[1, 1].grid(alpha=0.25)

fig = plt.gcf()
fig.suptitle('Square warm-up and loop normals: underconverged warm-up leaves '
             'the circulation to the initialisation', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=150)
print(f'wrote {OUT}')

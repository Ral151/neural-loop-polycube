"""Measure seam alignment of trained loop runs before any propagation.

The shared network is sampled at one global `z` grid, so index 0 of every
fitted loop is the network's `z = 0` seam.  Nothing in the fit ties that phase
across loops, but continuous conditioning (a triplane) might still leave
neighbouring loops with similar seams.  For each consecutive pair inside the
propagation tracks this compares

* `D0`, the normalized pair residual with no cyclic shift (seams left where the
  network put them) against `D`, the residual at the best shift;
* the best shift `delta` itself;
* the seam spread of each chain, raw and after the closed-form solve in
  `align_loops.synchronize`, normalized by the chain's mean loop radius.

Runs are finished `plc_main` outputs; the pairing is a step 2 `tracks.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _paths  # noqa: E402
from loop_propagation import (AX, OTHER, load_run, resample3)
from align_loops import (circular_shift, pair_residual, switch, synchronize,
                         wrap_turns)


def chain_segments(kept):
    segs, cur = [], [0]
    for k, kp in enumerate(kept):
        if kp:
            cur.append(k + 1)
        else:
            if len(cur) >= 2:
                segs.append(cur)
            cur = [k + 1]
    if len(cur) >= 2:
        segs.append(cur)
    return segs


def spread(seams):
    return max(float(np.linalg.norm(seams[i] - seams[j]))
               for i in range(len(seams)) for j in range(i))


def measure(run: Path, tracks, lam=0.02, min_switch=0.3):
    _, loops = load_run(run)
    by_key = {L['key']: L for L in loops}
    pairs, chains = [], []
    for t in tracks:
        axis = t['axis']
        keys = [k for k in t['keys'] if k in by_key]
        if len(keys) < 2:
            continue
        ls = [by_key[k] for k in keys]
        loops3, centers, offs = [], [], []
        for L in ls:
            r = resample3(L['block'])
            c = r.mean(0)
            x = r - c
            x[:, AX[axis]] = 0.0
            loops3.append(x)
            centers.append(c)
            offs.append(L['off'])
        centers = np.asarray(centers)
        res = synchronize(loops3, centers, offs, lam=lam, neighbors=1,
                          min_switch=min_switch)
        for p in res.pairs:
            xi, xj = loops3[p.i], loops3[p.j]
            sigma2 = 0.5 * (float(np.mean(np.sum(xi * xi, 1))) +
                            float(np.mean(np.sum(xj * xj, 1))))
            r0 = xi - xj
            D0 = float(np.mean(np.sum(r0 * r0, 1)) / max(sigma2, 1e-12))
            pairs.append(dict(track=t['id'], key_i=keys[p.i], key_j=keys[p.j],
                              delta=p.delta, D=p.D, D0=D0, s=p.s,
                              keep=bool(p.keep)))
        kept = [p.keep for p in res.pairs]
        ## how well one constant offset per loop explains the pair shifts; this
        ## is the jitter the solve cannot remove, in turns
        phase_rms = float(res.diagnostics.get('residual_rms', 0.0))
        for seg in chain_segments(kept):
            raw = np.stack([loops3[i][0][list(OTHER[axis])] for i in seg])
            shifted = np.stack([circular_shift(loops3[i], res.alpha[i])[0]
                                [list(OTHER[axis])] for i in seg])
            radius = float(np.mean([ls[i]['radius'] for i in seg]))
            alpha = wrap_turns(np.asarray([res.alpha[i] for i in seg]))
            ## the solve's absolute offset is a free gauge; only its spread
            ## across the chain says how far apart the network seams were
            alpha_spread = max(abs(wrap_turns(alpha[i] - alpha[0]))
                               for i in range(len(alpha)))
            ss = [res.pairs[k].s for k in range(len(kept))
                  if seg[0] <= k < seg[-1]]
            chains.append(dict(track=t['id'], axis=axis, n=len(seg),
                               radius=radius,
                               raw_spread=spread(raw),
                               post_spread=spread(shifted),
                               raw_rel=spread(raw) / max(radius, 1e-12),
                               post_rel=spread(shifted) / max(radius, 1e-12),
                               alpha_spread=float(alpha_spread),
                               phase_rms=phase_rms,
                               s_mean=float(np.mean(ss)) if ss else 1.0))
    return pairs, chains


def summarise(pairs, chains):
    d = np.asarray([abs(wrap_turns(p['delta'])) for p in pairs])
    D0 = np.asarray([p['D0'] for p in pairs])
    D = np.asarray([p['D'] for p in pairs])
    keep = np.asarray([p['keep'] for p in pairs])
    raw = np.asarray([c['raw_rel'] for c in chains])
    post = np.asarray([c['post_rel'] for c in chains])
    ar = np.asarray([c['alpha_spread'] for c in chains])
    pr = np.asarray([c['phase_rms'] for c in chains])
    return dict(
        pairs=int(len(pairs)), kept=int(keep.sum()),
        kept_frac=float(keep.mean()) if len(keep) else 0.0,
        median_D0=float(np.median(D0)) if len(D0) else 0.0,
        median_D=float(np.median(D)) if len(D) else 0.0,
        median_D0_over_D=float(np.median(D0 / np.maximum(D, 1e-9)))
        if len(D) else 0.0,
        median_abs_delta=float(np.median(d)) if len(d) else 0.0,
        p90_abs_delta=float(np.percentile(d, 90)) if len(d) else 0.0,
        max_abs_delta=float(np.max(d)) if len(d) else 0.0,
        frac_delta_le_0p02=float(np.mean(d <= 0.02)) if len(d) else 0.0,
        frac_delta_le_0p05=float(np.mean(d <= 0.05)) if len(d) else 0.0,
        chains=int(len(chains)),
        median_raw_rel=float(np.median(raw)) if len(raw) else 0.0,
        median_post_rel=float(np.median(post)) if len(post) else 0.0,
        median_alpha_spread=float(np.median(ar)) if len(ar) else 0.0,
        max_alpha_spread=float(np.max(ar)) if len(ar) else 0.0,
        median_phase_rms=float(np.median(pr)) if len(pr) else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='append', required=True,
                    help='name=path to a finished run, repeatable')
    ap.add_argument('--tracks-json', required=True,
                    help='step 2 tracks.json supplying the loop pairing')
    ap.add_argument('--out', default='log/align_after_training')
    ap.add_argument('--lam', type=float, default=0.02)
    ap.add_argument('--min-switch', type=float, default=0.3)
    args = ap.parse_args()

    tracks = json.loads(Path(args.tracks_json).read_text())['tracks']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report, curves = {}, {}
    for spec in args.run:
        name, path = spec.split('=', 1)
        print(f'== {name}: {path}')
        pairs, chains = measure(Path(path), tracks, args.lam,
                                args.min_switch)
        report[name] = summarise(pairs, chains)
        curves[name] = (pairs, chains)
        r = report[name]
        print(f"  pairs {r['pairs']} (kept {r['kept']}), chains {r['chains']}")
        print(f"  median D0 {r['median_D0']:.4f} -> D {r['median_D']:.4f} "
              f"(x{r['median_D0_over_D']:.1f})")
        print(f"  |delta| median {r['median_abs_delta']:.4f} turns, "
              f"p90 {r['p90_abs_delta']:.4f}, max {r['max_abs_delta']:.4f}; "
              f"{100 * r['frac_delta_le_0p02']:.0f}% <= 0.02, "
              f"{100 * r['frac_delta_le_0p05']:.0f}% <= 0.05")
        print(f"  seam spread / radius raw {r['median_raw_rel']:.4f} -> "
              f"post {r['median_post_rel']:.4f}, alpha spread "
              f"{r['median_alpha_spread']:.4f} (max "
              f"{r['max_alpha_spread']:.4f}) turns, phase rms "
              f"{r['median_phase_rms']:.4f} turns")

    (out / 'align_after_training.json').write_text(json.dumps(report, indent=1))

    names = list(report)
    colours = plt.cm.viridis(np.linspace(0.15, 0.8, len(names)))
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=150)

    ax = axs[0]
    bins = np.linspace(0, 0.5, 26)
    for name, c in zip(names, colours):
        d = [abs(wrap_turns(p['delta'])) for p in curves[name][0]]
        ax.hist(d, bins=bins, histtype='step', lw=1.5, color=c, label=name)
    ax.axvline(0.02, color='0.4', lw=1, ls=':')
    ax.set_xlabel('|best circular shift| (turns)')
    ax.set_ylabel('pairs')
    ax.set_title('how far the network seam is from alignment', fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axs[1]
    for name, c in zip(names, colours):
        D0 = [p['D0'] for p in curves[name][0]]
        D = [p['D'] for p in curves[name][0]]
        ax.scatter(D0, D, s=14, color=c, alpha=0.7, label=name)
    lim = [1e-4, 2.0]
    ax.plot(lim, lim, color='0.5', lw=1, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('D at zero shift (network seam)')
    ax.set_ylabel('D at best shift')
    ax.set_title('residual removed by shifting', fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axs[2]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, [report[n]['median_raw_rel'] for n in names], w,
           color='#bf1b1b', label='raw (z = 0)')
    ax.bar(x + w / 2, [report[n]['median_post_rel'] for n in names], w,
           color='#2a9d8f', label='after solve')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('median seam spread / loop radius')
    ax.set_title('seam scatter per chain', fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle('Alignment of trained loop runs, before propagation',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'align_after_training.png')
    print(f'wrote {out / "align_after_training.json"} and '
          f'{out / "align_after_training.png"}')


if __name__ == '__main__':
    main()

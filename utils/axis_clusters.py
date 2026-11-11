#!/usr/bin/env python3
"""Cluster one axis's loops from a finished loop_propagation run.

The propagation is already axis-pure (an edge only joins loops on adjacent
planes of the same axis), so filtering its step 1-4 outputs by axis gives the
clusters along that axis without rerunning the geometry.

    python utils/loop_propagation.py --run log/axis_seam10_armadillo \
        --out log/axis_seam10_armadillo_propagation --all --good-only
    python utils/axis_clusters.py --run log/axis_seam10_armadillo --axis z

Writes, under <propagation>/<axis>_clusters/:
  <axis>_loops.csv        one row per loop: track, chain, tube, stable, RMS
  <axis>_clusters.csv     one row per cluster (tracks plus the orphan group)
  <axis>_clusters.png     3D view, x-z and y-z side views, cluster sizes
  <axis>_clusters.ply     fitted loops coloured by track, orphans grey
"""

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
import trimesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import _paths  # noqa: E402
import loop_propagation as LP


def read_csv(path):
    return list(csv.DictReader(open(path)))


def literal(row, field):
    return ast.literal_eval(row[field])


## Stability and RMS live in step1's table, not on the loop dicts that
## `load_run` rebuilds from the PLYs.
def stable_of(step1, key):
    row = step1.get(key)
    return row is not None and row['stable'] == 'True'


def rms_of(step1, key):
    row = step1.get(key)
    return round(float(row['rms']), 6) if row is not None else ''


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', default='log/axis_seam10_armadillo')
    ap.add_argument('--prop', default=None,
                    help='propagation output (default: <run>_propagation)')
    ap.add_argument('--axis', default='z', choices='xyz')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    run = Path(args.run).resolve()
    prop = (Path(args.prop).resolve() if args.prop
            else run.with_name(run.name + '_propagation'))
    out = (Path(args.out).resolve() if args.out
           else prop / f'{args.axis}_clusters')
    out.mkdir(parents=True, exist_ok=True)

    planes, loaded = LP.load_run(run, good_only=True)
    loops = [L for L in loaded if L['axis'] == args.axis]
    step1 = {r['key']: r for r in read_csv(prop / 'step1/loops.csv')
             if r['axis'] == args.axis}
    tracks = [t for t in json.load(open(prop / 'step2/tracks.json'))['tracks']
              if t['axis'] == args.axis]
    chains = [r for r in read_csv(prop / 'step3/tracks.csv')
              if r['axis'] == args.axis]
    tubes = [r for r in read_csv(prop / 'step4/tubes.csv')
             if r['axis'] == args.axis]
    discarded = {r['key']: r for r in read_csv(prop / 'step4/discarded.csv')
                 if r['axis'] == args.axis}

    by_key = {L['key']: L for L in loops}
    track_of = {k: t['id'] for t in tracks for k in t['keys']}
    chain_of = {k: int(c['id']) for c in chains for k in literal(c, 'keys')}
    tube_of = {k: int(b['id']) for b in tubes for k in literal(b, 'keys')}
    chain_by_track = {int(c['track']): c for c in chains}
    tube_by_track = {int(b['track']): b for b in tubes}

    order = [t['id'] for t in tracks]
    colours = {tid: LP.PALETTE[i % len(LP.PALETTE)]
               for i, tid in enumerate(order)}
    orphan = '#9e9e9e'

    def colour_of(key):
        tid = track_of.get(key)
        return colours[tid] if tid is not None else orphan

    ## per-loop table
    loop_rows = []
    for L in sorted(loops, key=lambda L: (L['off'], L['key'])):
        tid = track_of.get(L['key'])
        chain = chain_of.get(L['key'])
        tube = tube_of.get(L['key'])
        loop_rows.append(dict(
            cluster=f't{tid}' if tid is not None else 'orphan',
            kind='track' if tid is not None else
                 discarded.get(L['key'], {}).get('kind', 'not reached'),
            track=tid if tid is not None else '',
            track_n=len(next(t['keys'] for t in tracks if t['id'] == tid))
                    if tid is not None else '',
            chain=chain if chain is not None else '',
            tube=tube if tube is not None else '',
            key=L['key'], plane=L['plane'], off=f"{L['off']:.4f}", k=L['k'],
            stable=stable_of(step1, L['key']),
            rms=rms_of(step1, L['key']),
            radius=round(L['radius'], 6)))
    fields = list(loop_rows[0])
    with open(out / f'{args.axis}_loops.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(loop_rows)

    ## per-cluster table
    cluster_rows = []
    for t in sorted(tracks, key=lambda t: t['id']):
        tid = t['id']
        keys = t['keys']
        c = chain_by_track.get(tid)
        b = tube_by_track.get(tid)
        if b is not None:
            status = f'tube b{b["id"]}'
        elif c is not None:
            status = 'chain, rejected by drift'
        else:
            status = 'seam cut'
        cluster_rows.append(dict(
            cluster=f't{tid}', kind='track', n=len(keys),
            n_stable=sum(t['stable']),
            off_min=f"{t['offsets'][0]:.4f}", off_max=f"{t['offsets'][-1]:.4f}",
            planes=' '.join(str(p) for p in t['planes']),
            chain=f"c{c['id']}" if c is not None else '',
            tube=f"b{b['id']}" if b is not None else '',
            status=status,
            s_min=round(float(c['s_min']), 4) if c is not None else '',
            axis_drift=round(float(c['axis_drift']), 4) if c is not None else '',
            seam_rms=round(float(c['seam_rms']), 6) if c is not None else '',
            tube_n=int(b['n']) if b is not None else 0))
    orphans = [L for L in loops if L['key'] not in track_of]
    cluster_rows.append(dict(
        cluster='orphan', kind='orphan', n=len(orphans),
        n_stable=sum(stable_of(step1, L['key']) for L in orphans),
        off_min=f"{min(L['off'] for L in orphans):.4f}" if orphans else '',
        off_max=f"{max(L['off'] for L in orphans):.4f}" if orphans else '',
        planes=' '.join(str(L['plane']) for L in
                        sorted(orphans, key=lambda L: L['plane'])),
        chain='', tube='', status='no accepted edge', s_min='', axis_drift='',
        seam_rms='', tube_n=0))
    with open(out / f'{args.axis}_clusters.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cluster_rows[0]))
        w.writeheader(); w.writerows(cluster_rows)

    ## PLY: fitted loops, resampled so one loop is a readable size
    verts, cols = [], []
    for L in loops:
        pts = LP.resample3(L['block'], 192)
        rgb = np.round(np.array(to_rgb(colour_of(L['key'])))
                       * 255).astype(np.uint8)
        verts.append(pts)
        cols.append(np.repeat(rgb[None], len(pts), axis=0))
    trimesh.PointCloud(np.vstack(verts), colors=np.vstack(cols)).export(
        out / f'{args.axis}_clusters.ply')

    ## figure
    slices = trimesh.load(LP.slices_file(run), process=False).vertices
    if len(slices) > 25000:
        slices = slices[::int(np.ceil(len(slices) / 25000))]
    fig = plt.figure(figsize=(13.5, 10.0))
    ax3 = fig.add_subplot(2, 2, 1, projection='3d')
    for L in loops:
        p = LP.resample3(L['block'], 192)
        ax3.plot(p[:, 0], p[:, 1], p[:, 2], color=colour_of(L['key']),
                 lw=0.8)
    for t in tracks:
        tid = t['id']
        pts = np.array([by_key[k]['centre'] for k in t['keys']
                        if k in by_key])
        ax3.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colours[tid],
                 lw=1.6, ls='-' if tid in tube_by_track else '--')
        ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=colours[tid],
                    s=12, depthshade=False)
    ax3.scatter(slices[:, 0], slices[:, 1], slices[:, 2], s=0.25,
                c='0.85', depthshade=False, rasterized=True)
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z')
    ax3.view_init(elev=20, azim=-58)
    rng = slices.max(0) - slices.min(0)
    ax3.set_box_aspect(tuple(rng))
    ax3.set_title(f'{args.axis}-axis loop clusters: solid = tube, '
                  f'dashed = cut', fontsize=9)

    for k, (a, name) in enumerate(((0, 'x'), (1, 'y'))):
        ax = fig.add_subplot(2, 2, k + 2)
        ax.scatter(slices[:, a], slices[:, 2], s=0.25, c='0.86',
                   rasterized=True)
        for L in loops:
            p = LP.resample3(L['block'], 128)
            ax.plot(p[:, a], p[:, 2], color=colour_of(L['key']), lw=0.55,
                    alpha=0.85)
        for t in tracks:
            tid = t['id']
            pts = np.array([by_key[k2]['centre'] for k2 in t['keys']
                            if k2 in by_key])
            ax.plot(pts[:, a], pts[:, 2], color=colours[tid], lw=1.4,
                    ls='-' if tid in tube_by_track else '--')
        ax.set_xlabel(name); ax.set_ylabel('z')
        ax.set_title(f'{args.axis} axis: {name}-z view', fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')

    ax = fig.add_subplot(2, 2, 4)
    labels, vals, cs = [], [], []
    for t in sorted(tracks, key=lambda t: t['id']):
        tid = t['id']
        labels.append(f"t{tid}")
        vals.append(len(t['keys']))
        cs.append(colours[tid])
    labels.append('orphan')
    vals.append(len(orphans))
    cs.append(orphan)
    y = np.arange(len(vals))
    bars = ax.barh(y, vals, color=cs, edgecolor='white')
    for bar, lab, v in zip(bars, labels, vals):
        tid = int(lab[1:]) if lab.startswith('t') else None
        if tid is not None:
            b = tube_by_track.get(tid)
            c = chain_by_track.get(tid)
            tag = f'b{b["id"]}' if b is not None else (
                'drift' if c is not None else 'cut')
        else:
            tag = ''
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2, f'{v} {tag}',
                va='center', fontsize=7.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel('loops in cluster')
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_title('cluster sizes; b = surviving tube', fontsize=9)
    ax.grid(alpha=0.25, axis='x')

    n_tube = len(tubes)
    n_tube_loops = sum(int(b['n']) for b in tubes)
    fig.suptitle(f'{run.name} — {args.axis} axis: {len(loops)} good loops, '
                 f'{len(tracks)} tracks, {n_tube} tubes '
                 f'({n_tube_loops} loops), {len(orphans)} orphans',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f'{args.axis}_clusters.png', dpi=170)
    plt.close(fig)

    print(f'{run.name} — {args.axis} axis')
    print(f'  good loops {len(loops)} '
          f'({sum(stable_of(step1, L["key"]) for L in loops)} stable)')
    print(f'  tracks {len(tracks)}, tubes {n_tube} ({n_tube_loops} loops), '
          f'orphans {len(orphans)}')
    header = ('cluster', 'n', 'stable', 'offset range', 'chain', 'tube',
              'status', 's_min', 'drift')
    print('  ' + '  '.join(f'{h:>13}' for h in header))
    for r in cluster_rows:
        vals = (r['cluster'], r['n'], r['n_stable'],
                f"{r['off_min']}..{r['off_max']}", r['chain'] or '-',
                r['tube'] or '-', r['status'], r['s_min'] or '-',
                r['axis_drift'] or '-')
        print('  ' + '  '.join(f'{str(v):>13}' for v in vals))
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()

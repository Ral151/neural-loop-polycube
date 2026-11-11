#!/usr/bin/env python3
"""Draw a finished run's fitted loops coloured by their code-cluster label.

The clusters in `<run>/*_code_clusters.csv` were computed from the loop
centroids' triplane codes.  This script joins those labels back to the fitted
loops in the per-plane PLYs and writes:

- `<out>.ply`                 all loops as thin tubes, coloured by cluster;
- `<out>_c<k>.ply`            one file per cluster, for isolating a group;
- `<out>.png`                 six views of all loops by cluster;
- `<out>_groups.png`          one panel per cluster (other loops in grey);
- `<out>.csv`                 loop key -> cluster;
- `<out>.txt`                 per-cluster axis/plane/offset summary.

`--groups-only` writes just `<out>_groups.png` and skips every other output.

    python utils/cluster_loops_viz.py --run log/axis_seam10_laurent_hand
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import trimesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import loop_propagation as lp


VIEWS = ((20, -60), (20, 120), (70, 30), (-70, 30), (0, 0), (0, 90))


def find_clusters(run: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f'no cluster csv: {path}')
        return path
    plain = run / f'{run.name}_code_clusters.csv'
    if plain.is_file():
        return plain
    hits = sorted(run.glob('*_code_clusters.csv'))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f'{run}: no *_code_clusters.csv; run code_clusters.py first')
    raise SystemExit(f'{run}: several cluster csvs, pass --clusters: '
                     + ', '.join(p.name for p in hits))


def read_clusters(path: Path):
    rows = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(
                axis=row['axis'],
                offset=float(row['offset']),
                slot=int(row['slot']),
                xyz=np.array([float(row['x']), float(row['y']), float(row['z'])]),
                cluster=int(row['cluster']),
            ))
    return rows


def index_clusters(rows):
    index = {}
    for row in rows:
        index.setdefault((row['axis'], row['slot']), []).append(row)
    for value in index.values():
        value.sort(key=lambda r: r['offset'])
    return index


def label_loop(loop, index):
    cands = index.get((loop['axis'], loop['k']), [])
    if not cands:
        return None
    row = min(cands, key=lambda r: abs(r['offset'] - loop['off']))
    return row['cluster'] if abs(row['offset'] - loop['off']) < 2e-3 else None


def resampled(loop, m):
    return lp.resample3(loop['block'], m)


def loop_mesh(loop, colour, radius, points):
    verts, faces = lp.ring_tube(resampled(loop, points), radius,
                                nring=6, up=loop['normal'])
    return lp.colored_mesh(verts, faces, colour)


def extent(loops):
    pts = np.vstack([L['block'] for L in loops])
    lo, hi = pts.min(0), pts.max(0)
    return lo, hi, 0.5 * (lo + hi), float(np.linalg.norm(hi - lo))


def set_axes(ax, centre, radius, view, zoom=1.0):
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(*view)
    ax.set_axis_off()


def draw_loops(ax, loops, colours, points, width=0.75, alpha=0.95):
    for L in loops:
        p = resampled(L, points)
        p = np.vstack([p, p[:1]])
        ax.plot(p[:, 0], p[:, 1], p[:, 2],
                color=colours[L['cluster']], lw=width, alpha=alpha)


def cluster_centres(loops, k):
    out = []
    for c in range(k):
        sel = [L['centre'] for L in loops if L['cluster'] == c]
        out.append(np.mean(sel, axis=0) if sel else None)
    return out


def cluster_view(loops, diag):
    """Centre and radius that frame one cluster and not the whole shape."""
    lo, hi, centre, cdiag = extent(loops)
    return centre, max(0.62 * cdiag, 0.08 * diag)


PAIRS = ((0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ'))
GREY = (0.72, 0.72, 0.72)


def draw_projection(ax, loops, sel, colours, c, points, pair, bounds):
    i, j, name = pair
    for L in loops:
        if L['cluster'] == c:
            continue
        p = resampled(L, points)
        ax.plot(p[:, i], p[:, j], color=GREY, lw=0.45, alpha=0.55)
    for L in sel:
        p = resampled(L, points)
        p = np.vstack([p, p[:1]])
        ax.plot(p[:, i], p[:, j], color=colours[c], lw=1.0, alpha=1.0)
    ax.set_xlim(bounds[0][i], bounds[1][i])
    ax.set_ylim(bounds[0][j], bounds[1][j])
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f'{name}', fontsize=8)


def dominant_pair(sel):
    lo, hi, _, _ = extent(sel)
    order = np.argsort(hi - lo)[::-1]
    i, j = int(order[0]), int(order[1])
    for pair in PAIRS:
        if set(pair[:2]) == {i, j}:
            return pair
    return PAIRS[0]


def page_bounds(sel, diag):
    """Frame one cluster tightly so tiny groups do not vanish in the full bbox."""
    lo, hi, _, cdiag = extent(sel)
    pad = max(0.12 * cdiag, 0.03 * diag)
    return lo - pad, hi + pad


def draw_cluster_page(path, run, loops, colours, sel, c, points, diag,
                      views=((20, -60),)):
    centre, radius = cluster_view(sel, diag)
    bounds = page_bounds(sel, diag)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.95))
    for j, pair in enumerate(PAIRS):
        ax = fig.add_subplot(gs[0, j])
        draw_projection(ax, loops, sel, colours, c, points, pair, bounds)
    ax = fig.add_subplot(gs[1, 1], projection='3d')
    others = [L for L in loops if L['cluster'] != c]
    draw_loops(ax, others, [GREY] * len(colours), points,
               width=0.40, alpha=0.55)
    draw_loops(ax, sel, colours, points, width=1.5, alpha=1.0)
    set_axes(ax, centre, radius, views[0], zoom=1.4)
    ax.set_title(f'perspective {views[0]}', fontsize=8)
    axes = {a: sum(L['axis'] == a for L in sel) for a in 'xyz'}
    fig.suptitle(f"{run}: cluster c{c}, {len(sel)} loops "
                 f"({' '.join(f'{a}{n}' for a, n in axes.items() if n)}) — "
                 f"others grey", fontsize=11, color=colours[c])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.04, hspace=0.12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', required=True, help='finished run folder')
    ap.add_argument('--clusters', default=None,
                    help='cluster csv (default: <run>/<run>_code_clusters.csv)')
    ap.add_argument('--out', default=None,
                    help='output stem (default: <run>/<run>_cluster_loops)')
    ap.add_argument('--points', type=int, default=96,
                    help='points per loop in the figures and PLYs')
    ap.add_argument('--radius', type=float, default=None,
                    help='tube radius (default: 0.0025 x bbox diagonal)')
    ap.add_argument('--no-split-ply', action='store_true',
                    help='skip the one-PLY-per-cluster files')
    ap.add_argument('--groups-only', action='store_true',
                    help='write only <out>_groups.png; skip the PLYs, the '
                         'six-view render, the per-cluster pages and csv/txt')
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_dir():
        raise SystemExit(f'no run folder: {run}')
    cluster_path = find_clusters(run, args.clusters)
    out = Path(args.out) if args.out else run / f'{run.name}_cluster_loops'
    out.parent.mkdir(parents=True, exist_ok=True)

    _, loops = lp.load_run(run)
    clusters = read_clusters(cluster_path)
    index = index_clusters(clusters)
    missing = []
    for L in loops:
        c = label_loop(L, index)
        if c is None:
            missing.append(L['key'])
            c = -1
        L['cluster'] = c
    k = max(L['cluster'] for L in loops) + 1
    unmatched = sum(L['cluster'] < 0 for L in loops)
    if unmatched:
        print(f'warning: {unmatched} loops have no cluster row '
              f'(coloured grey): {", ".join(missing[:8])}')

    pal = plt.get_cmap('tab20')
    colours = [tuple(np.asarray(pal(c % 20))[:3]) for c in range(k)]
    colours.append((0.65, 0.65, 0.65))
    grey = (0.80, 0.80, 0.80)
    lo, hi, centre, diag = extent(loops)
    radius = args.radius if args.radius is not None else 0.0025 * diag
    sphere = 0.004 * diag

    if not args.groups_only:
        # combined tube PLY and one file per cluster
        parts = []
        per_cluster = {c: [] for c in range(k)}
        for L in loops:
            rgb = np.round(np.array(colours[L['cluster']]) * 255).astype(np.uint8)
            mesh = loop_mesh(L, rgb, radius, args.points)
            parts.append(mesh)
            per_cluster.setdefault(L['cluster'], []).append(mesh)
            marker = trimesh.creation.icosphere(subdivisions=1, radius=sphere)
            marker.apply_translation(L['centre'])
            marker.visual.vertex_colors = np.tile(
                np.append(rgb, 255).astype(np.uint8), (len(marker.vertices), 1))
            parts.append(marker)
            per_cluster[L['cluster']].append(marker)
        trimesh.util.concatenate(parts).export(out.with_suffix('.ply'))
        if not args.no_split_ply:
            for c, group in sorted(per_cluster.items()):
                name = f'{out.name}_c{c}' if c >= 0 else f'{out.name}_unmatched'
                trimesh.util.concatenate(group).export(out.with_name(name + '.ply'))

        # six views of the full set
        fig = plt.figure(figsize=(18, 11))
        for si, view in enumerate(VIEWS):
            ax = fig.add_subplot(2, 3, si + 1, projection='3d')
            draw_loops(ax, loops, colours, args.points)
            set_axes(ax, centre, 0.62 * diag, view, zoom=1.45)
            ax.set_title(f'elev {view[0]}  azim {view[1]}', fontsize=9)
        handles = [plt.Line2D([], [], color=colours[c], lw=2,
                              label=f"c{c} ({sum(L['cluster'] == c for L in loops)})")
                   for c in range(k)]
        fig.legend(handles=handles, loc='lower center', ncol=min(k, 8),
                   frameon=False, fontsize=9)
        fig.suptitle(f'{run.name}: {len(loops)} fitted loops, {k} code clusters '
                     f'from {cluster_path.name}', fontsize=12)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(out.with_suffix('.png'), dpi=160)
        plt.close(fig)

    # one panel per cluster, each in its best 2-D projection, others in grey
    centres = cluster_centres(loops, k)
    bounds = (lo, hi)
    rows = int(np.ceil(k / 2))
    fig = plt.figure(figsize=(16, 3.4 * rows))
    for c in range(k):
        sel = [L for L in loops if L['cluster'] == c]
        ax = fig.add_subplot(rows, 2, c + 1)
        pair = dominant_pair(sel)
        draw_projection(ax, loops, sel, colours, c, args.points, pair, bounds)
        axes = {a: sum(L['axis'] == a for L in sel) for a in 'xyz'}
        ax.set_title(f"{pair[2]}  c{c}: {len(sel)} loops "
                     f"({' '.join(f'{a}{n}' for a, n in axes.items() if n)})",
                     fontsize=10, color=colours[c])
        if not args.groups_only:
            draw_cluster_page(out.with_name(f'{out.name}_c{c}.png'), run.name,
                              loops, colours, sel, c, args.points, diag)
    for j in range(k + 1, 2 * rows + 1):
        fig.add_subplot(rows, 2, j).axis('off')
    fig.suptitle(f'{run.name}: each centroid cluster framed on its own loops '
                 f'(others grey)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out.with_name(out.name + '_groups.png'), dpi=150)
    plt.close(fig)

    if not args.groups_only:
        rows = []
        lines = [f'run: {run}', f'clusters: {cluster_path}', f'loops: {len(loops)}',
                 f'clusters used: {k - (1 if unmatched else 0)}'
                 + (f' (+{unmatched} unmatched)' if unmatched else '')]
        for c in range(k):
            sel = [L for L in loops if L['cluster'] == c]
            axes = {a: [L for L in sel if L['axis'] == a] for a in 'xyz'}
            desc = []
            for a, group in axes.items():
                if group:
                    offs = sorted({round(L['off'], 3) for L in group})
                    desc.append(f'{a}: {len(group)} loops on {len(offs)} planes '
                                f'[{offs[0]:+.3f}, {offs[-1]:+.3f}]')
            ctr = centres[c]
            lines.append(f'  c{c}: {len(sel)} loops  '
                         + '; '.join(desc)
                         + (f'  centre {np.round(ctr, 3)}' if ctr is not None else ''))
            for L in sel:
                rows.append((L['key'], L['axis'], L['off'], L['k'], c))
        (out.with_suffix('.txt')).write_text('\n'.join(lines) + '\n')
        with open(out.with_suffix('.csv'), 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['key', 'axis', 'offset', 'slot', 'cluster'])
            writer.writerows(rows)
        print('\n'.join(lines))
        print(f'wrote {out.with_suffix(".ply")}')
        print(f'wrote {out.with_suffix(".png")}')
        print(f'wrote {out.with_suffix(".txt")} and {out.with_suffix(".csv")}')
    print(f'wrote {out.with_name(out.name + "_groups.png")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

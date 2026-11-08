"""Mark the z = 0 seam of every good loop of a fit run.

Vertex 0 of each loop block is its z = 0. Writes, next to <stem>_good_loops.ply:

    <stem>_good_loops_z0.ply   good loops with the seam painted green, plus a
                               green octahedron at every seam
    <stem>_good_loops_z0.png   four views: loops light blue, seams green, the
                               first tenth of each loop orange (its direction)

--sheet also draws every run's isometric view in one grid: a row per shape,
a column per run setting (the run path without the shape name).

    python main.py seam-view log/my_run                   # one run
    python main.py seam-view 'log/axis_*' --sheet log/seam_sheet.png
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh


# colours of the loops, the seam and the first stretch of each loop
LOOP = '#9ecae1'
SEAM = '#00b050'
ARC = '#ff7f0e'
MARKER_RADIUS = 0.012      # fraction of the largest extent
ARC_FRACTION = 0.10         # first part of the loop drawn in orange
SEAM_FRACTION = 0.01        # first part of the loop painted green in the PLY


# reading a run
def run_geometry(run: Path):
    """Good-loop vertices `(L, M, 3)`, their colours `(L, M, 4)` and M."""
    good = sorted(run.glob('*_good_loops.ply'))
    if not good:
        raise FileNotFoundError(f'no *_good_loops.ply in {run}')
    ply = good[0]
    mesh = trimesh.load(ply, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8).copy()

    # M = good vertices / good loops (from _loop_split.txt); older runs
    # without that file read M from the console log beside the run folder
    split = ply.with_name(ply.name.replace('_good_loops.ply', '_loop_split.txt'))
    log = Path(str(run) + '.log')
    if split.exists():
        n_good = split.read_text().strip().count('g')
        M = len(verts) // max(n_good, 1)
    elif log.exists():
        found = re.findall(r'M = (\d+) points each', log.read_text())
        if not found:
            raise ValueError(f'{log} does not say how many points each loop has')
        M = int(found[-1])
    else:
        raise FileNotFoundError(f'neither {split.name} nor {log} is there to say '
                                f'how many points each loop has')

    if len(verts) % M:
        raise ValueError(f'{ply}: {len(verts)} vertices do not split into blocks '
                         f'of M = {M}')
    return verts.reshape(-1, M, 3), colors.reshape(-1, M, 4), M


def seam_markers(seams: np.ndarray, radius: float):
    """Octahedron vertices and faces standing at every seam point."""
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                     [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    verts = (seams[:, None, :] + radius * dirs[None]).reshape(-1, 3)
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    n = len(seams)
    return verts, (faces[None] + 6 * np.arange(n)[:, None, None]).reshape(-1, 3)

# drawing
def draw(ax, loops: np.ndarray, elev: float, azim: float, title: str = ''):
    """Draw the loops, their z = 0 points and the first bit of each loop."""
    L, M, _ = loops.shape
    step = max(1, M // 160)
    idx = np.arange(0, M, step)
    for loop in loops:
        b = loop[np.r_[idx, 0]]
        ax.plot(b[:, 0], b[:, 1], b[:, 2], color=LOOP, lw=0.5, alpha=0.9)
        arc = loop[:max(3, int(ARC_FRACTION * M)) + 1]
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=ARC, lw=1.3, alpha=0.95)
    seams = loops[:, 0, :]
    ax.scatter(seams[:, 0], seams[:, 1], seams[:, 2], s=16, c=SEAM,
               edgecolors='none', depthshade=False, zorder=5)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect(np.ptp(loops.reshape(-1, 3), axis=0))
    except Exception:
        pass
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)


def draw_run(run: Path, out_png: Path, dpi: int = 150):
    """Write the four-view seam figure of a run; returns the loop count."""
    loops, _, _ = run_geometry(run)
    L = loops.shape[0]
    fig = plt.figure(figsize=(11, 9.5), dpi=dpi)
    views = [('isometric', 22, -55), ('front', 8, -90),
             ('side', 8, 0), ('top', 90, -90)]
    for k, (name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, k, projection='3d')
        draw(ax, loops, elev, azim, name)
    fig.suptitle(f'{run.name} — {L} good loops, {loops.shape[1]} samples each; '
                 f'green = z = 0, orange = z in [0, {ARC_FRACTION:g}]', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, facecolor='white')
    plt.close(fig)
    return L


def write_ply(run: Path):
    """Write <stem>_good_loops_z0.ply (seams painted and marked); returns its path."""
    loops, colors, _ = run_geometry(run)
    seams = loops[:, 0, :]
    extent = float(np.ptp(loops.reshape(-1, 3), axis=0).max())
    mverts, mfaces = seam_markers(seams, MARKER_RADIUS * extent)

    flat = loops.reshape(-1, 3)
    out = np.vstack([flat, mverts])
    cout = np.vstack([colors.reshape(-1, 4),
                      np.tile(np.array([0, 176, 80, 255], np.uint8),
                              (len(mverts), 1))])
    # also paint the first samples, for viewers that show only the points
    M = loops.shape[1]
    head = max(2, int(SEAM_FRACTION * M))
    first = (np.arange(len(seams))[:, None] * M
             + np.arange(head)[None, :]).reshape(-1)
    cout[first] = [0, 176, 80, 255]

    mesh = trimesh.Trimesh(vertices=out, faces=mfaces, vertex_colors=cout,
                           process=False)
    p = Path(str(run / sorted(run.glob('*_good_loops.ply'))[0].name
                 .replace('_good_loops.ply', '_good_loops_z0.ply')))
    mesh.export(p)
    return p

# contact sheet
def sheet(runs: list[Path], out_png: Path, dpi: int = 150):
    """Grid of isometric seam views: one row per shape, one column per run setting.

    The shape is the stem of a run's *_good_loops.ply; the setting (arm) is the run
    path with that stem removed, e.g. log/axis_seam10_<shape> -> log/axis_seam10.
    """
    named = {}
    for run in runs:
        shape = sorted(run.glob('*_good_loops.ply'))[0].name[:-len('_good_loops.ply')]
        arm = str(run).replace(shape, '').replace('//', '/').strip('_/') or run.name
        named[(shape, arm)] = run
    if not named:
        print('  sheet: no runs to draw; skipped')
        return
    shapes = sorted({s for s, _ in named})
    arms = sorted({a for _, a in named})
    fig, axs = plt.subplots(len(shapes), len(arms),
                            figsize=(4.2 * len(arms), 4.0 * len(shapes)),
                            dpi=dpi, subplot_kw={'projection': '3d'},
                            squeeze=False)
    for r, shape in enumerate(shapes):
        for c, arm in enumerate(arms):
            ax = axs[r][c]
            run = named.get((shape, arm))
            if run is None:
                ax.set_axis_off()
                ax.set_title(f'{shape} — {arm}\n(missing)', fontsize=9)
                continue
            loops, _, _ = run_geometry(run)
            draw(ax, loops, 22, -55,
                 f'{shape}\n{arm} — {len(loops)} loops')
    fig.suptitle('z = 0 seam on good loops (green dot; orange = first tenth)',
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, facecolor='white')
    plt.close(fig)


# main
def main(argv=None):
    """Write the seam PLY and figure of every matched run, and the optional sheet."""
    import argparse
    ap = argparse.ArgumentParser(prog='main.py seam-view',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('runs', nargs='+', help='run directories or globs')
    ap.add_argument('--sheet', default='', help='contact-sheet PNG to write')
    ap.add_argument('--dpi', type=int, default=150, help='figure resolution')
    a = ap.parse_args(argv)

    runs = []
    for pattern in a.runs:
        runs += [Path(p) for p in sorted(glob.glob(pattern))]
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        raise SystemExit('no run directories matched')

    done = []
    for run in runs:
        if not list(run.glob('*_good_loops.ply')):
            print(f'skip {run.name}: no good loops')
            continue
        loops, _, _ = run_geometry(run)
        p = write_ply(run)
        png = p.with_suffix('.png')
        draw_run(run, png, a.dpi)
        print(f'{run.name}: {len(loops)} loops -> {p.name}, {png.name}')
        done.append(run)

    if a.sheet:
        sheet(done, Path(a.sheet), a.dpi)
        print(f'sheet -> {a.sheet}')

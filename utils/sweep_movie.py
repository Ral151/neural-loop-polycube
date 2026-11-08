"""Movie revealing a run's good loops plane by plane along x, y and z.

Three panels sweep one axis each from low to high offset (the newest plane in
orange); the fourth shows all three together and rotates at the end. Needs
`ffmpeg` on the PATH for the mp4.

    python main.py sweep-movie --run log/my_run --preview   # one PNG
    python main.py sweep-movie --run log/my_run             # <run>/loops_axes_sweep.mp4
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.propagation import load_run


# loop colour, newest-plane colour and per-axis colours; every STRIDE-th point is drawn
BLUE = '#3b6ea5'
NEW = '#ff7f0e'
AXCOL = {'x': '#4575b4', 'y': '#2a9d8f', 'z': '#d1495b'}
STRIDE = 8


def load_loops(run: Path):
    """Good loops of a run, and their thinned closed polylines grouped by axis and offset."""
    _, loops = load_run(run, good_only=True)
    groups = {a: {} for a in 'xyz'}
    for L in loops:
        b = L['block']
        p = np.vstack([b[::STRIDE], b[:1]])
        groups[L['axis']].setdefault(L['off'], []).append(p)
    for a in groups:
        groups[a] = dict(sorted(groups[a].items()))
    return loops, groups


def build(run: Path):
    """Build the four-panel figure with every loop line hidden."""
    loops, groups = load_loops(run)
    allv = np.vstack([L['block'] for L in loops])
    lo, hi = allv.min(0), allv.max(0)
    mid, span = (lo + hi) / 2, np.maximum(hi - lo, 1e-6)

    fig = plt.figure(figsize=(12, 9), dpi=110)
    axs, art = {}, {}
    views = {'x': (20, -45), 'y': (20, -135), 'z': (18, -60)}
    for k, a in enumerate('xyz', 1):
        ax = fig.add_subplot(2, 2, k, projection='3d')
        axs[a] = ax
        art[a] = {}
        for off, polys in groups[a].items():
            art[a][off] = [ax.plot(p[:, 0], p[:, 1], p[:, 2], color=BLUE,
                                   lw=0.65, visible=False)[0] for p in polys]
        ax.view_init(*views[a])
        ax.set_box_aspect(span)
        ax.set_xlim(mid[0] - span[0] / 2, mid[0] + span[0] / 2)
        ax.set_ylim(mid[1] - span[1] / 2, mid[1] + span[1] / 2)
        ax.set_zlim(mid[2] - span[2] / 2, mid[2] + span[2] / 2)
        ax.set_axis_off()

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    axs['all'] = ax
    union = []
    for L in loops:
        b = L['block']
        p = np.vstack([b[::STRIDE], b[:1]])
        ln = ax.plot(p[:, 0], p[:, 1], p[:, 2],
                     color=AXCOL[L['axis']], lw=0.65, visible=False)[0]
        union.append((L['axis'], L['off'], ln))
    ax.view_init(20, -90)
    ax.set_box_aspect(span)
    ax.set_xlim(mid[0] - span[0] / 2, mid[0] + span[0] / 2)
    ax.set_ylim(mid[1] - span[1] / 2, mid[1] + span[1] / 2)
    ax.set_zlim(mid[2] - span[2] / 2, mid[2] + span[2] / 2)
    ax.set_axis_off()

    fig.suptitle(f'{run.name} — {len(loops)} good loops revealed plane by plane '
                 f'(newest plane orange)', fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, axs, art, union, groups


def reveal(u: float, n: int) -> int:
    """Index of the last revealed plane at sweep progress u in [0, 1]."""
    return int(round(np.clip(u, 0.0, 1.0) * (n - 1)))


def set_frame(axs, art, union, groups, f, sweep_frames, hold_frames,
              rotate_frames):
    """Show the loops revealed at frame f; rotate the union panel after the hold."""
    u = 1.0 if f >= sweep_frames else f / max(sweep_frames - 1, 1)
    k = {}
    for a in 'xyz':
        offs = list(groups[a])
        k[a] = reveal(u, len(offs))
        for j, off in enumerate(offs):
            vis = j <= k[a]
            new = vis and j == k[a] and u < 1.0
            for ln in art[a][off]:
                ln.set_visible(vis)
                ln.set_color(NEW if new else BLUE)
                ln.set_linewidth(1.5 if new else 0.65)
        axs[a].set_title(f'{a} loops — {k[a] + 1}/{len(offs)} planes '
                         f'(low → high)', fontsize=9)
    for a, off, ln in union:
        ln.set_visible(list(groups[a]).index(off) <= k[a])
    axs['all'].set_title('all axes — x blue, y green, z red', fontsize=9)

    start = sweep_frames + hold_frames
    if f >= start and rotate_frames > 1:
        r = min(1.0, (f - start) / (rotate_frames - 1))
        axs['all'].view_init(elev=20, azim=-90 + 360 * r)


def main(argv=None):
    """Write the sweep movie (or a single preview PNG) of a fit run."""
    ap = argparse.ArgumentParser(prog='main.py sweep-movie',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--run', required=True, help='a finished `main.py fit` run folder')
    ap.add_argument('--out', default='',
                    help='output mp4 (default <run>/loops_axes_sweep.mp4)')
    ap.add_argument('--fps', type=int, default=24, help='frames per second')
    ap.add_argument('--seconds', type=float, default=10.0, help='movie length')
    ap.add_argument('--preview', action='store_true',
                    help='only write one mid-sweep PNG')
    a = ap.parse_args(argv)

    run = Path(a.run)
    out = Path(a.out) if a.out else run / 'loops_axes_sweep.mp4'
    total = max(int(a.fps * a.seconds), 24)
    # frame budget: sweep, a short hold, then about 38% rotating
    hold = 12
    sweep = total - hold - int(total * 0.38)
    rotate = total - hold - sweep

    fig, axs, art, union, groups = build(run)
    if a.preview:
        set_frame(axs, art, union, groups, int(sweep * 0.62),
                  sweep, hold, rotate)
        png = out.with_suffix('.png')
        fig.savefig(png, facecolor='white')
        print(f'preview -> {png}')
        return

    # stream raw RGBA frames into ffmpeg
    W, H = fig.canvas.get_width_height()
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-s', f'{W}x{H}', '-pix_fmt', 'rgba',
           '-r', str(a.fps), '-i', '-', '-an', '-vcodec', 'libx264',
           '-pix_fmt', 'yuv420p', '-crf', '20', str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(total):
        set_frame(axs, art, union, groups, f, sweep, hold, rotate)
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if f % 24 == 0:
            print(f'  frame {f + 1}/{total}', flush=True)
    proc.stdin.close()
    if proc.wait():
        raise SystemExit('ffmpeg failed')
    print(f'movie -> {out} ({total} frames, {a.fps} fps)')

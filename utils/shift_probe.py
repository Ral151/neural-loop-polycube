"""Probe how the field generalises: decode a loop from a query moved along its axis.

Each loop's triplane query is shifted by a fraction of the plane spacing along
the plane normal; the decoded curve is compared with the loop actually fitted on
the neighbouring plane (Chamfer, coverage, axial placement). Needs a triplane
checkpoint; writes <run>/loop_shift/.

    python main.py shift-probe --run log/my_run
    python main.py shift-probe --run log/my_run --loops 3,17,42
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

from model.trainer import load_model


def resolve(run, model=None, loop_ply=None):
    """(checkpoint, combined loops PLY or None when only the good / bad split exists)."""
    run = Path(run)
    model = Path(model) if model else next(iter(sorted(run.glob('*_model.pt'))))
    if loop_ply:
        return model, Path(loop_ply)
    stem = model.name[:-len('_model.pt')]
    combined = model.with_name(f'{stem}_loops.ply')
    return model, combined if combined.is_file() else None


def read_loops(path, n_loops, block):
    """Fitted loops from a combined PLY, as n_loops x block x 3."""
    vertices = np.asarray(trimesh.load(path, process=False).vertices)
    if len(vertices) != n_loops * block:
        raise SystemExit(f'{path}: {len(vertices)} vertices, expected '
                         f'{n_loops} x {block} = {n_loops * block}')
    return vertices.reshape(n_loops, block, 3)


def read_split_loops(run, model_path, n_loops, block):
    """Rebuild the checkpoint's loop order from the good / bad PLYs and *_loop_split.txt."""
    run = Path(run)
    stem = Path(model_path).name[:-len('_model.pt')]
    flags_path = run / f'{stem}_loop_split.txt'
    if not flags_path.is_file():
        raise SystemExit(f'{run}: no {stem}_loops.ply and no '
                         f'{stem}_loop_split.txt; pass --loop-ply')
    flags = flags_path.read_text().strip()
    if len(flags) != n_loops or set(flags) - {'g', 'b'}:
        raise SystemExit(f'{flags_path}: expected {n_loops} g/b flags, '
                         f'got {len(flags)} characters')
    sides = {}
    for flag, name in (('g', 'good'), ('b', 'bad')):
        p = run / f'{stem}_{name}_loops.ply'
        if not p.is_file():
            sides[flag] = np.zeros((0, block, 3))
            continue
        v = np.asarray(trimesh.load(p, process=False).vertices)
        if len(v) % block:
            raise SystemExit(f'{p}: {len(v)} vertices do not split into '
                             f'blocks of {block}')
        sides[flag] = v.reshape(-1, block, 3)
    if sum(len(v) for v in sides.values()) != n_loops:
        raise SystemExit(f'{run}: split loops hold {sum(len(v) for v in sides.values())} '
                         f'blocks, expected {n_loops}')
    pos = {'g': 0, 'b': 0}
    fitted = np.empty((n_loops, block, 3))
    for i, flag in enumerate(flags):
        fitted[i] = sides[flag][pos[flag]]
        pos[flag] += 1
    return fitted


def plane_basis(normal):
    """In-plane basis (u, v): u = the least-aligned world axis projected into the plane."""
    a = int(np.argmin(np.abs(normal)))
    u = np.zeros(3)
    u[a] = 1.0
    u = u - normal * (u @ normal)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def decode(model, z, query, phase=0.0, axis=None):
    """Decode one loop at a world query point, with its seam phase and slice axis."""
    q = torch.as_tensor(query, dtype=torch.float32, device=z.device)[None]
    with torch.no_grad():
        feat = model.triplane(q)
        a = None if axis is None else torch.as_tensor(
            axis, dtype=torch.float32, device=z.device)[None]
        return model(torch.remainder(z + phase, 1.0)[None], None, feat, a)[0].cpu().numpy()


def chamfer(a, b):
    """Symmetric mean Chamfer distance between two point sets."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def coverage(a, b, tau):
    """Fraction of a within tau of some point of b."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float((d.min(1) <= tau).mean())


def area_2d(points, u, v):
    """Signed area of a closed curve projected onto (u, v)."""
    p = np.stack([points @ u, points @ v], 1)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def neighbours(keys, queries, plane_normal, offsets, axis, i, spacing):
    """Loop indices one plane before and after loop i (centroid nearest the shifted centroid)."""
    out = []
    for t in (1.0, -1.0):
        m = ((keys[:, 0] == axis)
             & (np.abs(keys[:, 1] - (offsets[i] + t * spacing)) < 0.25 * spacing))
        cand = np.where(m)[0]
        if not len(cand):
            out.append(None)
            continue
        target = queries[i] + t * spacing * plane_normal[i]
        out.append(cand[np.argmin(np.linalg.norm(queries[cand] - target, axis=1))])
    return out[0], out[1]


def main(argv=None):
    """Run the shift probe on a fit run and write its tables and figures."""
    ap = argparse.ArgumentParser(prog='main.py shift-probe', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True,
                    help='run folder with *_model.pt and the loop PLYs')
    ap.add_argument('--model', default=None,
                    help='checkpoint to probe (default: the run\'s *_model.pt)')
    ap.add_argument('--loop-ply', default=None,
                    help='loops PLY to compare against (default: alongside the model)')
    ap.add_argument('--loops', default=None, help='comma-separated loop indices to draw')
    ap.add_argument('--out', default=None,
                    help='output folder (default <run>/loop_shift)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='torch device (default cuda when available)')
    ap.add_argument('--radius', type=float, default=0.02,
                    help='nearest-sample radius for the coverage test, world units')
    args = ap.parse_args(argv)

    model_path, loops_path = resolve(args.run, args.model, args.loop_ply)
    out = Path(args.out) if args.out else Path(args.run) / 'loop_shift'
    out.mkdir(parents=True, exist_ok=True)

    model, meta = load_model(model_path, device=args.device)
    if meta['feature_mode'] != 'triplane':
        raise SystemExit('this probe needs a triplane checkpoint')
    keys = np.asarray(meta['keys'])
    plane_normal = np.asarray(meta['plane_normal'], np.float64)
    plane_normal /= np.linalg.norm(plane_normal, axis=1, keepdims=True) + 1e-12
    plane_point = np.asarray(meta['plane_point'], np.float64)
    if meta['queries'] is None:
        raise SystemExit('this checkpoint has no query points; re-run main.py fit to save them')
    queries = np.asarray(meta['queries'], np.float64)
    block = int(meta['latent_points'])
    fitted = (read_loops(loops_path, len(keys), block) if loops_path else
              read_split_loops(args.run, model_path, len(keys), block))
    z = torch.linspace(0, 1, block + 1, device=args.device)[:-1, None]
    # loops were exported at their seam phases, so decode at the same phases
    phases = meta.get('phases')
    phases = np.zeros(len(keys)) if phases is None else np.asarray(phases, np.float64)

    axis = keys[:, 0].astype(int)
    offsets = keys[:, 1]
    spacing = np.zeros(3)
    for a in range(3):
        lv = np.unique(offsets[axis == a])
        spacing[a] = float(np.median(np.diff(np.sort(lv)))) if len(lv) > 1 else 1.0

    def shift(i, t):
        """Decode loop i with its query moved t plane spacings: (plane-local, world) curve."""
        q = meta['queries'][i] + t * spacing[axis[i]] * plane_normal[i]
        curve = decode(model, z, q, float(phases[i]), plane_normal[i])
        local, world = curve, curve + plane_point[i] + t * spacing[axis[i]] * plane_normal[i]
        return local, world

    # sanity: t = 0 must reproduce the stored fit
    zero_err = max(np.abs(shift(i, 0.0)[1] - fitted[i]).max() for i in range(len(keys)))
    print(f'checkpoint {model_path.name}: {len(keys)} loops, {block} samples each')
    print(f'  t=0 replay max |error| vs stored fit: {zero_err:.2e}')

    # ---- all-loop summary, both directions ----
    rows = []
    for i in range(len(keys)):
        n6 = neighbours(keys, queries, plane_normal, offsets, axis[i], i, spacing[axis[i]])
        if n6[0] is None and n6[1] is None:
            continue
        for side, j in zip(('+', '-'), n6):
            if j is None:
                continue
            t = 1.0 if side == '+' else -1.0
            local, world = shift(i, t)
            nb = fitted[j]
            c0, c1 = fitted[i].mean(0), world.mean(0)
            u, v = plane_basis(plane_normal[i])
            step = t * spacing[axis[i]]
            rows.append(dict(
                i=i, j=j, side=side, axis='xyz'[axis[i]],
                gen_nb=chamfer(world, nb), gen_orig=chamfer(world, fitted[i]),
                nb_orig=chamfer(nb, fitted[i]),
                shape=chamfer(local, fitted[i] - plane_point[i]),
                cov_gen=coverage(world, nb, args.radius),
                cov_nb=coverage(nb, world, args.radius),
                step=step,
                # signed move along the normal, and its error vs one plane spacing
                axial=float((c1 - c0) @ plane_normal[i]),
                axial_res=float(abs((c1 - c0) @ plane_normal[i] - step)),
                drift=float(np.linalg.norm((c1 - c0) - ((c1 - c0) @ plane_normal[i]) * plane_normal[i])),
                r0=float(np.linalg.norm(fitted[i] - c0, axis=1).mean()),
                r1=float(np.linalg.norm(world - c1, axis=1).mean()),
                a0=area_2d(fitted[i], u, v), a1=area_2d(world, u, v)))
    if not rows:
        raise SystemExit('no loop has a neighbour on an adjacent plane')
    import csv
    with open(out / 'shift_pairs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    med = lambda k: float(np.median([r[k] for r in rows]))
    ratio = med('gen_orig') / max(med('nb_orig'), 1e-12)
    print(f'  {len(rows)} loop/neighbour pairs')
    print(f'  median chamfer: generated->neighbour {med("gen_nb"):.4f}, '
          f'generated->original {med("gen_orig"):.4f}, '
          f'real neighbour->original {med("nb_orig"):.4f} (ratio {ratio:.2f})')
    print(f'  median shape change from moving the query alone (same plane): '
          f'{med("shape"):.4f}')
    print(f'  median coverage of neighbour at {args.radius:g}: '
          f'{med("cov_gen"):.2f} generated, {med("cov_nb"):.2f} real')
    sp = np.median([abs(r['step']) for r in rows])
    print(f'  median axial placement error {med("axial_res"):.4f} '
          f'({np.mean([r["axial_res"] < 0.25 * abs(r["step"]) for r in rows]):.0%} '
          f'within a quarter plane, '
          f'{np.mean([r["axial_res"] < 0.5 * abs(r["step"]) for r in rows]):.0%} '
          f'within half), plane gap {sp:.4f}; in-plane drift {med("drift"):.4f}')
    for a in 'xyz':
        sub = [r for r in rows if r['axis'] == a]
        if not sub:
            continue
        g = np.median([r['gen_nb'] for r in sub])
        n = np.median([r['nb_orig'] for r in sub])
        e = np.median([r['axial_res'] for r in sub])
        better = np.mean([r['gen_nb'] < r['nb_orig'] for r in sub])
        print(f'    {a}: {len(sub)} pairs, gen->nbr {g:.4f} vs real nbr->orig {n:.4f}, '
              f'axial err {e:.4f}, gen closer than real in {better:.0%}')
    print(f'  median radius {med("r0"):.3f} -> {med("r1"):.3f}, '
          f'area |ratio| {np.median([abs(r["a1"] / r["a0"]) for r in rows if r["a0"]]):.2f}')

    # ---- figure: two examples per axis ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = args.loops.split(',') if args.loops else []
    if chosen:
        chosen = [(int(v), None) for v in chosen]
    else:
        chosen = []
        for a, name in enumerate('xyz'):
            ids = [r['i'] for r in rows if r['axis'] == name and r['side'] == '+']
            seen = []
            for i in ids:
                if i not in seen:
                    seen.append(i)
                if len(seen) == 2:
                    break
            chosen += [(i, name) for i in seen]
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    for ax, (i, _) in zip(axes.ravel(), chosen):
        n = plane_normal[i]
        u, v = plane_basis(n)
        origin = fitted[i].mean(0)
        proj = lambda p: np.stack([(p - origin) @ u, (p - origin) @ v], 1)
        n6 = neighbours(keys, queries, plane_normal, offsets, axis[i], i, spacing[axis[i]])
        ax.plot(*proj(fitted[i]).T, color='black', lw=2.2, label='original fit')
        for t, c in ((-1, '#4c72b0'), (-0.5, '#9ecae1'), (0.5, '#fdd0a2'), (1, '#c0392b')):
            _, world = shift(i, t)
            ax.plot(*proj(world).T, color=c, lw=1.4,
                    ls='--' if abs(t) == 1 else '-', label=f'query {t:+g} spacing')
        for side, j, c in (('+', n6[0], '#2ca02c'), ('-', n6[1], '#7f7f7f')):
            if j is not None:
                ax.plot(*proj(fitted[j]).T, color=c, lw=1.8, ls=':', label=f'real {side} neighbour')
        if n6[0] is not None:
            _, world = shift(i, 1.0)
            ax.set_title(f'{ "xyz"[axis[i]] } loop {i} (offset {offsets[i]:+.2f}) | '
                         f'gen->nbr {chamfer(world, fitted[n6[0]]):.3f}, '
                         f'nbr->orig {chamfer(fitted[n6[0]], fitted[i]):.3f}', fontsize=8)
        ax.set_aspect('equal'); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=6, loc='best')
    fig.suptitle('Loops decoded from an axially shifted centroid (dashed = one plane away)')
    fig.tight_layout()
    fig.savefig(out / 'loop_shift.png', dpi=170)
    print(f'  figure -> {out / "loop_shift.png"}')

    # ---- 3D view: the 2D panels drop the normal component the shift is about ----
    picks = (chosen[:3] if args.loops
             else [chosen[k] for k in range(0, len(chosen), 2)][:3])
    fig3 = plt.figure(figsize=(16, 5.4))
    for k, (i, _) in enumerate(picks):
        ax = fig3.add_subplot(1, 3, k + 1, projection='3d')
        n6 = neighbours(keys, queries, plane_normal, offsets, axis[i], i, spacing[axis[i]])
        _, world = shift(i, 1.0)
        ax.plot(*fitted[i].T, color='black', lw=2.2, label='original fit')
        ax.plot(*world.T, color='#c0392b', lw=1.6, ls='--', label='generated, +1 plane')
        if n6[0] is not None:
            ax.plot(*fitted[n6[0]].T, color='#2ca02c', lw=1.6, ls=':',
                    label='real + neighbour')
        title = f'{"xyz"[axis[i]]} loop {i}'
        if n6[0] is not None:
            title += f' | gen->nbr {chamfer(world, fitted[n6[0]]):.3f}, ' \
                     f'nbr->orig {chamfer(fitted[n6[0]], fitted[i]):.3f}'
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
    fig3.suptitle('3D: shifted query and plane origin together')
    fig3.tight_layout()
    fig3.savefig(out / 'loop_shift_3d.png', dpi=170)
    print(f'  3D figure -> {out / "loop_shift_3d.png"}')

    # ---- PLY of the chosen loops at t = 0, +1 and the real neighbour ----
    pts, cols = [], []
    palette = dict(orig=[20, 20, 20, 255], gen=[200, 40, 40, 255], nbr=[40, 140, 40, 255])
    for i, _ in chosen:
        n6 = neighbours(keys, queries, plane_normal, offsets, axis[i], i, spacing[axis[i]])
        _, world = shift(i, 1.0)
        for name, p in (('orig', fitted[i]), ('gen', world)):
            pts.append(p); cols.append(np.repeat([palette[name]], len(p), 0))
        if n6[0] is not None:
            pts.append(fitted[n6[0]]); cols.append(np.repeat([palette['nbr']], len(fitted[n6[0]]), 0))
    if pts:
        cloud = trimesh.PointCloud(np.concatenate(pts), np.concatenate(cols).astype(np.uint8))
        cloud.export(out / 'loop_shift.ply')
        print(f'  ply -> {out / "loop_shift.ply"}')

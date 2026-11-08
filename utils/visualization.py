"""Figures and per-plane files for the slices, fields and fitted loops."""

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from utils.ply_io import write_points
from utils.slicing import plane_basis, slice_fields


# error colour ramp: blue -> cyan -> yellow -> red
ERROR_RAMP = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]], np.float64)


def error_colors(t):
    """RGB (0-1) per value of t on the error ramp, t clipped to [0, 1]."""
    pos = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * (len(ERROR_RAMP) - 1)
    lo = np.floor(pos).astype(int)
    frac = (pos - lo)[:, None]
    rgb = (ERROR_RAMP[lo] * (1 - frac)
           + ERROR_RAMP[np.minimum(lo + 1, len(ERROR_RAMP) - 1)] * frac)
    # clip: blending can land a hair over 1, which scatter rejects
    return np.clip(rgb / 255.0, 0.0, 1.0)


# |field| below this is not drawn, so the padded grid does not show as a box
FIELD_FLOOR = 0.05


def plane_columns(p_normal):
    """Planes per axis: the length of the first run of equal normals (one figure row)."""
    cols = 1
    while cols < len(p_normal) and np.allclose(p_normal[cols], p_normal[0]):
        cols += 1
    return cols


def save_slice_grid(img_path, pc, indices, p_normal, p_point, colors, pcn=None, n_arrows=80,
                    centres=None, centre_scale=0.25, field=False, assign=False):
    """Figure with one panel per slice: its points projected onto the plane.

    One row per axis. Optional extras: normal arrows (pcn, n_arrows; None = all),
    loop centres coloured by residual (red at centre_scale of the longest extent),
    the Poisson field behind the points (field), or points coloured by loop (assign).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    pc = np.asarray(pc, dtype=np.float64)
    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))
    colors = np.asarray(colors, dtype=np.float64)[:, :3] / 255.0

    # residual colour scale, relative to the longest bounding-box side
    extent = float(np.ptp(pc, axis=0).max()) * centre_scale

    cols = plane_columns(p_normal)
    rows = -(-len(indices) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows),
                            dpi=120, squeeze=False)

    # project slices, normals and centres onto each plane's (u, v) basis
    uv, nv, cv, ce, cl, fields = [], [], [], [], [], []
    for i in range(len(indices)):
        u, v = plane_basis(p_normal[i])
        rel = pc[indices[i]] - p_point[i]
        uv.append((rel @ u, rel @ v))
        if pcn is not None:
            nv.append((pcn[indices[i]] @ u, pcn[indices[i]] @ v))
        if field:
            if pcn is None:
                raise ValueError('the implicit field is solved from the point normals')
            # normalised as in loop_centres, so this is the field the centres came from
            fn = np.stack(nv[i], axis=1)
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            fields.append(slice_fields(np.stack(uv[i], axis=1), fn))
        if centres is not None:
            cs, cr, lab = centres[i]
            rel_c = np.asarray(cs, dtype=np.float64).reshape(-1, 3) - p_point[i]
            cv.append((rel_c @ u, rel_c @ v))
            ce.append(np.asarray(cr, dtype=np.float64).reshape(-1))
            cl.append(np.asarray(lab, dtype=int))

    # one common window per row so the panels along a row stay comparable
    windows = {}
    for i, (uu, vv) in enumerate(uv):
        window = windows.setdefault(i // cols, [np.inf, -np.inf, np.inf, -np.inf])
        if uu.size:
            window[0], window[1] = min(window[0], uu.min()), max(window[1], uu.max())
            window[2], window[3] = min(window[2], vv.min()), max(window[3], vv.max())

    # diverging colour map, white at field = 0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)

    for i, (uu, vv) in enumerate(uv):
        ax = axs[i // cols][i % cols]
        # each field drawn on its own solve grid; faded when points are coloured by loop
        for chi, origin, h in (fields[i] if field else ()):
            ax.imshow(np.ma.masked_where(np.abs(chi) < FIELD_FLOOR, chi), origin='lower',
                      cmap='RdBu_r', norm=norm, zorder=0, interpolation='bilinear',
                      alpha=0.3 if assign else None,
                      extent=[origin[0], origin[0] + chi.shape[1] * h,
                              origin[1], origin[1] + chi.shape[0] * h])
        picked = None
        if assign and centres is not None and len(cv[i][0]):
            picked = cl[i]
        if picked is not None:
            # bigger dots per loop colour; unowned points grey
            face = plt.cm.tab10(np.where(picked < 0, 0, picked) % 10)
            face[picked < 0] = (0.62, 0.62, 0.62, 1.0)
            ax.scatter(uu, vv, s=2.2, c=face, linewidths=0)
        else:
            ax.scatter(uu, vv, s=0.8, c='k' if field else [colors[i]], linewidths=0)
        label = f'{"xyz"[i // cols]} = {p_point[i] @ p_normal[i]:.2f}' if rows <= 3 else f'p{i}'
        ax.set_title(f'{label} | {indices[i].size:,} pts', fontsize=7)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=5, length=2)
        lo_u, hi_u, lo_v, hi_v = windows[i // cols]
        pad = 0.06 * max(hi_u - lo_u, hi_v - lo_v, 1e-9)
        # set limits after imshow, which would otherwise widen them
        ax.set_xlim(lo_u - pad, hi_u + pad)
        ax.set_ylim(lo_v - pad, hi_v + pad)

        # n_arrows: 0 = none, None = all; arrows thinned and scaled to the panel
        if pcn is not None and len(uu) and n_arrows != 0:
            pick = (np.arange(len(uu)) if n_arrows is None or n_arrows >= len(uu)
                    else np.linspace(0, len(uu) - 1, n_arrows).astype(int))
            du, dv = nv[i]
            arrow = 0.15 * max(hi_u - lo_u, hi_v - lo_v)
            ax.quiver(uu[pick], vv[pick], du[pick] * arrow, dv[pick] * arrow,
                      angles='xy', scale_units='xy', scale=1,
                      color='k', width=0.004, headwidth=3, headlength=4)

        # centre marks: filled by residual, or hollow rings in the loop colour
        if centres is not None and len(cv[i][0]):
            if picked is not None:
                ax.scatter(*cv[i], s=60, marker='o', facecolors='none',
                           edgecolors=plt.cm.tab10(np.arange(len(cv[i][0])) % 10),
                           linewidths=1.6, zorder=3)
            else:
                ax.scatter(*cv[i], s=60, marker='o', facecolors=error_colors(ce[i] / extent),
                           edgecolors='k', linewidths=1.2, zorder=3)

    for i in range(len(uv), rows * cols):
        axs[i // cols][i % cols].axis('off')

    # title and colour bars
    if centres is not None and not assign:
        fig.suptitle(f'{len(uv)} slices projected onto their planes'
                     f'{" | implicit field behind the points" if field else ""}'
                     f' | loop centre residual, red at {centre_scale:.2g} of the '
                     f'{extent / centre_scale:.3g} bounding box extent', fontsize=9)
    elif centres is not None:
        fig.suptitle(f'{len(uv)} slices projected onto their planes'
                     f'{" | implicit field behind the points" if field else ""}'
                     f' | points coloured by the loop centre they were assigned to,'
                     f' grey where no loop owns them', fontsize=9)
    elif field:
        fig.suptitle(f'{len(uv)} slices projected onto their planes | implicit field, '
                     f'1 over the area a loop encloses, negative outside it', fontsize=9)
    else:
        fig.suptitle(f'{len(uv)} slices projected onto their planes', fontsize=9)

    fig.tight_layout()
    bars = []
    if field:
        bars.append((plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), 'implicit field'))
    if centres is not None and not assign:
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        ramp = LinearSegmentedColormap.from_list('centre', ERROR_RAMP / 255.0)
        bars.append((plt.cm.ScalarMappable(norm=Normalize(0, extent), cmap=ramp),
                     'centre residual'))
    if bars:
        # colour bars in axes of their own, below the panels
        fig.subplots_adjust(bottom=0.10)
        width = 0.36 if len(bars) == 2 else 0.4
        for slot, (mappable, name) in enumerate(bars):
            left = 0.08 + slot * 0.48 if len(bars) == 2 else 0.3
            cax = fig.add_axes([left, 0.035, width, 0.016])
            bar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
            bar.set_label(name, fontsize=6)
            bar.ax.tick_params(labelsize=6)

    fig.savefig(img_path)
    plt.close(fig)
    print(f'  {rows} x {cols} projected grid -> {img_path}')


def save_field_grid(img_path, pc, indices, p_normal, p_point, pcn, centres=None,
                    centre_scale=0.25, n_points=1500, k=8, c=2.0, min_points=100):
    """Figure with one panel per slice: the Poisson field of every loop in it.

    Each field is scaled by its own peak (1 inside, negative outside). n_points:
    slice points drawn per panel (0 none, None all); centres marked as in
    `save_slice_grid`; k, c, min_points: grouping, see `slice_loops`.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    pc = np.asarray(pc, dtype=np.float64)
    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))

    cols = plane_columns(p_normal)
    rows = -(-len(indices) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows),
                            dpi=120, squeeze=False)
    # diverging colour map, white at field = 0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    # same residual scale as save_slice_grid
    extent = float(np.ptp(pc, axis=0).max()) * centre_scale

    uv, fields, cv, ce = [], [], [], []
    for i in range(len(indices)):
        u, v = plane_basis(p_normal[i])
        rel = pc[indices[i]] - p_point[i]
        uv.append((rel @ u, rel @ v))
        nv = np.stack([pcn[indices[i]] @ u, pcn[indices[i]] @ v], axis=1)
        # keep only the in-plane direction of each normal
        nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
        fields.append(slice_fields(np.stack(uv[i], axis=1), nv, k, c, min_points))
        if centres is not None:
            cs, cr, _ = centres[i]
            rel_c = np.asarray(cs, dtype=np.float64).reshape(-1, 3) - p_point[i]
            cv.append((rel_c @ u, rel_c @ v))
            ce.append(np.asarray(cr, dtype=np.float64).reshape(-1))

    # one common window per row so the panels along a row stay comparable
    windows = {}
    for i, (uu, vv) in enumerate(uv):
        window = windows.setdefault(i // cols, [np.inf, -np.inf, np.inf, -np.inf])
        if uu.size:
            window[0], window[1] = min(window[0], uu.min()), max(window[1], uu.max())
            window[2], window[3] = min(window[2], vv.min()), max(window[3], vv.max())

    for i, (uu, vv) in enumerate(uv):
        ax = axs[i // cols][i % cols]
        lo_u, hi_u, lo_v, hi_v = windows[i // cols]
        pad = 0.06 * max(hi_u - lo_u, hi_v - lo_v, 1e-9)
        ax.set_xlim(lo_u - pad, hi_u + pad)
        ax.set_ylim(lo_v - pad, hi_v + pad)

        # each field drawn on its own solve grid, the faint padding masked
        for chi, origin, h in fields[i]:
            ax.imshow(np.ma.masked_where(np.abs(chi) < FIELD_FLOOR, chi), origin='lower',
                      cmap='RdBu_r', norm=norm, zorder=0, interpolation='bilinear',
                      extent=[origin[0], origin[0] + chi.shape[1] * h,
                              origin[1], origin[1] + chi.shape[0] * h])

        # thinned, translucent points over the field
        if n_points != 0 and len(uu):
            pick = (np.arange(len(uu)) if n_points is None or n_points >= len(uu)
                    else np.linspace(0, len(uu) - 1, n_points).astype(int))
            ax.scatter(uu[pick], vv[pick], s=0.8, c='k', alpha=0.6,
                       linewidths=0, zorder=1)
        label = f'{"xyz"[i // cols]} = {p_point[i] @ p_normal[i]:.2f}' if rows <= 3 else f'p{i}'
        ax.set_title(f'{label} | {indices[i].size:,} pts', fontsize=7)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=5, length=2)

        # centre marks filled by residual
        if centres is not None and len(cv[i][0]):
            ax.scatter(*cv[i], s=60, marker='o', facecolors=error_colors(ce[i] / extent),
                       edgecolors='k', linewidths=1.2, zorder=3)

    for i in range(len(uv), rows * cols):
        axs[i // cols][i % cols].axis('off')

    fig.suptitle(f'{len(uv)} slices projected onto their planes | implicit field, '
                 f'1 over the area a loop encloses, negative outside it', fontsize=9)
    fig.tight_layout()
    # colour bar in an axes of its own, below the panels
    fig.subplots_adjust(bottom=0.10)
    cax = fig.add_axes([0.3, 0.035, 0.4, 0.016])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                       cax=cax, orientation='horizontal')
    bar.ax.tick_params(labelsize=6)

    fig.savefig(img_path)
    plt.close(fig)
    print(f'  {rows} x {cols} field grid -> {img_path}')


def visualize_sliced_point_cloud(saved_path, pc, indices, p_normal=None, p_point=None,
                                 vis_to_img=True, pcn=None, centres=None, n_arrows=80,
                                 centre_scale=0.25, field=False, assign=False):
    """Write all slices to one PLY (a random colour per slice, normals if given).

    With vis_to_img, also write the projected grid (.png beside it) via
    `save_slice_grid`, which takes the remaining options.
    """
    if isinstance(indices, np.ndarray):
        indices = [indices]
    rng = np.random.default_rng(0)

    # one colour per slice; the seeded rng keeps runs comparable
    slice_colors = np.stack([np.append(rng.integers(0, 255, 3), 255) for _ in indices])
    colors = np.repeat(slice_colors, [idx.size for idx in indices], axis=0)
    points = np.concatenate([pc[idx] for idx in indices])
    print(f'  {len(indices)} slices, {len(points)} points -> {saved_path}')

    if pcn is None:
        # PointCloud: concatenated faceless meshes cannot be exported with colours
        cloud = trimesh.PointCloud(points, colors=colors.astype(np.uint8))
    else:
        # a faceless Trimesh, since a PointCloud cannot store normals
        cloud = trimesh.Trimesh(vertices=points, faces=np.zeros((0, 3), dtype=np.int64),
                                vertex_normals=np.concatenate([pcn[idx] for idx in indices]),
                                process=False)
        cloud.visual.vertex_colors = colors.astype(np.uint8)
    cloud.export(saved_path)

    if vis_to_img:
        if p_normal is None or p_point is None:
            raise ValueError('p_normal and p_point are needed to draw the projection')
        save_slice_grid(str(Path(saved_path).with_suffix('.png')), pc, indices,
                        p_normal, p_point, slice_colors, pcn=pcn, n_arrows=n_arrows,
                        centres=centres, centre_scale=centre_scale, field=field,
                        assign=assign)


# loop colours by the axis of the plane that cut them (x, y, z)
AXIS_COLORS = np.array([[0x45, 0x75, 0xb4, 255],
                        [0x2a, 0x9d, 0x8f, 255],
                        [0xd1, 0x49, 0x5b, 255]], dtype=np.uint8)


def fit_items(loops, preds, p_normal, p_point):
    """Project every loop's target and fit onto its plane, with its mean error.

    Returns ({slice: [[centre, (tu, tv), (pu, pv), err], ...]}, mean
    out-of-plane distance of each fit).
    """
    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))

    per, out_of_plane = {}, []
    for (s, c, target, *_), pred in zip(loops, preds):
        u, v = plane_basis(p_normal[s])
        target = np.asarray(target, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        rel, fit = target - p_point[s], pred - p_point[s]
        out_of_plane.append(float(np.abs(fit @ p_normal[s]).mean()))
        err = cKDTree(target).query(pred)[0]
        per.setdefault(s, []).append([c, (rel @ u, rel @ v), (fit @ u, fit @ v),
                                      float(err.mean())])
    return per, out_of_plane


def fit_normal_items(loops, pred_normals, p_normal):
    """Predicted normals projected onto each loop's plane, lengths kept: {slice: [(nu, nv)]}."""
    if pred_normals is None:
        return {}
    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    per = {}
    for (s, *_), normals in zip(loops, pred_normals):
        u, v = plane_basis(p_normal[s])
        normals = np.asarray(normals, dtype=np.float64)
        rel = normals - (normals @ p_normal[s])[:, None] * p_normal[s]
        per.setdefault(s, []).append((rel @ u, rel @ v))
    return per


def fit_window(items):
    """(lo_u, hi_u, lo_v, hi_v) holding all targets and fits of one panel."""
    lo_u = min(min(tu.min(), pu.min()) for _, (tu, _), (pu, _), _ in items)
    hi_u = max(max(tu.max(), pu.max()) for _, (tu, _), (pu, _), _ in items)
    lo_v = min(min(tv.min(), pv.min()) for _, (_, tv), (_, pv), _ in items)
    hi_v = max(max(tv.max(), pv.max()) for _, (_, tv), (_, pv), _ in items)
    return lo_u, hi_u, lo_v, hi_v


def error_scale(per, large=1.5):
    """(median per-plane mean error, the `large` multiple of it that marks a plane)."""
    means = [float(np.mean([e for *_, e in items])) for items in per.values()]
    typical = float(np.median(means)) if means else 0.0
    return typical, (large * typical if typical > 0 else np.inf)


def mark_error(err, max_err=None, digits=3):
    """Format an error, prefixed with "!" when it exceeds max_err."""
    return f'{err:.{digits}f}' if max_err is None or err <= max_err else f'!{err:.{digits}f}'


# fitted curves are coloured by their loop parameter; cyclic, since z = 0 and 1 meet
PARAM_CMAP = 'hsv'


def param_colorbar(fig, ax):
    """Colour bar mapping a fitted curve's colour back to its loop parameter z."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    sm = plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=PARAM_CMAP)
    bar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04,
                       pad=0.04, aspect=45)
    bar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    bar.set_label('loop parameter z in turns, cyclic: 0 and 1 meet at the ring',
                  fontsize=7)
    bar.ax.tick_params(labelsize=6, length=2)
    return bar


def draw_fit_panel(ax, items, window, max_err=None, normals=None, n_arrows=20):
    """Draw one fit panel: each loop's points and its fitted curve, in the loop colour.

    The curve is coloured by its parameter, with a ring at z = 0; a loop with error
    above max_err gets a heavy red casing and its error written at its centre.
    items / window: from `fit_items` / `fit_window`; normals: from
    `fit_normal_items`, drawn from the curve; n_arrows: per loop (0 none, None all).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    lo_u, hi_u, lo_v, hi_v = window
    # one arrow scale for the whole panel, so relative lengths are kept
    arrow = 0.04 * max(hi_u - lo_u, hi_v - lo_v, 1e-9)
    normals = [None] * len(items) if normals is None else normals

    for (c, (tu, tv), (pu, pv), err), normal in zip(items, normals):
        # dots in the loop colour, the curve casing darker (red when over the limit)
        over = max_err is not None and err > max_err
        colour = np.array(plt.cm.tab10(c % 10))
        dark = colour.copy()
        dark[:3] *= 0.55
        if over:
            dark = np.array([0.75, 0.10, 0.10, 1.0])
        ax.scatter(tu, tv, s=1.8, c=colour[None, :], alpha=0.5, linewidths=0)
        # closed curve, each segment coloured by its parameter
        if len(pu) > 1:
            ends = np.stack([np.append(pu, pu[0]), np.append(pv, pv[0])], axis=1)
            segs = np.stack([ends[:-1], ends[1:]], axis=1)
            z = np.linspace(0.0, 1.0, len(segs))
            # casing under the parameter colours
            ax.add_collection(LineCollection(
                segs, colors=[dark], linewidths=3.2 if over else 2.0,
                capstyle='round', joinstyle='round', zorder=4 if over else 3))
            ax.add_collection(LineCollection(
                segs, array=z, cmap=PARAM_CMAP, norm=Normalize(0.0, 1.0),
                linewidths=2.0 if over else 1.1, capstyle='round',
                joinstyle='round', zorder=4.5 if over else 3.5))
        ax.plot(pu[0], pv[0], marker='o', ms=4, mfc='none', mec=dark, mew=0.8)
        # predicted normals, drawn from the fitted curve
        if normal is not None and n_arrows != 0 and len(pu):
            pick = (np.arange(len(pu)) if n_arrows is None or n_arrows >= len(pu)
                    else np.linspace(0, len(pu) - 1, n_arrows).astype(int))
            nu, nv = normal
            ax.quiver(pu[pick], pv[pick], nu[pick] * arrow, nv[pick] * arrow,
                      angles='xy', scale_units='xy', scale=1, color=dark,
                      width=0.004, headwidth=3, headlength=4, zorder=3)
        if over:
            ax.annotate(f'{err:.3f}', (float(np.mean(tu)), float(np.mean(tv))),
                        color=dark, fontsize=5.5, ha='center', va='center', zorder=5)
    pad = 0.06 * max(hi_u - lo_u, hi_v - lo_v, 1e-9)
    ax.set_xlim(lo_u - pad, hi_u + pad)
    ax.set_ylim(lo_v - pad, hi_v + pad)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=5, length=2)


def save_fit_grid(img_path, loops, preds, p_normal, p_point, max_err=None, large=1.5,
                  pred_normals=None, n_arrows=20, align=None):
    """Figure with one panel per plane: its loops and the curves fitted to them.

    loops / preds: loop tuples and N x 3 predictions in the same order. Panels are
    marked red when a loop exceeds max_err or the plane's mean error exceeds
    `large` x the median; `align` (the run's normal penalty) goes in the title.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))

    cols = plane_columns(p_normal)
    rows = -(-len(p_normal) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows),
                            dpi=120, squeeze=False, constrained_layout=True)

    per, out_of_plane = fit_items(loops, preds, p_normal, p_point)
    normal_per = fit_normal_items(loops, pred_normals, p_normal)

    # one common window per row, targets and fits together
    windows = {}
    for s, items in per.items():
        window = windows.setdefault(s // cols, [np.inf, -np.inf, np.inf, -np.inf])
        lo_u, hi_u, lo_v, hi_v = fit_window(items)
        window[0], window[1] = min(window[0], lo_u), max(window[1], hi_u)
        window[2], window[3] = min(window[2], lo_v), max(window[3], hi_v)

    bad = 0
    typical, limit = error_scale(per, large)
    large_planes = 0
    for s in sorted(per):
        ax = axs[s // cols][s % cols]
        draw_fit_panel(ax, per[s], windows[s // cols], max_err,
                       normal_per.get(s), n_arrows)
        label = f'{"xyz"[s // cols]} = {p_point[s] @ p_normal[s]:.2f}'
        errs = [e for *_, e in per[s]]
        mean = float(np.mean(errs))
        over = 0 if max_err is None else sum(e > max_err for e in errs)
        bad += over
        # mark a plane with a loop over max_err, or a mean far above the median
        big = over > 0 or mean > limit
        large_planes += big
        mark = f', {over} over' if over else ''
        title = (f'{label} | {len(per[s])} loop{"" if len(per[s]) == 1 else "s"}, '
                 f'err {mean:.3f}{mark}')
        ax.set_title(f'! {title}' if big else title, fontsize=7,
                     color='#bf1b1b' if big else 'black')

    for i in range(len(p_normal), rows * cols):
        axs[i // cols][i % cols].axis('off')

    # figure title: legend, out-of-plane error, align and the mark counts
    over_all = '' if max_err is None else f' or a loop over {max_err:.3f}'
    align_part = ('' if align is None or not np.isfinite(align)
                  else f'| align {align:.3f} (0 = with the normals, 2 = against) ')
    fig.suptitle(f'{len(loops)} fitted loops over the {len(p_normal)} slice planes | '
                 f'line = mini nn, its colour the loop parameter, dots = the loop it '
                 f'was fitted to, ring = the seam, arrows = the fit normal\n'
                 f'mean out-of-plane {np.mean(out_of_plane):.3f} {align_part}'
                 f'| ! = mean err over {large:g}x the {typical:.3f} median plane'
                 f'{over_all}, {large_planes} planes, {bad} loops', fontsize=8,
                 wrap=True)
    param_colorbar(fig, axs.ravel().tolist())
    fig.savefig(img_path)
    plt.close(fig)
    print(f'  {rows} x {cols} fit grid -> {img_path}')


def save_plane_files(out_dir, pc, pcn, slices, centres, loops, preds,
                     p_normal, p_point, max_err=None, large=1.5,
                     pred_normals=None, n_arrows=20):
    """Write one set of files per plane, plus a per-plane and a bad-loop summary.

    <out_dir>/planes/plane_SS_<axis><offset>_{slice,loops,fit}.ply and _fit.png:
    the slice, its points coloured by loop, the fitted curves and their figure.
    <out_dir>/planes.txt lists counts and errors per plane ("!" = over max_err);
    <out_dir>/bad_loops.txt lists the loops over max_err.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))
    pc = np.asarray(pc, dtype=np.float64)
    cols = plane_columns(p_normal)
    per, _ = fit_items(loops, preds, p_normal, p_point)
    typical, limit = error_scale(per, large)

    # fitted points (and normals) per slice, in 3D
    fit_at, normal_at = {}, {}
    normal_per = fit_normal_items(loops, pred_normals, p_normal)
    for (s, c, *_), pred, normals in zip(loops, preds,
                                         pred_normals or [None] * len(preds)):
        fit_at.setdefault(s, []).append((c, np.asarray(pred, dtype=np.float64)))
        if normals is not None:
            normal_at.setdefault(s, []).append(np.asarray(normals, dtype=np.float64))

    out = Path(out_dir) / 'planes'
    out.mkdir(parents=True, exist_ok=True)
    lines = [f'{len(slices)} planes, {len(loops)} loops']
    size_of = {(s, c): len(pts) for s, c, pts, _ in loops}
    bad = []
    large_planes = 0
    for s, idx in enumerate(slices):
        axis = 'xyz'[s // cols] if s // cols < 3 else f'p{s // cols}'
        offset = p_point[s] @ p_normal[s]
        base = str(out / f'plane_{s:02d}_{axis}{offset:+.2f}')
        items = per.get(s, [])
        mean = float(np.mean([e for *_, e in items])) if items else 0.0
        over = max_err is not None and any(e > max_err for *_, e in items)
        # same marking rule as the grid titles
        big = bool(items) and (over or mean > limit)
        large_planes += big

        # the slice as cut, in grey
        grey = np.full((len(idx), 4), 200, dtype=np.uint8)
        grey[:, 3] = 255
        write_points(f'{base}_slice.ply', pc[idx], grey,
                     None if pcn is None else np.asarray(pcn)[idx])

        # the slice coloured by loop
        lab = np.full(len(idx), -1, dtype=int) if centres is None else centres[s][2]
        colors = np.full((len(idx), 4), 158, dtype=np.uint8)
        colors[:, 3] = 255
        for k in np.unique(lab[lab >= 0]):
            colors[lab == k] = (np.array(plt.cm.tab10(int(k) % 10)) * 255).astype(np.uint8)
        write_points(f'{base}_loops.ply', pc[idx], colors,
                     None if pcn is None else np.asarray(pcn)[idx])

        if fit_at.get(s):
            point_sets = [p for _, p in fit_at[s]]
            curve = np.concatenate(point_sets)
            tints = np.concatenate([
                np.repeat((np.array(plt.cm.tab10(int(c) % 10)) * 255).astype(np.uint8)[None],
                          len(p), axis=0) for c, p in fit_at[s]])
            write_points(f'{base}_fit.ply', curve, tints,
                         None if not normal_at.get(s)
                         else np.concatenate(normal_at[s]))

            fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=140,
                                   constrained_layout=True)
            draw_fit_panel(ax, items, fit_window(items), max_err,
                           normal_per.get(s), n_arrows)
            errs = ', '.join(mark_error(e, max_err) for *_, e in items)
            # a standalone figure, so state the multiple of the median
            title = (f'{axis} = {offset:.2f} | '
                     f'{len(items)} loop{"" if len(items) == 1 else "s"} | err {errs}')
            if big:
                title += f' | {mean / typical:.1f}x the {typical:.3f} median'
            ax.set_title(f'! {title}' if big else title, fontsize=8,
                         color='#bf1b1b' if big else 'black')
            param_colorbar(fig, ax)
            fig.savefig(f'{base}_fit.png')
            plt.close(fig)

        for c, _, _, e in items:
            if max_err is not None and e > max_err:
                bad.append(f'plane_{s:02d} {axis} {offset:+.3f} | centre {c} | '
                           f'{size_of.get((s, c), 0)} pts | err {e:.4f}')

        errs = ' '.join(mark_error(e, max_err, 4) for *_, e in items)
        sizes = ' '.join(str(len(pts)) for sl, _, pts, _ in loops if sl == s)
        lines.append(f'plane_{s:02d} {axis} {offset:+.3f} | '
                     f'{len(idx)} pts | {len(items)} loops | sizes {sizes or "-"} | '
                     f'err {errs or "-"}' + (' | large' if big else ''))

    (Path(out_dir) / 'planes.txt').write_text('\n'.join(lines) + '\n')
    head = 'no limit: every loop kept' if max_err is None else (
        f'max_err {max_err:.4f}: these {len(bad)} of {len(loops)} loops are red in the '
        f'figures and marked "!" in planes.txt')
    (Path(out_dir) / 'bad_loops.txt').write_text(head + '\n' + '\n'.join(bad) +
                                                 ('\n' if bad else ''))
    print(f'  {len(slices)} planes, {len(loops)} loops, {large_planes} planes over '
          f'{large:g}x the {typical:.4f} median, {len(bad)} loops over {max_err:.4f}: '
          f'-> {out}, {Path(out_dir) / "planes.txt"}, '
          f'{Path(out_dir) / "bad_loops.txt"}')

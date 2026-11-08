"""Cut a point cloud with axis-aligned planes and part each slice into loops.

Pure geometry: slabs, 2D projection, Poisson fields, loop centres and plane offsets.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def slice_point_cloud(pc, p_normal, p_point, threshold=0.01, pcn=None,
                      max_align=0.95):
    """Indices of the points within `threshold` of each plane (p_normal, p_point).

    With normals `pcn`, points whose normal is within cosine `max_align` of the
    plane normal are dropped from that plane: they lie flat in the slice and have
    no useful in-plane normal. One plane returns a bare index array, M planes a list.
    """
    pc = np.asarray(pc, dtype=np.float64)
    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))

    # unit normals, so the projection is a true point-to-plane distance
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    if len(p_point) == 1:
        p_point = np.repeat(p_point, len(p_normal), axis=0)

    # N x M signed point-to-plane distances
    signed = pc @ p_normal.T - np.einsum('mj,mj->m', p_point, p_normal)
    hit = np.abs(signed) < threshold
    if pcn is not None:
        pcn = np.asarray(pcn, dtype=np.float64)
        pcn = pcn / np.maximum(np.linalg.norm(pcn, axis=1, keepdims=True), 1e-12)
        hit &= np.abs(pcn @ p_normal.T) < max_align

    indices = [np.flatnonzero(hit[:, m]) for m in range(hit.shape[1])]
    return indices[0] if len(indices) == 1 else indices


def plane_basis(nrm):
    """Two unit vectors (u, v) spanning the plane with normal `nrm`."""
    helper = np.array([1.0, 0.0, 0.0] if abs(nrm[0]) < 0.9 else [0.0, 1.0, 0.0])
    u = np.cross(nrm, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(nrm, u)


def slice_loops(uv, k=8, c=2.0):
    """Group a projected slice (S x 2) into connected loops; returns one label per point.

    Two points connect when closer than c times the larger of their k-th-neighbour
    distances, so the link radius adapts to the local point spacing.
    """
    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) <= k + 1:
        return np.zeros(len(uv), dtype=int)
    tree = cKDTree(uv)
    dk = tree.query(uv, k=k + 1)[0][:, -1]

    # prune with the largest radius, then test each pair against its own spacing
    pairs = np.array(tuple(tree.query_pairs(c * dk.max())), dtype=int).reshape(-1, 2)
    reach = np.linalg.norm(uv[pairs[:, 0]] - uv[pairs[:, 1]], axis=1)
    keep = reach < c * np.maximum(dk[pairs[:, 0]], dk[pairs[:, 1]])
    graph = coo_matrix((np.ones(keep.sum()), (pairs[keep, 0], pairs[keep, 1])),
                       shape=(len(uv), len(uv)))
    return connected_components(graph, directed=False)[1]


def fit_centre(uv, nv, prior=None, ridge=1e-3):
    """In-plane point closest (least squares) to all normal lines of a loop.

    An optional `prior` with a small ridge keeps an open arc from sliding along itself.
    """
    proj = np.eye(2)[None] - nv[:, :, None] * nv[:, None, :]
    lhs = proj.sum(0)
    rhs = (proj @ uv[:, :, None])[:, :, 0].sum(0)
    if prior is not None:
        lhs = lhs + ridge * len(uv) * np.eye(2)
        rhs = rhs + ridge * len(uv) * np.asarray(prior, dtype=np.float64)
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def poisson_field(uv, nv, res=200, pad=0.2, sigma=2.0, blur=1.0):
    """Indicator of the area a loop of oriented 2D points encloses (positive inside).

    Solves laplacian(chi) = -div(V) by FFT on a res x res grid, V being the splatted
    and blurred normals. Returns (chi, grid origin, cell size).
    """
    uv = np.asarray(uv, dtype=np.float64)
    nv = np.asarray(nv, dtype=np.float64)
    span = max(float(np.ptp(uv, axis=0).max()), 1e-12) * (1 + 2 * pad)
    h = span / res
    origin = 0.5 * (uv.min(0) + uv.max(0)) - 0.5 * res * h

    grid = np.clip((uv - origin) / h, 0, res - 1.001)
    cell = grid.astype(int)
    frac = grid - cell
    vx = np.zeros((res, res))
    vy = np.zeros((res, res))
    for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1)):
        w = (frac[:, 0] if dj else 1 - frac[:, 0]) * (frac[:, 1] if di else 1 - frac[:, 1])
        np.add.at(vx, (cell[:, 1] + di, cell[:, 0] + dj), nv[:, 0] * w)
        np.add.at(vy, (cell[:, 1] + di, cell[:, 0] + dj), nv[:, 1] * w)
    if blur:
        vx = gaussian_filter(vx, blur, mode='constant')
        vy = gaussian_filter(vy, blur, mode='constant')
    div = np.gradient(vx, h, axis=1) + np.gradient(vy, h, axis=0)

    k = 2 * np.pi * np.fft.fftfreq(res)
    lam = (2 * np.cos(k[:, None] * h) - 2) / h**2 + (2 * np.cos(k[None, :] * h) - 2) / h**2
    # the constant mode would divide by zero
    lam[0, 0] = 1.0
    chi = np.real(np.fft.ifft2(np.fft.fft2(-div) / lam))
    chi[0, 0] = 0.0
    return (gaussian_filter(chi, sigma, mode='constant') if sigma else chi), origin, h


def slice_fields(uv, nv, k=8, c=2.0, min_points=100):
    """Poisson field of every loop group of a slice, each scaled by its own peak.

    Returns [(chi / peak, origin, h)] for every group of at least min_points points.
    """
    uv = np.asarray(uv, dtype=np.float64)
    # too few points to group: treat as one group
    lp = (slice_loops(uv, k, c) if len(uv) >= min_points
          else np.zeros(len(uv), dtype=int))
    fields = []
    for m in (lp == j for j, size in enumerate(np.bincount(lp)) if size >= min_points):
        chi, origin, h = poisson_field(uv[m], nv[m])
        peak = float(chi.max())
        fields.append((chi / peak if peak > 0 else chi, origin, h))
    return fields


def field_parts(uv, nv, level=0.0, min_frac=0.02, min_area=0.01, **kw):
    """Enclosed regions of a point group: connected parts of its Poisson field > level.

    Parts smaller than min_frac of the largest, or than min_area of the group's box,
    are dropped. Returns ([(centre 2, area)] largest first, label per point or -1).
    """
    uv = np.asarray(uv, dtype=np.float64)
    nv = np.asarray(nv, dtype=np.float64)
    none = np.full(len(uv), -1, dtype=int)
    # degenerate group: nothing enclosed
    if len(uv) < 3 or float(np.ptp(uv, axis=0).max()) <= 0:
        return [], none
    chi, origin, h = poisson_field(uv, nv, **kw)
    blob, count = label(chi > level * chi.max())
    if not count:
        return [], none

    # centroid weighted by the field inside each part
    rows, cols = np.indices(chi.shape)
    found = []
    for b in range(1, count + 1):
        mask = blob == b
        mass = np.where(mask, chi, 0.0)
        if mass.sum() <= 0:
            continue
        centroid = np.array([(mass * cols).sum() / mass.sum(),
                             (mass * rows).sum() / mass.sum()]) * h + origin
        # snap the centroid to the part's nearest cell (concave parts)
        cells = np.argwhere(mask)
        xy = np.stack([cells[:, 1], cells[:, 0]], axis=1) * h + origin
        centre = xy[np.argmin(((xy - centroid) ** 2).sum(axis=1))]
        found.append((b, centre, int(mask.sum())))
    if not found:
        return [], none
    box = max(float(np.prod(np.ptp(uv, axis=0))), 1e-12)
    top = max(a for *_, a in found)
    kept = sorted((f for f in found if f[2] >= min_frac * top and f[2] * h**2 >= min_area * box),
                  key=lambda f: -f[2])

    # each point takes the nearest kept part to its cell
    keep = np.isin(blob, [b for b, _, _ in kept])
    near = distance_transform_edt(~keep, return_distances=False, return_indices=True)
    at = np.clip((uv - origin) / h, 0, chi.shape[0] - 1.001).astype(int)
    hit = blob[near[0][at[:, 1], at[:, 0]], near[1][at[:, 1], at[:, 0]]]

    order = np.full(count + 1, -1, dtype=int)
    order[[b for b, _, _ in kept]] = np.arange(len(kept))
    return [(c, float(a)) for _, c, a in kept], order[hit]


def field_loops(uv, nv):
    """Centres (k x 2) of the regions a group encloses, and a part label per point.

    Each part is re-solved from its own points and split recursively if it reveals
    several regions. A group enclosing nothing (an open arc) gets `fit_centre`.
    """
    uv = np.asarray(uv, dtype=np.float64)
    nv = np.asarray(nv, dtype=np.float64)
    parts, lab = field_parts(uv, nv)
    if not parts:
        return fit_centre(uv, nv)[None], np.zeros(len(uv), dtype=int)

    centres = []
    labels = np.full(len(uv), -1, dtype=int)

    def split(ids, centre):
        """Keep this part, or recursively keep all parts its own normals reveal."""
        if len(ids) < 3:
            labels[ids] = len(centres)
            centres.append(centre)
            return
        subparts, sublab = field_parts(uv[ids], nv[ids])
        nonempty = [r for r in range(len(subparts)) if np.any(sublab == r)]
        if len(nonempty) <= 1:
                # a single re-solved part only refines the centre
            labels[ids] = len(centres)
            centres.append(subparts[nonempty[0]][0] if nonempty else centre)
            return
        for r in nonempty:
            split(ids[sublab == r], subparts[r][0])

    for r, (centre, _) in enumerate(parts):
        ids = np.flatnonzero(lab == r)
        if len(ids):
            split(ids, centre)
    return np.asarray(centres).reshape(-1, 2), labels


def loop_centres(pc, indices, p_normal, p_point, pcn, k=8, c=2.0, min_points=100):
    """Loop centres of every slice, from the point normals.

    Each slice is grouped (`slice_loops`, groups under min_points dropped), and
    every region a group encloses (`field_loops`) becomes one loop with a 3D centre.
    The rms is the distance from the centre to its points' normal lines (0 = the
    normals meet in one spot, as on a circle).

    Returns one (m x 3 centres, m rms, labels) per slice; labels index the centres
    per point, -1 where no loop owns the point.
    """
    if pcn is None:
        raise ValueError('loop centres are solved from the point normals')
    if isinstance(indices, np.ndarray):
        indices = [indices]
    pc = np.asarray(pc, dtype=np.float64)
    pcn = np.asarray(pcn, dtype=np.float64)
    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))
    if len(p_point) == 1:
        p_point = np.repeat(p_point, len(indices), axis=0)

    found = []
    for s, idx in enumerate(indices):
        u_ax, v_ax = plane_basis(p_normal[s])
        rel = pc[idx] - p_point[s]
        uv = np.stack([rel @ u_ax, rel @ v_ax], axis=1)
        nv = np.stack([pcn[idx] @ u_ax, pcn[idx] @ v_ax], axis=1)
        labels = np.full(len(idx), -1, dtype=int)
        if len(uv) < min_points:
            found.append((np.zeros((0, 3)), np.zeros(0), labels))
            continue
        # keep only the in-plane direction of each normal
        nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)

        lp = slice_loops(uv, k, c)
        centres, rms = [], []
        for m in (lp == j for j, size in enumerate(np.bincount(lp)) if size >= min_points):
            gp = np.flatnonzero(m)
            # the regions this group encloses, and which one each point is on
            at, aim = field_loops(uv[m], nv[m])
            for r, ctr in enumerate(at):
                own = gp[aim == r]
                if not len(own):
                    continue
                off = uv[own] - ctr
                # offset of the centre from each point's normal line
                lean = off - (off * nv[own]).sum(1)[:, None] * nv[own]
                rms.append(np.sqrt((lean**2).sum(1).mean()))
                labels[own] = len(centres)
                centres.append(p_point[s] + ctr[0] * u_ax + ctr[1] * v_ax)
        found.append((np.array(centres).reshape(-1, 3), np.array(rms), labels))
    return found


def assigned_loops(pc, indices, centres, pcn=None):
    """The loops as point sets: the slice points assigned to each centre.

    Returns [(slice, slot, S x 3 points, S x 3 normals or None)]; unowned points
    are left out.
    """
    loops = []
    for s, (_, _, lab) in enumerate(centres):
        for r in range(int(lab.max()) + 1 if len(lab) else 0):
            own = indices[s][lab == r]
            if len(own):
                nrm = None if pcn is None else np.asarray(pcn)[own]
                loops.append((s, r, np.asarray(pc)[own], nrm))
    return loops


def bisect_offsets(offsets, add=1):
    """`offsets` with `add` midpoints inserted, each bisecting the widest gap.

    Existing offsets are kept exactly, so their slices and loops stay the same
    across refinement stages. Returns a sorted array of len(offsets) + add.
    """
    offs = np.sort(np.asarray(offsets, dtype=np.float64).reshape(-1))
    if add < 0:
        raise ValueError(f'add must not be negative, got {add}')
    if add and len(offs) < 2:
        raise ValueError(f'a gap needs two planes to lie between, got {len(offs)}')
    for _ in range(add):
        gaps = np.diff(offs)
        k = int(np.argmax(gaps))
        if not gaps[k] > 0:
            raise ValueError(f'no gap to bisect: the planes at {offs} coincide')
        # ties go to the gap nearest the middle, so planes spread over the shape
        widest = gaps >= gaps[k] - 1e-12
        if int(widest.sum()) > 1:
            middle = 0.5 * (offs[0] + offs[-1])
            middles = 0.5 * (offs[:-1] + offs[1:])
            k = int(np.argmin(np.where(widest, np.abs(middles - middle), np.inf)))
        offs = np.insert(offs, k + 1, 0.5 * (offs[k] + offs[k + 1]))
    return offs


def bisect_offsets_trace(offsets, add=1):
    """`bisect_offsets` that also returns (inserted, left, right) per added plane, in order."""
    offs = np.sort(np.asarray(offsets, dtype=np.float64).reshape(-1))
    inserts = []
    for _ in range(add):
        nxt = bisect_offsets(offs, 1)
        # the first position where the arrays differ is the inserted one
        diff = np.nonzero(nxt[:-1] != offs)[0] if len(offs) else np.array([], int)
        k = int(diff[0]) if len(diff) else len(offs)
        left = float(nxt[k - 1]) if k > 0 else None
        right = float(nxt[k + 1]) if k + 1 < len(nxt) else None
        inserts.append((float(nxt[k]), left, right))
        offs = nxt
    return offs, inserts


def plane_keys(p_normal, p_point, loops):
    """Stable id (axis, offset, slot) per loop: its plane and its index on that plane.

    Unlike plane indices these do not change when refinement inserts planes.
    """
    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))
    axis = np.argmax(np.abs(p_normal), axis=1)
    offset = np.einsum('mj,mj->m', p_point, p_normal)
    taken, keys = {}, []
    for s, *_ in loops:
        plane = (int(axis[s]), float(offset[s]))
        keys.append(plane + (taken.get(plane, 0),))
        taken[plane] = taken.get(plane, 0) + 1
    return keys

"""Cross-section loop extraction: the geometry helpers and the class that runs them.

`plc_main` is the entry point and this module has no command line of its own.  It
holds the plane slicing, the Poisson field, the loop centres and the assignment
geometry, the rendering of the extraction figures, and `LoopExtractor`, one object
per point cloud that walks those stages and keeps each result on itself.
"""

# import libraries
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

## the file `plc_main` reads when no path is named
DEFAULT_PLY = str(Path(__file__).resolve().parents[1] / 'Datasets' / 'normalized_shapes_cuboid'
                  / 'normalized_shapes_cuboid' / 'laurent_hand_normal_cubic_lam3.ply')

# functions

# load a ply as an N x 3 point array and its N x 3 normals, if the file stores them
def load_points(path):
    mesh = trimesh.load(path, maintain_order=True, process=False)
    points = np.asarray(mesh.vertices, dtype=np.float64)

    ## a faceless ply loads as a PointCloud, which drops the normals, but trimesh
    ## keeps the raw vertex fields and nx/ny/nz survive there
    raw = mesh.metadata.get('_ply_raw', {}).get('vertex', {}).get('data')
    if raw is not None and raw.dtype.names and {'nx', 'ny', 'nz'} <= set(raw.dtype.names):
        normals = np.stack([raw['nx'], raw['ny'], raw['nz']], axis=1).astype(np.float64)
    elif getattr(mesh, 'vertex_normals', None) is not None:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    else:
        ## a point cloud without normals, e.g. one re-exported from a sliced result
        normals = None
    return points, normals

# slicing point cloud given plane normals and points
def slice_point_cloud(pc, p_normal, p_point, threshold=0.01, intersection=False,
                      pcn=None, max_align=0.95):
    """
    Indices of the points within `threshold` of the planes (p_normal, p_point),
    leaving out the points whose own normal runs along the plane normal.

    pc: N x 3
    p_normal: 3 or M x 3 normals (scaled to unit length here)
    p_point: 3 or M x 3 points on the planes; a lone point is shared by all
    threshold: float (default: 0.01)
    intersection: False -> one index set per plane, True -> near every plane
    pcn: optional N x 3 normals of pc; without them nothing can be left out
    max_align: points whose normal lies within this cosine of a plane normal are
             left out of that plane's slice, so 0.95 drops everything within
             about 18 degrees of it and 1.0 keeps every point. Those points are
             the surface running flat through the slice rather than a contour of
             it: what they add is the cut face's own sheet, whose in-plane normal
             is mostly noise. Lower than this starts eating the walls of a
             faceted shape, whose rims carry the flat faces' normals

    The drop is made per plane, because a point can lie flat in one slice and cut
    across another, and it is made here rather than downstream so that the solve,
    the panels, the field and the exported PLY all see the same points.

    A 1-D normal/point pair counts as M = 1 and returns a bare index array.
    """
    pc = np.asarray(pc, dtype=np.float64)
    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))

    ## normalize so the projection onto the normal is a true point-to-plane distance
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    ## a lone plane point is shared by every plane
    if len(p_point) == 1:
        p_point = np.repeat(p_point, len(p_normal), axis=0)

    ## N x M signed point-to-plane distances, plane axis kept separate
    signed = pc @ p_normal.T - np.einsum('mj,mj->m', p_point, p_normal)
    hit = np.abs(signed) < threshold
    ## a normal parallel to the plane normal leaves nothing of itself in the plane to
    ## solve with: those points are the surface lying flat in the slice, not a contour
    ## through it, so they are dropped from the plane whose normal they run along
    if pcn is not None:
        pcn = np.asarray(pcn, dtype=np.float64)
        pcn = pcn / np.maximum(np.linalg.norm(pcn, axis=1, keepdims=True), 1e-12)
        hit &= np.abs(pcn @ p_normal.T) < max_align

    if intersection:
        return np.flatnonzero(hit.all(axis=1))
    indices = [np.flatnonzero(hit[:, m]) for m in range(hit.shape[1])]
    return indices[0] if len(indices) == 1 else indices

# two unit vectors spanning the plane whose normal is nrm
def plane_basis(nrm):
    helper = np.array([1.0, 0.0, 0.0] if abs(nrm[0]) < 0.9 else [0.0, 1.0, 0.0])
    u = np.cross(nrm, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(nrm, u)

# group a projected slice into loops, single linkage over a spacing-adaptive graph
def slice_loops(uv, k=8, c=2.0):
    """
    Loop label per point of a slice projected onto its plane.

    Density varies sharply along a slice: the tips of an elongated loop collect
    far more points per unit of arc than its flanks, so a fixed k-NN graph uses
    up its k links inside the dense clump and never reaches the sparse ones,
    cutting a single loop into arcs. Each point instead joins every point within
    c times the wider of their two k-th-neighbour distances, which is the local
    spacing measured at whichever end is sparser. Loops that come within that
    spacing of each other still merge, which is the failure at the other end.

    uv: S x 2 projected slice
    k: neighbours used to measure the local spacing
    c: multiple of that spacing that still counts as connected
    """
    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) <= k + 1:
        return np.zeros(len(uv), dtype=int)
    tree = cKDTree(uv)
    dk = tree.query(uv, k=k + 1)[0][:, -1]

    ## the candidate radius is global only to prune pairs; each pair is then
    ## tested against its own two spacings
    pairs = np.array(tuple(tree.query_pairs(c * dk.max())), dtype=int).reshape(-1, 2)
    reach = np.linalg.norm(uv[pairs[:, 0]] - uv[pairs[:, 1]], axis=1)
    keep = reach < c * np.maximum(dk[pairs[:, 0]], dk[pairs[:, 1]])
    graph = coo_matrix((np.ones(keep.sum()), (pairs[keep, 0], pairs[keep, 1])),
                       shape=(len(uv), len(uv)))
    return connected_components(graph, directed=False)[1]

# the point closest to all normal lines of one loop, in its plane
def fit_centre(uv, nv, prior=None, ridge=1e-3):
    """(I - n n^T) projects onto the line along n, so summing it asks for the
    point that lies on every normal line at once.

    A loop that is only a short arc leaves the fit free to slide along the arc,
    along the one direction its normals do not pin down. `prior` is the point to
    fall back on there: the ridge is a thousandth of the loop's own weight, which
    moves a closed loop by nothing and keeps an arc where it was."""
    proj = np.eye(2)[None] - nv[:, :, None] * nv[:, None, :]
    lhs = proj.sum(0)
    rhs = (proj @ uv[:, :, None])[:, :, 0].sum(0)
    if prior is not None:
        lhs = lhs + ridge * len(uv) * np.eye(2)
        rhs = rhs + ridge * len(uv) * np.asarray(prior, dtype=np.float64)
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

# the region one loop encloses, as an indicator solved for by poisson
def poisson_field(uv, nv, res=200, pad=0.2, sigma=2.0, blur=1.0):
    """
    Indicator of the area a closed loop of oriented points bounds.

    The normals of a closed loop point out of the region they enclose, so the
    divergence of the normal field is a sink along the boundary and solving
    laplacian(chi) = -div(V) makes chi near 1 inside and near 0 outside. The
    division is by the 5-point Laplacian in Fourier space, with the constant mode
    dropped so the solution has zero mean: the exterior then reads slightly
    negative instead of being lifted, and the sign inside is unambiguous.

    V is splatted bilinearly and blurred before the divergence is taken. A
    nearest-cell splat leaves V so spiky that the peak of chi is a single cell of
    noise, and every such peak above the threshold registers as another region.

    uv, nv: S x 2 projected slice and its in-plane unit normals
    res: grid cells along the longer side; the walls between two nearby loops
         need a few cells between them to stay apart
    pad: margin around the loop, as a fraction of its longer side
    sigma, blur: smoothing of chi and of V, in cells

    Returns chi on the res x res grid, plus the grid origin and cell size.
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
    ## the constant mode carries no information and would divide by zero
    lam[0, 0] = 1.0
    chi = np.real(np.fft.ifft2(np.fft.fft2(-div) / lam))
    chi[0, 0] = 0.0
    return (gaussian_filter(chi, sigma, mode='constant') if sigma else chi), origin, h

# the implicit field of every loop of one slice, each over its own peak
def slice_fields(uv, nv, k=8, c=2.0, min_points=100):
    """
    What poisson_field solves for, per loop of one slice, divided by its own peak.

    Each field carries its own amplitude: a loop whose normals disagree solves to
    values thousands of times larger than a clean one, and on one shared scale the
    clean ones would be flat.

    uv, nv: S x 2 projected slice and its in-plane unit normals
    k, c, min_points: the grouping of the slice, see slice_loops

    Returns [(chi over its peak, origin, h)] one per group worth solving.
    """
    uv = np.asarray(uv, dtype=np.float64)
    ## fewer points than a group needs is nothing to group, and the tree inside
    ## slice_loops would have no spacing to scale a radius by
    lp = (slice_loops(uv, k, c) if len(uv) >= min_points
          else np.zeros(len(uv), dtype=int))
    fields = []
    for m in (lp == j for j, size in enumerate(np.bincount(lp)) if size >= min_points):
        chi, origin, h = poisson_field(uv[m], nv[m])
        peak = float(chi.max())
        fields.append((chi / peak if peak > 0 else chi, origin, h))
    return fields

# the parts one group's normals enclose, and the part each point of it is on
def field_parts(uv, nv, level=0.0, min_frac=0.02, min_area=0.01, **kw):
    """
    The areas a group's normals enclose, one connected part of the solved
    indicator each, and the part every point of the group sits on.

    The solve is a step through the wall: positive on the side the normals point
    away from, negative on the other, so the level that separates inside from out
    is 0. A level at a fraction of the peak instead cuts the field into one patch
    per stretch of wall, and a single enclosed area then registers as several
    parts -- which the fit draws as several curves wandering across the one loop.

    Every positive connected component is a candidate loop interior. Only the
    relative and absolute area filters below reject one; there is no rim-support
    test, so an open arc that happens to enclose a positive region is retained too.

    level: level of the field that counts as inside, as a fraction of its peak
    min_frac: parts smaller than this fraction of the largest are dropped
    min_area: ... and so are parts under this fraction of the group's own box

    Returns ([(centre 2, area)] largest first, one label per point into that list,
    or -1 for every point when the group encloses nothing.
    """
    uv = np.asarray(uv, dtype=np.float64)
    nv = np.asarray(nv, dtype=np.float64)
    none = np.full(len(uv), -1, dtype=int)
    ## three points at one spot enclose nothing, and the grid below would have no
    ## width to solve on
    if len(uv) < 3 or float(np.ptp(uv, axis=0).max()) <= 0:
        return [], none
    chi, origin, h = poisson_field(uv, nv, **kw)
    blob, count = label(chi > level * chi.max())
    if not count:
        return [], none

    ## only the inside of a part counts towards its centroid, never the grid cells
    ## that merely share its label
    rows, cols = np.indices(chi.shape)
    found = []
    for b in range(1, count + 1):
        mask = blob == b
        mass = np.where(mask, chi, 0.0)
        if mass.sum() <= 0:
            continue
        centroid = np.array([(mass * cols).sum() / mass.sum(),
                             (mass * rows).sum() / mass.sum()]) * h + origin
        ## a concave part's centroid can fall in its hole or outside its rim. Snap it
        ## to the nearest positive cell so the centre is always inside the red region
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

    ## a point's own cell already carries the answer: the field is positive on the
    ## side of the wall the point is on, so the part is read rather than decided. A
    ## point on the outside of the step reads zero or a hair below it and is one cell
    ## from its part, so the nearest part takes it rather than leaving a loop's own
    ## wall unattached to it
    keep = np.isin(blob, [b for b, _, _ in kept])
    near = distance_transform_edt(~keep, return_distances=False, return_indices=True)
    at = np.clip((uv - origin) / h, 0, chi.shape[0] - 1.001).astype(int)
    hit = blob[near[0][at[:, 1], at[:, 0]], near[1][at[:, 1], at[:, 0]]]

    order = np.full(count + 1, -1, dtype=int)
    order[[b for b, _, _ in kept]] = np.arange(len(kept))
    return [(c, float(a)) for _, c, a in kept], order[hit]

## blue -> cyan -> yellow -> red, the ramp mini_nn uses for prediction error, so a
## warm mark means the same thing in both figures
ERROR_RAMP = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]], np.float64)

# RGB per error, taking 0-1 and running blue at 0 through to red at 1
def error_colors(t):
    """One RGB row per value of t, clipped to 0-1 so an off-scale error cannot
    wrap around the end of the ramp."""
    pos = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * (len(ERROR_RAMP) - 1)
    lo = np.floor(pos).astype(int)
    frac = (pos - lo)[:, None]
    rgb = (ERROR_RAMP[lo] * (1 - frac)
           + ERROR_RAMP[np.minimum(lo + 1, len(ERROR_RAMP) - 1)] * frac)
    ## the blend can land a hair over 255 where the stops agree, and scatter
    ## rejects any channel above 1 rather than clamping it
    return np.clip(rgb / 255.0, 0.0, 1.0)

# the loops one group of points encloses, each centre solved from its own part
def field_loops(uv, nv):
    """
    One centre per part of the group's field, and the part each point of it is on.

    First solve the indicator from the points and their normals, then take its
    positive connected components. Finally rebuild the field from each component's
    own points: if those normals enclose several positive components, keep every
    one of them and repeat the same check on each, rather than discarding all but
    the first. Point labels are remapped to the final components as this happens.

    A centre is a point, so every point has a direction to it, and choosing the
    centre a point faces can let two centres claim one area. A field component is
    an area instead: a point is on it or it is not.

    uv, nv: S x 2 points of one group and their in-plane unit normals

    Returns (k x 2 centres, one label per point of uv; -1 where no part holds it).
    """
    uv = np.asarray(uv, dtype=np.float64)
    nv = np.asarray(nv, dtype=np.float64)
    parts, lab = field_parts(uv, nv)
    ## an open arc encloses nothing, and the normal-line fit is all that is left
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
            ## one re-solved part only refines the centre; it is not a split
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

# one centre per loop of a slice, from the point normals
def loop_centres(pc, indices, p_normal, p_point, pcn, k=8, c=2.0, min_points=100):
    """
    For each slice, one 3D centre per enclosed region and the rms distance to the
    normals of the points that region answers for.

    The centre is a positive-field cell nearest the centroid of the area the
    loop's normals enclose, solved for as a Poisson problem. Unlike the centroid
    itself, that cell cannot fall outside a concave region. The point closest to
    all the normal lines, which is what this used to report, is pulled
    off a kinked or folded loop and can land outside it. Which is also why the
    grouping never consults the fitted centre -- an arc of an ellipse fits a
    centre far from the real one, so agreeing centres cannot decide that two arcs
    are one loop.

    A group whose points are two loops merged together still encloses two areas,
    and gets one centre each, which is how two rings too close to separate by
    their points alone stay two. A group that encloses nothing, an open arc being
    the usual case, gets the normal-line fit instead.

    Every loop is answered for by its own points alone. The parting comes first:
    the positive part of the solved field a point sits in is the loop it belongs
    to, read off an area rather than chosen among centres. Each part is then solved
    again from its own points and normals, and every smaller positive part found by
    that solve is kept and checked in turn. What a centre reports is therefore the
    centre of a final positive component, not of a group holding several.

    The rms is the quality tell. Zero means the normals do agree on one spot, as
    they do on a circular or spherical loop. Large means they do not: the loop is
    concave, or elliptical enough that its normals never meet, and no single
    point is the centre of it.

    indices: list of index arrays, one per slice (a bare array counts as one),
             already holding only the points that cut across their plane, which is
             what slice_point_cloud returns; a point lying flat in the slice has no
             in-plane normal to solve with
    p_normal, p_point: the plane of each slice, same order as indices
    pcn: N x 3 normals of pc; without normals there is nothing to solve with
    k, c: how tightly a slice is grouped into loops, see slice_loops
    min_points: groups with fewer points count as sampling debris and are dropped

    Returns one (m x 3 centres, m rms, labels) triple per slice. labels holds
    one index per point of that slice, into its own centres, or -1 where no loop
    owns the point: a group too small to be a loop, and the points of a centre
    that ended up with none of them.
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
        ## one label per point of the slice, so a caller drawing the slice is not
        ## left to guess the parting from the centres alone
        labels = np.full(len(idx), -1, dtype=int)
        if len(uv) < min_points:
            found.append((np.zeros((0, 3)), np.zeros(0), labels))
            continue
        ## only the in-plane part of a normal says anything about an in-plane centre
        nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)

        lp = slice_loops(uv, k, c)
        centres, rms = [], []
        for m in (lp == j for j, size in enumerate(np.bincount(lp)) if size >= min_points):
            gp = np.flatnonzero(m)
            ## the parts the normals enclose and the part each point is on: a group
            ## merged from two rings holds two areas, while one closed area holds one
            ## however many stretches of wall it is made of
            at, aim = field_loops(uv[m], nv[m])
            for r, ctr in enumerate(at):
                own = gp[aim == r]
                if not len(own):
                    continue
                off = uv[own] - ctr
                ## distance from the centre to each normal line, since off minus
                ## its component along n is what is left of the line's lever arm
                lean = off - (off * nv[own]).sum(1)[:, None] * nv[own]
                rms.append(np.sqrt((lean**2).sum(1).mean()))
                labels[own] = len(centres)
                centres.append(p_point[s] + ctr[0] * u_ax + ctr[1] * v_ax)
        found.append((np.array(centres).reshape(-1, 3), np.array(rms), labels))
    return found

# one point set per loop: the slice points that were assigned to one centre
def assigned_loops(pc, indices, centres, pcn=None):
    """
    The loops themselves, as point sets, one per centre the points were assigned
    to.

    A loop is the partition of a slice between its centres, so a plane cutting two
    rings gives two point sets and a plane whose points all lie flat gives none.
    The points no loop owns are left out: they are the surface running flat inside
    the band, not part of any contour.

    pc: N x 3 points
    indices: one index array per slice, same order as centres
    centres: (centres, rms, labels) per slice, as loop_centres returns them
    pcn: N x 3 normals, one per point of pc, or None

    Returns [(slice, centre, S x 3 points, S x 3 normals)]. The normals are the ones
    of the very points the loop holds, in the same order, so a fit can be asked which
    way the surface it follows faces -- which is the one thing the coordinates cannot
    say when two stretches of a contour pass close to each other.
    """
    loops = []
    for s, (_, _, lab) in enumerate(centres):
        for r in range(int(lab.max()) + 1 if len(lab) else 0):
            own = indices[s][lab == r]
            if len(own):
                nrm = None if pcn is None else np.asarray(pcn)[own]
                loops.append((s, r, np.asarray(pc)[own], nrm))
    return loops

## how far a field has to leave its peak before it is drawn at all: the exterior
## decays slowly rather than ending, and painting the faint tail would draw the
## padded grid's own rectangle into every panel
FIELD_FLOOR = 0.05

# how many planes of the first axis there are, which is one row of every grid
def plane_columns(p_normal):
    """The planes arrive grouped by axis, so the first run of equal normals is one
    row of the figure and one axis of the shape."""
    cols = 1
    while cols < len(p_normal) and np.allclose(p_normal[cols], p_normal[0]):
        cols += 1
    return cols

# draw every slice as a 2D panel, projected onto its own plane
def save_slice_grid(img_path, pc, indices, p_normal, p_point, colors, pcn=None, n_arrows=80,
                    centres=None, centre_scale=0.25, field=False, assign=False):
    """
    One panel per slice, holding the slice points projected onto its plane.

    indices: list of index arrays, one per slice
    p_normal, p_point: the plane of each slice, same order as indices
    colors: one RGBA colour per slice, matching the exported PLY
    pcn: optional N x 3 normals of pc, drawn as arrows in each panel
    n_arrows: arrows per panel, thinned out of the slice points; None draws all
    centres: optional list of (m x 3 centres, m residual, per-point labels) per
             slice, as loop_centres returns them; each mark is filled by its
             residual over the largest bounding box extent, so blue sits near zero
             and red at a residual as big as the shape itself
    centre_scale: residual that fills the ramp red, as a fraction of that extent;
             a loop whose normals miss each other by a quarter of the shape is
             already a bad loop, and nothing sane reaches the full extent
    field: draw the implicit field of every loop behind its points, as
             save_field_grid does, 1 over the area the loop encloses and
             negative outside it; needs pcn
    assign: colour each panel's points by the centre they were assigned to, and
             its centre marks with the same colour, so a slice holding several
             loops shows which points went to which; the marks then say which
             loop rather than how big its residual is. The labels are the ones
             loop_centres read off, so a point only joins a loop its own group
             walled in, and the points no loop owns stay grey

    The points a slice holds are already the ones cutting across its plane:
    slice_point_cloud leaves out those lying flat in it, so a panel and the solve
    behind its centres see the same points.

    Panels are laid out one row per plane group and one column per sample along
    it, so each row reads as a sweep through the shape.
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

    ## one yardstick for the whole figure and every model: the residual measured
    ## against the longest side of the bounding box, never against the slice's own
    ## spread, so a colour means the same absolute thing in every panel; the ramp
    ## fills at centre_scale of that, since a residual as large as the whole shape
    ## would be a failure nobody needs a colour for
    extent = float(np.ptp(pc, axis=0).max()) * centre_scale

    ## the slices arrive grouped by plane, so the first run of equal normals is one row
    cols = plane_columns(p_normal)
    rows = -(-len(indices) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows),
                            dpi=120, squeeze=False)

    ## project every slice: the normal component cancels out of both dot products
    ## with the in-plane basis, which is exactly the projection onto the plane
    uv, nv, cv, ce, cl, fields = [], [], [], [], [], []
    for i in range(len(indices)):
        u, v = plane_basis(p_normal[i])
        rel = pc[indices[i]] - p_point[i]
        uv.append((rel @ u, rel @ v))
        ## a vertex normal projected the same way: what is left of it lies in the
        ## panel, and its length is how far the surface tilts within the slice
        if pcn is not None:
            nv.append((pcn[indices[i]] @ u, pcn[indices[i]] @ v))
        if field:
            if pcn is None:
                raise ValueError('the implicit field is solved from the point normals')
            ## the normalisation loop_centres makes, so the field drawn is the one
            ## the centres were read off and not a neighbouring one
            fn = np.stack(nv[i], axis=1)
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            fields.append(slice_fields(np.stack(uv[i], axis=1), fn))
        ## a centre is a point like any other, so the same two dot products drop
        ## it onto the panel; it is built from u and v in the first place, so it
        ## lands back inside the loop it came from
        if centres is not None:
            cs, cr, lab = centres[i]
            rel_c = np.asarray(cs, dtype=np.float64).reshape(-1, 3) - p_point[i]
            cv.append((rel_c @ u, rel_c @ v))
            ce.append(np.asarray(cr, dtype=np.float64).reshape(-1))
            ## already one label per point of the slice as it was sliced, so it
            ## lines up with the points drawn without being dropped by anything
            cl.append(np.asarray(lab, dtype=int))

    ## one common window per row so the panels along a row stay comparable
    windows = {}
    for i, (uu, vv) in enumerate(uv):
        window = windows.setdefault(i // cols, [np.inf, -np.inf, np.inf, -np.inf])
        if uu.size:
            window[0], window[1] = min(window[0], uu.min()), max(window[1], uu.max())
            window[2], window[3] = min(window[2], vv.min()), max(window[3], vv.max())

    ## white at zero, so the enclosed area and the ground outside it part ways at
    ## the level the field crosses on its way out
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)

    for i, (uu, vv) in enumerate(uv):
        ax = axs[i // cols][i % cols]
        ## each field is drawn on the grid it was solved on, at that grid's own place
        ## and cell size, so the panel shows the field the centres came from rather
        ## than a resampling of it; faded under an assignment, whose point colours
        ## would otherwise lose to a saturated background
        for chi, origin, h in (fields[i] if field else ()):
            ax.imshow(np.ma.masked_where(np.abs(chi) < FIELD_FLOOR, chi), origin='lower',
                      cmap='RdBu_r', norm=norm, zorder=0, interpolation='bilinear',
                      alpha=0.3 if assign else None,
                      extent=[origin[0], origin[0] + chi.shape[1] * h,
                              origin[1], origin[1] + chi.shape[0] * h])
        ## the loop each point went to, when the panel is to show the parting; a
        ## loop colour is one of ten reused round the panel, so it names a centre
        ## within its own slice and nowhere else
        picked = None
        if assign and centres is not None and len(cv[i][0]):
            picked = cl[i]
        ## a field behind them would fight a per-slice colour for the same blues and
        ## reds, so the points go dark and say only where the loop runs
        if picked is not None:
            ## a colour per loop has to be read off the dots themselves, so they are
            ## drawn bigger than the single-colour ones; a point no loop owns is
            ## grey, since a colour would claim it for a loop it is not part of
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
        ## after the fields, which would otherwise stretch the panel to their padding
        ax.set_xlim(lo_u - pad, hi_u + pad)
        ax.set_ylim(lo_v - pad, hi_v + pad)

        ## 0 means no arrows but None means all of them, so the test is against 0
        ## rather than a plain truthiness check
        if pcn is not None and len(uu) and n_arrows != 0:
            ## thin the points out, otherwise the arrows merge into a hairball, and
            ## scale them to the panel so a tilt of 1 spans 15% of the window; one
            ## factor for both axes, or the arrow direction would be skewed
            pick = (np.arange(len(uu)) if n_arrows is None or n_arrows >= len(uu)
                    else np.linspace(0, len(uu) - 1, n_arrows).astype(int))
            du, dv = nv[i]
            arrow = 0.15 * max(hi_u - lo_u, hi_v - lo_v)
            ax.quiver(uu[pick], vv[pick], du[pick] * arrow, dv[pick] * arrow,
                      angles='xy', scale_units='xy', scale=1,
                      color='k', width=0.004, headwidth=3, headlength=4)

        ## filled and ringed, not a bare cross: the slice points and the normal
        ## arrows are dense and both dark, so only a light disc stays findable. A
        ## loop named by colour instead is left hollow, since a small loop's own
        ## points fit under its mark and would be hidden by a filled one
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

    ## the marks and the field need a scale or they say nothing: one bar each, side
    ## by side under the panels so neither is read off the other's ramp; coloured by
    ## loop instead, the marks carry no value and there is no bar to draw
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
        ## an axes of its own rather than make_axes: that one is positioned
        ## against the pre-tight_layout panels and lands on top of them
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


# the field each loop's normals enclose, drawn as a colour map per panel
def save_field_grid(img_path, pc, indices, p_normal, p_point, pcn, centres=None,
                    centre_scale=0.25, n_points=1500, k=8, c=2.0, min_points=100):
    """
    One panel per slice, holding the implicit field of every loop in it.

    The field is what poisson_field solves for, divided by its own peak: 1 over
    the area a loop encloses and negative outside it, because the solve is asked
    for zero mean. It is smooth rather than compact, so the loop does not sit at
    zero -- measured on the cuboid hand, the points lie where the field has
    fallen to 0.14-0.37 of its peak and the zero itself crosses about 0.1, a
    tenth of the slice, outside them. The level that runs along the loop is
    therefore the region threshold, half the peak, which is why the centres are
    taken at that level and land inside. So the colour map shows the same thing
    the centres are read off: a loop whose points leave a stretch unwalled fills
    no area at all, and its centre falls back to the normal-line fit.

    Each field carries its own peak, so each is shown at its own scale. A loop
    whose normals disagree solves to values thousands of times larger than a
    clean one, and on one shared scale the clean ones would be flat white.

    Each field is drawn on the grid it was solved on, at that grid's own place
    and cell size, so a panel is a picture of the field the centres came from
    rather than a resampling of it.

    img_path: png to write
    pc, indices, p_normal, p_point, pcn: as loop_centres
    centres: optional (m x 3 centres, m rms, labels) triples per slice, marked as
             they are in the slice grid
    centre_scale: the residual the centre ramp calls red, as in save_slice_grid
    n_points: loop points drawn per panel, thinned out; 0 for none, None for all
    k, c, min_points: the grouping of each slice, see slice_loops

    The points a slice holds are already the ones cutting across its plane, as
    slice_point_cloud leaves the rest out, so the field is solved from exactly the
    points loop_centres solved it from.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    pc = np.asarray(pc, dtype=np.float64)
    p_normal = np.atleast_2d(np.asarray(p_normal, dtype=np.float64))
    p_normal = p_normal / np.linalg.norm(p_normal, axis=1, keepdims=True)
    p_point = np.atleast_2d(np.asarray(p_point, dtype=np.float64))

    ## the slices arrive grouped by plane, so the first run of equal normals is one row
    cols = plane_columns(p_normal)
    rows = -(-len(indices) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows),
                            dpi=120, squeeze=False)
    ## white sits at zero, so the filled area and the ground outside it part ways
    ## at the level the loop runs along
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    ## the same yardstick save_slice_grid fills its centres by
    extent = float(np.ptp(pc, axis=0).max()) * centre_scale

    uv, fields, cv, ce = [], [], [], []
    for i in range(len(indices)):
        u, v = plane_basis(p_normal[i])
        rel = pc[indices[i]] - p_point[i]
        uv.append((rel @ u, rel @ v))
        nv = np.stack([pcn[indices[i]] @ u, pcn[indices[i]] @ v], axis=1)
        ## only the in-plane part of a normal says anything about an in-plane field
        nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
        fields.append(slice_fields(np.stack(uv[i], axis=1), nv, k, c, min_points))
        if centres is not None:
            cs, cr, _ = centres[i]
            rel_c = np.asarray(cs, dtype=np.float64).reshape(-1, 3) - p_point[i]
            cv.append((rel_c @ u, rel_c @ v))
            ce.append(np.asarray(cr, dtype=np.float64).reshape(-1))

    ## one common window per row so the panels along a row stay comparable
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

        ## the grid's own corners are its extent, so the map lands where it was
        ## solved; the padding is masked instead of drawn, or every panel would
        ## carry a pale rectangle whose edge reads as a boundary of the shape
        for chi, origin, h in fields[i]:
            ax.imshow(np.ma.masked_where(np.abs(chi) < FIELD_FLOOR, chi), origin='lower',
                      cmap='RdBu_r', norm=norm, zorder=0, interpolation='bilinear',
                      extent=[origin[0], origin[0] + chi.shape[1] * h,
                              origin[1], origin[1] + chi.shape[0] * h])

        ## the points say where the loop runs, but a dense slice would cover the
        ## field they were solved from, so they are thinned and left translucent
        if n_points != 0 and len(uu):
            pick = (np.arange(len(uu)) if n_points is None or n_points >= len(uu)
                    else np.linspace(0, len(uu) - 1, n_points).astype(int))
            ax.scatter(uu[pick], vv[pick], s=0.8, c='k', alpha=0.6,
                       linewidths=0, zorder=1)
        label = f'{"xyz"[i // cols]} = {p_point[i] @ p_normal[i]:.2f}' if rows <= 3 else f'p{i}'
        ax.set_title(f'{label} | {indices[i].size:,} pts', fontsize=7)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=5, length=2)

        ## filled and ringed, not a bare cross: the slice points and the normal
        ## arrows are dense and both dark, so only a light disc stays findable
        if centres is not None and len(cv[i][0]):
            ax.scatter(*cv[i], s=60, marker='o', facecolors=error_colors(ce[i] / extent),
                       edgecolors='k', linewidths=1.2, zorder=3)

    for i in range(len(uv), rows * cols):
        axs[i // cols][i % cols].axis('off')

    fig.suptitle(f'{len(uv)} slices projected onto their planes | implicit field, '
                 f'1 over the area a loop encloses, negative outside it', fontsize=9)
    fig.tight_layout()
    ## an axes of its own rather than make_axes, which is positioned against the
    ## pre-tight_layout panels and lands on top of them
    fig.subplots_adjust(bottom=0.10)
    cax = fig.add_axes([0.3, 0.035, 0.4, 0.016])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                       cax=cax, orientation='horizontal')
    bar.ax.tick_params(labelsize=6)

    fig.savefig(img_path)
    plt.close(fig)
    print(f'  {rows} x {cols} field grid -> {img_path}')


# save to ply, optionally as a projected image grid
def visualize_sliced_point_cloud(saved_path, pc, indices, p_normal=None, p_point=None,
                                 vis_to_img=True, pcn=None, centres=None, n_arrows=80,
                                 centre_scale=0.25, field=False, assign=False):
    """
    Export the sliced points to one PLY, one colour per slice.

    indices: list of index arrays, one per slice (a bare array counts as one)
    p_normal, p_point: plane of each slice, needed for the projected image grid
    vis_to_img: also write the .png next to saved_path with every slice projected
    pcn: optional N x 3 normals of pc, carried into the PLY as nx/ny/nz
    centres: optional list of (m x 3 centres, m residual) pairs per slice, marked
             in the grid only -- they stay out of the PLY, which must hold slice
             points alone so it can be loaded back and sliced again
    n_arrows: normals drawn per panel, 0 for none, None for every one of them
    centre_scale: residual that fills the centre ramp red, as a fraction of the
             largest bounding box extent, see save_slice_grid
    field: draw the implicit field behind the points of each panel
    assign: colour the points of each panel by the loop centre they belong to
    """
    if isinstance(indices, np.ndarray):
        indices = [indices]
    rng = np.random.default_rng(0)

    ## one colour per slice; a seeded rng keeps runs comparable
    slice_colors = np.stack([np.append(rng.integers(0, 255, 3), 255) for _ in indices])
    colors = np.repeat(slice_colors, [idx.size for idx in indices], axis=0)
    points = np.concatenate([pc[idx] for idx in indices])
    print(f'  {len(indices)} slices, {len(points)} points -> {saved_path}')

    if pcn is None:
        ## a PointCloud, not a faceless Trimesh: concatenating faceless meshes gives
        ## a (0,) face array that export_ply cannot write and drops the vertex colours
        cloud = trimesh.PointCloud(points, colors=colors.astype(np.uint8))
    else:
        ## a PointCloud export has no nx/ny/nz field, so hand the already merged
        ## points to one faceless Trimesh, which writes them
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


## loop vertex colours by the axis of the plane that cut them, in x/y/z order
AXIS_COLORS = np.array([[0x45, 0x75, 0xb4, 255],
                        [0x2a, 0x9d, 0x8f, 255],
                        [0xd1, 0x49, 0x5b, 255]], dtype=np.uint8)
AXIS_HEX = tuple('#' + ''.join(f'{c:02x}' for c in col[:3]) for col in AXIS_COLORS)


# write one point set to a ply, with normals when it has them
def write_points(path, points, colors, normals=None):
    """
    points: S x 3
    colors: S x 4 uint8
    normals: optional S x 3; without them the cloud is a trimesh PointCloud, which
             has no nx/ny/nz field, and with them one faceless Trimesh, which does
    """
    if normals is None:
        trimesh.PointCloud(np.asarray(points), colors=np.asarray(colors)).export(path)
        return
    cloud = trimesh.Trimesh(vertices=np.asarray(points),
                            faces=np.zeros((0, 3), dtype=np.int64),
                            vertex_normals=np.asarray(normals), process=False)
    cloud.visual.vertex_colors = colors
    cloud.export(path)

# the target and the fit of every loop, projected onto its plane, with its error
def fit_items(loops, preds, p_normal, p_point):
    """
    One entry per loop, (centre, target in plane, fit in plane, mean error),
    grouped by slice, and the mean out-of-plane distance of the fits.

    The projection is the same pair of dot products that drew the slices, so the
    panel and the points agree; how far a fit leaves its plane is dropped by it and
    therefore has to be measured here or it would go unreported. The error is the
    distance from every predicted point to the loop it was fitted to, taken on the
    whole loop rather than the M-point sample the trainer happened to draw.
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


# the predicted normal of every loop, projected onto its plane like the fit itself
def fit_normal_items(loops, pred_normals, p_normal):
    """
    One (nu, nv) arrow array per loop, grouped by slice and in the same order as
    fit_items returns that slice's loops.

    normals are 3D vectors, so they are projected with the same plane basis the
    target and fit points use, and the projection keeps the length it arrives with
    instead of being scaled to one: the arrows stay in proportion to each other, so
    a longer normal draws a longer arrow on the same panel scale.
    """
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

# the box that holds every loop point and every fitted point of one panel
def fit_window(items):
    """lo_u, hi_u, lo_v, hi_v over the targets and the fits together: a curve that
    runs off its loop is a result too, and clipping it would hide it."""
    lo_u = min(min(tu.min(), pu.min()) for _, (tu, _), (pu, _), _ in items)
    hi_u = max(max(tu.max(), pu.max()) for _, (tu, _), (pu, _), _ in items)
    lo_v = min(min(tv.min(), pv.min()) for _, (_, tv), (_, pv), _ in items)
    hi_v = max(max(tv.max(), pv.max()) for _, (_, tv), (_, pv), _ in items)
    return lo_u, hi_u, lo_v, hi_v

# the run's own error scale, and the multiple of it that counts as large
def error_scale(per, large=1.5):
    """The median of the per-plane mean errors, and the value past which a plane is
    called large.

    An absolute limit alone marks nothing on a run whose fits all sit inside it,
    however much worse its worst plane is than the rest, which is the case for a
    well behaved shape: 45 planes, none over 3% of the extent, yet the worst is two
    and a half times the median. The median says where the run itself sits, so the
    planes worth opening are named either way.
    """
    means = [float(np.mean([e for *_, e in items])) for items in per.values()]
    typical = float(np.median(means)) if means else 0.0
    return typical, (large * typical if typical > 0 else np.inf)

# one loop's error, with the mark that says it is past the limit
def mark_error(err, max_err=None, digits=3):
    """A "!" in front of an error past max_err, so a table or a title says which of
    its numbers the figure drew red without needing a second column."""
    return f'{err:.{digits}f}' if max_err is None or err <= max_err else f'!{err:.{digits}f}'

# draw one panel of the fit figure
## the fitted curve is coloured by the loop parameter it was predicted at.  The
## parameter is periodic, so the map has to be cyclic: the colour at one full turn is
## the colour at zero, which is the seam the ring on each curve already marks
PARAM_CMAP = 'hsv'


def param_colorbar(fig, ax):
    """The bar that reads a fitted curve's colour back as a loop parameter.

    Shared by the fit grid and the single-plane figures so a curve means the same
    parameter in both.  The ticks are turns of the parameter rather than radians,
    since the network is evaluated on one uniform walk from 0 to 1.
    """
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
    """
    The points of every loop on one plane and the curve fitted to each, both
    projected onto that plane, in the loop's own colour.

    A loop past max_err is one the fit did not follow.  It keeps its points in the
    assignment colour, so it is still identifiable beside the others, but its curve
    casing turns red and heavier and its error is written at its centre: that mark
    is what a loop is meant to be removed by.

    The curve itself is coloured by the loop parameter, one turn of a cyclic map
    from the first predicted point back to it, so the direction and pace of the walk
    show on the curve: a stretch that changes colour slowly is one the fit crosses
    quickly.  The thin casing under the colours keeps the loop's own colour.

    items: one entry per loop, as fit_items returns them
    window: lo_u, hi_u, lo_v, hi_v to frame, as fit_window returns it
    max_err: absolute error past which a loop counts as badly fitted, or None
    normals: optional (nu, nv) arrays, as fit_normal_items returns them, one per
             item in the same order; arrows are drawn from the fitted curve along
             them exactly as predicted, lengths included, with no outward sign
             correction and nothing scaled to unit length
    n_arrows: predicted-normal arrows per loop, 0 for none, None for every point
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    lo_u, hi_u, lo_v, hi_v = window
    ## one factor for every arrow on the panel, so their lengths stay in the
    ## proportion the prediction gives them; it is small because a normal is about
    ## one loop perimeter long, not one, and the arrows are read beside the curve
    arrow = 0.04 * max(hi_u - lo_u, hi_v - lo_v, 1e-9)
    normals = [None] * len(items) if normals is None else normals

    for (c, (tu, tv), (pu, pv), err), normal in zip(items, normals):
        ## the loop colour of the assignment panel, so a curve and the points it came
        ## from are the same loop at a glance; the line is the darker of the two,
        ## since a curve drawn over its own dots would hide them
        over = max_err is not None and err > max_err
        colour = np.array(plt.cm.tab10(c % 10))
        dark = colour.copy()
        dark[:3] *= 0.55
        if over:
            dark = np.array([0.75, 0.10, 0.10, 1.0])
        ax.scatter(tu, tv, s=1.8, c=colour[None, :], alpha=0.5, linewidths=0)
        ## closed onto the first point, since the parameter wraps and the fit is
        ## periodic in it; a ring marks where that seam was.  Each segment carries
        ## the parameter it starts at as a colour, so the order the network walks
        ## the loop in can be read off the curve instead of only its shape, and a
        ## curve that doubles back shows the same colour twice
        if len(pu) > 1:
            ends = np.stack([np.append(pu, pu[0]), np.append(pv, pv[0])], axis=1)
            segs = np.stack([ends[:-1], ends[1:]], axis=1)
            z = np.linspace(0.0, 1.0, len(segs))
            ## a casing in the loop's own colour, heavier where the loop is bad, so
            ## the curve still belongs to its points and still reads red past the
            ## limit under the parameter colours
            ax.add_collection(LineCollection(
                segs, colors=[dark], linewidths=3.2 if over else 2.0,
                capstyle='round', joinstyle='round', zorder=4 if over else 3))
            ax.add_collection(LineCollection(
                segs, array=z, cmap=PARAM_CMAP, norm=Normalize(0.0, 1.0),
                linewidths=2.0 if over else 1.1, capstyle='round',
                joinstyle='round', zorder=4.5 if over else 3.5))
        ax.plot(pu[0], pv[0], marker='o', ms=4, mfc='none', mec=dark, mew=0.8)
        ## predicted normals are a property of the fit, so they start on the fitted
        ## curve rather than on the source points; only the one panel factor below
        ## is applied to them, so the prediction's own lengths survive and scale
        ## with each other instead of all reading as unit vectors
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

# project the fitted curves back onto the planes their loops were cut from
def save_fit_grid(img_path, loops, preds, p_normal, p_point, max_err=None, large=1.5,
                  pred_normals=None, n_arrows=20, align=None):
    """
    One panel per slice, holding the loop it was fitted to and the curve the fit
    produced, both projected onto the slice's own plane.

    The curves are 3D points the network predicts at one parameter value each, so
    the projection is the same pair of dot products that drew the slices: what the
    plane component takes away is the only part of the fit the panel cannot show,
    which is why it is reported as a distance rather than drawn.

    loops: (slice, centre, S x 3 points) per loop, as assigned_loops returns them
    preds: N x 3 predictions, one per loop of loops and in the same order; the
             order along the parameter is the order of the rows, so the curve is
             drawn through them and closed back onto the first
    p_normal, p_point: the plane of each slice, indexed as the loops' slice
    max_err: absolute error past which a loop is drawn red, and its panel's title
             red with the count of them, so a badly fitted plane stands out from the
             thirty around it
    large: multiple of the run's median plane error past which a panel's title is
             marked and turned red even when no loop is past max_err, which is what
             names the worst planes of a run that is otherwise inside the limit
    pred_normals: optional N x 3 predicted in-plane normal per fitted point, one
             array per loop and in the same order as preds; drawn from the fitted
             curve as arrows, facing the way the target normals do rather than
             against them, and without being scaled to unit length, so a longer
             normal draws a longer arrow
    n_arrows: predicted-normal arrows per loop, 0 for none, None for every point
    align: optional alignment penalty over the whole run, 0 when every fitted curve
             runs the way its target points' normals say, 2 when it runs against
             them; written into the figure's title, since no panel can show it

    Each loop keeps the colour its centre was drawn with in the assignment panels
    for its points and for the casing under its curve; the curve's own colour is
    the loop parameter, with one bar under the figure to read it by.
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

    ## one common window per row, targets and fits together: a curve that runs off
    ## its loop is a result too, and clipping it would hide it
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
        ## a plane is named when one of its loops is past the absolute limit or when
        ## its own mean is far above the run's, since the two say different things:
        ## one loop badly fitted against every loop a little badly fitted
        big = over > 0 or mean > limit
        large_planes += big
        mark = f', {over} over' if over else ''
        title = (f'{label} | {len(per[s])} loop{"" if len(per[s]) == 1 else "s"}, '
                 f'err {mean:.3f}{mark}')
        ax.set_title(f'! {title}' if big else title, fontsize=7,
                     color='#bf1b1b' if big else 'black')

    for i in range(len(p_normal), rows * cols):
        axs[i // cols][i % cols].axis('off')

    ## the out-of-plane part is the one number the projection cannot carry, so it is
    ## stated once for the figure rather than left out; the counts say how much of the
    ## run is marked without counting panels.  The alignment penalty is one number
    ## over the whole run too, so it sits in the same line.
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


# every plane on its own: its slice, its loops and the curve fitted to each
def save_plane_files(out_dir, pc, pcn, slices, centres, loops, preds,
                     p_normal, p_point, max_err=None, large=1.5,
                     pred_normals=None, n_arrows=20, write_files=True):
    """
    One set of files per slice plane, so a plane can be loaded, coloured and
    compared on its own instead of only inside the run's combined figures.

    With write_files, under <out_dir>/planes/:
      plane_SS_<axis><offset>_slice.ply   the plane's points as they were sliced
      plane_SS_<axis><offset>_loops.ply   the same points coloured by the loop each
                                          was assigned to, grey where none
      plane_SS_<axis><offset>_fit.ply     the curve the fit produced for each loop,
                                          in that loop's colour, with the predicted
                                          in-plane normal in nx/ny/nz when available
      plane_SS_<axis><offset>_fit.png     the loops and their fits projected onto
                                          the plane
    and <out_dir>/planes.txt, one line per plane with its counts and its errors,
    where a "!" marks an error past max_err.

    <out_dir>/bad_loops.txt lists those loops alone, one per line with the plane,
    the centre, the point count and the error, since that is the list a loop is
    dropped from. The two text files are written whether or not the per-plane
    files are: they are the run's summary, not the per-plane detail.

    pc, pcn, slices, centres: as slice_point_cloud and loop_centres return them
    loops, preds: as save_fit_grid takes them
    p_normal, p_point: the plane of each slice, indexed as the loops' slice
    max_err: absolute error past which a loop is marked, or None to mark none
    large: multiple of the run's median plane error past which a plane's own figure
             is marked too, so a panel opened on its own carries the same verdict it
             has in the grid
    pred_normals: optional N x 3 predicted in-plane normal per fitted point, one
             array per loop and in the same order as preds; written into the fit
             PLY as nx/ny/nz and drawn as arrows without an outward sign correction
    n_arrows: predicted-normal arrows per loop, 0 for none, None for every point
    write_files: write the per-plane PLYs and PNGs under <out_dir>/planes/; False
             skips that folder entirely and writes only the two text files
    """
    if write_files:
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

    ## the fits as 3D points, kept beside the projection so the curves can be loaded
    ## next to the loops they came from
    fit_at, normal_at = {}, {}
    normal_per = fit_normal_items(loops, pred_normals, p_normal)
    for (s, c, *_), pred, normals in zip(loops, preds,
                                         pred_normals or [None] * len(preds)):
        fit_at.setdefault(s, []).append((c, np.asarray(pred, dtype=np.float64)))
        if normals is not None:
            normal_at.setdefault(s, []).append(np.asarray(normals, dtype=np.float64))

    out = Path(out_dir) / 'planes'
    if write_files:
        out.mkdir(parents=True, exist_ok=True)
    lines = [f'{len(slices)} planes, {len(loops)} loops']
    size_of = {(s, c): len(pts) for s, c, pts, _ in loops}
    bad = []
    large_planes = 0
    for s, idx in enumerate(slices):
        axis = 'xyz'[s // cols] if s // cols < 3 else f'p{s // cols}'
        offset = p_point[s] @ p_normal[s]
        base = str(out / f'plane_{s:02d}_{axis}{offset:+.2f}') if write_files else None
        items = per.get(s, [])
        mean = float(np.mean([e for *_, e in items])) if items else 0.0
        over = max_err is not None and any(e > max_err for *_, e in items)
        ## named when a loop of it is past the absolute limit or when the plane's own
        ## mean is far above the run's, the same rule the grid titles use
        big = bool(items) and (over or mean > limit)
        large_planes += big

        if write_files:
            ## the slice as it was cut: one grey, so a viewer shows the contour itself
            grey = np.full((len(idx), 4), 200, dtype=np.uint8)
            grey[:, 3] = 255
            write_points(f'{base}_slice.ply', pc[idx], grey,
                         None if pcn is None else np.asarray(pcn)[idx])

            ## the same points again, coloured by loop, which is the parting the solve saw
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
                ## this figure is opened on its own, so it carries the multiple rather
                ## than leaving the reader to remember the grid's median
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
    limit_txt = 'no max_err' if max_err is None else f'{max_err:.4f}'
    dest = (f'per-plane files in {out}, ' if write_files else '')
    print(f'  {len(slices)} planes, {len(loops)} loops, {large_planes} planes over '
          f'{large:g}x the {typical:.4f} median, {len(bad)} loops over {limit_txt}: '
          f'-> {dest}{Path(out_dir) / "planes.txt"}, '
          f'{Path(out_dir) / "bad_loops.txt"}')


# the folder a shape's outputs go to: log/<input stem>, or the root the caller names
def run_dir(path, root=None):
    """One folder per shape under `log`, so two shapes never share one and a re-run
    overwrites only its own. A caller that names a root gets exactly that folder,
    which is what makes side-by-side experiments possible."""
    return Path('log') / Path(path).stem if root is None else Path(root)


def bisect_offsets(offsets, add=1):
    """
    `offsets` with the midpoints of its `add` widest gaps inserted, kept sorted.

    Refining by bisection rather than by re-spacing the axis is what lets a fit be
    handed on: an offset that was there before is still there, at the same value, so
    its slice holds the same points and the loops read off it are the same loops.
    A re-spaced axis would move every plane a little, and every loop with it, so no
    loop of one stage would be the counterpart of any loop of the next.

    add: planes to insert; each one bisects the widest gap left, so a stage that
             asks for more plates than the axis has gaps still gets them, by
             bisecting the halves it has just made

    Returns a sorted 1-D array `len(offsets) + add` long.
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
        ## the widest gaps are usually tied, and the first of them would add every
        ## plane from one end of the axis; the tie goes to the one nearest the middle
        ## instead, so a stage spreads its planes over the shape rather than crawling
        ## in from a corner
        widest = gaps >= gaps[k] - 1e-12
        if int(widest.sum()) > 1:
            middle = 0.5 * (offs[0] + offs[-1])
            middles = 0.5 * (offs[:-1] + offs[1:])
            k = int(np.argmin(np.where(widest, np.abs(middles - middle), np.inf)))
        offs = np.insert(offs, k + 1, 0.5 * (offs[k] + offs[k + 1]))
    return offs


def bisect_offsets_trace(offsets, add=1):
    """
    `bisect_offsets`, plus where every inserted plane came from.

    Returns (offsets, inserts), where `inserts` is one (inserted, left, right) per
    plane added, in the order they were added.  A plane added into a gap that an
    earlier insertion of the same call had already split has that earlier plane as
    a parent and comes after it in the list, so a caller can seed each inserted
    plane from rows its parents already hold.

    `bisect_offsets` itself is used unchanged, one insertion at a time: an inserted
    offset is the one value the sorted array did not have before, and its two
    neighbours in the new array are the planes it lies between.
    """
    offs = np.sort(np.asarray(offsets, dtype=np.float64).reshape(-1))
    inserts = []
    for _ in range(add):
        nxt = bisect_offsets(offs, 1)
        ## one value was added and the order is kept, so the first position where
        ## the two arrays differ is the inserted one; an insertion at the far end
        ## leaves every earlier position equal and lands at len(offs)
        diff = np.nonzero(nxt[:-1] != offs)[0] if len(offs) else np.array([], int)
        k = int(diff[0]) if len(diff) else len(offs)
        left = float(nxt[k - 1]) if k > 0 else None
        right = float(nxt[k + 1]) if k + 1 < len(nxt) else None
        inserts.append((float(nxt[k]), left, right))
        offs = nxt
    return offs, inserts


def plane_keys(p_normal, p_point, loops):
    """
    For every loop, the plane it was cut on and its slot among that plane's loops.

    A plane is named by the axis it cuts along and where along it, never by its index,
    which shifts as soon as a plane is added before it. Bisection keeps the offset of
    every plane already cut, so the same plane is the same pair of numbers at every
    stage of a refined run, and a retained plane's slice, centres and loops are the
    same ones in the same order.

    `(axis, offset, slot)` is therefore stable for the whole run, and it is what a
    loop's embedding row is looked up by: the row is never renumbered, so a fit is
    still where it was when a stage adds planes around it. A plane that cuts two
    rings needs two rows, which is what the slot is for -- slot 0 is the plane's own
    code, 1 its second loop, and so on, in the order `assigned_loops` gives them.

    p_normal, p_point: the planes of the extraction, axis-major
    loops: assigned_loops output, (slice, centre, points, normals) per loop

    Returns one (axis, offset, slot) tuple per loop, in the order they came in.
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


# the cross-section loop pipeline of one shape, as one object
class LoopExtractor:
    """
    Slice a point cloud with axis-aligned planes, then part every slice into loops.

    One object is one shape and one run folder.  The stages are separate methods,
    so a caller can stop after any of them and look at what it has, and `extract`
    is the whole chain in order.  Nothing is read until `load` runs.

    After `extract`:
      points, normals      N x 3 cloud and its N x 3 normals, None when the file
                           stores none
      p_normal, p_point    M x 3 planes, axis-major: slice i * samples + k is
                           sample k along axis i
      slices               M index arrays into points, one per plane
      centres              one (m x 3 centres, m rms, labels) per plane, as
                           loop_centres returns them; None without normals
      loops                (slice, centre, S x 3 points, S x 3 normals) per loop

    samples planes are laid down per axis, evenly spaced inside the bounding box
    and inset by `margin` of each axis extent, so no plane grazes the tips of the
    shape.  A point belongs to a slice when its distance to that plane is below
    `band` of the longest extent, which keeps the band width model-relative
    instead of tied to one copy of the fixture.
    """

    def __init__(self, path, samples=5, margin=0.1, band=0.0075, max_align=0.95,
                 out=None):
        """
        path: ply to read
        samples: planes sampled per axis
        margin: inset from each bounding box face, as a fraction of the axis extent
        band: half-width of the slab a point has to fall in to count as on a plane,
                 as a fraction of the longest extent. It is also the depth of
                 surface that gets projected flat onto the plane, so it is what
                 smears a contour along its own thickness; narrower is a crisper
                 loop but fewer points, and under about 0.006 the thinnest walls
                 stop being resolved into two regions, so the marginal splits close
                 up again
        max_align: points whose normal lies within this cosine of a plane normal are
                 left out of that plane's slice, see slice_point_cloud; they are the
                 surface lying flat in the slice, so nothing downstream sees them.
                 0.95 is about 18 degrees, which is the width of a lobe of the same
                 order as the sheet itself rather than a sampling artefact, and it
                 takes a third of the cuboid's slice points with it; 0.999 is 2.6
                 degrees, and 1.0 keeps every point
        out: the folder to write in, defaulting to log/<input stem>/, so two shapes
                 never share one and a shape's re-run overwrites only its own
        """
        if not 0.0 <= margin < 0.5:
            raise ValueError(f'margin fraction must be in [0, 0.5), got {margin}')
        if samples < 1:
            raise ValueError(f'samples must be at least 1, got {samples}')
        if band <= 0.0:
            raise ValueError(f'band must be positive, got {band}')
        if not 0.0 < max_align <= 1.0:
            raise ValueError(f'max_align must be in (0, 1], got {max_align}')

        self.path = Path(path)
        ## named after the input, so a re-run can never overwrite the source
        self.stem = self.path.stem
        self.samples = samples
        self.margin_frac = margin
        self.band = band
        self.max_align = max_align
        self.out = run_dir(self.path, out)

        self.points, self.normals = None, None
        self.threshold, self.offsets = None, None
        self.p_normal, self.p_point, self.slices = None, None, None
        self.centres, self.loops = None, []
        ## how this shape was last asked to save, so a re-cut can write the same
        ## artefacts again without the caller repeating the switches
        self.save_opts = None

    def load(self):
        """Read the cloud and its normals, and say what the file holds."""
        self.points, self.normals = load_points(self.path)
        print(f'{self.path}\n  {self.points.shape[0]} points, '
              f'{0 if self.normals is None else self.normals.shape[0]} normals, '
              f'extent {np.ptp(self.points, axis=0).round(2)}')
        return self

    def make_planes(self, offsets=None):
        """
        The planes to cut along, axis-major, and the band they are cut with.

        offsets: one sorted array of positions per axis, instead of the `samples`
                 evenly spaced planes. This is what a re-cut uses: a stage that keeps
                 every plane the stage before had and bisects some of its gaps cannot
                 say that with a count, and passing the positions is what keeps a
                 plane, its slice and its loops the same across the stages
        """
        ## a point counts as "on" a plane when it is within `band` of the longest
        ## extent. The band is what smears a contour: everything inside it is
        ## projected flat onto the plane, so a band a hundredth of the shape deep
        ## drags the surface along its own thickness into the loop. Three quarters of
        ## a percent costs a third of the points per plane and keeps the loop
        ## structure on the smooth hand exactly, which is the shape of the trade:
        ## measured, the cuboid's first x plane gains a sliver the wide band had
        ## merged and its marginal rings thin out, and below 0.006 they stop
        ## clearing the region gates at all
        self.threshold = self.band * np.ptp(self.points, axis=0).max()
        print(f'  band {self.band * 100:.2f}% of the longest extent '
              f'({self.threshold:.4f})')
        ## the other half of what a slice holds: which normals count as cutting
        ## across it
        if self.normals is not None:
            print(f'  max_align {self.max_align}: normals within '
                  f'{np.degrees(np.arccos(min(self.max_align, 1.0))):.1f} deg of a '
                  f'plane normal are left out of that plane')

        ## the planes are sampled along three axes, kept off the bounding box faces
        ## by a per-axis margin so none grazes the tips of the shape
        lo, hi = self.points.min(axis=0), self.points.max(axis=0)
        margin = self.margin_frac * np.ptp(self.points, axis=0)
        if offsets is None:
            offsets = np.linspace(lo + margin, hi - margin, self.samples).T
        else:
            offsets = [np.sort(np.asarray(o, dtype=np.float64).reshape(-1))
                       for o in offsets]
            if len(offsets) != 3:
                raise ValueError(f'one offset array per axis, got {len(offsets)}')
            if len({len(o) for o in offsets}) != 1:
                raise ValueError(f'every axis needs the same number of planes, got '
                                 f'{[len(o) for o in offsets]}')
            self.samples = len(offsets[0])
        self.offsets = np.stack(offsets)                                # per axis
        self.p_normal = np.repeat(np.eye(3), self.samples, axis=0)      # axis i, n times
        self.p_point = np.tile(0.5 * (lo + hi), (3 * self.samples, 1))  # box centre
        ## only the coordinate along the plane normal enters the distance, so the
        ## sampled offset replaces that one coordinate and the rest stay centred
        self.p_point[np.arange(3 * self.samples),
                     np.repeat(np.arange(3), self.samples)] = self.offsets.ravel()
        return self

    def cut(self):
        """The points of every plane, one index array each, axis-major."""
        self.slices = slice_point_cloud(self.points, self.p_normal, self.p_point,
                                        self.threshold, pcn=self.normals,
                                        max_align=self.max_align)
        margin = self.margin_frac * np.ptp(self.points, axis=0)
        for i, axis in enumerate('xyz'):
            counts = [idx.size
                      for idx in self.slices[i * self.samples:(i + 1) * self.samples]]
            print(f'  {axis}: {self.samples} planes at '
                  f'{self.offsets[i].min():.2f}..{self.offsets[i].max():.2f} '
                  f'(margin {margin[i]:.2f}), {sum(counts)} points, '
                  f'{min(counts)}-{max(counts)} per plane')
        return self

    def find_centres(self, assign=False):
        """
        One centre per loop of every slice, and the loop each point went to.

        A plane may cut several loops, and the normals of each one name its centre.
        assign: also print how many points each centre took, and how many no loop
                  owns
        """
        if self.normals is None:
            self.centres = None
            return self
        self.centres = loop_centres(self.points, self.slices, self.p_normal,
                                    self.p_point, self.normals)
        for i, axis in enumerate('xyz'):
            row = self.centres[i * self.samples:(i + 1) * self.samples]
            print(f'  {axis}: loops per plane {[len(c) for c, _, _ in row]}')
            ## which loop each point went to: the grouping already answers that, so
            ## a point is only ever counted for a loop its own group walled in. The
            ## rest are the flat points the panels leave out and the debris of the
            ## grouping
            if assign:
                sizes = [[int((lab == k).sum()) for k in range(len(c))]
                         for c, _, lab in row]
                loose = [int((lab < 0).sum()) for _, _, lab in row]
                print(f'  {axis}: points per centre {sizes}, unused {loose}')
        return self

    def group_loops(self):
        """
        The loops themselves, one point set per centre the points were assigned to.

        The training targets are the loops, not the slices: a plane cutting two
        rings is two loops, and a plane that only grazes the shape is none. Each
        loop carries the normals of its own points, in the same order, so a fit can
        ask which way the surface it follows faces.
        """
        self.loops = ([] if self.centres is None
                      else assigned_loops(self.points, self.slices, self.centres,
                                          self.normals))
        if self.loops:
            sizes = [len(p) for _, _, p, _ in self.loops]
            print(f'  loops: {len(self.loops)} from {len(self.slices)} planes, '
                  f'{min(sizes)}-{max(sizes)} points each')
        elif self.centres is not None:
            print(f'  loops: none from {len(self.slices)} planes')
        return self

    def save(self, figures=True, n_arrows=80, field=False, assign=False):
        """
        Write the slices into the run folder, one colour per slice.

        Writes <stem>_slices.ply, carrying the input's nx/ny/nz when it had them,
        and, unless `figures` is off, <stem>_slices.png with every slice projected
        onto its own plane, one row per axis and one column per sample.

        figures: also write the projected slice grid png
        n_arrows: normals drawn per panel, 0 for none, None for every one of them
        field: draw the implicit field the loop centres are read off behind the
                 slice points, and also write <stem>_field.png holding the field
                 alone, as a colour map per panel
        assign: colour the slice points by the loop centre each one belongs to, as
                 its slice's own grouping decided, so a plane holding several loops
                 shows how its points part between them
        """
        if field and self.normals is None:
            raise ValueError('the implicit field is solved from the point normals')
        self.save_opts = (figures, n_arrows, field, assign)
        ## the name follows the input, and the folder the shape's own inside the run
        ## root, so a re-run overwrites its own output and nothing else
        self.out.mkdir(parents=True, exist_ok=True)
        saved_path = self.out / f'{self.stem}_slices.ply'
        visualize_sliced_point_cloud(saved_path, self.points, self.slices,
                                     p_normal=self.p_normal, p_point=self.p_point,
                                     vis_to_img=figures, pcn=self.normals,
                                     n_arrows=n_arrows, centres=self.centres,
                                     field=field, assign=assign)
        if field:
            save_field_grid(self.out / f'{self.stem}_field.png', self.points,
                            self.slices, self.p_normal, self.p_point, self.normals,
                            centres=self.centres)
        return self

    def extract(self, figures=True, n_arrows=80, field=False, assign=False):
        """The whole chain in order, leaving every stage on the object."""
        return (self.load().make_planes().cut().find_centres(assign)
                .group_loops().save(figures, n_arrows, field, assign))

    def restage(self, offsets=None):
        """
        Cut again, with `offsets` per axis or with `samples` even planes, and leave
        the slices, centres and loops of that plane set on the object.

        Only the planes that are new are cut. A plane is the same plane when it sits
        on the same axis at the same offset, and a refinement stage keeps every plane
        the stage before had, so most of a re-cut is already on the object and its
        slice, centres and loops are read off rather than solved again. That is what
        a stage can afford: cutting is by far the expensive half of this pipeline --
        on the hand, 27 planes take 14 s to cut and the fit over them 2 s per 100
        epochs -- so a stage that cut every plane again would spend more on the planes
        it already had than on training.

        This is the whole chain after `load`. Nothing is read again, and the artefacts
        are left alone until `resave` is called for the plane set the run ended on.
        """
        if self.points is None:
            raise ValueError('nothing to re-cut: load the cloud first')
        ## a plane is named the way the fit names it, axis and offset, not by index:
        ## a stage inserts planes, so index 5 is a different plane afterwards
        old_keys = {}
        if getattr(self, 'offsets', None) is not None:
            old_samples = self.samples
            for axis, row in enumerate(self.offsets):
                for j, o in enumerate(row):
                    old_keys[(axis, float(o))] = axis * old_samples + j
        old_slices, old_centres = self.slices, self.centres

        self.make_planes(offsets)
        slices = [None] * len(self.p_normal)
        centres = [None] * len(self.p_normal)
        for axis, row in enumerate(self.offsets):
            for j, o in enumerate(row):
                i = axis * self.samples + j
                src = old_keys.get((axis, float(o)))
                if src is not None:
                    slices[i], centres[i] = old_slices[src], old_centres[src]

        new = [i for i, s in enumerate(slices) if s is None]
        if new:
            fresh = slice_point_cloud(self.points, self.p_normal[new], self.p_point[new],
                                      self.threshold, pcn=self.normals,
                                      max_align=self.max_align)
            fresh_centres = (None if self.normals is None else
                             loop_centres(self.points, fresh, self.p_normal[new],
                                          self.p_point[new], self.normals))
            for i, s, c in zip(new, fresh,
                               fresh_centres if fresh_centres is not None
                               else [None] * len(new)):
                slices[i], centres[i] = s, c
        self.slices, self.centres = slices, centres

        for i, axis in enumerate('xyz'):
            keep = [bool(old_keys.get((i, float(o))) is not None)
                    for o in self.offsets[i]]
            counts = [idx.size
                      for idx in self.slices[i * self.samples:(i + 1) * self.samples]]
            print(f'  {axis}: {self.samples} planes at '
                  f'{self.offsets[i].min():.2f}..{self.offsets[i].max():.2f}, '
                  f'{sum(keep)} kept, {len(keep) - sum(keep)} cut now, '
                  f'{min(counts)}-{max(counts)} points per plane')
            if self.centres is not None:
                row = self.centres[i * self.samples:(i + 1) * self.samples]
                print(f'  {axis}: loops per plane {[len(c) for c, _, _ in row]}')
        return self.group_loops()

    def resave(self):
        """Write the extraction artefacts again for the plane set on the object,
        with the switches the shape was last saved with."""
        if self.save_opts is None:
            return self.save()
        return self.save(*self.save_opts)

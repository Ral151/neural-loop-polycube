import numpy as np

from .curves import UNIT_CUBE_HALF, in_plane_u
from .square import Square


def cube_loops(planes=4, samples=256, half=UNIT_CUBE_HALF, fill=0.8):
    """
    The unit cube cut by `planes` axis-aligned planes per axis, as the unit-cube
    mode of the warm-up's loop set.

    Each loop is the cube's cross-section on the plane `axis = d`: a square of
    half-side `half` centred at `d * axis`, walked counter-clockwise seen from
    `+axis`. The target handed back is in the plane's own frame -- the world point
    minus the plane's centre -- so every loop's target is the same unit square; the
    offset is carried by the query point, which is where the field is read. That is
    the split the real fit uses, where a loop's target is in its plane's frame and
    the loop's place in the shape is its query.

    Loops adjacent in offset along one axis are paired, so the seam distance D has
    the pairs that "z = 0 close to the neighbours" is asked over.

    There is no plane through the exact centre: the three centre cross-sections,
    x = 0, y = 0 and z = 0, all have their centroid at the origin, so one field
    feature would be asked to produce three different squares. A real shape does not
    repeat that coincidence -- its centre cuts have different centroids -- so the
    cube skips the centre and every loop's query is its own.

    planes: planes per axis, an even number, spread evenly over +-fill * half
    samples: points round each loop, at the parameters k / samples
    half: the cube's half-side, `UNIT_CUBE_HALF` by default -- a definition, never
          a measurement
    fill: how far the outermost plane sits from the centre, as a fraction of `half`.
          0.8 keeps every plane off the cube's faces

    Returns a dict:
      targets: K x samples x 3 plane-local points on the cube
      normals: K x samples x 3 outward normals of the faces there
      queries: K x 3 world points at the loops' centres
      axes:    K x 3 plane normals
      pairs:   P x 2 loop indices adjacent along one axis
    """
    if planes < 2 or planes % 2:
        raise ValueError(f'the cube needs an even number of planes per axis, none '
                         f'at the centre, got {planes}')
    if samples < 4:
        raise ValueError(f'a square loop needs at least four samples, got {samples}')
    if half <= 0:
        raise ValueError(f'the cube half-side must be positive, got {half}')
    if not 0.0 < fill < 1.0:
        raise ValueError(f'fill must be inside (0, 1), got {fill}')
    offsets = np.linspace(-fill * half, fill * half, planes)
    targets, normals, queries, axes, pairs = [], [], [], [], []
    for a in range(3):
        axis = np.eye(3)[a]
        square = Square(half, centre=None, axis=axis, u=in_plane_u(axis))
        points, outward, _ = square.loop(samples)
        base = a * planes
        for k, d in enumerate(offsets):
            targets.append(points)
            normals.append(outward)
            axes.append(axis)
            queries.append(d * axis)
            if k:
                pairs.append((base + k - 1, base + k))
    return dict(targets=np.stack(targets), normals=np.stack(normals),
                queries=np.stack(queries), axes=np.stack(axes),
                pairs=np.asarray(pairs, dtype=np.int64))


def random_cube_loops(planes=4, samples=256, count=1, half_range=(0.2, 0.45),
                      fill=0.8, domain=UNIT_CUBE_HALF, rng=None):
    """A batch of cubes of random size and place, cut like `cube_loops` cuts one.

    Each cube draws a half-side from `half_range` and a centre uniform over the part
    of the domain where the whole cube fits, so the loops land in the grid's
    interior rather than on its border. A cube's cross-sections are congruent
    squares in the same plane-local frame `cube_loops` returns -- the size is the
    cube's own half, the placement is carried by the query -- so the fit, direction
    and seam terms read a batch exactly as before. Seam pairs stay inside one cube:
    two cubes in the same batch are not neighbours of each other.

    count: cubes in the batch, each contributing 3 * planes loops
    half_range: (min, max) cube half-side; both below `domain`, so the centre has
            room to move
    fill: how far the outermost plane sits from the cube centre, as a fraction of
            the half-side; the centre itself is skipped
    domain: half-side of the box the triplane covers, the unit cube by default
    rng: numpy Generator; None draws a fresh one, so a run is reproducible only
            when the caller passes a seeded generator

    Returns the dict `cube_loops` returns plus `halves` and `centres`, one entry per
    cube, for reporting and for drawing a representative cube.
    """
    if count < 1:
        raise ValueError(f'count must be at least 1, got {count}')
    if planes < 2 or planes % 2:
        raise ValueError(f'the cube needs an even number of planes per axis, none '
                         f'at the centre, got {planes}')
    if samples < 4:
        raise ValueError(f'a square loop needs at least four samples, got {samples}')
    low, high = half_range
    if not 0.0 < low <= high < domain:
        raise ValueError(f'half_range must sit inside (0, {domain}), got '
                         f'{half_range}')
    if not 0.0 < fill < 1.0:
        raise ValueError(f'fill must be inside (0, 1), got {fill}')
    rng = np.random.default_rng() if rng is None else rng
    targets, normals, queries, axes, pairs = [], [], [], [], []
    halves, centres = [], []
    offsets = None
    for _ in range(count):
        half = float(rng.uniform(low, high))
        room = domain - half
        centre = rng.uniform(-room, room, 3)
        ## the planes are the same in every cube's own frame; only their place moves
        offsets = np.linspace(-fill * half, fill * half, planes)
        base = len(targets)
        for a in range(3):
            axis = np.eye(3)[a]
            square = Square(half, centre=None, axis=axis, u=in_plane_u(axis))
            points, outward, _ = square.loop(samples)
            for k, d in enumerate(offsets):
                targets.append(points)
                normals.append(outward)
                axes.append(axis)
                queries.append(centre + d * axis)
                if k:
                    pairs.append((base + a * planes + k - 1,
                                  base + a * planes + k))
        halves.append(half)
        centres.append(centre)
    return dict(targets=np.stack(targets), normals=np.stack(normals),
                queries=np.stack(queries), axes=np.stack(axes),
                pairs=np.asarray(pairs, dtype=np.int64),
                halves=np.asarray(halves), centres=np.stack(centres))

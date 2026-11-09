"""The unit cube and its square cross-sections: the warm-up's exact targets and
the in-plane frame every loop is walked in."""

import numpy as np


# half-side of the warm-up cube (side 1, centred at the origin), the same for every shape
UNIT_CUBE_HALF = 0.5


def in_plane_u(axis):
    """In-plane x direction for a plane normal: the least-aligned world axis.

    Ties go to the lower index, so a normal on a world axis always gets the same
    frame regardless of floating-point noise.
    """
    mag = np.abs(np.asarray(axis, dtype=np.float64).reshape(3))
    return np.eye(3)[int(np.flatnonzero(mag <= mag.min() + 1e-9)[0])]


class Square:
    """A square curve with an exact parameterisation p(t), t in [0, 1).

    Defined by half-width, centre, plane normal `axis` and in-plane x `u`
    (v = axis x u). It is walked counter-clockwise seen from +axis, so its outward
    edge normals match the ones its tangent implies. `at(t)` evaluates the curve,
    `loop(n)` samples it.
    """

    # the four edges walked counter-clockwise from a corner, each as
    # (start corner, unit walk direction, outward normal) for half-width 1
    EDGES = (
        ((1.0, -1.0), (0.0, 1.0), (1.0, 0.0)),
        ((1.0, 1.0), (-1.0, 0.0), (0.0, 1.0)),
        ((-1.0, 1.0), (0.0, -1.0), (-1.0, 0.0)),
        ((-1.0, -1.0), (1.0, 0.0), (0.0, -1.0)),
    )

    def __init__(self, half=1.0, centre=None, axis=None, u=None):
        """half: centre to edge; centre: default origin; axis: plane normal,
        default +z; u: square's x direction, default +x (projected into the plane)."""
        if half <= 0:
            raise ValueError(f'half must be positive, got {half}')
        self.half = float(half)
        self.centre = (np.zeros(3) if centre is None else
                       np.asarray(centre, dtype=np.float64).reshape(3))
        axis = np.array([0.0, 0.0, 1.0]) if axis is None else axis
        axis = np.asarray(axis, dtype=np.float64).reshape(3)
        self.axis = axis / np.linalg.norm(axis)
        u = np.array([1.0, 0.0, 0.0]) if u is None else u
        u = np.asarray(u, dtype=np.float64).reshape(3)
        # keep only the in-plane part of u
        u = u - (u @ self.axis) * self.axis
        if np.linalg.norm(u) < 1e-9:
            raise ValueError('u must not run along the plane normal')
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(self.axis, self.u)

    @classmethod
    def from_points(cls, points, half=None):
        """Square fitted to a loop (S x 3): its centre, plane and, by default, mean radius."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        centre = pts.mean(0)
        axis = np.linalg.svd(pts - centre)[2][-1]
        # fix the SVD's sign: largest normal component positive
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
        if half is None:
            half = float(np.linalg.norm(pts - centre, axis=1).mean())
        return cls(half, centre, axis, np.eye(3)[int(np.argmin(np.abs(axis)))])

    def at(self, t):
        """Points (k x 3) at parameters t (wrapped into [0, 1)) and their edge indices."""
        q = np.mod(np.asarray(t, dtype=np.float64).reshape(-1), 1.0)
        side = np.minimum(np.floor(4 * q).astype(int), len(self.EDGES) - 1)
        frac = 4 * q - side
        starts = np.array([e[0] for e in self.EDGES])
        walks = np.array([e[1] for e in self.EDGES])
        # 2D position on the square, then mapped into the plane
        xy = self.half * (starts[side] + walks[side] * 2 * frac[:, None])
        points = self.centre + xy[:, :1] * self.u + xy[:, 1:2] * self.v
        return points, side

    def loop(self, n):
        """n even samples round the square: (points, outward normals, arc-length params)."""
        if n < 1:
            raise ValueError(f'a square needs at least one sample, got {n}')
        points, side = self.at(np.arange(n) / n)
        out = np.array([e[2] for e in self.EDGES])[side]
        normals = out[:, :1] * self.u + out[:, 1:2] * self.v
        # arc-length fraction of the perimeter at each sample
        steps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        params = np.concatenate([[0.0], np.cumsum(steps)[:-1]]) / steps.sum()
        return points, normals, params


def cube_loops(planes=4, samples=256, half=UNIT_CUBE_HALF, fill=0.8):
    """The warm-up loop set: square cross-sections of the cube, `planes` per axis.

    Targets are in each plane's own frame (all the same square); the loop's
    position is carried by its query point. `planes` must be even so no plane cuts
    the centre, where three loops would share one query. Planes span +-fill * half.

    Returns a dict of targets / normals (K x samples x 3), queries / axes (K x 3)
    and pairs (P x 2, loops adjacent along one axis, used by the seam term).
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

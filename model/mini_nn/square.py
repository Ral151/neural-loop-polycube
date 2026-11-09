import numpy as np


## the one curve whose parameterisation is known exactly: a square, p(z), held as
## data -- how big it is and where it stands -- so every question the warm-up asks
## about it is answered by the same object
class Square:
    """
    A square: half-width, centre, and the plane it lies in, given as its unit normal
    `axis` and the in-plane direction `u` its own x runs along. `v` is not taken, it is
    closed by v = axis x u, so a walk that is counter-clockwise in (u, v) is
    counter-clockwise seen from the plane normal.

    That is the whole point of carrying the placement in the data rather than drawing
    a square at the origin and moving it later: the walk's sense and the plane's normal
    then agree by construction, so the square's own outward edge normals are the ones
    its walk implies, which is what the direction term rewards and what lets a loop's
    fit start already the right way round.

    The default is the unit square at the origin in z = 0, u along x, v along y: one
    cross-section of the unit cube, which is what the cube warm-up fits. A loop's own
    square is built with `Square.from_points`, which reads the placement off the loop.

    `at(t)` is the curve itself at any parameter -- so a coordinate drawn anywhere gets
    the point that really sits there, with no correspondence search and no quantising
    to a sample -- and `loop(n)` is that same curve laid down as samples, with the
    outward normal of the edge each sits on and its arc-length parameter.
    """

    ## the four edges, walked counter-clockwise seen from the plane normal and starting
    ## at a corner: up the right, across the top, down the left, along the bottom, each
    ## as (start corner, unit walk, outward unit normal) on a square of half-width 1
    EDGES = (
        ((1.0, -1.0), (0.0, 1.0), (1.0, 0.0)),
        ((1.0, 1.0), (-1.0, 0.0), (0.0, 1.0)),
        ((-1.0, 1.0), (0.0, -1.0), (-1.0, 0.0)),
        ((-1.0, -1.0), (1.0, 0.0), (0.0, -1.0)),
    )

    def __init__(self, half=1.0, centre=None, axis=None, u=None):
        """
        half: centre to edge, so the side is 2 * half
        centre: the square's centre in world coordinates; default the origin
        axis: unit normal of the plane the square lies in; default +z
        u: the in-plane direction the square's own x runs along; default +x, and only
           its part inside the plane is used, the rest being dropped
        """
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
        ## u has to lie in the plane, or the square would not: keep its in-plane part
        u = u - (u @ self.axis) * self.axis
        if np.linalg.norm(u) < 1e-9:
            raise ValueError('u must not run along the plane normal')
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(self.axis, self.u)

    @classmethod
    def from_points(cls, points, half=None):
        """
        The square standing in a loop's place: the loop's own centre and plane, and --
        unless `half` is given -- the loop's own mean radius.

        The plane normal's sign is fixed here rather than left to the SVD, so the same
        loop always gets the same square and the same way round: the largest component
        is made positive. The square's x then runs along the world axis the normal is
        least aligned with, which is the one that projects into the plane with the most
        length left; a loop in the z = 0 plane therefore comes out as u = x, v = y,
        exactly the square `Square(half)` describes.

        points: S x 3 points of one loop, in its plane
        half: centre to edge. Default: the mean distance of the loop's points from
              their centre, so every square is the size of the loop it stands for
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        centre = pts.mean(0)
        axis = np.linalg.svd(pts - centre)[2][-1]
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
        if half is None:
            half = float(np.linalg.norm(pts - centre, axis=1).mean())
        return cls(half, centre, axis, np.eye(3)[int(np.argmin(np.abs(axis)))])

    def at(self, t):
        """
        k x 3 points p(z) at the k parameters t, and the index of the edge each one
        lands on.

        The walk is counter-clockwise seen from the plane normal, which is the sense
        the direction term rewards: it asks the tangent to be axis x normal, and with
        an in-plane outward normal that is this way round. A parameter landing exactly
        on a corner takes the edge that starts there.

        t: parameters in turns, of any shape; wrapped into [0, 1), so 1.0 reads as 0.0

        Returns (points, edges): k x 3 points and k edge indices.
        """
        q = np.mod(np.asarray(t, dtype=np.float64).reshape(-1), 1.0)
        side = np.minimum(np.floor(4 * q).astype(int), len(self.EDGES) - 1)
        frac = 4 * q - side
        starts = np.array([e[0] for e in self.EDGES])
        walks = np.array([e[1] for e in self.EDGES])
        ## the square in its own two directions, then carried into the plane
        xy = self.half * (starts[side] + walks[side] * 2 * frac[:, None])
        points = self.centre + xy[:, :1] * self.u + xy[:, 1:2] * self.v
        return points, side

    def loop(self, n):
        """
        n x 3 points round the square, one per even split of [0, 1), the n x 3 outward
        normals of the edges they sit on, and the n arc-length parameters they sit at.

        The points are `at` read at the even parameters k / n; the normals and
        parameters travel with them so a fit can be shown which way the surface faces
        there. What holds a coordinate to where it belongs is `at` itself, since a
        square's parameterisation is known in closed form.

        n: samples round the perimeter, laid down evenly by arc length, so parameter
           step and distance travelled agree

        Returns (points, normals, params); a sample landing on a corner is given the
        edge that starts there.
        """
        if n < 1:
            raise ValueError(f'a square needs at least one sample, got {n}')
        points, side = self.at(np.arange(n) / n)
        out = np.array([e[2] for e in self.EDGES])[side]
        normals = out[:, :1] * self.u + out[:, 1:2] * self.v
        ## the parameter is the distance walked, as a fraction of the whole perimeter:
        ## read off the samples themselves, so it comes out at k / n for even spacing
        ## and would still be right for any other spacing
        steps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        params = np.concatenate([[0.0], np.cumsum(steps)[:-1]]) / steps.sum()
        return points, normals, params

    def normals_at(self, t):
        """
        k x 3 outward normals of the edges the k parameters t land on.

        The same normals `loop` hands back with its samples, read at arbitrary
        parameters: constant along an edge and jumping a right angle at each corner.
        They are what the pre-training direction term is held against, so the curve
        is taught the square's own sense of outward even where it is still far from
        the square in position.

        t: parameters in turns, of any shape; wrapped into [0, 1)
        """
        _, side = self.at(t)
        out = np.array([e[2] for e in self.EDGES])[side]
        return out[:, :1] * self.u + out[:, 1:2] * self.v

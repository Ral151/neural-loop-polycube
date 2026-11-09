import numpy as np

from .square import Square


## fixed-size batches out of point sets holding different numbers of points
class DataGenerator:
    def __init__(self, point_sets, max_points=None, batch_size=1, seed=0, normals=None):
        """
        Batches of one shape from targets of varying size.

        point_sets: list of S_i x 3 arrays, e.g. the sliced cross-section loops
        max_points: M, points per target after sampling (default: median S_i)
        batch_size: targets per batch
        normals: list of S_i x 3 unit normals, one per point of each set, or None.
                 They are drawn with the same indices as the points, so a normal never
                 comes loose from the point it describes

        A target with more M points is subsampled without replacement, one
        with fewer is padded by repeating its points, so M is the same on every
        iteration and a batch is always batch_size x M x 3.

        The loops' plane normals are part of the data too: each set is a
        cross-section, so `Square.from_points` reads the plane it was cut on.
        They are kept as `axes`, one unit normal per point set, for the trainer
        and the normal terms to index by `index_batches`' ids.
        """
        self.point_sets = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in point_sets]
        if any(len(p) == 0 for p in self.point_sets):
            raise ValueError('every point set needs at least one point')

        self.normals = None if normals is None else [
            np.asarray(n, dtype=np.float64).reshape(-1, 3) for n in normals]
        if self.normals is not None and any(len(n) != len(p) for n, p in
                                            zip(self.normals, self.point_sets)):
            raise ValueError('every normal set needs one normal per point')

        ## the plane each set lies in, read off its own points. The sign is the one
        ## Square.from_points fixes, so this is the axis the warm-up's square is walked
        ## counter-clockwise around. Loop-level metadata, indexed by `index_batches`' ids
        self.axes = np.stack([Square.from_points(p).axis for p in self.point_sets])

        sizes = np.array([len(p) for p in self.point_sets])
        self.max_points = int(np.median(sizes)) if max_points is None else int(max_points)
        if self.max_points < 1:
            raise ValueError(f'max_points must be positive, got {self.max_points}')
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        ## batches in one pass over the point sets
        return -(-len(self.point_sets) // self.batch_size)

    def pick(self, i):
        """M indices into point set i, repeating points if it holds fewer than M."""
        pts = self.point_sets[i]
        if len(pts) >= self.max_points:
            return self.rng.choice(len(pts), self.max_points, replace=False)
        ## tile shuffled copies so the repeats spread out instead of clustering
        reps = -(-self.max_points // len(pts))
        return np.tile(self.rng.permutation(len(pts)), reps)[:self.max_points]

    def index_batches(self):
        """Yields (point set indices, batch, normal batch or None), so callers can tell
        the sets apart."""
        ## reshuffle the point sets every pass, so the batches differ per epoch
        order = self.rng.permutation(len(self.point_sets))
        ## cycle the tail back to the start, every batch keeps the full batch_size
        order = np.resize(order, len(self) * self.batch_size)
        for start in range(0, len(order), self.batch_size):
            ids = order[start:start + self.batch_size]
            ## one draw of indices feeds the points and their normals both, so a normal
            ## never comes loose from the point it describes
            picks = [self.pick(i) for i in ids]
            batch = np.stack([self.point_sets[i][p] for i, p in zip(ids, picks)])
            yield (ids, batch,
                   None if self.normals is None else
                   np.stack([self.normals[i][p] for i, p in zip(ids, picks)]))

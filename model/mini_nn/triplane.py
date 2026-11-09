import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneField(nn.Module):
    """A continuous 3D feature read from three orthogonal 2D grids.

    The three planes are indexed XY, XZ, YZ. A query point is projected onto
    each, bilinearly sampled, and the three samples are concatenated, so the
    feature is a function of position rather than a row looked up by loop id.
    Two loops close in space therefore start from similar features even if the
    network has never fitted either of them; the grid is what makes the feature
    spatial and the bilinear interpolation is what makes it continuous across a
    cell boundary.

    The grids live on the unit cube, `[-UNIT_CUBE_HALF, UNIT_CUBE_HALF]^3`,
    subdivided uniformly and the same for every shape. `centre` and `half` say
    which world box that cube is laid over -- the cloud's own box when fitting, so
    the shape fills the grid -- and a query is carried into the cube by
    (x - centre) / half, then clamped at the border. That map is affine and the
    field is only a feature map, so stretching its domain changes where a cell sits
    in the world, not what the field is: a grid learned on one shape reads the same
    way on another once the frame is known. The frame is stored with the grid, so a
    checkpoint is queried in world coordinates.

    The grids start at zero, so a cell the warm-up never reads stays the zero
    vector; every query reading only such cells shares one condition. The warm-up
    is what puts structure on the cells it reads.
    """

    def __init__(self, centre=None, half=None, resolution=32, channels=16, tv=0.0,
                 lo=None, hi=None):
        super().__init__()
        ## `lo`/`hi` are the older spelling of the same frame, kept so a checkpoint
        ## written before the cube became explicit still loads
        if centre is None or half is None:
            if lo is None or hi is None:
                raise ValueError('a triplane field needs the world box it is mapped '
                                 'onto, as centre and half (or lo and hi)')
            lo = torch.as_tensor(lo, dtype=torch.float32).reshape(3)
            hi = torch.as_tensor(hi, dtype=torch.float32).reshape(3)
            if torch.any(hi <= lo):
                raise ValueError(f'degenerate triplane box {lo.tolist()} .. '
                                 f'{hi.tolist()}')
            centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        centre = torch.as_tensor(centre, dtype=torch.float32).reshape(3)
        half = torch.as_tensor(half, dtype=torch.float32).reshape(3)
        if torch.any(half <= 0):
            raise ValueError(f'triplane half-extents must be positive, got '
                             f'{half.tolist()}')
        ## `box_` not `centre`/`half`: `half` is a method on nn.Module
        self.register_buffer('box_centre', centre)
        self.register_buffer('box_half', half)
        self.resolution = int(resolution)
        self.channels = int(channels)
        ## (plane, channel, row, col); XY samples (x, y), XZ (x, z), YZ (y, z).
        ## Zero, so every loop starts from the same curve until the warm-up or the
        ## fit moves the cells it reads
        self.grid = nn.Parameter(torch.zeros(3, self.channels, self.resolution,
                                             self.resolution))
        self.tv = float(tv)
        self.pairs = ((0, 1), (0, 2), (1, 2))

    @property
    def feat_dim(self):
        return 3 * self.channels

    @property
    def lo(self):
        """World corner of the box the cube is mapped onto."""
        return self.box_centre - self.box_half

    @property
    def hi(self):
        """World corner of the box the cube is mapped onto."""
        return self.box_centre + self.box_half

    def sample(self, xyz):
        """B x 3 world points -> B x (3 * channels) features."""
        xyz = xyz.reshape(-1, 3)
        ## the world box onto the unit cube's [-1, 1], and held there so a query
        ## outside the box reads the border rather than an extrapolated value
        p = (xyz - self.box_centre) / self.box_half
        p = p.clamp(-1.0, 1.0)
        out = []
        for plane, (a, b) in enumerate(self.pairs):
            ## grid_sample takes x first (width) and y second (height)
            g = torch.stack([p[:, a], p[:, b]], dim=-1).view(1, -1, 1, 2)
            s = F.grid_sample(self.grid[plane:plane + 1], g, mode='bilinear',
                              padding_mode='border', align_corners=True)
            out.append(s.reshape(self.channels, -1).T)
        return torch.cat(out, dim=-1)

    def forward(self, xyz):
        return self.sample(xyz)

    def cell_ids(self, xyz):
        """B x 3 world points -> B x 3 flat indices, one per plane.

        Each point is carried into the cube and snapped to its nearest grid node.
        With `align_corners=True` node k sits at -1 + 2k / (R - 1), so the snap is
        the rounded position; the flat index is (height, width) because
        `grid_sample` reads the first stored coordinate as width and the second as
        height, which is why the index is `q[b] * R + q[a]`.
        """
        xyz = xyz.reshape(-1, 3)
        p = (xyz - self.box_centre) / self.box_half
        p = p.clamp(-1.0, 1.0)
        q = ((p + 1.0) * (0.5 * (self.resolution - 1))).round().long()
        q = q.clamp(0, self.resolution - 1)
        return torch.stack([q[:, b] * self.resolution + q[:, a]
                            for a, b in self.pairs], dim=-1)

    def pool_cells(self, cells, masks):
        """One feature per loop: the mean of the grid nodes it occupies.

        cells: three L x K long tensors, plane p's K unique node indices per loop
        masks: three L x K tensors, 0 where a row is padding

        Each occupied node counts once, however many target points fall in it, so
        the pooled code describes the loop's footprint rather than its sampling
        density. The three plane means are concatenated, keeping the feature width
        `3 * channels` the centroid sample has.
        """
        out = []
        for plane in range(3):
            flat = self.grid[plane].reshape(self.channels, -1)
            values = flat[:, cells[plane]]                    # C x L x K
            mask = masks[plane].to(values.dtype)              # L x K
            pooled = (values * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
            out.append(pooled.T)
        return torch.cat(out, dim=-1)

    def tv_penalty(self):
        """Mean squared difference between neighbouring cells, both directions."""
        g = self.grid
        if g.shape[-1] < 2:
            return g.new_zeros(())
        dx = (g[..., 1:, :] - g[..., :-1, :]).pow(2).mean()
        dy = (g[..., :, 1:] - g[..., :, :-1]).pow(2).mean()
        return 0.5 * (dx + dy)


def cell_pool_structure(triplane, point_sets):
    """Padded unique grid nodes each loop occupies, per plane, for `pool_cells`.

    point_sets: list of S_i x 3 world points, one set per loop
    Returns (cells, masks), each a 3-tuple of L x K tensors. K is the widest
    occupied-node count over every loop and plane, so the three planes share one
    padding width and a batch of loops is one gather rather than a Python loop per
    step. The node indices depend only on the frame and the points, both fixed for
    a fit, so this is built once.
    """
    device = triplane.grid.device
    per_plane = ([], [], [])
    for points in point_sets:
        xyz = torch.as_tensor(np.asarray(points, dtype=np.float32),
                              device=device)
        ids = triplane.cell_ids(xyz)
        for plane in range(3):
            per_plane[plane].append(torch.unique(ids[:, plane]))
    width = max((int(u.numel()) for plane in per_plane for u in plane), default=0)
    width = max(width, 1)
    cells, masks = [], []
    for plane in range(3):
        index = torch.zeros((len(point_sets), width), dtype=torch.long,
                            device=device)
        mask = torch.zeros((len(point_sets), width), dtype=torch.float32,
                           device=device)
        for i, unique in enumerate(per_plane[plane]):
            index[i, :unique.numel()] = unique
            mask[i, :unique.numel()] = 1.0
        cells.append(index)
        masks.append(mask)
    return tuple(cells), tuple(masks)

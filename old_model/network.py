"""The shared curve network (`miniNN`) and the triplane field that conditions it."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneField(nn.Module):
    """A continuous 3D feature read from three orthogonal 2D grids (XY, XZ, YZ).

    A query is projected onto each plane, bilinearly sampled and the three samples
    concatenated, so nearby loops get similar features. The grids span the unit
    cube; `centre`/`half` give the world box that cube is laid over (the cloud's
    box when fitting). The grids start at zero.
    """

    def __init__(self, centre=None, half=None, resolution=32, channels=16, tv=0.0,
                 lo=None, hi=None):
        super().__init__()
        # `lo`/`hi` are an older spelling of the box, kept so old checkpoints load
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
        # named `box_*` because `half` is already a method on nn.Module
        self.register_buffer('box_centre', centre)
        self.register_buffer('box_half', half)
        self.resolution = int(resolution)
        self.channels = int(channels)
        # (plane, channel, row, col); XY samples (x, y), XZ (x, z), YZ (y, z)
        self.grid = nn.Parameter(torch.zeros(3, self.channels, self.resolution,
                                             self.resolution))
        self.tv = float(tv)
        self.pairs = ((0, 1), (0, 2), (1, 2))

    @property
    def feat_dim(self):
        """Length of one feature vector (3 planes x channels)."""
        return 3 * self.channels

    @property
    def lo(self):
        """Lower world corner of the box the cube is mapped onto."""
        return self.box_centre - self.box_half

    @property
    def hi(self):
        """Upper world corner of the box the cube is mapped onto."""
        return self.box_centre + self.box_half

    def sample(self, xyz):
        """B x 3 world points -> B x (3 * channels) features."""
        xyz = xyz.reshape(-1, 3)
        # world box -> [-1, 1]; clamped so points outside read the border
        p = (xyz - self.box_centre) / self.box_half
        p = p.clamp(-1.0, 1.0)
        out = []
        for plane, (a, b) in enumerate(self.pairs):
            # grid_sample takes x first (width) and y second (height)
            g = torch.stack([p[:, a], p[:, b]], dim=-1).view(1, -1, 1, 2)
            s = F.grid_sample(self.grid[plane:plane + 1], g, mode='bilinear',
                              padding_mode='border', align_corners=True)
            out.append(s.reshape(self.channels, -1).T)
        return torch.cat(out, dim=-1)

    def forward(self, xyz):
        """Same as `sample`."""
        return self.sample(xyz)

    def tv_penalty(self):
        """Total-variation smoothness: mean squared difference of neighbouring cells."""
        g = self.grid
        if g.shape[-1] < 2:
            return g.new_zeros(())
        dx = (g[..., 1:, :] - g[..., :-1, :]).pow(2).mean()
        dy = (g[..., :, 1:] - g[..., :, :-1]).pow(2).mean()
        return 0.5 * (dx + dy)


class miniNN(nn.Module):
    """Shared MLP mapping a loop parameter z in [0, 1] and a condition to a 3D point.

    The condition is a per-loop feature (an embedding row or a triplane sample),
    optionally followed by the slice axis. `self.model` is the MLP itself.
    """

    def __init__(self, num_models, num_neurons=64, feat_dim=32, pe=True,
                 feature_mode='embedding', triplane=None, num_layers=2,
                 axis_condition=False):
        super(miniNN, self).__init__()

        self.feature_mode = feature_mode
        self.num_layers = int(num_layers)
        self.axis_condition = bool(axis_condition)
        if self.num_layers < 1:
            raise ValueError('the network needs at least one hidden layer')
        self.triplane = None
        if feature_mode == 'embedding':
            self.embedding = nn.Embedding(num_models, feat_dim)
            # zero rows: every loop starts from the same curve
            nn.init.zeros_(self.embedding.weight)
            self.embedding_dim = feat_dim
        elif feature_mode == 'triplane':
            if not triplane:
                raise ValueError('triplane mode needs the field configuration')
            self.triplane = TriplaneField(**triplane)
            self.embedding_dim = self.triplane.feat_dim
        else:
            raise ValueError(f'unknown feature mode {feature_mode!r}')

        # input = (sin, cos) of z or raw z, then the feature, then the axis
        self.pe = pe
        width = self.embedding_dim + (2 if pe else 1) + (3 if self.axis_condition else 0)
        blocks = [nn.Linear(width, num_neurons),
                  nn.Softplus(beta=200)]
        for _ in range(self.num_layers - 1):
            blocks += [nn.Linear(num_neurons, num_neurons), nn.Softplus(beta=200)]
        blocks.append(nn.Linear(num_neurons, 3))
        self.model = nn.Sequential(*blocks)

    def forward(self, z, idx=None, feat=None, axis=None):
        """Decode B x N x 1 parameters (or N x 1 for one loop) to B x N x 3 points.

        idx: B embedding rows; None (and no feat) uses an all-zero feature
        feat: B x F features from the triplane; overrides idx
        axis: B x 3 slice normals, required when the network is axis-conditioned
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)

        if self.pe:
            # sin/cos of 2*pi*z makes z = 0 and z = 1 the same input, so the loop closes
            ang = 2 * np.pi * z
            z = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)        # B x N x 2
        # one condition vector per loop, repeated over its N points
        if feat is not None:
            emb = feat[:, None, :].expand(-1, z.shape[1], -1)
        elif idx is None:
            emb = torch.zeros(z.shape[0], z.shape[1], self.embedding_dim,
                              dtype=z.dtype, device=z.device)
        else:
            if self.feature_mode != 'embedding':
                raise ValueError('a triplane network needs feat, not idx')
            emb = self.embedding(idx)[:, None, :].expand(-1, z.shape[1], -1)
        parts = [z, emb]
        if self.axis_condition:
            if axis is None:
                raise ValueError('this network is axis-conditioned; pass one unit '
                                 'plane normal per sample as axis')
            axis = torch.as_tensor(axis, dtype=z.dtype, device=z.device).reshape(-1, 3)
            if axis.shape[0] != z.shape[0]:
                raise ValueError(f'{axis.shape[0]} axes for {z.shape[0]} samples')
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            parts.append(axis[:, None, :].expand(-1, z.shape[1], -1))
        return self.model(torch.cat(parts, dim=-1))

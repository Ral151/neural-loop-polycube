import numpy as np
import torch
import torch.nn as nn

from .triplane import TriplaneField


## the shared map: sin/cos coords plus a condition vector -> a point on the curve.
## Which condition that is belongs to the child: a learned row per part, or a
## feature read from the triplane field
class miniNN(nn.Module):
    feature_mode = None

    def __init__(self, embedding_dim, num_neurons=64, num_layers=2, pe=True):
        super(miniNN, self).__init__()
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError('the network needs at least one hidden layer')
        self.embedding_dim = int(embedding_dim)
        self.triplane = None

        ## the sine encoding swaps the single coord for sin/cos, so the first layer
        ## must be built for that width: it cannot be a forward-time switch
        self.pe = pe
        width = self.embedding_dim + (2 if pe else 1)
        blocks = [nn.Linear(width, num_neurons),
                  nn.Softplus(beta=200)]
        for _ in range(self.num_layers - 1):
            blocks += [nn.Linear(num_neurons, num_neurons), nn.Softplus(beta=200)]
        blocks.append(nn.Linear(num_neurons, 3))
        self.model = nn.Sequential(*blocks)

        ## the inverse of the map above: a point on the curve and the condition it
        ## was drawn under -> the parameter z it belongs to. It is fed the
        ## concatenated [xyz, condition], the same layout `model` reads for
        ## [coords, condition], so a caller hands it the network's own output and
        ## the condition that produced it. Nothing reads it yet; it is the half a
        ## cycle-consistency term would need, and it is built from the same
        ## num_neurons / num_layers so the two maps stay comparable in size
        cycle_width = 3 + self.embedding_dim
        cycle_blocks = [nn.Linear(cycle_width, num_neurons), nn.Softplus(beta=200)]
        for _ in range(self.num_layers - 1):
            cycle_blocks += [nn.Linear(num_neurons, num_neurons),
                             nn.Softplus(beta=200)]
        cycle_blocks.append(nn.Linear(num_neurons, 1))
        self.cycle = nn.Sequential(*cycle_blocks)

    def encode(self, z):
        """The coord input, with the two ends of the parameter mapped together."""
        if not self.pe:
            return z
        ## [0-1] to [0-2pi] and on to sin/cos: the two ends of the parameter then
        ## map to the same input, so the loop closes instead of leaving a seam;
        ## the raw coord is dropped on purpose, it would break that equality
        ang = 2 * np.pi * z
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)        # B x N x 2

    def condition(self, z, idx, feat):
        """B x N x F condition vectors matching the encoded coords."""
        raise NotImplementedError

    def zero_condition(self, z):
        """The shared base curve's condition: an all-zero feature, no part at all."""
        return torch.zeros(z.shape[0], z.shape[1], self.embedding_dim,
                           dtype=z.dtype, device=z.device)

    def forward(self, z, idx=None, feat=None):
        """
        z: B x N x 1 parameter coords in [0-1], or N x 1 for a single part
        idx: part indices, one per sample, or None for the base curve, which is
             conditioned on an all-zero feature instead of a part's own
        feat: B x F feature vectors, when the condition comes from a field rather
             than the embedding table
        """
        if z.dim() == 2:
            ## a bare coord list belongs to one part, so give it the batch axis
            z = z.unsqueeze(0)
        zpe = self.encode(z)
        emb = self.condition(z, idx, feat)

        ## cycle loss
        xyz = self.model(torch.cat([zpe, emb], dim=-1))
        zpred = self.cycle(torch.cat([xyz.detach(), emb], dim=-1))
        return xyz, zpred


## one learned row per part: the table is the whole condition, and every row
## starts at zero so every loop starts from the same curve
class EmbeddingNN(miniNN):
    feature_mode = 'embedding'

    def __init__(self, num_models, num_neurons=64, feat_dim=32, pe=True,
                 num_layers=2):
        super(EmbeddingNN, self).__init__(feat_dim, num_neurons=num_neurons,
                                          num_layers=num_layers, pe=pe)
        self.embedding = nn.Embedding(num_models, feat_dim)
        ## zero weights, so every loop starts from the same curve; the fit is
        ## what moves each row off it and onto its own loop
        nn.init.zeros_(self.embedding.weight)

    def condition(self, z, idx, feat):
        if feat is not None:
            raise ValueError('an embedding network conditions on idx, not feat')
        if idx is None:
            return self.zero_condition(z)
        ## the part embedding is a single vector per sample, so repeat it over the
        ## N coords before the concat, which needs equal dims on every axis but -1
        return self.embedding(idx)[:, None, :].expand(-1, z.shape[1], -1)


## a feature read from the triplane field, one per sample, which may carry
## gradients back into the grid
class TriplaneNN(miniNN):
    feature_mode = 'triplane'

    def __init__(self, num_neurons=64, num_layers=2, pe=True, triplane=None):
        if not triplane:
            raise ValueError('triplane mode needs the field configuration')
        field = TriplaneField(**triplane)
        super(TriplaneNN, self).__init__(field.feat_dim, num_neurons=num_neurons,
                                         num_layers=num_layers, pe=pe)
        self.triplane = field

    def condition(self, z, idx, feat):
        if feat is None:
            if idx is not None:
                raise ValueError('a triplane network needs feat, not idx')
            return self.zero_condition(z)
        ## a field supplies one vector per sample, so repeat it over the N coords;
        ## an idx may still be passed through, but the feature is what conditions
        return feat[:, None, :].expand(-1, z.shape[1], -1)


def make_network(feature_mode='embedding', *, num_models=1, num_neurons=64,
                 num_layers=2, feat_dim=32, pe=True, triplane=None):
    """Build the network that reads the requested kind of condition."""
    if feature_mode == 'triplane':
        return TriplaneNN(num_neurons=num_neurons, num_layers=num_layers,
                          pe=pe, triplane=triplane)
    if feature_mode == 'embedding':
        return EmbeddingNN(num_models, num_neurons=num_neurons,
                           num_layers=num_layers, feat_dim=feat_dim, pe=pe)
    raise ValueError(f'unknown feature mode {feature_mode!r}')

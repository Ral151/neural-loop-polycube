"""The neural fit: shared network, curve geometry, losses and training loop.

Formerly one `mini_nn.py`; each class now lives in its own module. The public
names are re-exported here, so `from model import mini_nn` and
`mini_nn.Trainer` keep working unchanged.
"""

from .chamfer import CD_Loss
from .cubes import cube_loops, random_cube_loops
from .curves import (UNIT_CUBE_HALF, bend_penalty, curve_curvature,
                     curve_normal, curve_tangent, curve_winding,
                     in_plane_normal, in_plane_u, normal_cosine,
                     normal_penalty, project_normals, winding_number)
from .data import DataGenerator
from .network import EmbeddingNN, TriplaneNN, make_network, miniNN
from .square import Square
from .trainer import Trainer, load_model, save_model
from .triplane import TriplaneField, cell_pool_structure
from .warmup import load_warmup, save_warmup, warm_on_cube

__all__ = [
    'CD_Loss', 'DataGenerator', 'EmbeddingNN', 'Square', 'Trainer',
    'TriplaneField', 'TriplaneNN',
    'UNIT_CUBE_HALF', 'bend_penalty', 'cell_pool_structure',
    'cube_loops', 'curve_curvature', 'curve_normal', 'curve_tangent',
    'curve_winding', 'in_plane_normal', 'in_plane_u', 'load_model',
    'load_warmup', 'make_network', 'miniNN', 'normal_cosine', 'normal_penalty',
    'project_normals', 'random_cube_loops', 'save_model', 'save_warmup',
    'warm_on_cube', 'winding_number',
]

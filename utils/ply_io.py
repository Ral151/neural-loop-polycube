"""Read and write the point-cloud PLY files the pipeline uses."""

import numpy as np
import trimesh


def load_points(path):
    """Load a PLY as (N x 3 points, N x 3 normals or None)."""
    mesh = trimesh.load(path, maintain_order=True, process=False)
    points = np.asarray(mesh.vertices, dtype=np.float64)

    # a faceless PLY loads as a PointCloud, which drops the normals; read them
    # from the raw vertex fields trimesh keeps instead
    raw = mesh.metadata.get('_ply_raw', {}).get('vertex', {}).get('data')
    if raw is not None and raw.dtype.names and {'nx', 'ny', 'nz'} <= set(raw.dtype.names):
        normals = np.stack([raw['nx'], raw['ny'], raw['nz']], axis=1).astype(np.float64)
    elif getattr(mesh, 'vertex_normals', None) is not None:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    else:
        normals = None
    return points, normals


def write_points(path, points, colors, normals=None):
    """Write coloured points (S x 3, S x 4 uint8) to a PLY, with normals if given.

    Goes through trimesh, so coordinates are stored as float32.
    """
    if normals is None:
        trimesh.PointCloud(np.asarray(points), colors=np.asarray(colors)).export(path)
        return
    # a PointCloud has no normal field, so a faceless Trimesh carries them
    cloud = trimesh.Trimesh(vertices=np.asarray(points),
                            faces=np.zeros((0, 3), dtype=np.int64),
                            vertex_normals=np.asarray(normals), process=False)
    cloud.visual.vertex_colors = colors
    cloud.export(path)


def write_cloud(path, points, normals):
    """Write points and normals (S x 3 each) as a binary PLY of doubles.

    Same layout as Open3D; used for clouds that are themselves inputs (e.g.
    stylized shapes) so they keep full precision.
    """
    fields = ('x', 'y', 'z', 'nx', 'ny', 'nz')
    rows = np.empty(len(points), dtype=[(name, '<f8') for name in fields])
    for k, name in enumerate(fields):
        rows[name] = (points if k < 3 else normals)[:, k % 3]
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(rows)}\n'
              + ''.join(f'property double {name}\n' for name in fields)
              + 'end_header\n')
    with open(path, 'wb') as fh:
        fh.write(header.encode('ascii'))
        fh.write(rows.tobytes())

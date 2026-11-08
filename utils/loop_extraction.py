"""`LoopExtractor`: runs the extraction stages for one point cloud (load, planes,
slices, loop centres, loops, figures) and keeps every result on itself."""

from pathlib import Path

import numpy as np

from utils.ply_io import load_points
from utils.slicing import assigned_loops, loop_centres, slice_point_cloud
from utils.visualization import save_field_grid, visualize_sliced_point_cloud


# the cloud `main.py extract` / `main.py fit` read when no --path is given
DATASETS = Path(__file__).resolve().parents[1] / 'Datasets' / 'normalized_shapes_cuboid' / 'normalized_shapes_cuboid'
DEFAULT_PLY = str(DATASETS / 'laurent_hand_normal_cubic_lam3.ply')


def run_dir(path, root=None):
    """Output folder of a shape: `root` if given, else log/<input stem>."""
    return Path('log') / Path(path).stem if root is None else Path(root)


class LoopExtractor:
    """Slice a point cloud with axis-aligned planes and part every slice into loops.

    Each stage is a method (`extract` runs them all). Results kept on the object:
      points, normals     N x 3 cloud and normals (None if the file has none)
      p_normal, p_point   planes, axis-major: index i * samples + k = plane k on axis i
      slices              one index array into points per plane
      centres             (m x 3 centres, m rms, labels) per plane, see `loop_centres`
      loops               (slice, slot, S x 3 points, S x 3 normals) per loop
    """

    def __init__(self, path, samples=5, margin=0.1, band=0.0075, max_align=0.95,
                 out=None):
        """path: PLY to read; samples: planes per axis; margin: inset from the box
        faces (fraction of each extent); band: slab half-width (fraction of the
        longest extent); max_align: see `slice_point_cloud`; out: output folder.
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
        # outputs are named after the input
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
        # the last save() options, so resave() can repeat them
        self.save_opts = None

    def load(self):
        """Read the cloud and its normals, and say what the file holds."""
        self.points, self.normals = load_points(self.path)
        print(f'{self.path}\n  {self.points.shape[0]} points, '
              f'{0 if self.normals is None else self.normals.shape[0]} normals, '
              f'extent {np.ptp(self.points, axis=0).round(2)}')
        return self

    def make_planes(self, offsets=None):
        """Lay out the planes (evenly, or at the given per-axis `offsets`) and the band."""
        # slab half-width in world units
        self.threshold = self.band * np.ptp(self.points, axis=0).max()
        print(f'  band {self.band * 100:.2f}% of the longest extent '
              f'({self.threshold:.4f})')
        if self.normals is not None:
            print(f'  max_align {self.max_align}: normals within '
                  f'{np.degrees(np.arccos(min(self.max_align, 1.0))):.1f} deg of a '
                  f'plane normal are left out of that plane')

        # planes kept `margin` inside the bounding box on each axis
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
        # each plane point = box centre with its axis coordinate set to the offset
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
        """Find each slice's loop centres (needs normals); `assign` also prints counts."""
        if self.normals is None:
            self.centres = None
            return self
        self.centres = loop_centres(self.points, self.slices, self.p_normal,
                                    self.p_point, self.normals)
        for i, axis in enumerate('xyz'):
            row = self.centres[i * self.samples:(i + 1) * self.samples]
            print(f'  {axis}: loops per plane {[len(c) for c, _, _ in row]}')
            if assign:
                sizes = [[int((lab == k).sum()) for k in range(len(c))]
                         for c, _, lab in row]
                loose = [int((lab < 0).sum()) for _, _, lab in row]
                print(f'  {axis}: points per centre {sizes}, unused {loose}')
        return self

    def group_loops(self):
        """Split every slice into its loops (the training targets), with their normals."""
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
        """Write <stem>_slices.ply and, if `figures`, <stem>_slices.png.

        n_arrows: normals drawn per panel (0 none, None all); field: draw the Poisson
        field behind the points and write <stem>_field.png; assign: colour points by loop.
        """
        if field and self.normals is None:
            raise ValueError('the implicit field is solved from the point normals')
        self.save_opts = (figures, n_arrows, field, assign)
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
        """Re-cut at new `offsets` (refinement), reusing planes that were already cut.

        A plane with the same axis and offset keeps its slice and centres; only the
        new planes are sliced and solved. Artefacts are rewritten only by `resave`.
        """
        if self.points is None:
            raise ValueError('nothing to re-cut: load the cloud first')
        # planes are matched by (axis, offset), since indices shift on insertion
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
        """Rewrite the extraction artefacts for the current planes (last save options)."""
        if self.save_opts is None:
            return self.save()
        return self.save(*self.save_opts)

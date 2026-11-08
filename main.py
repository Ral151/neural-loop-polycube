"""Neural polycube generation from a point cloud: the one entry point.

    python main.py <command> [options]        # python main.py <command> --help

Pipeline, in order:

    warmup        train the cube warm-up checkpoint (once per network size)
    extract       cut the cloud with x/y/z planes and extract the loops
    fit           extract, then fit every loop with the shared network
    propagate     stable loops -> tracks -> aligned seams -> tubes

Input preparation:

    stylize       cubic stylization of a point cloud (makes the *_cubic_lam3.ply inputs)

Inspection of a finished fit run:

    seam-check    seam distance D between neighbouring loops of a checkpoint
    shift-probe   decode loops at queries shifted to the neighbouring planes
    seam-view     mark each loop's z = 0 seam in a PLY and a figure
    sweep-movie   movie revealing the fitted loops plane by plane
    clusters      cluster the loops' triplane codes and show them in 3D
"""

import importlib
import sys

# command -> (module, kwargs for its main(), one-line help)
COMMANDS = {
    'warmup': ('utils.warmup_cube', {}, 'train the cube warm-up checkpoint'),
    'extract': ('model.fit', {'train': False}, 'extract the cross-section loops'),
    'fit': ('model.fit', {'train': True}, 'extract the loops and fit them'),
    'propagate': ('utils.propagation', {}, 'tracks, aligned seams and tubes'),
    'stylize': ('utils.cubic_stylize', {}, 'cubic stylization of a point cloud'),
    'seam-check': ('utils.seam_check', {}, 'seam distance of a checkpoint'),
    'shift-probe': ('utils.shift_probe', {}, 'shifted-query decoding probe'),
    'seam-view': ('utils.seam_view', {}, 'mark the z = 0 seams of a run'),
    'sweep-movie': ('utils.sweep_movie', {}, 'plane-by-plane loop movie'),
    'clusters': ('utils.code_clusters', {}, 'cluster the loops\' triplane codes'),
}


def main(argv=None):
    """Dispatch `main.py <command> ...` to that command's module; returns the exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0 if argv else 1
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f'unknown command {command!r}; choose one of: '
              f'{", ".join(COMMANDS)}', file=sys.stderr)
        return 2
    module, kwargs, _ = COMMANDS[command]
    # import lazily so one command does not pay for every module's imports
    importlib.import_module(module).main(rest, **kwargs)
    return 0


if __name__ == '__main__':
    sys.exit(main())

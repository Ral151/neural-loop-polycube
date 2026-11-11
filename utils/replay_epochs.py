"""Replay a run's first epochs and keep the states.

The run writes one checkpoint, at the end, so the curves at early epochs are not
on disk.  This re-runs the same command -- same data, same warm-up, same seed,
and the same 1000-epoch schedule, since the anneal spans the call -- and records
the model after each of the first `--until` epochs, then stops.  The schedule is
why this trains under epochs=1000 and is cut short rather than being asked for a
few epochs: a short call would anneal the direction term to nothing by its last
update, which is not what the run did.

  python utils/replay_epochs.py [out_dir] [--until 13]

The default out dir is log/axis_seam10_horse_repro/replay; `state_eN.pt` holds
the model after N updates, which is the state the run's diagnostics row N
describes.  These are what `plot_epoch_sheets.py` reads.
"""
import argparse
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('out', nargs='?',
                default='log/axis_seam10_horse_repro/replay')
ap.add_argument('--until', type=int, default=2,
                help='save the state after each of the first N epochs')
ap.add_argument('--freeze-f-epochs', type=int, default=0,
                help='replay with the shared MLP frozen for the first N epochs')
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

sys.argv = [
    'plc_main.py',
    '--path', 'Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid/horse_points_cubic_lam3.ply',
    '--train',
    '--epochs', '1000',
    '--samples', '29',
    '--normal', '0.01',
    '--distill', '10',
    ## this replays the original axis-on/centroid run, which the current code can
    ## no longer build; the pooling flag is kept so the other settings still match
    '--feature-pool', 'centroid',
    '--warmup', 'warmup/cube_warmup_axis.pt',
    '--distill-warmup', 'warmup/cube_warmup_axis.pt',
    '--out', str(out),
    '--snapshot-every', '0',
]
if args.freeze_f_epochs:
    sys.argv += ['--freeze-f-epochs', str(args.freeze_f_epochs)]

import _paths  # noqa: E402
from model import mini_nn
import plc_main


class Stop(Exception):
    pass


original = mini_nn.Trainer.train


def train(self, epochs=1, every=None, on_epoch=None):
    def wrap(n):
        if on_epoch is not None:
            on_epoch(n)
        state = {k: v.detach().cpu().clone()
                 for k, v in self.model.state_dict().items()}
        torch.save(state, out / f'state_e{n}.pt')
        print(f'  saved {out / f"state_e{n}.pt"}')
        if n >= args.until:
            raise Stop()
    return original(self, epochs, every, wrap)


mini_nn.Trainer.train = train
try:
    plc_main.main()
except Stop:
    print(f'stopped after epoch {args.until}')

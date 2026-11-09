"""The trainer that fits one shared network to every loop, and its checkpoint."""

import numpy as np
import torch
import torch.nn as nn

from model.cube import Square
from model.losses import CD_Loss, bend_penalty, cosine_anneal, curve_curvature, curve_tangent, curve_winding, normal_cosine, normal_penalty
from model.network import miniNN


class DataGenerator:
    """Fixed-size random batches (batch_size x M x 3) from point sets of varying size."""

    def __init__(self, point_sets, max_points=None, batch_size=1, seed=0, normals=None):
        """point_sets: list of S_i x 3 arrays; max_points: M (default median S_i);
        normals: optional matching S_i x 3 normals, drawn with the same indices.
        Sets larger than M are subsampled, smaller ones padded by repetition.
        """
        self.point_sets = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in point_sets]
        if any(len(p) == 0 for p in self.point_sets):
            raise ValueError('every point set needs at least one point')

        self.normals = None if normals is None else [
            np.asarray(n, dtype=np.float64).reshape(-1, 3) for n in normals]
        if self.normals is not None and any(len(n) != len(p) for n, p in
                                            zip(self.normals, self.point_sets)):
            raise ValueError('every normal set needs one normal per point')

        sizes = np.array([len(p) for p in self.point_sets])
        self.max_points = int(np.median(sizes)) if max_points is None else int(max_points)
        if self.max_points < 1:
            raise ValueError(f'max_points must be positive, got {self.max_points}')
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        """Number of batches in one pass over the point sets."""
        return -(-len(self.point_sets) // self.batch_size)

    def pick(self, i):
        """M indices into point set i, repeating points if it holds fewer than M."""
        pts = self.point_sets[i]
        if len(pts) >= self.max_points:
            return self.rng.choice(len(pts), self.max_points, replace=False)
        # tile shuffled copies so the repeats spread out
        reps = -(-self.max_points // len(pts))
        return np.tile(self.rng.permutation(len(pts)), reps)[:self.max_points]

    def index_batches(self):
        """Yield (set indices, B x M x 3 points, B x M x 3 normals or None) for one epoch."""
        # reshuffle every pass; wrap the tail around so every batch is full
        order = self.rng.permutation(len(self.point_sets))
        order = np.resize(order, len(self) * self.batch_size)
        for start in range(0, len(order), self.batch_size):
            ids = order[start:start + self.batch_size]
            # one index draw for both points and normals, so they stay paired
            picks = [self.pick(i) for i in ids]
            batch = np.stack([self.point_sets[i][p] for i, p in zip(ids, picks)])
            yield (ids, batch,
                   None if self.normals is None else
                   np.stack([self.normals[i][p] for i, p in zip(ids, picks)]))


class Trainer:
    """Fits one shared network to every loop of a shape, with optional extra terms."""
    def __init__(self, point_sets, num_models=1, max_points=None, batch_size=None,
                 latent_points=None, lr=1e-2, seed=0, num_neurons=64, num_layers=2,
                 feat_dim=32,
                 normals=None, align=0.0, codes=None,
                 code_pairs=None, code_weights=None, code_smooth=0.0,
                 feature_mode='embedding', triplane=None, query_points=None,
                 feature_pairs=None, feature_tv=0.0, axis_condition=False,
                 distill=0.0, distill_warmup=None, distill_pairs=None,
                 distill_weights=None,
                 seam_align=0.0, seam_pairs=None, seam_weights=None, seam_keys=None,
                 seam_every=10, seam_start=0, seam_steps=100, seam_lam=0.02,
                 seam_lr=2e-2, seam_mu=1e-3, freeze_f_epochs=0):
        """Set up the network, data and loss terms for a fit.

        Data
          point_sets: list of S_i x 3 loop targets, each in its plane's frame
          normals: matching S_i x 3 outward normals (needed when align > 0)
          max_points, batch_size, latent_points: M points per target, targets per
              batch (default all), N parameter samples per curve (default M)
        Network
          feature_mode: 'triplane' (feature sampled at each loop's query point) or
              'embedding' (one learnable row per code)
          triplane: field config (centre, half, resolution, channels)
          query_points: one world xyz per target, triplane mode only
          num_models, codes: embedding rows and the row of each target
          num_neurons, num_layers, feat_dim: MLP width, depth, embedding length
          axis_condition: also feed each loop's slice normal to the network
        Loss terms (0 disables)
          align: weight of the normal-direction term, cosine-annealed to 0
          code_smooth with code_pairs / feature_pairs (+ code_weights): keep the
              conditions of neighbouring loops close
          feature_tv: total-variation smoothness of the triplane grids
          distill (+ distill_warmup, distill_pairs, distill_weights): keep each
              neighbour pair's seam distance D at or below the warm-up teacher's
        Seam phase solve (seam_align > 0)
          seam_pairs, seam_weights: neighbour pairs; seam_keys: loop ids for `inherit`
          seam_every, seam_start: solve every N epochs, starting at epoch seam_start
          seam_steps, seam_lr: Adam steps and step size per solve
          seam_lam: pair vote scale, lam / (lam + D_best); seam_mu: zero-mean prior
        freeze_f_epochs: for the first N epochs train only the triplane, MLP frozen
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_mode = feature_mode
        if feature_mode == 'triplane':
            if query_points is None:
                raise ValueError('triplane mode needs one query point per target')
            query_points = np.asarray(query_points, dtype=np.float32)
            if query_points.shape != (len(point_sets), 3):
                raise ValueError(f'one xyz query per target: {len(point_sets)} '
                                 f'targets, shape {query_points.shape}')
            self.queries = torch.as_tensor(query_points, device=self.device)
            self.num_models = len(point_sets)
            self.codes = np.arange(len(point_sets), dtype=np.int64)
        else:
            self.queries = None
            self.num_models = num_models
            if codes is None:
                self.codes = np.arange(len(point_sets)) % num_models
            else:
                self.codes = np.asarray(codes, dtype=np.int64).reshape(-1)
                if len(self.codes) != len(point_sets):
                    raise ValueError(f'one code per target: {len(point_sets)} targets, '
                                     f'{len(self.codes)} codes')
                if self.codes.min(initial=0) < 0 or self.codes.max(initial=-1) >= num_models:
                    raise ValueError(f'codes must name rows of the {num_models}-row '
                                     f'table, got {self.codes.min()}..{self.codes.max()}')
        # default: every target in one batch
        self.dataloader = DataGenerator(point_sets, max_points,
                                        batch_size or len(point_sets), seed, normals)
        
        self.model = miniNN(1 if feature_mode == 'triplane' else num_models,
                            num_neurons=num_neurons, num_layers=num_layers,
                            feat_dim=feat_dim, feature_mode=feature_mode,
                            triplane=triplane,
                            axis_condition=axis_condition).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.align = align
        self.code_smooth = float(code_smooth)
        self.code_pairs = None
        self.code_weights = None
        self.feature_pairs = None
        self.feature_weights = None
        self.feature_tv = float(feature_tv)
        if feature_mode == 'triplane':
            if code_pairs is not None and len(code_pairs):
                raise ValueError('triplane mode takes feature_pairs, not code_pairs')
            if feature_pairs is not None and len(feature_pairs):
                self.feature_pairs, self.feature_weights = self._pair_tensors(
                    feature_pairs, code_weights, len(point_sets), 'feature')
        elif feature_pairs is not None and len(feature_pairs):
            raise ValueError('feature_pairs only apply in triplane mode')
        else:
            if code_pairs is not None and len(code_pairs):
                self.code_pairs, self.code_weights = self._pair_tensors(
                    code_pairs, code_weights, num_models, 'code')
        self.loss_fn = CD_Loss()

        # each loop's plane normal, read off its points; used by the direction
        # term and as the network's axis condition
        if align and self.dataloader.normals is None:
            raise ValueError('align needs the normals of every point set')
        self.axes = (torch.as_tensor(
            np.stack([Square.from_points(p).axis for p in self.dataloader.point_sets]),
            dtype=torch.float32, device=self.device)
            if (align or axis_condition) else None)

        # N parameter samples in [0, 1); the curve closes at 1
        latent_points = self.dataloader.max_points if latent_points is None else latent_points
        self.z = torch.linspace(0, 1, latent_points + 1, device=self.device)[:-1, None]  # N x 1

        # seam distillation against the warm-up teacher (see _build_distill)
        self.distill = float(distill)
        if self.distill < 0:
            raise ValueError(f'distill must be non-negative, got {self.distill}')
        self.distill_pairs = None
        self.distill_weights = None
        self.distill_scale = None
        self.distill_target = None
        self.distill_count = None
        if self.distill:
            self._build_distill(distill_warmup, distill_pairs, distill_weights)

        # per-loop seam phase in turns; applied when predicting, never in training
        self.seam_align = float(seam_align)
        self.seam_every = int(seam_every)
        self.seam_start = int(seam_start)
        self.seam_steps = int(seam_steps)
        self.seam_lam = float(seam_lam)
        self.seam_lr = float(seam_lr)
        self.seam_mu = float(seam_mu)
        self.phase = None
        self.phase_optimizer = None
        self.seam_keys = None
        self.seam_pairs = None
        self.seam_weights = None
        if self.seam_align:
            if seam_pairs is None or not len(seam_pairs):
                raise ValueError('seam_align needs seam_pairs to align across')
            if self.seam_lam <= 0:
                raise ValueError(f'seam_lam must be positive, got {self.seam_lam}')
            if self.seam_lr <= 0:
                raise ValueError(f'seam_lr must be positive, got {self.seam_lr}')
            if self.seam_mu < 0:
                raise ValueError(f'seam_mu must be non-negative, got {self.seam_mu}')
            if self.seam_every < 0 or self.seam_steps < 1:
                raise ValueError(f'seam_every must be non-negative and seam_steps at '
                                 f'least 1, got {self.seam_every} and {self.seam_steps}')
            if self.seam_start < 0:
                raise ValueError(f'seam_start must be non-negative, got '
                                 f'{self.seam_start}')
            if seam_keys is not None:
                self.seam_keys = list(seam_keys)
                if len(self.seam_keys) != len(point_sets):
                    raise ValueError(f'one seam key per target: {len(point_sets)} '
                                     f'targets, {len(self.seam_keys)} keys')
            self.seam_pairs, self.seam_weights = self._pair_tensors(
                seam_pairs, seam_weights, len(point_sets), 'seam')
            self.phase = nn.Parameter(torch.zeros(len(point_sets),
                                                 device=self.device))
            self.phase_optimizer = torch.optim.Adam([self.phase], lr=self.seam_lr)
        elif seam_keys is not None:
            raise ValueError('seam_keys only mean something with seam_align set')

        # per-epoch readings by slice axis and per-loop winding, to catch loops
        # that run the wrong way round
        self.epoch_stats = []
        self.winding_history = []
        self._last_reversed = None

        # frozen start; epochs_done carries across refinement stages
        self.freeze_f_epochs = int(freeze_f_epochs)
        if self.freeze_f_epochs < 0:
            raise ValueError(f'freeze_f_epochs must be non-negative, got '
                             f'{self.freeze_f_epochs}')
        self.epochs_done = 0
        self.f_trainable = True

    def set_f_trainable(self, trainable):
        """Freeze or release the shared MLP; the triplane grid always trains."""
        trainable = bool(trainable)
        if trainable == self.f_trainable:
            return
        for parameter in self.model.model.parameters():
            parameter.requires_grad_(trainable)
        self.f_trainable = trainable
        print(f'  f map {"released" if trainable else "frozen"}: '
              f'{"every parameter trains" if trainable else "only the field trains"}')

    def _epoch_diagnostics(self, out, target, near, pair, d1, axis):
        """Per-loop fit, direction, winding and fold for one batch, by axis."""
        with torch.no_grad():
            near_t = target.gather(1, near[..., None].expand(-1, -1, 3))
            winding, fold = curve_winding(out, axis, centre=target.mean(1))
            loop = dict(fit=((out - near_t) ** 2).sum(-1).mean(1),
                        winding=winding, fold=fold)
            if d1 is not None and pair is not None:
                loop['align'] = (1.0 - normal_cosine(d1, pair, axis)).mean(1)
        return {name: value.cpu().numpy() for name, value in loop.items()}

    def _build_distill(self, warmup, pairs, weights):
        """Compute the warm-up teacher's seam distance D for every pair, once.

        Training then penalises only a student D above the teacher's (a ceiling).
        Each D uses its own curves' size; a fixed per-pair target radius turns the
        excess into Chamfer units.
        """
        if self.feature_mode != 'triplane':
            raise ValueError('the seam distillation reads the teacher through the '
                             'triplane field; fit with feature_mode="triplane"')
        if warmup is None:
            raise ValueError('the seam distillation needs the warm-up checkpoint to '
                             'build its teacher; pass distill_warmup')
        try:
            ckpt = torch.load(warmup, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            raise ValueError(f'the seam distillation needs a warm-up checkpoint, and '
                             f'there is none at {warmup}') from None
        if ckpt.get('kind') != 'cube_warmup':
            raise ValueError(f'{warmup} is not a cube warm-up checkpoint')
        if bool(ckpt.get('axis_condition', False)) != self.model.axis_condition:
            raise ValueError(
                f'{warmup} was warmed '
                f'{"with" if ckpt.get("axis_condition") else "without"} axis '
                f'conditioning and this network is '
                f'{"axis-conditioned" if self.model.axis_condition else "not"}; '
                f'the teacher and the student must read the same condition')
        if int(ckpt['embedding_dim']) != self.model.embedding_dim:
            raise ValueError(
                f'the teacher in {warmup} supplies {int(ckpt["embedding_dim"])}-D '
                f'features and this network reads {self.model.embedding_dim}-D; '
                f'warm up with the same triplane channels')

        teacher = miniNN(1, num_neurons=int(ckpt['num_neurons']),
                         num_layers=int(ckpt['num_layers']),
                         pe=bool(ckpt['pe']), feature_mode='triplane',
                         triplane=dict(ckpt['triplane']),
                         axis_condition=bool(ckpt['axis_condition']))
        teacher.load_state_dict(ckpt['state_dict'])
        teacher.to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        pairs, weights = self._pair_tensors(
            pairs, weights, len(self.dataloader.point_sets), 'distill')
        # fixed per-pair target size, so the term cannot drop by shrinking curves
        radius = np.asarray([
            float(((p - p.mean(0)) ** 2).sum(-1).mean())
            for p in self.dataloader.point_sets])
        index = pairs.cpu().numpy()
        scale = torch.as_tensor(0.5 * (radius[index[:, 0]] + radius[index[:, 1]]),
                                dtype=torch.float32, device=self.device)

        # map the shape's queries into the teacher's unit-cube frame
        triplane = self.model.triplane
        unit = (self.queries - triplane.box_centre) / triplane.box_half
        points = (teacher.triplane.box_centre
                  + unit * teacher.triplane.box_half)
        z = self.z[None].expand(len(self.queries), -1, -1)
        with torch.no_grad():
            feat = teacher.triplane(points)
            out = teacher(z, None, feat, self.axes)
            v = out - out.mean(1, keepdim=True)
            vi, vj = v[pairs[:, 0]], v[pairs[:, 1]]
            # each D is normalised by its own curves' size
            t_radius = (v ** 2).sum(-1).mean(1)
            t_scale = 0.5 * (t_radius[pairs[:, 0]] + t_radius[pairs[:, 1]])
            target = (((vi - vj) ** 2).sum(-1).mean(1)
                      / t_scale.clamp_min(1e-12))

        self.distill_pairs = pairs
        self.distill_weights = weights
        self.distill_scale = scale
        self.distill_target = target.detach()
        self.distill_count = len(self.dataloader.point_sets)
        print(f'  seam distillation: teacher {warmup}, {len(pairs)} pairs over '
              f'{len(self.queries)} loops, teacher D mean '
              f'{float(target.mean()):.6f} max {float(target.max()):.6f}, '
              f'weight {self.distill:g}')

    def align_phases(self):
        """Solve each loop's seam phase so neighbouring z = 0 points line up.

        Curves are decoded once and detached; only `self.phase` is optimised, so
        the shapes never change. Each pair votes lam / (lam + D_best), D_best being
        its distance at the best circular shift, so unrelated loops barely count.
        Returns the final median D.
        """
        if self.phase is None:
            raise ValueError('align_phases needs seam_align and seam_pairs')
        count = len(self.dataloader.point_sets)
        n = self.z.shape[0]
        with torch.no_grad():
            idx = (None if self.feature_mode == 'triplane'
                   else torch.as_tensor(self.codes, device=self.device))
            feat = self.feature_values() if self.feature_mode == 'triplane' else None
            curves = self.model(self.z[None].expand(count, -1, -1),
                                idx, feat, self.axes).detach()
        # centroids fixed before shifting, so D compares shape and seam only
        centres = curves.mean(1, keepdim=True)

        def drawn(ph):
            """Centred curves resampled after phase shifts ph (periodic linear interp)."""
            u = torch.remainder(self.z[:, 0][None, :] + ph[:, None], 1.0) * n
            lo = torch.floor(u)
            k0 = lo.long() % n
            w = (u - lo)[..., None]
            c0 = curves.gather(1, k0[..., None].expand(-1, -1, 3))
            c1 = curves.gather(1, ((k0 + 1) % n)[..., None].expand(-1, -1, 3))
            return c0 * (1.0 - w) + c1 * w - centres

        def seam_distance(v, scale_grad=True):
            """Per-pair seam distance D; scale_grad=False detaches the size normaliser."""
            vi, vj = v[self.seam_pairs[:, 0]], v[self.seam_pairs[:, 1]]
            s2 = 0.5 * ((vi ** 2).sum(-1).mean(1) + (vj ** 2).sum(-1).mean(1))
            if not scale_grad:
                # detached so a pair cannot lower D by changing its size
                s2 = s2.detach()
            return ((vi - vj) ** 2).sum(-1).mean(1) / s2.clamp_min(1e-12)

        with torch.no_grad():
            start = seam_distance(drawn(self.phase), scale_grad=False)
        # each pair's vote: its best D over all circular shifts (FFT cross-correlation)
        with torch.no_grad():
            vi = curves[self.seam_pairs[:, 0]] - centres[self.seam_pairs[:, 0]]
            vj = curves[self.seam_pairs[:, 1]] - centres[self.seam_pairs[:, 1]]
            corr = torch.fft.irfft((torch.fft.rfft(vj, dim=1)
                                    * torch.fft.rfft(vi, dim=1).conj()).sum(-1),
                                   n=n, dim=1)
            var = (vi ** 2).sum(-1).mean(1) + (vj ** 2).sum(-1).mean(1)
            best = (var[:, None] - 2.0 * corr / n).min(1).values
            agreement = (best / (0.5 * var).clamp_min(1e-12)).clamp_min(0.0)
            weight = self.seam_weights * self.seam_lam / (self.seam_lam + agreement)
            weight = weight / weight.sum()
        for _ in range(self.seam_steps):
            self.phase_optimizer.zero_grad()
            d = seam_distance(drawn(self.phase))
            energy = (weight * d).sum()
            energy = energy + self.seam_mu * self.phase.pow(2).mean()
            (self.seam_align * energy).backward()
            self.phase_optimizer.step()
            # wrap phases into [-0.5, 0.5) turns
            with torch.no_grad():
                self.phase.copy_(torch.remainder(self.phase + 0.5, 1.0) - 0.5)
        with torch.no_grad():
            end = seam_distance(drawn(self.phase), scale_grad=False)
        print(f'  seam align {self.seam_steps} steps on {len(self.seam_pairs)} pairs: '
              f'D median {float(start.median()):.4f} -> {float(end.median()):.4f}, '
              f'phase median |p| {float(self.phase.abs().median()):.4f} turns')
        return float(end.median())

    def _pair_tensors(self, pairs, weights, count, what):
        """Validate a P x 2 pair list and its weights, as device tensors."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.min() < 0 or pairs.max() >= count:
            raise ValueError(f'{what} pairs name entries outside 0..{count - 1}, '
                             f'got {pairs.min()}..{pairs.max()}')
        weights = (np.ones(len(pairs), dtype=np.float32) if weights is None
                   else np.asarray(weights, dtype=np.float32).reshape(-1))
        if len(weights) != len(pairs):
            raise ValueError(f'one weight per {what} pair: {len(pairs)} pairs, '
                             f'{len(weights)} weights')
        if not np.all(weights >= 0.0) or weights.sum() <= 0.0:
            raise ValueError(f'{what} pair weights must be non-negative and not all '
                             f'zero')
        return (torch.as_tensor(pairs, dtype=torch.long, device=self.device),
                torch.as_tensor(weights, dtype=torch.float32, device=self.device))

    def feature_values(self, ids=None):
        """The field feature of every target, or of the targets named by ids."""
        if self.feature_mode != 'triplane':
            raise ValueError('feature_values only exists in triplane mode')
        feat = self.model.triplane(self.queries)
        if ids is not None:
            feat = feat[torch.as_tensor(ids, dtype=torch.long, device=self.device)]
        return feat


    def train(self, epochs=1, every=None, on_epoch=None):
        """Train for `epochs` epochs; returns the mean loss per epoch.

        The learning rate and the align weight are cosine-annealed per batch over
        this call. on_epoch(n) is called after epoch n whenever n % every == 0.
        """
        # reset the lr so a second call does not start from the annealed minimum
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr
        # the anneal counts batches, so it is smooth within an epoch
        total = epochs * len(self.dataloader)
        step = 0
        history = []
        for e in range(epochs):
            # frozen start: only the triplane trains while epochs_done < freeze_f_epochs
            frozen = self.epochs_done < self.freeze_f_epochs
            self.set_f_trainable(not frozen)
            losses, fits, aligns, weights = [], [], [], []
            code_pens, tvs, distills = [], [], []
            diag = None
            if self.axes is not None:
                count = len(self.dataloader.point_sets)
                diag = {name: np.zeros(count) for name in
                        ('fit', 'winding', 'fold')}
                diag_seen = np.zeros(count)
            for ids, target, n_target in self.dataloader.index_batches():
                self.optimizer.zero_grad()

                target = torch.from_numpy(target).float().to(self.device)      # B x M x 3
                # embedding row of each target (unused in triplane mode)
                idx = torch.as_tensor(self.codes[ids], device=self.device)
                # slice axis of each target, shared by the condition and the direction term
                ids_t = torch.as_tensor(ids, device=self.device)
                axis = None if self.axes is None else self.axes[ids_t]
                feat = self.feature_values(ids) if self.feature_mode == 'triplane' else None
                # fresh leaf each step: the tangent is a derivative w.r.t. z
                z = self.z.expand(target.shape[0], -1, -1).clone().requires_grad_(True)
                out = self.model(z, idx, feat, axis)       # B x N x 3
                fit, near = self.loss_fn(out, target)
                loss = fit
                d1 = curve_tangent(out, z) if self.align else None
                pair = None
                if self.align:
                    # normals of the target points Chamfer matched to each prediction
                    n_t = torch.from_numpy(n_target).float().to(self.device)
                    pair = n_t.gather(1, near[..., None].expand(-1, -1, 3))
                    align = normal_penalty(d1, pair, axis)
                    # --normal is the starting weight, annealed to 0 over the call
                    weight = self.align * cosine_anneal(step, total)
                    loss = loss + weight * align
                    aligns.append(align.item())
                    weights.append(weight)

                if diag is not None:
                    # per-loop diagnostics from this step's curves and pairing
                    reading = self._epoch_diagnostics(out, target, near, pair,
                                                      d1, axis)
                    where = np.asarray(ids)
                    diag_seen[where] += 1
                    for name, value in reading.items():
                        if name not in diag:
                            diag[name] = np.zeros(count)
                        diag[name][where] += value

                if self.code_smooth and self.feature_pairs is not None:
                    # keep sampled features of neighbouring loops close (weighted by proximity)
                    all_feat = self.feature_values()
                    diff = (all_feat[self.feature_pairs[:, 0]]
                            - all_feat[self.feature_pairs[:, 1]])
                    code_pen = (self.feature_weights *
                                diff.pow(2).sum(1)).sum() / self.feature_weights.sum()
                    loss = loss + self.code_smooth * code_pen
                    code_pens.append(code_pen.item())
                elif self.code_smooth and self.code_pairs is not None:
                    # same, on the embedding rows
                    table = self.model.embedding.weight
                    diff = table[self.code_pairs[:, 0]] - table[self.code_pairs[:, 1]]
                    code_pen = (self.code_weights *
                                diff.pow(2).sum(1)).sum() / self.code_weights.sum()
                    loss = loss + self.code_smooth * code_pen
                    code_pens.append(code_pen.item())

                if self.feature_tv and self.feature_mode == 'triplane':
                    tv = self.model.triplane.tv_penalty()
                    loss = loss + self.feature_tv * tv
                    tvs.append(tv.item())

                if self.distill:
                    # seam distillation on the pairs whose two loops are both in this batch
                    position = np.full(self.distill_count, -1, dtype=np.int64)
                    position[np.asarray(ids)] = np.arange(len(ids))
                    local = position[self.distill_pairs.cpu().numpy()]
                    keep = (local[:, 0] >= 0) & (local[:, 1] >= 0)
                    if keep.any():
                        li = torch.as_tensor(local[keep, 0], dtype=torch.long,
                                             device=self.device)
                        lj = torch.as_tensor(local[keep, 1], dtype=torch.long,
                                             device=self.device)
                        v = out - out.mean(1, keepdim=True)
                        # student D normalised by its own curves' size, like the teacher's
                        r = (v ** 2).sum(-1).mean(1)
                        own = 0.5 * (r[li] + r[lj])
                        seen = (((v[li] - v[lj]) ** 2).sum(-1).mean(1)
                                / own.clamp_min(1e-12))
                        w = self.distill_weights[keep]
                        # penalise only the excess over the teacher's D, in Chamfer units
                        scale = self.distill_scale[keep]
                        gap = (seen - self.distill_target[keep]).clamp_min(0.0)
                        hold = (w * scale * gap).sum() / w.sum()
                        loss = loss + self.distill * hold
                        distills.append(hold.item())

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                fits.append(fit.item())
                step += 1
            # epoch line with each term printed separately
            parts = [f'fit {np.mean(fits):.4f}']
            if aligns:
                parts.append(f'align {np.mean(aligns):.3f} '
                             f'(w {np.mean(weights):.4f})')
            if code_pens:
                parts.append(f'{"feat" if self.feature_mode == "triplane" else "code"} '
                             f'{np.mean(code_pens):.4f}')
            if tvs:
                parts.append(f'tv {np.mean(tvs):.5f}')
            if distills:
                parts.append(f'distill {np.mean(distills):.6f}')
            print(f' {e} epoch: loss: {np.mean(losses):.4f} '
                  f'({", ".join(parts)})'
                  f'{" [f frozen]" if frozen else ""}')
            self.epochs_done += 1
            history.append(float(np.mean(losses)))
            if diag is not None:
                # average the per-loop diagnostics and summarise them per slice axis
                diag_seen_safe = np.maximum(diag_seen, 1)
                for name in diag:
                    diag[name] = np.where(diag_seen > 0,
                                          diag[name] / diag_seen_safe, np.nan)
                labels = self.axes.abs().argmax(-1).cpu().numpy()
                per_axis = {}
                for a, name in enumerate(('x', 'y', 'z')):
                    mask = labels == a
                    if not mask.any():
                        continue
                    row = dict(loops=int(mask.sum()),
                               fit=float(np.nanmean(diag['fit'][mask])),
                               winding=float(np.nanmean(diag['winding'][mask])),
                               fold=float(np.nanmean(diag['fold'][mask])),
                               reversed=int(np.nansum(diag['winding'][mask] < 0)))
                    if 'align' in diag:
                        row['align'] = float(np.nanmean(diag['align'][mask]))
                    per_axis[name] = row
                self.epoch_stats.append(dict(epoch=e + 1,
                                             loss=float(np.mean(losses)),
                                             axes=per_axis))
                # per-loop winding history: which loop flipped, and when
                self.winding_history.append(diag['winding'].copy())
                now = tuple(per_axis.get(name, {}).get('reversed', 0)
                            for name in ('x', 'y', 'z'))
                if now != self._last_reversed:
                    print(f'   loops wound the other way round (winding < 0) '
                          f'x/y/z: {now[0]}/{now[1]}/{now[2]}')
                    self._last_reversed = now
            # re-solve the seam phases periodically, before the on_epoch hook
            if (self.phase is not None and self.seam_every
                    and e + 1 >= self.seam_start
                    and (e + 1) % self.seam_every == 0):
                self.align_phases()
            if on_epoch is not None and every and (e + 1) % every == 0:
                on_epoch(e + 1)
        return history

    def inherit(self, previous, source=None):
        """Start from a previous stage's trainer (refinement with more planes).

        Copies the network (in triplane mode the whole model) and carries seam phases
        over by seam key. In embedding mode rows are copied by `source` (one previous
        row per row here, -1 = none; default identity, since codes are only appended).
        Optimiser moments restart.
        """
        if self.feature_mode != previous.feature_mode:
            raise ValueError('the two networks condition on features in different '
                             f'ways ({previous.feature_mode} and {self.feature_mode})')
        # carry each loop's seam phase over by key; new loops start at 0
        if self.phase is not None and self.seam_keys:
            carried = {}
            if previous.phase is not None and previous.seam_keys:
                carried = {key: float(previous.phase[i])
                           for i, key in enumerate(previous.seam_keys)}
            with torch.no_grad():
                for i, key in enumerate(self.seam_keys):
                    if key in carried:
                        self.phase[i] = carried[key]
            print(f'  {sum(1 for k in self.seam_keys if k in carried)} of '
                  f'{len(self.seam_keys)} loops carry their seam phase over')
        if self.feature_mode == 'triplane':
            # the field is positional, so copying the whole model carries every condition
            if previous.model.triplane.grid.shape != self.model.triplane.grid.shape:
                raise ValueError('the two triplanes have different grid shapes '
                                 f'({tuple(previous.model.triplane.grid.shape)} and '
                                 f'{tuple(self.model.triplane.grid.shape)})')
            self.model.load_state_dict(previous.model.state_dict())
            print(f'  {len(self.codes)} of {len(self.codes)} loops read the inherited '
                  f'triplane; added planes need no row seed')
            return self
        if previous.model.embedding.embedding_dim != self.model.embedding.embedding_dim:
            raise ValueError('the two networks condition on features of different '
                             f'length ({previous.model.embedding.embedding_dim} and '
                             f'{self.model.embedding.embedding_dim})')
        rows = previous.model.embedding.weight.detach()
        if source is None:
            # codes are only appended, so row i maps to row i
            source = np.arange(self.num_models)
        else:
            source = np.asarray(source, dtype=np.int64).reshape(-1)
            if len(source) != self.num_models:
                raise ValueError(f'one source per row: {self.num_models} rows, '
                                 f'{len(source)} named')
            if source.max(initial=-1) >= len(rows):
                raise ValueError(f'source names row {int(source.max())} of a trainer '
                                 f'that has {len(rows)}')
        shared = {name: value for name, value in previous.model.state_dict().items()
                  if name != 'embedding.weight'}
        # the embedding table is copied row by row below
        self.model.load_state_dict(shared, strict=False)
        have = np.zeros(self.num_models, dtype=bool)
        with torch.no_grad():
            for i, src in enumerate(source):
                if -1 < src < len(rows):
                    self.model.embedding.weight[i] = rows[src]
                    have[i] = True
        # count loops, not rows: some rows are planes this set does not cut
        kept = int(have[self.codes].sum())
        print(f'  {kept} of {len(self.codes)} loops carry their fit over from the '
              f'previous plane set, {len(self.codes) - kept} start from a square')
        return self

    def phase_coords(self, i=None):
        """Parameter coords of loop i with its seam phase applied (base coords for None)."""
        if i is None or self.phase is None:
            return self.z
        return torch.remainder(self.z + self.phase[i], 1.0)

    def predict(self, i=None):
        """N x 3 points predicted for loop i (the zero-feature base curve when None)."""
        if i is None:
            idx, feat = None, None
        elif self.feature_mode == 'triplane':
            idx, feat = None, self.feature_values([i])
        else:
            idx = torch.as_tensor([self.codes[i]], device=self.device)
            feat = None
        axis = None
        if self.axes is not None:
            if i is None:
                raise ValueError('an axis-conditioned network has no base curve: '
                                 'pass the loop whose slice axis to read')
            axis = self.axes[i:i + 1]
        with torch.no_grad():
            return self.model(self.phase_coords(i), idx, feat, axis)[0]

    def predict_tangent(self, i=None):
        """N x 3 analytic tangent d(prediction)/dz of loop i, at `predict`'s coords."""
        if i is None:
            idx, feat = None, None
        elif self.feature_mode == 'triplane':
            idx, feat = None, self.feature_values([i])
        else:
            idx = torch.as_tensor([self.codes[i]], device=self.device)
            feat = None
        axis = None
        if self.axes is not None:
            if i is None:
                raise ValueError('an axis-conditioned network has no base curve: '
                                 'pass the loop whose slice axis to read')
            axis = self.axes[i:i + 1]
        # needs the graph, so redo the forward pass with a fresh leaf, outside no_grad
        z = self.phase_coords(i)[None].clone().requires_grad_(True)
        return curve_tangent(self.model(z, idx, feat, axis), z)[0].detach().cpu().numpy()

    def bend(self):
        """Bend reading (circle = 39.5) over all fitted curves; reported, not optimised."""
        idx = torch.as_tensor(self.codes, device=self.device)
        feat = self.feature_values() if self.feature_mode == 'triplane' else None
        z = self.z.expand(len(idx), -1, -1).clone().requires_grad_(True)
        d1 = curve_tangent(self.model(z, idx, feat, self.axes), z)
        return float(bend_penalty(d1, curve_curvature(d1, z)).detach())


def save_model(path, trainer, **meta):
    """Save the fitted network, its condition (rows or grid), codes, queries, axes
    and seam phases, plus any `meta`, so curves can be decoded without retraining."""
    tp = getattr(trainer.model, 'triplane', None)
    triplane = None
    if tp is not None:
        triplane = dict(centre=tp.box_centre.detach().cpu().numpy(),
                        half=tp.box_half.detach().cpu().numpy(),
                        resolution=tp.resolution, channels=tp.channels, tv=tp.tv)
    payload = dict(
        feature_mode=trainer.feature_mode,
        num_neurons=int(trainer.model.model[0].out_features),
        num_layers=int(trainer.model.num_layers),
        pe=bool(trainer.model.pe),
        embedding_dim=int(trainer.model.embedding_dim),
        axis_condition=bool(trainer.model.axis_condition),
        num_models=int(trainer.num_models),
        latent_points=int(trainer.z.shape[0]),
        codes=np.asarray(trainer.codes, dtype=np.int64),
        queries=(None if trainer.queries is None
                 else trainer.queries.detach().cpu().numpy()),
        axes=(None if trainer.axes is None
              else trainer.axes.detach().cpu().numpy()),
        triplane=triplane,
        phases=(None if trainer.phase is None
                else trainer.phase.detach().cpu().numpy()),
        state_dict={k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
    )
    payload.update(meta)
    torch.save(payload, path)
    return path


def load_model(path, device='cpu'):
    """Rebuild a `save_model` checkpoint as (model in eval mode, meta dict).

    Decode with z = torch.linspace(0, 1, meta['latent_points'] + 1)[:-1, None].
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_layers = ckpt.get('num_layers', 2)
    if ckpt['feature_mode'] == 'triplane':
        model = miniNN(1, num_neurons=ckpt['num_neurons'], num_layers=num_layers,
                       pe=ckpt['pe'], feature_mode='triplane',
                       triplane=ckpt['triplane'],
                       axis_condition=bool(ckpt.get('axis_condition', False)))
    else:
        model = miniNN(ckpt['num_models'], num_neurons=ckpt['num_neurons'],
                       num_layers=num_layers, feat_dim=ckpt['embedding_dim'],
                       pe=ckpt['pe'], feature_mode='embedding',
                       axis_condition=bool(ckpt.get('axis_condition', False)))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != 'state_dict'}
    meta['device'] = device
    return model, meta

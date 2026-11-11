import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .chamfer import CD_Loss
from .curves import (bend_penalty, curve_curvature, curve_normal,
                     curve_tangent, curve_winding, normal_cosine,
                     normal_penalty, project_normals)
from .data import DataGenerator
from .network import make_network
from .triplane import cell_pool_structure


class Trainer:
    def __init__(self, point_sets, num_models=1, max_points=None, batch_size=None,
                 latent_points=None, lr=1e-2, seed=0, num_neurons=64, num_layers=2,
                 feat_dim=32,
                 normals=None, align_normal=0.0, match_angle=150.0,
                 cycle=0.0, repel=0.0, repel_gap=0.05, repel_samples=1000,
                 codes=None,
                 code_pairs=None, code_weights=None, code_smooth=0.0,
                 feature_mode='embedding', triplane=None, query_points=None,
                 query_sets=None, feature_pool='centroid',
                 feature_pairs=None, feature_tv=0.0,
                 distill=0.0, distill_warmup=None, distill_pairs=None,
                 distill_weights=None,
                 seam_align=0.0, seam_pairs=None, seam_weights=None, seam_keys=None,
                 seam_every=10, seam_start=0, seam_steps=100, seam_lam=0.02,
                 seam_lr=2e-2, seam_mu=1e-3, freeze_f_epochs=0):
        """
        Fit a list of point sets, one target per sample.

        point_sets: list of S_i x 3 targets of varying size, e.g. the loop slices
        num_models: rows in the embedding table, i.e. how many curves the shared
                     network is conditioned on
        codes: the row each target is conditioned on, one per point set. This is what
                     ties a target to a feature code that outlives the list: a plane
                     keeps its row when the set around it is rebuilt, so the fit stays
                     on the loop it was put on and nothing has to be matched up by
                     position. None gives one row per target, i % num_models
        max_points: M, points per target (default: median S_i), see DataGenerator
        batch_size: targets per iteration (default: all of them, one batch)
        latent_points: N, parameter coords the network is evaluated at (default: M)
        num_neurons: width of the shared MLP; the whole set shares one, so a target
                     with a lot of structure in it needs a wider one
        feat_dim: length of the latent vector each target is conditioned on, which is
                     what lets one MLP serve many different targets
        normals: list of S_i x 3 outward unit normals, one per point of each set.
                     Chamfer distance pairs points by distance alone, so a curve can
                     cut across the surface it is meant to run along wherever two
                     stretches of it come close; the outward normals are what say which
                     way the surface faces and which way round the loop runs
        align_normal: weight of the normal term, 0 to fit by position only. It reads 0
                     when a curve runs the way the outward normals say, 1 when it runs
                     across them, and 2 when it runs the other way round, so it is a
                     fraction of the loss rather than a distance
        match_angle: degrees; chamfer pairs a predicted point to a target point only
                     when the angle between the curve's implied normal and the
                     target's outward normal, read in the slice plane, is no
                     greater than this. 150 by default, which passes over a point on the far
                     side of a thin feature but keeps almost every true pair; a point
                     with no allowed partner is left unmatched that step. 180 matches
                     by distance alone, as before
        cycle: weight of a cycle-consistency term, 0 (default) to leave the inverse
                     map untrained. The network also carries the other half of the
                     map: a point on the curve and the condition it was drawn under
                     are read back to the parameter z it belongs to. The term is the
                     mean squared error between that reading and z, so a curve that
                     folds two distant parameters onto the same point cannot be
                     inverted and the term says so. The point branch is detached
                     inside forward, so the gradient trains the inverse map and the
                     condition, not the fitted curve; the fit is not pulled off its
                     points by a term in different units
        repel: weight of a repulsion term on the curve, 0 (default) to leave the
                     parameterisation free. Anchors spread around the closed loop are
                     charged when a pair closer than a gap is not an immediate
                     parameter neighbour: two parameters that are far apart on the
                     loop have no business landing on the same 3D point, while
                     adjacent samples of any curve may. Unlike `cycle` the term acts
                     on the points themselves, so the fit feels it; it is a squared
                     distance, in the same units as the chamfer loss
        repel_gap: the separation a non-neighbour pair is held apart by, as a
                     fraction of the target loop's own RMS radius, so one value works
                     for a large loop and a small one. 0.05 holds a pair apart by a
                     twentieth of the loop's size
        repel_samples: how many anchors the dense repulsion term is evaluated on,
                     spread evenly in z. The full sample set can be thousands of
                     points per loop, so a dense N x N loss is not affordable; the
                     anchors cost a K x K matrix and the network is smooth in z
                     between them
        code_pairs: P x 2 embedding rows that should stay close, e.g. the codes of
                     loops on neighbouring planes. The term is the weighted mean of
                     their squared code distance, so it constrains the latent table
                     rather than the curves directly
        code_weights: one non-negative weight per pair, so a neighbour that is close
                     in the plane counts more than one at the edge of the gate
        code_smooth: weight of the code-smoothness term, 0 to learn every row freely.
                     It is in the same units as the chamfer loss, a squared distance
                     summed over xyz, so it competes with the fit rather than replacing
                     it; the row still moves to its own loop, just less independently
                     of its neighbours
        feature_mode: 'embedding' keeps the per-target row table. 'triplane' replaces
                     it with a continuous field: every target is conditioned on the
                     feature sampled at its query point, and there is no row to hand
                     out, inherit or blend. Neighbouring loops then start from similar
                     features because the field is continuous, not because a term made
                     their rows similar
        triplane: configuration for the field: the world box its unit cube is
                     mapped onto (centre, half), resolution and channels; required
                     in triplane mode
        query_points: one xyz per target, in world coordinates, only in triplane
                     mode. A loop's own centre is the natural query; the target is in
                     its plane's local frame, so the caller adds that frame's origin
        query_sets: one S_i x 3 world-point set per target, only with
                     feature_pool='occupied_cells'. These are the target points in
                     world coordinates, the observation the fit is given, not the
                     fitted curve
        feature_pool: how a loop's feature is read off the field. 'centroid'
                     (default) bilinearly samples the single query point.
                     'occupied_cells' snaps every target point to its nearest grid
                     node in each plane, then averages the grid vectors of the
                     unique nodes, so the code describes the loop's footprint and not
                     only its centre; two loops whose centroids share a cell still
                     read different cells when their extents differ. Width stays
                     3 * channels and gradients still reach the grid
        feature_pairs: P x 2 target indices defining the neighbour graph in triplane
                     mode; the epoch's `pair cos` reading compares the sampled
                     features of each pair
        feature_tv: weight of a total-variation penalty on the three grids, 0 to leave
                     every cell free
        distill: weight of a teacher-student seam distillation, 0 to train on the
                     shape alone. The warm-up teacher already draws the shape and
                     runs it the right way round; what the shape fit is free to lose
                     is where z = 0 falls, since no fit term sees it. The teacher is
                     therefore read once at the shape's own queries, its
                     per-pair seam distance D is the target, and every step's student
                     D is pulled to that profile. The curves themselves are never
                     matched pointwise: a common deformation costs nothing, so shape
                     and orientation stay with the fit and the warm-up. The term is
                     the weighted mean absolute difference of the two D profiles,
                     carried back to a squared length by each pair's target radius,
                     so it is in the same units as the chamfer loss and is exactly
                     zero when the student's seam agreement is the teacher's
        distill_warmup: the cube-warm-up checkpoint the teacher is built from.
                     `plc_main` passes its own `--warmup`, so the teacher is the
                     checkpoint the fit started from unless another one is named
        distill_pairs, distill_weights: P x 2 loop indices whose seams the profile
                     is read over, and one non-negative weight per pair, e.g. the
                     same neighbour graph the phase solve uses
        seam_align: weight of the seam-alignment energy, 0 to leave every loop's
                     parameterisation alone. The fit cannot see where z = 0 falls on a
                     closed curve -- sliding a loop through its own parameter changes
                     no chamfer or normal term -- so that phase is one number per
                     loop, learned against its neighbours and nothing else. The curves
                     are read once, detached, and only the phases take a gradient, so
                     the shape never moves: what changes is where the first sample of
                     an exported loop sits
        seam_pairs: P x 2 target indices whose seams are held together, e.g. the
                     mutual-nearest loops of `code_neighbour_graph` in `plc_main`
        seam_weights: one non-negative weight per pair, as for `code_pairs`
        seam_keys: one key per target, used only by `inherit` to carry a phase over
                     to the next plane set by what the loop is rather than by where
                     it sits in the list
        seam_every: solve the phases every this many epochs, 0 to never solve them
        seam_start: the first epoch the solve runs at, counted from 1
        seam_steps: Adam steps per solve; each solve starts from the phases the last
                     one left, so ten steps every ten epochs is one continuing
                     optimisation rather than ten cold starts
        seam_lam: the scale that decides how much a pair votes, in the same units as
                     the seam distance D (mean squared separation over the two loops'
                     mean squared radius). Each pair is matched at its best circular
                     shift first, and its vote is lam / (lam + D_best): a pair that
                     agrees after a shift is the same cross-section with its seams in
                     the wrong place and votes nearly in full, while two different
                     loops -- a torso against a limb -- vote at nearly nothing. The
                     energy itself is D, so a seam far out is pulled in proportion to
                     how far out it is rather than sinking into a saturating switch
        seam_lr: Adam step size on the phases, in turns
        seam_mu: weight of a zero-mean prior on the phases, which is what keeps the
                     whole run from rotating together: only differences are observable,
                     and this picks the one nearest the phases it started from
        freeze_f_epochs: freeze the shared MLP -- the map from (sin z, cos z,
                     feature) to the curve -- for the first N epochs and train
                     nothing but the conditioning field. The warm-up's map is what
                     makes every loop close, circulate the same way and start from the
                     same square; the fit's first job is to say where each shape loop
                     is, which is the field's job. Frozen, the map keeps that prior
                     while the grid moves the curves onto their slices; released after
                     N epochs, it refines the shapes. 0 (default) trains both from the
                     first step, as every run before this did
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_mode = feature_mode
        if feature_pool not in ('centroid', 'occupied_cells'):
            raise ValueError(f"feature_pool must be 'centroid' or 'occupied_cells', "
                             f"got {feature_pool!r}")
        if feature_pool != 'centroid' and feature_mode != 'triplane':
            raise ValueError('occupied-cell pooling only exists in triplane mode')
        self.feature_pool = feature_pool
        self.query_sets = None
        self.cell_cells = None
        self.cell_masks = None
        self.cell_counts = None
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
            if feature_pool == 'occupied_cells':
                if query_sets is None:
                    raise ValueError('occupied-cell pooling needs the world points '
                                     'of every target; pass query_sets')
                if len(query_sets) != len(point_sets):
                    raise ValueError(f'one point set per target: {len(point_sets)} '
                                     f'targets, {len(query_sets)} point sets')
                self.query_sets = [np.asarray(p, dtype=np.float32).reshape(-1, 3)
                                   for p in query_sets]
                if any(len(p) == 0 for p in self.query_sets):
                    raise ValueError('every pooled point set needs at least one '
                                     'point')
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
        ## one batch of every target by default, not one per row: with a code per plane
        ## the table is larger than the target list, and a batch sized by the table
        ## would repeat targets to fill it
        self.dataloader = DataGenerator(point_sets, max_points,
                                        batch_size or len(point_sets), seed, normals)
        
        self.model = make_network(feature_mode, num_models=num_models,
                                  num_neurons=num_neurons, num_layers=num_layers,
                                  feat_dim=feat_dim, triplane=triplane).to(self.device)

        ## the occupied nodes depend only on the frame and the fixed target
        ## points, so the gather structure is built once; the grid values it
        ## indexes are read fresh every step
        if self.feature_pool == 'occupied_cells':
            self.cell_cells, self.cell_masks = cell_pool_structure(
                self.model.triplane, self.query_sets)
            self.cell_counts = torch.stack(
                [m.sum(-1).long() for m in self.cell_masks], dim=-1)
            print(f'  occupied-cell pooling: {len(self.query_sets)} loops, median '
                  f'{int(self.cell_counts.median())} nodes per loop, widest '
                  f'{int(self.cell_counts.max())}')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.align_normal = align_normal
        self.cycle = float(cycle)
        if self.cycle < 0:
            raise ValueError(f'cycle must be non-negative, got {self.cycle}')
        self.repel = float(repel)
        self.repel_gap = float(repel_gap)
        self.repel_mask = None
        self.repel_stride = 1
        if self.repel < 0:
            raise ValueError(f'repel must be non-negative, got {self.repel}')
        if self.repel_gap < 0:
            raise ValueError(f'repel_gap must be non-negative, got {self.repel_gap}')
        self.match_angle = float(match_angle)
        if not 0.0 < self.match_angle <= 180.0:
            raise ValueError('match_angle must be in (0, 180], got '
                             f'{match_angle}')
        self.code_smooth = float(code_smooth)
        self.code_pairs = None
        self.code_weights = None
        self.feature_pairs = None
        self.feature_tv = float(feature_tv)
        if feature_mode == 'triplane':
            if code_pairs is not None and len(code_pairs):
                raise ValueError('triplane mode takes feature_pairs, not code_pairs')
            if feature_pairs is not None and len(feature_pairs):
                self.feature_pairs, _ = self._pair_tensors(
                    feature_pairs, code_weights, len(point_sets), 'feature')
        elif feature_pairs is not None and len(feature_pairs):
            raise ValueError('feature_pairs only apply in triplane mode')
        else:
            if code_pairs is not None and len(code_pairs):
                self.code_pairs, self.code_weights = self._pair_tensors(
                    code_pairs, code_weights, num_models, 'code')
        self.loss_fn = CD_Loss(self.match_angle)
        self.scheduler = None

        ## the plane each target lies in is read off the target by the dataloader: a
        ## loop is a cross-section, so the direction it is flattest in is the plane it
        ## was cut with. The direction term, the winding report and the normal-aware
        ## pairing all need them to project into the loop's plane
        if align_normal and self.dataloader.normals is None:
            raise ValueError('align_normal needs the normals of every point set')
        self.axes = torch.as_tensor(self.dataloader.axes, dtype=torch.float32,
                                    device=self.device)

        ## self.z are the [0-1] parameter coords; the latent vector of a part is
        ## appended inside forward, from that part's embedding, so it stays learnable
        latent_points = self.dataloader.max_points if latent_points is None else latent_points
        self.z = torch.linspace(0, 1, latent_points + 1, device=self.device)[:-1, None]  # N x 1

        ## the repulsion term is read on anchors spread around the closed loop: the
        ## dense K x K matrix over a handful of them covers the whole parameter
        ## range, while the full N x N one over thousands of samples would not fit
        if self.repel:
            if repel_samples < 3:
                raise ValueError(f'repel_samples must be at least 3, got '
                                 f'{repel_samples}')
            self.repel_stride = max(1, latent_points // int(repel_samples))
            anchors = len(self.z[::self.repel_stride])
            ## the immediate neighbours on either side of the cycle are 'quite
            ## close' and left free; every other pair of parameters is charged
            step = (torch.arange(anchors, device=self.device)[:, None]
                    - torch.arange(anchors, device=self.device)[None]).abs()
            step = torch.minimum(step, anchors - step)
            self.repel_mask = (step >= 2).to(torch.float32)

        ## the teacher-student seam distillation: the warm-up maps the shape's own
        ## queries to curves, and its per-pair D is the ceiling the fit is held
        ## under. The curves are never matched, so this term sees only the seam
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

        ## where z = 0 falls on each loop, in turns. Read at predict time only: the
        ## training forward pass keeps the base coords, so with the phases at zero the
        ## fit is bit for bit the run without this, and a solved phase is a
        ## reparameterisation of the same curve rather than a different one
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

        ## one row of readings per epoch, split by slice axis, and one winding per
        ## loop per epoch. The epoch line is a mean over every loop, so a family that
        ## runs the other way round is invisible in it; these say which family and
        ## when it turned over
        self.epoch_stats = []
        self.winding_history = []
        self._last_reversed = None

        ## the shared map's first epochs: frozen, only the field trains. The counter
        ## is the trainer's own, so a stage handed the trainer continues the same
        ## budget rather than restarting it
        self.freeze_f_epochs = int(freeze_f_epochs)
        if self.freeze_f_epochs < 0:
            raise ValueError(f'freeze_f_epochs must be non-negative, got '
                             f'{self.freeze_f_epochs}')
        self.epochs_done = 0
        self.f_trainable = True

    def set_f_trainable(self, trainable):
        """Freeze or release the shared MLP, the f map, without rebuilding the run.

        The triplane grid is a parameter of `self.model` but not of `self.model.model`,
        so it keeps training either way. Freezing removes the map's gradients too, so
        the frozen branch is cheaper as well as constrained; the optimizer keeps its
        parameter list, and a parameter without a gradient is simply not stepped.
        """
        trainable = bool(trainable)
        if trainable == self.f_trainable:
            return
        for parameter in self.model.model.parameters():
            parameter.requires_grad_(trainable)
        self.f_trainable = trainable
        print(f'  f map {"released" if trainable else "frozen"}: '
              f'{"every parameter trains" if trainable else "only the field trains"}')

    def _epoch_diagnostics(self, out, target, near, pair, d1, axis,
                           near_ok=None):
        """Per-loop fit, direction, winding and fold for one batch, by axis.

        fit is the distance to the plain nearest target point, whether or not the
        normal test accepted that pair: it reads how far the curve sits from the
        cloud, which is a measurement rather than a term. The direction reading
        is masked to the accepted pairs, because an unmatched point carries an
        arbitrary nearest normal and charging it to the curve would report a
        direction the fit never used.
        """
        with torch.no_grad():
            near_t = target.gather(1, near[..., None].expand(-1, -1, 3))
            winding, fold = curve_winding(out, axis, centre=target.mean(1))
            loop = dict(fit=((out - near_t) ** 2).sum(-1).mean(1),
                        winding=winding, fold=fold)
            if d1 is not None and pair is not None:
                turn = 1.0 - normal_cosine(d1, pair, axis)
                if near_ok is None:
                    loop['align'] = turn.mean(1)
                else:
                    w = near_ok.to(turn.dtype)
                    loop['align'] = (turn * w).sum(1) / w.sum(1).clamp_min(1.0)
        return {name: value.cpu().numpy() for name, value in loop.items()}

    def _build_distill(self, warmup, pairs, weights):
        """
        Read the teacher's seam profile at the shape's own conditions.

        The warm-up is a teacher of shape and orientation, and both are already in
        the student's weights. What the fit cannot see is where z = 0 falls: the
        chamfer and normal terms are invariant to sliding a closed curve
        through its own parameter. The teacher can see it, so the teacher is read
        once at this shape's own queries and axes, and the per-pair seam distance D
        of its curves is the ceiling the student is held under: the term charges
        only the excess over the teacher, so it can lower D and never raise it.

        The teacher's field lives on the unit cube, so a shape query is carried into
        the teacher's own frame before it is sampled; the shape's frame and the
        teacher's then name the same cell, which is what makes the profile the one
        the student starts from. The curves themselves are never matched: only their
        D is, so a common deformation is free and shape and orientation stay with
        the fit. D is scale-free, so each side reads its own curve size -- the
        teacher draws at cube scale (the sizes the warm-up trained on), the
        student at the shape's -- and
        the two relative disagreements are what is compared. At step 0 the curves
        are the same and the two D values are equal. The pair's fixed target radius
        only carries the dimensionless gap into chamfer units, so neither D can be
        lowered by shrinking a curve.

        warmup: path of the cube-warm-up checkpoint the teacher is built from
        pairs: P x 2 loop indices whose seams the profile is read over
        weights: one non-negative weight per pair
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
        if int(ckpt['embedding_dim']) != self.model.embedding_dim:
            raise ValueError(
                f'the teacher in {warmup} supplies {int(ckpt["embedding_dim"])}-D '
                f'features and this network reads {self.model.embedding_dim}-D; '
                f'warm up with the same triplane channels')
        teacher_pool = ckpt.get('feature_pool', 'centroid')
        if teacher_pool != self.feature_pool:
            raise ValueError(
                f'{warmup} was warmed with {teacher_pool} features and this fit '
                f'pools them as {self.feature_pool}; the teacher and the student '
                f'must read the same feature')

        teacher = make_network('triplane',
                               num_neurons=int(ckpt['num_neurons']),
                               num_layers=int(ckpt['num_layers']),
                               pe=bool(ckpt['pe']),
                               triplane=dict(ckpt['triplane']))
        ## checkpoints written before the inverse head carry no cycle.* keys
        teacher.load_state_dict(ckpt['state_dict'], strict=False)
        teacher.to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        pairs, weights = self._pair_tensors(
            pairs, weights, len(self.dataloader.point_sets), 'distill')
        ## the target scale is each loop's own mean squared radius on its points,
        ## fixed once: it carries the dimensionless D gap into chamfer units, so the
        ## term cannot be lowered by shrinking a curve
        radius = np.asarray([
            float(((p - p.mean(0)) ** 2).sum(-1).mean())
            for p in self.dataloader.point_sets])
        index = pairs.cpu().numpy()
        scale = torch.as_tensor(0.5 * (radius[index[:, 0]] + radius[index[:, 1]]),
                                dtype=torch.float32, device=self.device)

        ## the teacher's grid is laid on the unit cube, so the shape's normalized
        ## query is carried into the teacher's own frame before sampling
        triplane = self.model.triplane
        unit = (self.queries - triplane.box_centre) / triplane.box_half
        points = (teacher.triplane.box_centre
                  + unit * teacher.triplane.box_half)
        z = self.z[None].expand(len(self.queries), -1, -1)
        with torch.no_grad():
            if self.feature_pool == 'occupied_cells':
                ## pool the same target points the student pools, carried into the
                ## teacher's frame: a shape point set is normalized by the student's
                ## box and laid on the teacher's cube, exactly as the queries are
                teacher_sets = [teacher.triplane.box_centre.cpu().numpy()
                                + ((p - triplane.box_centre.cpu().numpy())
                                   / triplane.box_half.cpu().numpy())
                                * teacher.triplane.box_half.cpu().numpy()
                                for p in self.query_sets]
                cells, masks = cell_pool_structure(teacher.triplane, teacher_sets)
                feat = teacher.triplane.pool_cells(cells, masks)
            else:
                feat = teacher.triplane(points)
            out, _ = teacher(z, None, feat)
            v = out - out.mean(1, keepdim=True)
            vi, vj = v[pairs[:, 0]], v[pairs[:, 1]]
            ## D is a ratio, so each side reads its own curve size: the teacher
            ## draws at cube scale (the warm-up's training sizes), and a shape-target denominator
            ## would inflate its D wherever the shape loop is small
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
        """
        Move where z = 0 falls on every loop until neighbouring seams line up.

        The curves are read once from the network and detached, and every sample
        after that is a periodic linear interpolation of that fixed array, so the
        only thing the energy differentiates against is `self.phase`. The shape
        terms are therefore untouched by construction rather than by a small weight:
        the fitted curves come out of this exactly as they went in, drawn from a
        different starting point.

        The energy is a pair-weighted seam distance: for each pair the two seams are
        compared as they stand, D being their mean squared separation over the two
        loops' mean squared radius. A pair's vote is not the same for every pair,
        though: the loops are first matched at their best circular shift, and the
        distance they reach there says whether they are the same cross-section at
        all. A pair that only agrees after a shift is one where the seams are simply
        in the wrong place, and it votes at full weight; a pair that disagrees even
        at its best shift is two different loops -- a torso against a limb -- and
        votes at lam / (lam + D_best), which is nearly nothing. The pull itself is D
        and not the switched value, so a badly misplaced seam is pulled as hard as a
        slightly misplaced one rather than falling into the tail of a saturating
        function. The zero-mean prior on the phases fixes the one gauge the pairs
        cannot: every phase moved by the same amount changes no difference at all.
        """
        if self.phase is None:
            raise ValueError('align_phases needs seam_align and seam_pairs')
        count = len(self.dataloader.point_sets)
        n = self.z.shape[0]
        with torch.no_grad():
            idx = (None if self.feature_mode == 'triplane'
                   else torch.as_tensor(self.codes, device=self.device))
            feat = self.feature_values() if self.feature_mode == 'triplane' else None
            curves = self.model(self.z[None].expand(count, -1, -1), idx,
                                feat)[0].detach()
        ## the centroid is read before any phase is applied, so the centred curve a
        ## phase draws is the same set of points slid through the coords: on a dense
        ## loop the sampled mean barely moves, and fixing it keeps the distance about
        ## shape and seam rather than about where this sample set's centre landed
        centres = curves.mean(1, keepdim=True)

        def drawn(ph):
            u = torch.remainder(self.z[:, 0][None, :] + ph[:, None], 1.0) * n
            lo = torch.floor(u)
            k0 = lo.long() % n
            w = (u - lo)[..., None]
            c0 = curves.gather(1, k0[..., None].expand(-1, -1, 3))
            c1 = curves.gather(1, ((k0 + 1) % n)[..., None].expand(-1, -1, 3))
            return c0 * (1.0 - w) + c1 * w - centres

        def seam_distance(v, scale_grad=True):
            vi, vj = v[self.seam_pairs[:, 0]], v[self.seam_pairs[:, 1]]
            s2 = 0.5 * ((vi ** 2).sum(-1).mean(1) + (vj ** 2).sum(-1).mean(1))
            if not scale_grad:
                ## the normaliser is the pair's own size, which the phase does not
                ## change; letting it into the gradient would let a pair grow its way
                ## out of a mismatch
                s2 = s2.detach()
            return ((vi - vj) ** 2).sum(-1).mean(1) / s2.clamp_min(1e-12)

        with torch.no_grad():
            start = seam_distance(drawn(self.phase), scale_grad=False)
        ## how much each pair votes, read once: the circular cross-correlation of the
        ## two centred loops over every shift gives the separation they could reach if
        ## the seams were placed freely, so a pair of the same cross-section is told
        ## apart from a pair of two different loops without depending on where the
        ## phases happen to be when the solve starts
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
            ## the phases are angles in turns: a step is taken in the unwrapped
            ## parameterisation and then folded back, which the geometry cannot see
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
        if self.feature_pool == 'occupied_cells':
            feat = self.model.triplane.pool_cells(self.cell_cells,
                                                  self.cell_masks)
        else:
            feat = self.model.triplane(self.queries)
        if ids is not None:
            feat = feat[torch.as_tensor(ids, dtype=torch.long, device=self.device)]
        return feat


    def train(self, epochs=1, every=None, on_epoch=None):
        """
        One pass over the sampled batches per epoch; returns the mean loss per epoch.

        The cosine anneal spans this call, so building it here rather than in
        __init__ is what makes T_max the epochs actually run.

        The direction term is taken at the constant weight `align_normal`. It settles
        which way round each loop runs: two stretches of a contour passing close are
        told apart by the way they run rather than by where they are. The printed
        `align` value is the term itself, unweighted, so the columns stay comparable
        across epochs.

        every, on_epoch: `on_epoch(n)` is called after epoch n (counted from 1)
        whenever n is a multiple of `every`, which is where a caller watches a run
        rather than only judging it: the state handed over is the one the epoch
        printed, so a picture taken there is the picture of that line. Both default
        to None, which watches nothing.
        """
        ## a second call would otherwise anneal from the dead eta_min the first one
        ## ended on, leaving every later step with a frozen learning rate
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-4)
        history = []
        for e in range(epochs):
            ## the first N epochs keep the warm-up's map and move only the field;
            ## the counter is the trainer's, so a stage boundary does not refund the
            ## frozen epochs or restart the budget
            frozen = self.epochs_done < self.freeze_f_epochs
            self.set_f_trainable(not frozen)
            ## the rate every batch of this epoch trains with: the scheduler steps
            ## once per epoch, at the bottom, so this is read before it moves
            rate = self.optimizer.param_groups[0]['lr']
            losses, fits, normal_loss = [], [], []
            matches, reroutes = [], []
            smooth_pens, distills = [], []
            cycles, repels = [], []
            ## per-loop readings split by slice axis: the axes are always read off
            ## the data, so the record is kept for every run
            count = len(self.dataloader.point_sets)
            diag = {name: np.zeros(count) for name in
                    ('fit', 'winding', 'fold')}
            diag_seen = np.zeros(count)
            for ids, target, n_target in self.dataloader.index_batches():
                self.optimizer.zero_grad()

                target = torch.from_numpy(target).float().to(self.device)      # B x M x 3
                ## the row is each point set's own feature code, so a target keeps its
                ## curve however the list around it is rebuilt; every sample shares the
                ## same coords, so one call covers the batch
                idx = torch.as_tensor(self.codes[ids], device=self.device)
                ## the slice axis each target was cut on, by its own id: the direction
                ## term and the normal-aware pairing read it
                ids_t = torch.as_tensor(ids, device=self.device)
                axis = self.axes[ids_t]
                feat = self.feature_values(ids) if self.feature_mode == 'triplane' else None
                ## a leaf of its own every step: the penalties differentiate with
                ## respect to the coords, so they have to be a graph input, and a view
                ## of the shared self.z would pile every step's gradient onto one leaf
                z = self.z.expand(target.shape[0], -1, -1).clone().requires_grad_(True)
                out, zpred = self.model(z, idx, feat)       # B x N x 3, B x N x 1
                ## both terms read the cloud's normals: the pairing compares them in
                ## the curve's own plane, the direction term projects the raw vector
                ## itself
                n_t = (None if n_target is None else
                       torch.from_numpy(n_target).float().to(self.device))
                ## carried into the slice plane once, here, rather than inside the
                ## loss, so the loss only asks about directions
                n_plane = None if n_t is None else project_normals(n_t, axis)
                ## the tangent is needed by the direction term and by the normal read
                ## off the curve for the pairing, so it is taken once, before the fit.
                ## The derivative is only paid for when one of them will read it
                uses_match = self.match_angle < 180.0 and n_t is not None
                d1 = curve_tangent(out, z) if (self.align_normal or uses_match) else None
                pred_normal = curve_normal(d1, axis) if d1 is not None else None
                fit, near, match_meta = self.loss_fn(
                    out, target, n_plane, pred_normal, self.match_angle)
                loss = fit
                if match_meta is not None:
                    matches.append(match_meta['matched'])
                    reroutes.append(match_meta['rerouted'])
                near_ok = None if match_meta is None else match_meta['near_ok']
                pair = None
                if self.align_normal:
                    ## the outward normal of the target point chamfer matched to each
                    ## predicted one, against the direction the curve travels there:
                    ## the normals pick out one way round the loop, and two stretches
                    ## of a contour passing close are told apart by the way they run,
                    ## not by where they are. A point the normal test left unmatched
                    ## has no such pairing, so it is left out of the mean
                    pair = n_t.gather(1, near[..., None].expand(-1, -1, 3))
                    ## the plane normal is the one of the slice each target was cut on,
                    ## so it is read by the target's own id and not by its row
                    loss_normal = normal_penalty(d1, pair, axis, mask=near_ok)
                    loss = loss + self.align_normal * loss_normal
                    normal_loss.append(loss_normal.item())

                if diag is not None:
                    ## read off the same curves and the same pairing the step just
                    ## used, so the record is the step's own state rather than a new
                    ## draw of it
                    reading = self._epoch_diagnostics(out, target, near, pair,
                                                      d1, axis, near_ok)
                    where = np.asarray(ids)
                    diag_seen[where] += 1
                    for name, value in reading.items():
                        if name not in diag:
                            diag[name] = np.zeros(count)
                        diag[name][where] += value

                if self.feature_mode == 'triplane':
                    if self.feature_tv:
                        tv = self.model.triplane.tv_penalty()
                        loss = loss + self.feature_tv * tv
                        smooth_pens.append(tv.item())
                elif self.feature_mode == 'embedding':
                    if self.code_smooth and self.code_pairs is not None:
                        ## the rows are shared by every sample, so this is one term
                        ## over the table rather than one per batch element. A pair's
                        ## weight falls with its in-plane distance, so a close
                        ## neighbour is held to its code and a distant one is free to
                        ## differ
                        table = self.model.embedding.weight
                        diff = (table[self.code_pairs[:, 0]]
                                - table[self.code_pairs[:, 1]])
                        code_pen = (self.code_weights *
                                    diff.pow(2).sum(1)).sum() / self.code_weights.sum()
                        loss = loss + self.code_smooth * code_pen
                        smooth_pens.append(code_pen.item())
                else:
                    raise ValueError(f'unknown feature mode {self.feature_mode!r}')

                if self.distill:
                    ## the student's own seam profile, read on the curves this step
                    ## already drew and paired only where both loops are in the
                    ## batch. Holding it under the teacher's ceiling leaves the curves
                    ## themselves free: a common deformation, or a rotation, costs
                    ## nothing, and only the relative seam placement is held
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
                        ## D is a ratio, so the student reads its own curve size
                        ## just as the teacher does; with the teacher's curves
                        ## reproduced at step 0 the two D values are then equal
                        r = (v ** 2).sum(-1).mean(1)
                        own = 0.5 * (r[li] + r[lj])
                        seen = (((v[li] - v[lj]) ** 2).sum(-1).mean(1)
                                / own.clamp_min(1e-12))
                        w = self.distill_weights[keep]
                        ## the teacher's D is a ceiling, not a target: carry the
                        ## dimensionless excess back to a squared length by the
                        ## pair's fixed target radius before it competes with the fit
                        scale = self.distill_scale[keep]
                        gap = (seen - self.distill_target[keep]).clamp_min(0.0)
                        hold = (w * scale * gap).sum() / w.sum()
                        loss = loss + self.distill * hold
                        distills.append(hold.item())

                if self.cycle:
                    ## the inverse map reads the points this step drew, detached
                    ## inside forward, and has to recover the parameter that made
                    ## them: two parameters folded onto the same point cannot both
                    ## be read back, which is the collapse the term is against
                    loss_cycle = ((zpred - z) ** 2).mean()
                    loss = loss + self.cycle * loss_cycle
                    cycles.append(loss_cycle.item())

                if self.repel_mask is not None:
                    ## anchors every repel_stride-th parameter; all pairs except the
                    ## immediate neighbours have to keep apart by the gap. The gap is
                    ## read from the target's own RMS radius, so a collapse of the
                    ## output cannot shrink the threshold it is measured against
                    anchors = out[:, ::self.repel_stride]
                    size = ((target - target.mean(1, keepdim=True)) ** 2
                            ).sum(-1).mean(1)
                    span = ((anchors[:, :, None] - anchors[:, None]) ** 2).sum(-1)
                    gap = (self.repel_gap ** 2) * size[:, None, None]
                    pen = (self.repel_mask * (gap - span).clamp_min(0.0)).sum()
                    pen = pen / (self.repel_mask.sum() * len(ids))
                    loss = loss + self.repel * pen
                    repels.append(pen.item())

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                fits.append(fit.item())
            self.scheduler.step()
            ## how far apart neighbouring loops' codes sit: the field is what the
            ## fit moves, and while f is frozen it is the only moving part, so this
            ## says whether the conditioning is splitting the loops or leaving them
            ## together
            pair_cos = None
            if self.feature_mode == 'triplane' and self.feature_pairs is not None:
                with torch.no_grad():
                    all_feat = self.feature_values()
                    a = all_feat[self.feature_pairs[:, 0]]
                    b = all_feat[self.feature_pairs[:, 1]]
                    pair_cos = float(1.0 - F.cosine_similarity(a, b, dim=1).mean())
            ## the terms separately: a weight is chosen by what it costs in the fit
            ## against what it buys in smoothness, which one total cannot show
            parts = [f'fit {np.mean(fits):.4f}']
            if matches:
                ## the share of predicted points the normal test found a partner for,
                ## and the share it moved off the plain nearest: the first says how
                ## much of the curve the step could fit, the second how much the rule
                ## changed the pairing
                parts.append(f'match {np.mean(matches):.1%} '
                             f'(moved {np.mean(reroutes):.1%})')
            if normal_loss:
                parts.append(f'align {np.mean(normal_loss):.3f}')
            if pair_cos is not None:
                parts.append(f'pair cos {pair_cos:.4f}')
            if smooth_pens:
                ## the one smoothness term of whichever condition is in use
                if self.feature_mode == 'triplane':
                    parts.append(f'tv {np.mean(smooth_pens):.5f}')
                else:
                    parts.append(f'code {np.mean(smooth_pens):.4f}')
            if distills:
                parts.append(f'distill {np.mean(distills):.6f}')
            if cycles:
                parts.append(f'cycle {np.mean(cycles):.6f}')
            if repels:
                parts.append(f'repel {np.mean(repels):.6f}')
            parts.append(f'lr {rate:.2e}')
            print(f' {e} epoch: loss: {np.mean(losses):.4f} '
                  f'({", ".join(parts)})'
                  f'{" [f frozen]" if frozen else ""}')
            self.epochs_done += 1
            history.append(float(np.mean(losses)))
            if diag is not None:
                ## the axis a loop belongs to is the plane it was cut on, so the
                ## labels come from the trainer's own axes, not from a name in the
                ## data that a caller could order differently
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
                                             pair_cos=pair_cos,
                                             axes=per_axis))
                ## kept per loop rather than per axis: which loop turned over, and at
                ## which epoch, is the question this record exists to answer
                self.winding_history.append(diag['winding'].copy())
                now = tuple(per_axis.get(name, {}).get('reversed', 0)
                            for name in ('x', 'y', 'z'))
                if now != self._last_reversed:
                    print(f'   loops wound the other way round (winding < 0) '
                          f'x/y/z: {now[0]}/{now[1]}/{now[2]}')
                    self._last_reversed = now
            ## the phases follow the curves: a solve is cheap and starts where the
            ## last one left, so re-solving on the way through keeps the seam where
            ## the current shape puts it. Before the watcher, so a picture of epoch n
            ## is drawn with the phases that epoch ended on
            if (self.phase is not None and self.seam_every
                    and e + 1 >= self.seam_start
                    and (e + 1) % self.seam_every == 0):
                self.align_phases()
            ## after the line it belongs to, and at the last epoch too: a watcher gets
            ## the run at 50, 100, ... including whatever count the run ends on, so
            ## the pictures stand in order without a hole where the last one fell
            if on_epoch is not None and every and (e + 1) % every == 0:
                on_epoch(e + 1)
        return history

    def inherit(self, previous, source=None):
        """
        Continue from another trainer's fit: its shared network, and every embedding
        row the two tables have in common.

        previous: the trainer the stage before this one ended on
        source: one row of `previous` per row here, or -1 for a row with no
                  counterpart, which keeps the zero feature. None is the plane-code
                  case: a row is a plane's code and codes are only ever appended, so
                  row i here is row i there, and the table of this trainer starts
                  with the whole of the last one

        The embedding table is the one thing that cannot be copied whole, because a
        refinement stage has rows the stage before it did not. When rows are handed
        out per plane code and never moved, the rows of the earlier stage are a prefix
        of the later one and this is just a copy of that prefix: no loop is matched up
        by position or by where its points sit, so nothing can be handed the wrong
        curve when a stage inserts planes among the ones already cut.

        Nothing else moves: the optimiser was built on these parameters and they are
        copied into in place, but its moments start again, so a stage is a fresh set of
        steps rather than a continuation of the last stage's.
        """
        if self.feature_mode != previous.feature_mode:
            raise ValueError('the two networks condition on features in different '
                             f'ways ({previous.feature_mode} and {self.feature_mode})')
        ## where z = 0 falls belongs to the loop, not to a row: the stages are matched
        ## by seam key, which names the plane and the slot on it and survives planes
        ## being inserted around it. A loop this stage did not have last time starts at
        ## zero, and is moved by the first solve
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
            ## the field is a function of position, so there is no row to match up:
            ## copying the grids and the decoder copies the whole condition, and a
            ## loop on a plane both stages cut reads the same feature at the same
            ## query point. Added planes need no seed for the same reason.
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
            ## row i is row i: what this table is short of is the planes this stage
            ## added, which are the rows past the end of the last one
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
        ## strict off, since the one table left out on purpose is the one that differs
        self.model.load_state_dict(shared, strict=False)
        have = np.zeros(self.num_models, dtype=bool)
        with torch.no_grad():
            for i, src in enumerate(source):
                if -1 < src < len(rows):
                    self.model.embedding.weight[i] = rows[src]
                    have[i] = True
        ## counted over the loops, not the rows: the rows without a loop here are the
        ## codes of planes this plane set does not cut, and they are not a fit either
        kept = int(have[self.codes].sum())
        print(f'  {kept} of {len(self.codes)} loops carry their fit over from the '
              f'previous plane set, {len(self.codes) - kept} start from a square')
        return self

    def phase_coords(self, i=None):
        """
        The parameter coords point set i is drawn at, with its seam phase applied.

        This is the one place the phase is read into a prediction, so every caller
        -- the fit report, the normals, the exported loops -- draws the same curve
        from the same starting point. Point set None is the base curve, which has
        no seam to move.
        """
        if i is None or self.phase is None:
            return self.z
        return torch.remainder(self.z + self.phase[i], 1.0)

    def predict(self, i=None):
        """
        N x 3 points the network predicts for point set i, or for the base curve when
        i is None.
        """
        if i is None:
            idx, feat = None, None
        elif self.feature_mode == 'triplane':
            idx, feat = None, self.feature_values([i])
        else:
            idx = torch.as_tensor([self.codes[i]], device=self.device)
            feat = None
        with torch.no_grad():
            return self.model(self.phase_coords(i), idx, feat)[0][0]

    def predict_tangent(self, i=None):
        """
        N x 3 analytic tangent of the curve predicted for point set i, or for the base
        curve when i is None.

        This is d(prediction)/dz at the same parameter values the points are drawn
        at, so it is the exact tangent of the fitted curve rather than a central
        difference between neighbouring points.  Its length is not thrown away when
        predict_with_normals turns it into an in-plane normal: it becomes that
        normal's length, which is what keeps the drawn arrows in proportion.  Read on
        the warm-up's own curve it is how fast that curve travels per step of the
        parameter, which the square says should be its own perimeter.
        """
        if i is None:
            idx, feat = None, None
        elif self.feature_mode == 'triplane':
            idx, feat = None, self.feature_values([i])
        else:
            idx = torch.as_tensor([self.codes[i]], device=self.device)
            feat = None
        ## predict() runs under no_grad, but a derivative needs the graph, so this
        ## repeats the forward pass with a fresh leaf. The explicit batch axis keeps
        ## curve_tangent's result 1 x N x 3; a bare N x 1 coord drops that axis. The
        ## coords carry the loop's seam phase, so the tangent is read at the same
        ## points predict draws rather than one loop's worth further round
        z = self.phase_coords(i)[None].clone().requires_grad_(True)
        return curve_tangent(self.model(z, idx, feat)[0], z)[0].detach().cpu().numpy()

    def predict_with_normals(self, i, axis=None):
        """
        N x 3 predicted points and their N x 3 in-plane normals.

        The network predicts points, not normals, so the normal is read off the fitted
        curve: the curve lies in its slice, which leaves one in-plane direction square
        to its tangent, and the quarter turn is taken in the sense the direction term
        asks the tangent for, so the arrow is the surface normal the curve implies
        rather than its negation. The order of the cross product is that whole choice:
        crossing the tangent with the plane normal is one quarter turn, crossing the
        plane normal with the tangent is the same turn the other way, and the direction
        term asks for the tangent to be axis x normal. A quarter turn applied twice is
        a half turn, axis x (axis x t) = -t, so the surface normal is t x axis; the
        other order turns the tangent the wrong way for a surface the curve is a
        contour of.

        That length is left exactly as the turn gives it rather than scaled to one: the
        curve runs in its slice, so it is |tangent|, which is how far the fit travels
        per step of the parameter. A loop drawn larger, or one the network sweeps
        faster, therefore carries longer vectors, and only a tangent lifted out of the
        plane shortens them. On a fit that satisfies the direction term this is the
        cloud's own outward normal; a loop running against it draws inward with it,
        which is the state that term is there to correct.

        i: point set index, or None for the curve conditioned on no feature at all
        axis: plane normal of that set's slice, in the prediction's frame; required,
                 since nothing else in a bare point list says which plane it is in
        """
        if axis is None:
            raise ValueError('predict_with_normals needs the plane normal to cross with')
        points = self.predict(i).cpu().numpy()
        tangent = self.predict_tangent(i)
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        if len(points) < 2:
            return points, np.zeros_like(points)
        return points, np.cross(tangent, axis[None, :])

    def bend(self):
        """
        The bend reading over every fitted curve as it now stands, all sets at once.

        A circle reads 39.5, so the number says how many times round the fitted curves
        run in the course of one: it is the only way to tell a straightened fit from
        an equally well fitting wiggly one. Nothing optimises this; the fit has no
        bend term, and the number is printed so the curves' own shape stays visible.
        """
        idx = torch.as_tensor(self.codes, device=self.device)
        feat = self.feature_values() if self.feature_mode == 'triplane' else None
        z = self.z.expand(len(idx), -1, -1).clone().requires_grad_(True)
        ## a derivative needs the graph, so this cannot sit under no_grad like predict;
        ## the graph is dropped with the return value
        d1 = curve_tangent(self.model(z, idx, feat)[0], z)
        return float(bend_penalty(d1, curve_curvature(d1, z)))

    def nearest_distance(self, points, target):
        """Distance from each row of points to the nearest row of target."""
        p = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        ## chunked: a full slice against M predicted points does not fit as one cdist
        return torch.cat([torch.cdist(chunk, t).min(1).values
                          for chunk in p.split(2048)]).cpu().numpy()

    ## blue -> cyan -> yellow -> red, so a good fit reads blue and a bad one red
    RAMP = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]], np.float64)

    def error_colors(self, errors):
        """RGBA colours for an error array, ramped over its own range."""
        span = errors.max() - errors.min()
        pos = (errors - errors.min()) / (span if span > 0 else 1) * (len(self.RAMP) - 1)
        lo = np.floor(pos).astype(int)
        frac = (pos - lo)[:, None]
        rgb = (self.RAMP[lo] * (1 - frac)
               + self.RAMP[np.minimum(lo + 1, len(self.RAMP) - 1)] * frac)
        return np.concatenate([rgb, np.full((len(errors), 1), 255)], axis=1).astype(np.uint8)


def save_model(path, trainer, **meta):
    """Write the fitted network so it can be replayed without the training run.

    The file holds the shared MLP, the condition it reads (embedding table or
    triplane grid), the parameter grid length, the per-loop codes/queries and the
    seam phase each loop was drawn from. Anything a caller needs to locate the
    curves (keys, plane normals and plane points, target counts) is passed through
    `meta` and stored beside them.
    """
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
        feature_pool=getattr(trainer, 'feature_pool', 'centroid'),
        cell_counts=(None if getattr(trainer, 'cell_counts', None) is None
                     else trainer.cell_counts.detach().cpu().numpy()),
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
    """Rebuild the network `save_model` wrote, as (model, meta) with meta on CPU.

    The returned model is in eval mode and needs only the parameter grid and a
    feature: `z = torch.linspace(0, 1, meta['latent_points'] + 1)[:-1, None]`.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = make_network(ckpt['feature_mode'],
                         num_models=ckpt['num_models'],
                         num_neurons=ckpt['num_neurons'],
                         num_layers=ckpt.get('num_layers', 2),
                         feat_dim=ckpt['embedding_dim'],
                         pe=ckpt['pe'], triplane=ckpt['triplane'])
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != 'state_dict'}
    meta['device'] = device
    return model, meta

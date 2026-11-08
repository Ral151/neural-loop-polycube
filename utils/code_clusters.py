"""Cluster a run's per-loop triplane codes and show the clusters in 3D.

Each loop's code is the 48-D feature the field returns at its query point.
Average-linkage agglomerative clustering on cosine distance, k chosen by the
silhouette score unless --k is given. Writes <stem>_code_clusters[_tag].png
(world view, code-PCA view, silhouette curve), .csv and .ply (one coloured
sphere per loop).

    python main.py clusters log/<run>/<stem>_triplane.npz --k 10
    python main.py clusters <npz> --k auto --kmax 8
    python main.py clusters <npz> --state log/<run>/states/<stem>_state_e0100.pt --tag e0100

--state re-reads the codes from a saved network (`main.py fit --save-states`,
or a *_model.pt); the queries and keys still come from the npz.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from model.network import TriplaneField


def axis_of_keys(keys):
    """Per-loop (axis letter, offset, slot); keys are (axis, offset, slot)."""
    a = keys[:, 0].astype(int)
    return np.array(['xyz'[i] for i in a]), keys[:, 1], keys[:, 2].astype(int)


def sample_state(path, xyz):
    """Codes at the loop queries, sampled from a saved network's triplane grid.

    Accepts a bare state dict (`--save-states`) or a *_model.pt ('state_dict')."""
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    grid = sd['triplane.grid']
    field = TriplaneField(centre=sd['triplane.box_centre'], half=sd['triplane.box_half'],
                          resolution=grid.shape[-1], channels=grid.shape[1])
    with torch.no_grad():
        field.grid.copy_(grid)
        feats = field.sample(torch.as_tensor(xyz, dtype=torch.float32))
    return feats.double().numpy()


def cluster_ply(xyz, labels, colours, radius):
    """One sphere per loop centroid, painted in its cluster's colour."""
    template = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    parts = []
    for p, c in zip(xyz, labels):
        m = template.copy()
        m.apply_translation(p)
        rgba = np.round(np.asarray(colours[c][:3]) * 255).astype(np.uint8)
        m.visual.vertex_colors = np.tile(np.append(rgba, 255), (len(m.vertices), 1))
        parts.append(m)
    return trimesh.util.concatenate(parts)


def main(argv=None):
    """Cluster the codes of a run's loops and write the figure, CSV and PLY."""
    ap = argparse.ArgumentParser(prog='main.py clusters',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('npz', help='<run>_triplane.npz from a finished run')
    ap.add_argument('--k', default='auto', help='number of clusters or "auto"')
    ap.add_argument('--kmax', type=int, default=12,
                    help='largest k tried by the silhouette search')
    ap.add_argument('--out', default=None,
                    help='output PNG (default <run>/<stem>_code_clusters[_tag].png)')
    ap.add_argument('--sphere-radius', type=float, default=None,
                    help='PLY sphere radius (default: 0.8%% of the cloud diameter)')
    ap.add_argument('--state', default=None,
                    help='a saved network (main.py fit --save-states, or a '
                         '*_model.pt); resample the codes from it instead of the npz '
                         'features (queries/keys still come from the npz)')
    ap.add_argument('--tag', default='',
                    help='suffix for the output names, e.g. e0100')
    args = ap.parse_args(argv)

    data = np.load(args.npz)
    xyz = data['queries'].astype(np.float64)
    keys = data['keys'].astype(np.float64)
    feats = (sample_state(args.state, xyz) if args.state
             else data['features'].astype(np.float64))
    axis, off, slot = axis_of_keys(keys)
    n = len(feats)

    # cosine similarity of the unit-length codes
    unit = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    sim = unit @ unit.T
    print(f'{Path(args.npz).name}: {n} loops, codes {feats.shape[1]}-D')
    print(f'  cosine similarity: p5 {np.percentile(sim, 5):.3f}, '
          f'p50 {np.percentile(sim, 50):.3f}, '
          f'p95 {np.percentile(sim, 95):.3f}')

    # silhouette score for every candidate k
    scores = {}
    for kk in range(2, min(args.kmax, n - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=kk, metric='cosine',
                                      linkage='average').fit_predict(unit)
        scores[kk] = silhouette_score(unit, lab, metric='cosine')
    if args.k == 'auto':
        k = max(scores, key=scores.get)
    else:
        k = int(args.k)
    labels = AgglomerativeClustering(n_clusters=k, metric='cosine',
                                     linkage='average').fit_predict(unit)
    # a fixed k outside the searched range still needs its score
    if k not in scores:
        scores[k] = silhouette_score(unit, labels, metric='cosine')

    # order clusters by size so the legend and colours stay stable
    order = sorted(np.unique(labels), key=lambda c: -(labels == c).sum())
    remap = {c: i for i, c in enumerate(order)}
    labels = np.array([remap[c] for c in labels])
    colours = plt.get_cmap('tab10' if k <= 10 else 'tab20')(np.arange(k) % 10)
    print(f'  k = {k}, silhouette {scores[k]:.3f}')
    for c in range(k):
        sel = labels == c
        intra = sim[np.ix_(sel, sel)][np.triu_indices(sel.sum(), 1)].mean() \
            if sel.sum() > 1 else np.nan
        inter = sim[np.ix_(sel, ~sel)].mean()
        mix = {a: int((axis[sel] == a).sum()) for a in 'xyz' if (axis[sel] == a).any()}
        print(f'    c{c}: {sel.sum():3d} loops ({" ".join(f"{a}{v}" for a, v in mix.items())})'
              f'  intra cos {intra:+.3f}  inter {inter:+.3f}')

    # feature PCA for the middle panel
    centred = feats - feats.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    pc = centred @ vt[:3].T

    fig = plt.figure(figsize=(16.5, 5.6))
    axw = fig.add_subplot(1, 3, 1, projection='3d')
    axf = fig.add_subplot(1, 3, 2, projection='3d')
    axs = fig.add_subplot(1, 3, 3)

    handles = []
    for c in range(k):
        sel = labels == c
        colour = colours[c]
        axw.scatter(xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], s=26, color=colour,
                    depthshade=False, label=f'c{c} ({sel.sum()})')
        axf.scatter(pc[sel, 0], pc[sel, 1], pc[sel, 2], s=26, color=colour,
                    depthshade=False)
        for ax, pts in ((axw, xyz[sel]), (axf, pc[sel])):
            ax.text(*(pts.mean(0) + 0.04 * (pts.max(0) - pts.min(0))),
                    f'c{c}', color=colour, fontsize=9, weight='bold')
        handles.append(plt.Line2D([], [], marker='o', ls='', color=colour,
                                  label=f'c{c} ({sel.sum()})'))

    for ax in (axw, axf):
        ax.grid(alpha=0.2)
    axw.set_xlabel('x'); axw.set_ylabel('y'); axw.set_zlabel('z')
    axw.set_title('loops in world space', fontsize=10)
    axw.set_box_aspect(xyz.max(0) - xyz.min(0))
    axw.view_init(elev=20, azim=-58)

    axf.set_xlabel(f'PC1 {ev[0] * 100:.1f}%')
    axf.set_ylabel(f'PC2 {ev[1] * 100:.1f}%')
    axf.set_zlabel(f'PC3 {ev[2] * 100:.1f}%')
    axf.set_title('loops in the code PCA basis', fontsize=10)
    axf.view_init(elev=20, azim=-58)

    ks = sorted(scores)
    axs.plot(ks, [scores[i] for i in ks], 'o-', color='#4c72b0')
    axs.axvline(k, color='#d62728', ls='--', lw=1,
                label=f'chosen k = {k}')
    axs.set_xlabel('clusters k')
    axs.set_ylabel('silhouette (cosine)')
    axs.set_title('cluster count', fontsize=10)
    axs.grid(alpha=0.25)
    axs.legend(fontsize=8, frameon=False)

    fig.legend(handles=handles, loc='lower center', ncol=min(k, 8),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'{Path(args.npz).stem}: loops clustered on the cosine '
                 f'similarity of their {feats.shape[1]}-D triplane codes '
                 f'(k = {k}, silhouette {scores[k]:.3f})', fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    tag = f'_{args.tag}' if args.tag else ''
    out = Path(args.out or Path(args.npz).parent /
               f'{Path(args.npz).stem.removesuffix("_triplane")}_code_clusters{tag}.png')
    fig.savefig(out, dpi=170)

    with open(out.with_suffix('.csv'), 'w') as f:
        f.write('axis,offset,slot,x,y,z,cluster\n')
        for i in range(n):
            f.write(f'{axis[i]},{off[i]:.4f},{slot[i]},{xyz[i, 0]:.4f},'
                    f'{xyz[i, 1]:.4f},{xyz[i, 2]:.4f},{labels[i]}\n')
    # sphere radius scales with the shape's size
    radius = args.sphere_radius or 0.008 * np.linalg.norm(xyz.max(0) - xyz.min(0))
    clouds = cluster_ply(xyz, labels, colours, radius)
    clouds.export(out.with_suffix('.ply'))
    print(f'wrote {out}')
    print(f'wrote {out.with_suffix(".csv")}')
    print(f'wrote {out.with_suffix(".ply")} '
          f'({len(clouds.vertices)} verts, {len(clouds.faces)} faces, '
          f'sphere radius {radius:.4f})')


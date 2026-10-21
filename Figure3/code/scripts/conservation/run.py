#!/usr/bin/env python3
"""Recompute cell-type-specific X_HENOSIS centroid similarities and clustered views."""

from itertools import combinations
from math import ceil
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from common.io import load_config, write_plot_csv
from common.names import SpeciesNames
from common.embedding import load_embedding

matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
MIN_CELLS, MIN_SPECIES, CAP, SEED = 20, 5, 300, 20260930
MAIN_CELLTYPES = ('Cardiomyocyte', 'Endothelial cell')
VERTEBRATES = {'Chiloscyllium_plagiosum', 'Danio_rerio', 'Protopterus_annectens',
               'Ambystoma_mexicanum', 'Xenopus_laevis', 'Trachemys_scripta_elegans',
               'Taeniopygia_guttata', 'Mus_musculus', 'Homo_sapiens'}
GROUP_COLORS = {'Vertebrate': '#336f9b', 'Other animal': '#b17443'}


def group(species):
    return 'Vertebrate' if species in VERTEBRATES else 'Other animal'


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else np.nan


def build_centroids(species, labels, embedding, eligible, seed=SEED):
    rng = np.random.default_rng(seed)
    centers, rows = {}, []
    for sp in sorted(set(species)):
        for ct in eligible:
            idx = np.flatnonzero((species == sp) & (labels == ct))
            rows.append({'species': sp, 'celltype': ct, 'n_cells': len(idx),
                         'eligible': len(idx) >= MIN_CELLS})
            if len(idx) >= MIN_CELLS:
                chosen = rng.choice(idx, min(CAP, len(idx)), replace=False)
                centers[(sp, ct)] = embedding[chosen].mean(axis=0).astype(float)
    return centers, pd.DataFrame(rows)


def make_pairs(centers, counts, labels, species):
    n = counts.set_index(['species', 'celltype']).n_cells.to_dict()
    rows = []
    for ct in labels:
        available = [sp for sp in species if (sp, ct) in centers]
        for a, b in combinations(available, 2):
            rows.append({'celltype': ct, 'species_a': a, 'species_b': b,
                         'similarity': cosine(centers[a, ct], centers[b, ct]),
                         'n_cells_a': n[a, ct], 'n_cells_b': n[b, ct]})
    return pd.DataFrame(rows)


def panel_matrix(eligibility, pairs, celltype):
    names = sorted(eligibility[(eligibility.celltype == celltype) &
                               (eligibility.eligible)].species.tolist())
    if len(names) < 2:
        raise ValueError(f'{celltype}: fewer than two eligible species')
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    np.fill_diagonal(matrix.values, 1.)
    for row in pairs[pairs.celltype == celltype].itertuples():
        matrix.loc[row.species_a, row.species_b] = row.similarity
        matrix.loc[row.species_b, row.species_a] = row.similarity
    if not np.isfinite(matrix.values).all() or not np.allclose(matrix, matrix.T):
        raise ValueError(f'{celltype}: missing/asymmetric similarity')
    return matrix


def cluster_linkage(matrix):
    distance = np.maximum(1 - matrix.to_numpy(float), 0)
    np.fill_diagonal(distance, 0)
    return linkage(squareform(distance, checks=True), method='average',
                   optimal_ordering=True)


def cluster_order(matrix):
    return matrix.index[leaves_list(cluster_linkage(matrix))].tolist()


def plot_rows(matrix, order, celltype):
    rows = []
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            rows.append({'celltype': celltype, 'row_species': a, 'column_species': b,
                         'row_position': i + 1, 'column_position': j + 1,
                         'similarity': matrix.loc[a, b], 'display_diagonal': i == j,
                         'row_group': group(a), 'column_group': group(b)})
    return rows


def draw_selected(eligibility, pairs, labels, stem, names, columns, size, title):
    rows = ceil(len(labels) / columns)
    compact = len(labels) <= 2
    fig = plt.figure(figsize=size)
    left = .20 if len(labels) == 1 else .09
    outer = fig.add_gridspec(rows, columns, left=left, right=.89,
                             top=.88 if compact else .93,
                             bottom=.20 if compact else .10,
                             wspace=.48, hspace=.55)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#ffffff')
    norm = TwoSlopeNorm(vmin=min(-.65, float(pairs.similarity.min())),
                        vcenter=0, vmax=1)
    all_rows, order_rows, image, axes = [], [], None, []
    for position, ct in enumerate(labels):
        matrix = panel_matrix(eligibility, pairs, ct)
        tree = cluster_linkage(matrix)
        order = matrix.index[leaves_list(tree)].tolist()
        for i, sp in enumerate(order):
            count = eligibility[(eligibility.celltype == ct) &
                                (eligibility.species == sp)].n_cells.iloc[0]
            order_rows.append({'celltype': ct, 'position': i+1, 'species': sp,
                               'group': group(sp), 'n_cells': int(count)})
        all_rows.extend(plot_rows(matrix, order, ct))
        inner = outer[position // columns, position % columns].subgridspec(
            2, 1, height_ratios=[.26, 1], hspace=.015)
        ax_tree = fig.add_subplot(inner[0])
        ax = fig.add_subplot(inner[1])
        dendrogram(tree, ax=ax_tree, no_labels=True, link_color_func=lambda _: '#555555',
                   above_threshold_color='#555555')
        ax_tree.set_xlim(0, 10 * len(order))
        ax_tree.set_xticks([])
        ax_tree.set_yticks([])
        for spine in ax_tree.spines.values():
            spine.set_visible(False)
        ax_tree.set_title(f'{ct} (n={len(order)})', fontsize=10, pad=5)
        data = matrix.loc[order, order].to_numpy(float).copy()
        np.fill_diagonal(data, np.nan)
        image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm,
                          aspect='equal', interpolation='nearest')
        labels_short = [names.display(sp, compact=True) for sp in order]
        ax.set_xticks(range(len(order)), labels_short, rotation=90, fontsize=6)
        ax.set_yticks(range(len(order)), labels_short, fontsize=6)
        for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tick, sp in zip(ticks, order):
                tick.set_color(GROUP_COLORS[group(sp)])
                tick.set_fontstyle('italic')
        ax.tick_params(length=0)
        axes.append((ax_tree, ax))
    fig.canvas.draw()
    for ax_tree, ax in axes:
        tb, hb = ax_tree.get_position(), ax.get_position()
        ax_tree.set_position([hb.x0, tb.y0, hb.width, tb.height])
    cax = fig.add_axes([.91, .34, .014, .34])
    fig.colorbar(image, cax=cax, label='X_HENOSIS centroid cosine similarity')
    fig.suptitle(title, x=.07, y=.975, ha='left', fontsize=15)
    fig.text(.5, .055 if compact else .035,
             'Average linkage on 1 − cosine similarity; <20-cell species omitted; diagonal blank.',
             ha='center', fontsize=8)
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_plot_csv(stem.with_suffix('.csv'), pd.DataFrame(all_rows),
                   ['celltype', 'row_species', 'column_species', 'similarity'])
    for ext in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=250, facecolor='white')
    plt.close(fig)
    return pd.DataFrame(order_rows)


def make_references(species, config):
    ages = pd.read_csv(config['pairwise_ages'])
    age_lookup = {tuple(sorted((r.species_a, r.species_b))): r.age_mya
                  for r in ages.itertuples()}
    genes = {}
    for sp in species:
        frame = pd.read_csv(config['orthology_root'] / f'{sp}.Homo_sapiens.all_relationships.csv',
                            dtype=str, keep_default_na=False)
        typ = next(c for c in frame if c.endswith('orthology_type'))
        human = 'hsapiens_ensembl_gene_id' if 'hsapiens_ensembl_gene_id' in frame else 'Homo_sapiens_gene_id'
        genes[sp] = set(frame.loc[frame[typ].eq('ortholog_one2one') & frame[human].ne(''), human])
    rows = []
    for a, b in combinations(species, 2):
        shared = len(genes[a] & genes[b])
        union = len(genes[a] | genes[b])
        rows.append({'species_a': a, 'species_b': b,
                     'age_mya': age_lookup[tuple(sorted((a, b)))],
                     'shared_1to1_human_ids': shared,
                     'one_to_one_coverage_jaccard': shared / union if union else np.nan})
    return pd.DataFrame(rows)


def validation_stats(pairs, labels, permutations=999):
    rows = []
    rng = np.random.default_rng(SEED)
    for ct in labels:
        part = pairs[pairs.celltype == ct]
        members = sorted(set(part.species_a) | set(part.species_b))
        if len(members) < MIN_SPECIES:
            continue
        for field in ('age_mya', 'one_to_one_coverage_jaccard'):
            lookup = {tuple(sorted((r.species_a, r.species_b))): getattr(r, field)
                      for r in part.itertuples()}
            observed = float(spearmanr(part.similarity, part[field]).statistic)
            extreme = 0
            for _ in range(permutations):
                shuffled = dict(zip(members, rng.permutation(members)))
                y = [lookup[tuple(sorted((shuffled[a], shuffled[b])))]
                     for a, b in zip(part.species_a, part.species_b)]
                rho = spearmanr(part.similarity, y).statistic
                extreme += np.isfinite(rho) and abs(rho) >= abs(observed) - 1e-12
            rows.append({'celltype': ct, 'reference': field, 'rho': observed,
                         'qap_p_two_sided': (extreme + 1) / (permutations + 1),
                         'n_species': len(members), 'n_pairs': len(part),
                         'permutations': permutations})
    return pd.DataFrame(rows)


def main():
    config = load_config(ROOT / 'config/inputs.yaml')
    names = SpeciesNames.from_csv(ROOT / 'config/species_names.csv')
    species, labels, embedding = load_embedding(config['xmot'])
    counts_raw = pd.crosstab(species, labels)
    eligible = sorted(ct for ct in counts_raw if ct not in {'Others', 'unknown', ''}
                      and (counts_raw[ct] >= MIN_CELLS).sum() >= MIN_SPECIES)
    centers, counts = build_centroids(species, labels, embedding, eligible)
    out = ROOT / 'results/conservation_landscape'
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / 'tables/celltype_species_eligibility.csv', index=False)
    pairs = make_pairs(centers, counts, eligible, names.order)
    refs = make_references(names.order, config)
    refs.to_csv(out / 'tables/species_pair_references.csv', index=False)
    pairs = pairs.merge(refs, on=['species_a', 'species_b'], validate='many_to_one')
    pairs.to_csv(out / 'tables/celltype_species_pair_similarity.csv', index=False)
    stats = validation_stats(pairs, eligible)
    stats.to_csv(out / 'tables/phylogeny_and_coverage_tests.csv', index=False)
    all_order = draw_selected(counts, pairs, eligible, out / 'overview/clustered_all_celltypes',
                              names, 4, (20, 16), 'Cell-type-specific X_HENOSIS similarity')
    all_order.to_csv(out / 'tables/clustered_species_order.csv', index=False)
    for ct in eligible:
        slug = ct.lower().replace(' ', '_').replace('/', '_')
        draw_selected(counts, pairs, [ct], out / 'individual_celltypes' / slug,
                      names, 1, (8, 8.5), f'{ct} | cross-species X_HENOSIS similarity')
    draw_selected(counts, pairs, MAIN_CELLTYPES,
                  out / 'cardiomyocyte_endothelial/cardiomyocyte_endothelial',
                  names, 2, (15, 8), 'Cardiomyocyte and endothelial representation similarity')
    other = [ct for ct in eligible if ct not in MAIN_CELLTYPES]
    draw_selected(counts, pairs, other,
                  out / 'other_celltypes/other_celltypes_supplementary',
                  names, 3, (17, 17), 'Other cardiac cell types | X_HENOSIS similarity')
    print(f'Conservation: {len(eligible)} cell types, {len(pairs)} pair similarities', flush=True)


if __name__ == '__main__':
    main()

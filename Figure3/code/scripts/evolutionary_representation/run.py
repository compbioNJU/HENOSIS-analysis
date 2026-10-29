#!/usr/bin/env python3
"""Three separately reproducible representation panels: all-cell, pair contrast, spectrum."""

from itertools import combinations
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from common.io import load_config, write_plot_csv, results_root
from common.names import SpeciesNames
from conservation.run import build_centroids, cosine, VERTEBRATES
from common.embedding import load_embedding

matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
GROUPS = ['vertebrate–vertebrate', 'vertebrate–nonvertebrate',
          'nonvertebrate–nonvertebrate']
COLORS = dict(zip(GROUPS, ['#236c89', '#a56642', '#8a739a']))


def pair_group(a, b):
    return GROUPS[2 - int(a in VERTEBRATES) - int(b in VERTEBRATES)]


def aggregate_pairs(celltype_pairs):
    result = celltype_pairs.groupby(['species_a', 'species_b'], as_index=False).agg(
        similarity=('similarity', 'mean'), n_shared_celltypes=('celltype', 'nunique'))
    result['pair_group'] = [pair_group(a, b) for a, b in
                            zip(result.species_a, result.species_b)]
    return result


def type_specific_contrast(centers, species, labels):
    rows = []
    for a, b in combinations(species, 2):
        shared = [ct for ct in labels if (a, ct) in centers and (b, ct) in centers]
        if len(shared) < 2:
            continue
        for ct in shared:
            matched = cosine(centers[a, ct], centers[b, ct])
            mismatched = [cosine(centers[a, ct], centers[b, other])
                          for other in shared if other != ct]
            mismatched += [cosine(centers[a, other], centers[b, ct])
                           for other in shared if other != ct]
            rows.append({'celltype': ct, 'species_a': a, 'species_b': b,
                         'matched_similarity': matched,
                         'mismatched_mean': float(np.mean(mismatched)),
                         'contrast': matched - float(np.mean(mismatched)),
                         'n_shared_celltypes': len(shared)})
    return pd.DataFrame(rows)


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)


def draw_all_cell(pair, names, out):
    order = [sp for sp in names.order if sp in VERTEBRATES] + [
        sp for sp in names.order if sp not in VERTEBRATES]
    matrix = pd.DataFrame(np.nan, index=order, columns=order)
    for row in pair.itertuples():
        matrix.loc[row.species_a, row.species_b] = row.similarity
        matrix.loc[row.species_b, row.species_a] = row.similarity
    plot = pd.DataFrame([{'row_species': a, 'column_species': b,
                          'row_position': i + 1, 'column_position': j + 1,
                          'mean_similarity': matrix.loc[a, b],
                          'n_shared_celltypes': int(pair.loc[
                              ((pair.species_a == a) & (pair.species_b == b)) |
                              ((pair.species_a == b) & (pair.species_b == a)),
                              'n_shared_celltypes'].iloc[0]) if i != j else 0}
                         for i, a in enumerate(order) for j, b in enumerate(order)])
    stem = out / 'all_celltype_similarity/all_celltype_similarity'
    write_plot_csv(stem.with_suffix('.csv'), plot,
                   ['row_species', 'column_species', 'mean_similarity'])
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#eeeeee')
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)),
                      cmap=cmap, norm=TwoSlopeNorm(vmin=min(-.5, pair.similarity.min()),
                                                    vcenter=0, vmax=1))
    labels = [names.display(sp, compact=True) for sp in order]
    ax.set_xticks(range(15), labels, rotation=55, ha='right', fontsize=8)
    ax.set_yticks(range(15), labels, fontsize=8)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontstyle('italic')
    ax.axhline(8.5, color='#262626', lw=1.4)
    ax.axvline(8.5, color='#262626', lw=1.4)
    ax.tick_params(length=0)
    ax.set_title('Across all eligible Celltypes', loc='left', fontsize=13)
    fig.colorbar(image, ax=ax, fraction=.035, pad=.025,
                 label='Mean X_HENOSIS centroid cosine similarity')
    fig.tight_layout()
    save(fig, stem)


def draw_pair_contrast(pair, group, out):
    stem = out / 'vertebrate_pair_contrast/vertebrate_pair_contrast'
    write_plot_csv(stem.with_suffix('.csv'), pair,
                   ['species_a', 'species_b', 'pair_group', 'similarity'])
    group.to_csv(stem.parent / 'pair_group_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    rng = np.random.default_rng(42)
    data = [pair.loc[pair.pair_group == label, 'similarity'].to_numpy()
            for label in GROUPS]
    bp = ax.boxplot(data, patch_artist=True, widths=.46, showfliers=False,
                    medianprops={'color': '#202020', 'lw': 1.3})
    for patch, label in zip(bp['boxes'], GROUPS):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(.2)
        patch.set_edgecolor(COLORS[label])
    for i, (label, values) in enumerate(zip(GROUPS, data), 1):
        ax.scatter(i + rng.uniform(-.16, .16, len(values)), values,
                   s=15, alpha=.7, color=COLORS[label], edgecolors='none')
        mean = float(values.mean())
        ax.scatter(i, mean, marker='D', color=COLORS[label], s=58,
                   edgecolor='white', linewidth=.6, zorder=5)
        ax.text(i, min(1.03, max(values.max() + .045, mean + .06)),
                f'{mean:.3f}', ha='center', fontsize=9, color=COLORS[label])
    ax.set_xticks(range(1, 4), ['V–V\n(n=36)', 'V–nonV\n(n=54)',
                                'nonV–nonV\n(n=15)'])
    ax.set_ylabel('Mean similarity across shared cell types')
    ax.set_ylim(min(-.6, pair.similarity.min() - .05), 1.08)
    ax.set_title('Vertebrate and nonvertebrate pairs', loc='left', fontsize=13)
    ax.grid(axis='y', color='#e6e6e6', lw=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, stem)


def draw_spectrum(spectrum, contrast, out):
    stem = out / 'celltype_conservation_spectrum/celltype_conservation_spectrum'
    palette_table = pd.read_csv(ROOT / 'config/celltype_colors.csv')
    if palette_table.label.duplicated().any():
        raise ValueError('Duplicate Celltype colors')
    palette = palette_table.set_index('label').color
    missing = set(spectrum.celltype) - set(palette.index)
    if missing:
        raise ValueError(f'Missing Celltype colors: {sorted(missing)}')
    spectrum = spectrum.copy()
    spectrum['color'] = spectrum.celltype.map(palette)
    write_plot_csv(stem.with_suffix('.csv'), spectrum,
                   ['celltype', 'mean_contrast', 'n_species', 'color'])
    contrast.to_csv(stem.parent / 'celltype_specificity_pairs.csv', index=False)
    ordered = spectrum.sort_values('mean_contrast').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    ax.barh(range(len(ordered)), ordered.mean_contrast,
            color=ordered.color.tolist(), height=.68)
    ax.set_yticks(range(len(ordered)),
                  [f'{r.celltype}  ({int(r.n_species)} sp)'
                   for r in ordered.itertuples()], fontsize=9)
    ax.axvline(0, color='#333333', lw=.8)
    ax.set_xlabel('Same-type − mismatched-type cosine similarity')
    ax.set_title('Cell-type conservation / specificity spectrum', loc='left', fontsize=13)
    ax.grid(axis='x', color='#e6e6e6', lw=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, stem)


def main():
    names = SpeciesNames.from_csv(ROOT / 'config/species_names.csv')
    config = load_config(ROOT / 'config/inputs.yaml')
    landscape = results_root(ROOT) / 'conservation_landscape/tables'
    celltype_pairs = pd.read_csv(landscape / 'celltype_species_pair_similarity.csv')
    eligibility = pd.read_csv(landscape / 'celltype_species_eligibility.csv')
    pair = aggregate_pairs(celltype_pairs)
    if len(pair) != 105 or pair.n_shared_celltypes.lt(2).any():
        raise ValueError('Expected 105 species pairs each sharing at least two cell types')
    group = pair.groupby('pair_group', as_index=False).agg(
        mean_similarity=('similarity', 'mean'),
        median_similarity=('similarity', 'median'),
        sd_similarity=('similarity', 'std'),
        n_species_pairs=('similarity', 'size'),
        min_shared_celltypes=('n_shared_celltypes', 'min'),
        max_shared_celltypes=('n_shared_celltypes', 'max'))
    species, labels, embedding = load_embedding(config['xmot'])
    valid = sorted(eligibility.celltype.unique())
    centers, _ = build_centroids(species, labels, embedding, valid)
    contrast = type_specific_contrast(centers, names.order, valid)
    spectrum = contrast.groupby('celltype', as_index=False).agg(
        mean_contrast=('contrast', 'mean'), median_contrast=('contrast', 'median'),
        q25_contrast=('contrast', lambda x: x.quantile(.25)),
        q75_contrast=('contrast', lambda x: x.quantile(.75)),
        n_species_pairs=('contrast', 'size'))
    spectrum['n_species'] = spectrum.celltype.map(
        eligibility[eligibility.eligible].groupby('celltype').size())
    out = results_root(ROOT) / 'evolutionary_representation'
    draw_all_cell(pair, names, out)
    draw_pair_contrast(pair, group, out)
    draw_spectrum(spectrum, contrast, out)
    print(group.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

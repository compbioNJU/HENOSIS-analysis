#!/usr/bin/env python3
"""Render the unsupervised full-data 15-species heart UMAPs."""

import argparse
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))
from common.io import load_config, write_plot_csv, results_root  # noqa: E402
from common.names import SpeciesNames  # noqa: E402


def read_category(obs, key):
    group = obs[key]
    if isinstance(group, h5py.Group):
        categories = group['categories'].asstr()[:]
        codes = group['codes'][:]
        values = np.full(len(codes), '', dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values.astype(str)
    return group.asstr()[:]


def load_umap_data(path):
    with h5py.File(path) as h:
        if 'obsm/X_umap' not in h or 'obs/Celltype' not in h:
            raise ValueError(f'{path}: missing X_umap or Celltype')
        species_key = 'species' if 'species' in h['obs'] else 'Species'
        if species_key not in h['obs']:
            raise ValueError(f'{path}: missing species')
        coords = np.asarray(h['obsm/X_umap'], dtype=np.float32)
        major = read_category(h['obs'], 'Celltype')
        species = read_category(h['obs'], species_key)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError(f'{path}: invalid UMAP coordinates')
    if len(coords) != len(major) or len(coords) != len(species):
        raise ValueError(f'{path}: UMAP/metadata row mismatch')
    return coords, major, species


def palette(path):
    frame = pd.read_csv(path, dtype=str)
    if set(frame) != {'label', 'color'} or frame.label.duplicated().any():
        raise ValueError(f'Invalid palette: {path}')
    return dict(zip(frame.label, frame.color))


def figure_data(coords, labels, names, color_by, colors):
    missing = sorted(set(labels) - set(colors))
    if missing:
        raise ValueError(f'Missing {color_by} palette labels: {missing}')
    frame = pd.DataFrame({'umap_1': coords[:, 0], 'umap_2': coords[:, 1]})
    if color_by == 'Species':
        frame['species_id'] = labels
        frame['display_name'] = [names.display(value) for value in labels]
    else:
        frame['celltype'] = labels
        frame['display_name'] = labels
    frame['color'] = [colors[value] for value in labels]
    return frame


def save_view(frame, colors, path, kind):
    fig, ax = plt.subplots(figsize=(6, 6))
    size = max(.15, min(8., 120000. / len(frame)))
    ax.scatter(frame.umap_1, frame.umap_2, c=frame.color.to_numpy(),
               s=size, alpha=.9, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for suffix in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + suffix), dpi=300, transparent=True)
    plt.close(fig)
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=6,
                      markerfacecolor=color, markeredgecolor='none', label=label)
               for label, color in colors.items()]
    legend, lax = plt.subplots(figsize=(5.5, max(2., .3 * len(handles) + 1)))
    lax.axis('off')
    lax.legend(handles=handles, loc='center', frameon=False, title=kind,
               fontsize=8, prop={'style': 'italic' if kind == 'Species' else 'normal'})
    legend.savefig(path.with_name(path.stem + '_legend.pdf'),
                   bbox_inches='tight', transparent=True)
    plt.close(legend)


def render_umap(input_h5ad, output_dir, names, palettes, condition):
    coords, major, species = load_umap_data(input_h5ad)
    if len(coords) != 430338:
        raise ValueError(f'{condition}: expected full 430338 cells, found {len(coords)}')
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for kind, labels, colors in [('Celltype', major, palettes[0]),
                                 ('Species', species, palettes[1])]:
        frame = figure_data(coords, labels, names, kind, colors)
        stem = output_dir / f'umap_{condition}_{kind}'
        write_plot_csv(stem.with_suffix('.csv'), frame, ['umap_1', 'umap_2', 'color'])
        present = {names.display(key) if kind == 'Species' else key: colors[key]
                   for key in sorted(set(labels))}
        legend_rows = pd.DataFrame([{'display_name': label, 'color': color}
                                    for label, color in present.items()])
        write_plot_csv(stem.with_name(stem.stem + '_legend.csv'), legend_rows,
                       ['display_name', 'color'])
        save_view(frame, present, stem, kind)
        outputs.append(stem)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('15S',), default='15S')
    args = parser.parse_args()
    config = load_config(ROOT / 'config/inputs.yaml')
    names = SpeciesNames.from_csv(ROOT / 'config/species_names.csv')
    palettes = (palette(ROOT / 'config/celltype_colors.csv'),
                palette(ROOT / 'config/species_colors.csv'))
    conditions = (args.condition,)
    for condition in conditions:
        key = 'umap_15s'
        outputs = render_umap(config[key], results_root(ROOT) / 'umap' / condition,
                              names, palettes, condition)
        print(f'{condition}: {len(outputs)} full-data views', flush=True)


if __name__ == '__main__':
    main()

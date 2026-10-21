#!/usr/bin/env python3
"""Combine the existing five-gene expression examples for all heart species."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from common.names import SpeciesNames

matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                            'font.family': 'DejaVu Sans'})

CELLTYPES = {'Cardiomyocyte': 'cardiomyocyte',
             'Endothelial cell': 'endothelial_cell'}
TARGET_COLOR = '#226a9b'
OTHER_COLOR = '#969696'


def load_selected(source_dir: Path, species_order: list[str], celltype: str) -> pd.DataFrame:
    required = {'species', 'celltype', 'gene', 'plot_order',
                'detection_in', 'detection_out'}
    parts = []
    for species in species_order:
        path = source_dir / f'{species}.csv'
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        selected = frame.loc[frame.celltype.eq(celltype)].copy()
        if len(selected) != 5 or selected.gene.duplicated().any():
            raise ValueError(f'{path}: expected five distinct {celltype} genes')
        if not selected.species.eq(species).all():
            raise ValueError(f'{path}: species column does not match file name')
        if sorted(selected.plot_order.tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError(f'{path}: expected plot_order 1–5')
        rates = selected[['detection_in', 'detection_out']].to_numpy(dtype=float)
        if not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
            raise ValueError(f'{path}: detection fractions must be within [0, 1]')
        parts.append(selected.sort_values('plot_order', kind='mergesort'))
    return pd.concat(parts, ignore_index=True)


def make_figure(frame: pd.DataFrame, names: SpeciesNames, celltype: str) -> plt.Figure:
    species_order = names.order
    n_rows = (len(species_order) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14.5, 2.0 * n_rows + 1.1),
                             sharex=True)
    axes = np.asarray(axes).reshape(n_rows, 2)
    for i, species in enumerate(species_order):
        ax = axes[i % n_rows, i // n_rows]
        part = frame.loc[frame.species.eq(species)].sort_values('plot_order')
        y = np.arange(len(part))
        ax.hlines(y, part.detection_out, part.detection_in,
                  color='#b7b7b7', linewidth=1.25, zorder=1)
        ax.scatter(part.detection_out, y, color=OTHER_COLOR, s=35,
                   marker='o', zorder=2)
        ax.scatter(part.detection_in, y, color=TARGET_COLOR, s=48,
                   marker='D', zorder=3)
        ax.set_yticks(y, part.gene.tolist(), fontsize=8.5)
        ax.set_ylim(4.5, -.5)
        ax.set_xlim(-.02, 1.02)
        ax.set_title(names.display(species), loc='left', fontstyle='italic',
                     fontsize=10, pad=7)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#e8e8e8', linewidth=.6)
        ax.set_axisbelow(True)
    for i in range(len(species_order), n_rows * 2):
        fig.delaxes(axes[i % n_rows, i // n_rows])
    for ax in (axes[-1, 0], axes[-2, 1] if len(species_order) % 2 else axes[-1, 1]):
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xticks(np.linspace(0, 1, 5), ['0', '25', '50', '75', '100'])
        ax.set_xlabel('Cells with nonzero raw counts (%)', fontsize=9)
    handles = [
        Line2D([0], [0], marker='D', linestyle='none', color=TARGET_COLOR,
               markersize=6, label=celltype),
        Line2D([0], [0], marker='o', linestyle='none', color=OTHER_COLOR,
               markersize=6, label='Other major cell types'),
    ]
    fig.suptitle(celltype, fontsize=13, y=.988)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .972),
               ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.13, right=.98, top=.925, bottom=.055,
                        wspace=.35, hspace=.73)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path,
                        default=ROOT / 'results/expression_examples/species')
    parser.add_argument('--out-dir', type=Path,
                        default=ROOT / 'results/expression_examples/all_species_by_celltype')
    args = parser.parse_args()
    names = SpeciesNames.from_csv(ROOT / 'config/species_names.csv')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for celltype, filename in CELLTYPES.items():
        frame = load_selected(args.source_dir, names.order, celltype)
        fig = make_figure(frame, names, celltype)
        stem = args.out_dir / filename
        for suffix in ('.png', '.pdf'):
            fig.savefig(stem.with_suffix(suffix), dpi=300,
                        bbox_inches='tight', facecolor='white')
        plt.close(fig)
        frame.to_csv(stem.with_suffix('.csv'), index=False)
        print(f'Saved {stem}.{{png,pdf,csv}} ({len(frame)} genes)')


if __name__ == '__main__':
    main()

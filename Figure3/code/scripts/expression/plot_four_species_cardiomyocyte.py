#!/usr/bin/env python3
"""Plot existing ranked-gene cardiomyocyte expression examples for four species."""

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
from common.io import results_root
from common.names import SpeciesNames

matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                            'font.family': 'DejaVu Sans'})

SPECIES = (
    'Homo_sapiens',
    'Danio_rerio',
    'Drosophila_melanogaster',
    'Mizuhopecten_yessoensis',
)
TARGET_COLOR = '#226a9b'
OTHER_COLOR = '#969696'


def load_selected(source_dir: Path) -> pd.DataFrame:
    parts = []
    required = {'species', 'celltype', 'gene', 'plot_order',
                'detection_in', 'detection_out'}
    for species in SPECIES:
        path = source_dir / f'{species}.csv'
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        selected = frame.loc[frame.celltype.eq('Cardiomyocyte')].copy()
        if len(selected) != 5 or selected.gene.duplicated().any():
            raise ValueError(f'{path}: expected five distinct cardiomyocyte genes')
        if not selected.species.eq(species).all():
            raise ValueError(f'{path}: species column does not match file name')
        if sorted(selected.plot_order.tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError(f'{path}: expected plot_order 1–5')
        rates = selected[['detection_in', 'detection_out']].to_numpy(dtype=float)
        if not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
            raise ValueError(f'{path}: detection fractions must be within [0, 1]')
        parts.append(selected.sort_values('plot_order', kind='mergesort'))
    return pd.concat(parts, ignore_index=True)


def draw(frame: pd.DataFrame, names: SpeciesNames, stem: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 10.0), sharex=True)
    for ax, species in zip(axes, SPECIES):
        part = frame.loc[frame.species.eq(species)].sort_values('plot_order')
        y = np.arange(len(part))
        ax.hlines(y, part.detection_out, part.detection_in,
                  color='#b7b7b7', linewidth=1.25, zorder=1)
        ax.scatter(part.detection_out, y, color=OTHER_COLOR, s=38,
                   marker='o', zorder=2)
        ax.scatter(part.detection_in, y, color=TARGET_COLOR, s=53,
                   marker='D', zorder=3)
        ax.set_yticks(y, part.gene.tolist(), fontsize=9)
        ax.set_ylim(4.5, -.5)
        ax.set_xlim(-.02, 1.02)
        ax.set_title(names.display(species), loc='left', fontstyle='italic',
                     fontsize=11, pad=8)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#e8e8e8', linewidth=.6)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(np.linspace(0, 1, 5), ['0', '25', '50', '75', '100'])
    axes[-1].set_xlabel('Cells with nonzero raw counts (%)', fontsize=10)
    handles = [
        Line2D([0], [0], marker='D', linestyle='none', color=TARGET_COLOR,
               markersize=7, label='Cardiomyocytes'),
        Line2D([0], [0], marker='o', linestyle='none', color=OTHER_COLOR,
               markersize=7, label='Other major cell types'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .965),
               ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.21, right=.98, top=.91, bottom=.08, hspace=.42)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path,
                        default=results_root(ROOT) / 'expression_examples/species')
    parser.add_argument('--out-dir', type=Path,
                        default=results_root(ROOT) / 'expression_examples/four_species_cardiomyocyte')
    args = parser.parse_args()
    frame = load_selected(args.source_dir)
    names = SpeciesNames.from_csv(ROOT / 'config/species_names.csv')
    stem = args.out_dir / 'four_species_cardiomyocyte'
    draw(frame, names, stem)
    frame.to_csv(stem.with_suffix('.csv'), index=False)
    print(f'Saved {stem}.{{png,pdf,csv}} ({len(frame)} cardiomyocyte genes)')


if __name__ == '__main__':
    main()

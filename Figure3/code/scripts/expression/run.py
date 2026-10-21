#!/usr/bin/env python3
"""Recompute adaptive ranked-gene expression examples from primary ranked/evidence files."""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from common.io import load_config, write_plot_csv
from common.names import SpeciesNames

matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                            'font.family': 'DejaVu Sans'})
CELLTYPES = ('Cardiomyocyte', 'Endothelial cell')
RANKED_FOLDERS = {'Cardiomyocyte': 'Cardiomyocyte',
                  'Endothelial cell': 'Endothelial_cell'}


def canonical_key(name):
    text = re.sub(r'^gene-', '', str(name).strip(), flags=re.I)
    return text.split(':', 1)[1].lower() if ':' in text else text.lower()


def top_n(df, n, seed=20260928):
    if len(df) <= n:
        return df.gene.tolist()
    score = df.score.to_numpy()
    if score[n - 1] != score[n]:
        return df.gene.iloc[:n].tolist()
    cut = score[n - 1]
    above = int((score > cut).sum())
    tied = np.flatnonzero(score == cut)
    chosen = np.random.default_rng(seed).choice(tied, n - above, replace=False)
    return df.gene.iloc[:above].tolist() + df.gene.iloc[np.sort(chosen)].tolist()


def select_panel(ranked, expression, target=5):
    """Search top20/30/40/50 at 20%; only then relax to 10%."""
    top_sets = {n: top_n(ranked, n) for n in (20, 30, 40, 50)}
    relevant = {gene for names in top_sets.values() for gene in names}
    audit = ranked.loc[ranked.gene.isin(relevant), ['gene', 'score', 'rank']].copy()
    audit['key'] = audit.gene.map(canonical_key)
    evidence = expression.copy()
    evidence['key'] = evidence.gene.map(canonical_key)
    evidence = evidence.drop_duplicates('key', keep='first')
    cols = ['key', 'detection_in', 'detection_out', 'deg_logfc_full',
            'deg_padj_full', 'n_cells_in']
    audit = audit.merge(evidence[cols], on='key', how='left', validate='one_to_one')
    audit['detection_delta'] = audit.detection_in - audit.detection_out
    counts = []
    for threshold in (.20, .10):
        for n, genes in top_sets.items():
            part = audit[audit.key.isin({canonical_key(g) for g in genes})]
            if len(part) != len(genes) or part.detection_in.isna().any():
                raise ValueError('Ranked topN contains genes without expression evidence')
            supported = part[(part.detection_delta >= threshold - 1e-12) &
                             (part.deg_logfc_full >= 1) &
                             (part.deg_padj_full < .05)]
            counts.append({'threshold': threshold, 'topN': n,
                           'n_supported': len(supported)})
            if len(supported) >= target or (threshold == .10 and n == 50):
                selected = supported.sort_values(['detection_delta', 'rank'],
                                                 ascending=[False, True],
                                                 kind='mergesort').head(target).copy()
                selected['plot_order'] = range(1, len(selected) + 1)
                return {'threshold': threshold, 'top_n': n,
                        'n_supported': len(supported), 'selected': selected,
                        'audit': audit, 'search_counts': pd.DataFrame(counts)}
    raise AssertionError('Unreachable adaptive search')


def draw_panel(ax, result, celltype, compact=False):
    selected = result['selected']
    ax.set_title(f'{celltype} · {result["n_supported"]}/{result["top_n"]} · '
                 f'Δ≥{result["threshold"]:.0%}', loc='left', fontsize=8 if compact else 10)
    ax.set_xlim(-.02, 1.28)
    ax.set_ylim(4.5, -.55)
    ax.set_xticks(np.arange(0, 1.01, .2), [f'{n}%' for n in range(0, 101, 20)])
    ax.set_yticks(range(len(selected)), selected.gene.tolist(), fontsize=7 if compact else 9)
    if selected.empty:
        ax.text(.5, .52, 'No supported ranked gene', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)
    else:
        y = np.arange(len(selected))
        ax.hlines(y, selected.detection_out, selected.detection_in,
                  color='#a5a5a5', lw=1.2)
        ax.scatter(selected.detection_out, y, color='#969696', s=23, marker='o', zorder=3)
        ax.scatter(selected.detection_in, y, color='#226a9b', s=30, marker='D', zorder=3)
        for pos, logfc in zip(y, selected.deg_logfc_full):
            ax.text(1.03, pos, f'log2FC {logfc:.1f}', va='center', fontsize=6.5 if compact else 8)
    ax.set_xlabel('Cells with nonzero raw counts', fontsize=7 if compact else 9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#e8e8e8', lw=.6)
    ax.set_axisbelow(True)


def save_figure(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def render_individual(species, panels, names, out):
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.5), sharex=True)
    for ax, ct in zip(axs, CELLTYPES):
        draw_panel(ax, panels[ct], ct)
    fig.suptitle(names.display(species) + ' | ranked-gene expression support',
                 x=.04, y=.98, ha='left', fontsize=12, fontstyle='italic')
    fig.text(.5, .09, 'Δ = target detection − other Celltypes detection; log2FC ≥1, adjusted p <0.05.',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=.09, right=.98, top=.78, bottom=.23, wspace=.27)
    save_figure(fig, out / 'species' / species)


def render_supplement(panels, names, out):
    fig, axs = plt.subplots(len(names.order), 2, figsize=(13, 39), squeeze=False)
    for row, species in enumerate(names.order):
        for col, ct in enumerate(CELLTYPES):
            draw_panel(axs[row, col], panels[species][ct], ct, compact=True)
            if col == 0:
                axs[row, col].text(-.58, .5, names.display(species, compact=True),
                                   transform=axs[row, col].transAxes, ha='right',
                                   va='center', fontsize=9, fontstyle='italic')
    fig.suptitle('Ranked-gene expression support across 15 species', fontsize=16, y=.997)
    fig.text(.5, .014, 'Adaptive 20% then 10% detection contrast; top20→50; log2FC ≥1; adjusted p <0.05.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.20, right=.98, top=.982, bottom=.035, hspace=.9, wspace=.30)
    save_figure(fig, out / 'all_15_species_supplementary')


def main():
    config = load_config(ROOT / 'config' / 'inputs.yaml')
    names = SpeciesNames.from_csv(ROOT / 'config' / 'species_names.csv')
    evidence = pd.read_parquet(config['expression_evidence'])
    out = ROOT / 'results' / 'expression_examples'
    all_panels, selections, summary, traces = {}, [], [], []
    for species in names.order:
        panels = {}
        for celltype in CELLTYPES:
            folder = RANKED_FOLDERS[celltype]
            path = config['ranked_root'] / folder / f'heart_{folder}_{species}.csv'
            ranked = pd.read_csv(path)
            ranked['gene'] = ranked.gene.astype(str).str.strip()
            ranked = ranked.drop_duplicates('gene').reset_index(drop=True)
            ranked['rank'] = np.arange(1, len(ranked) + 1)
            part = evidence[(evidence.species == species) & (evidence.celltype == celltype)]
            result = select_panel(ranked, part)
            panels[celltype] = result
            selected = result['selected'].assign(species=species, celltype=celltype,
                                                 threshold=result['threshold'], top_n=result['top_n'])
            selections.append(selected)
            summary.append({'species': species, 'celltype': celltype,
                            'threshold': result['threshold'], 'top_n': result['top_n'],
                            'n_supported': result['n_supported'], 'n_plotted': len(selected)})
            traces.append(result['search_counts'].assign(species=species, celltype=celltype))
        all_panels[species] = panels
        render_individual(species, panels, names, out)
        pair = pd.concat([panels[ct]['selected'].assign(species=species, celltype=ct,
                         threshold=panels[ct]['threshold'], top_n=panels[ct]['top_n'])
                         for ct in CELLTYPES], ignore_index=True)
        write_plot_csv(out / 'species' / f'{species}.csv', pair,
                       ['species', 'celltype', 'gene', 'detection_in', 'detection_out'])
        print(species, [(ct, panels[ct]['threshold'], panels[ct]['top_n'],
                         panels[ct]['n_supported']) for ct in CELLTYPES], flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / 'panel_summary.csv', index=False)
    pd.concat(selections, ignore_index=True).to_csv(out / 'selected_genes.csv', index=False)
    pd.concat(traces, ignore_index=True).to_csv(out / 'search_trace.csv', index=False)
    render_supplement(all_panels, names, out)
    write_plot_csv(out / 'all_15_species_supplementary.csv',
                   pd.concat(selections, ignore_index=True),
                   ['species', 'celltype', 'gene', 'detection_in', 'detection_out'])


if __name__ == '__main__':
    main()

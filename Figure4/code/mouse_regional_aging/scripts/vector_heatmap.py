"""Editable, explicit per-cell RGB rectangles for Illustrator-compatible heatmaps."""
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def draw_vector_heatmap(ax, values):
    mappable = ScalarMappable(norm=Normalize(0, 40), cmap='YlOrRd')
    for (row, col), value in np.ndenumerate(values):
        color = mappable.to_rgba(float(value))
        patch = Rectangle((col-.5, row-.5), 1, 1,
                          facecolor=color, edgecolor='none', linewidth=0,
                          rasterized=False, clip_on=False)
        patch.set_gid(f'heatmap_cell_{row}_{col}')
        ax.add_patch(patch)
    ax.set_xlim(-.5, values.shape[1]-.5)
    ax.set_ylim(values.shape[0]-.5, -.5)
    ax.set_aspect('auto')
    return mappable

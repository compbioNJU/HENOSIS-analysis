"""Figure defaults, and the rule that every PDF keeps its text editable.

The requirement
---------------
**Every label, tick and title in every PDF must remain selectable and editable in
Illustrator / Inkscape / Acrobat.** A PDF whose text has been converted to
outlines or baked into a bitmap cannot be relabelled or translated without
redrawing it, which defeats the point of shipping a vector figure.

How that is enforced here, rather than hoped for
------------------------------------------------
Three things would break it, and each is handled:

  1. **Non-embedded / Type-3 fonts.** `pdf.fonttype = 42` and `ps.fonttype = 42`
     write TrueType with a Unicode cmap, which Illustrator can edit. The
     matplotlib default (`pdf.fonttype = 3`) writes Type-3 fonts that cannot be
     edited.
  2. **Rasterising an artist that has text in it.** `rasterized=True` is applied
     only to scatter clouds (hundreds of thousands of points, where vector output
     would be hundreds of megabytes). It is never applied to an `Axes`, a legend
     or a colourbar, because that would take the text with it.
  3. **Saving a figure as a screenshot.** Figures are always saved through
     `save_figure`, which writes the PDF from the vector canvas, and never by
     embedding a PNG in a PDF.

`verify_pdf_text` re-opens the written file and extracts its text, so a figure
that lost its text is caught at write time instead of at submission time. Every
analysis in this repository calls it on every PDF it produces.

Deliberately not used
---------------------
`bbox_inches="tight"` is avoided on PDFs: it can crop through a label and, more
importantly, silently changes the figure size between the PNG and the PDF, so a
size quoted in a caption stops being true for one of them. Layouts are sized
explicitly instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------

FONT_STACK: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")

DPI_PNG: int = 300


def apply_style(dpi: int = DPI_PNG) -> None:
    """Idempotent global style. Call once, at the top of every script's main()."""
    matplotlib.rcParams.update({
        # -- editable text in vector output --------------------------------
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # -- typography ----------------------------------------------------
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_STACK),
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "legend.frameon": False,
        # -- marks ---------------------------------------------------------
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": dpi,
        "savefig.bbox": None,          # never crop; see the module docstring
        "figure.autolayout": False,
    })


def clean_axis(ax, *, keep_frame: bool = False) -> None:
    """A UMAP-style axis: no ticks, no labels, optionally no spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if not keep_frame:
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")


def scatter_points(ax, xy: np.ndarray, *, color, size: float, alpha: float = 0.9,
                   zorder: int = 2, **kwargs) -> None:
    """A rasterised point cloud.

    Rasterising is confined to the marks. Whatever text the axes carries is added
    by the caller as separate artists and stays vector.
    """
    ax.scatter(
        xy[:, 0], xy[:, 1], c=color, s=size, alpha=alpha, linewidths=0,
        rasterized=True, zorder=zorder, **kwargs,
    )


def render_point_layer(
    xy: np.ndarray,
    colors: np.ndarray | str,
    *,
    size: float,
    alpha: float = 0.9,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    background_xy: np.ndarray | None = None,
    background_color: str = "#D5D5D5",
    background_size: float = 0.15,
    background_alpha: float = 0.25,
    inches: float = 6.0,
    dpi: int = 300,
) -> np.ndarray:
    """Draw a point cloud once, off-screen, and return it as an RGBA array.

    The point cloud is the one part of a UMAP panel that cannot be vector
    without producing an unusable file. Rendering it once into a bitmap and
    pasting that bitmap into a vector axes -- instead of rasterising the whole
    axes -- is what lets the trajectories, the centroid markers and the stage
    labels on top of it stay editable text in the PDF.

    The returned array is drawn with `origin="upper"` and the extent
    `(*xlim, *ylim)`, so `paste_point_layer` is the only correct way to place it.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as _Figure

    fig = _Figure(figsize=(inches, inches), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    if background_xy is not None and len(background_xy):
        ax.scatter(background_xy[:, 0], background_xy[:, 1], c=background_color,
                   s=background_size, alpha=background_alpha, linewidths=0)
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=size, alpha=alpha, linewidths=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    return buffer


def paste_point_layer(ax, layer: np.ndarray, xlim: tuple[float, float],
                      ylim: tuple[float, float], *, zorder: int = 0) -> None:
    """Place a layer from `render_point_layer` so its coordinates are 1:1.

    `aspect="auto"` is required: the layer is square in pixels while the data
    limits are not, and letting the aspect adjust would rescale the cloud
    relative to the annotations drawn on top of it.
    """
    ax.imshow(layer, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
              origin="upper", aspect="auto", zorder=zorder)


# seaborn registers its own colormaps ('mako', 'rocket', 'flare', 'crest', ...)
# only once it has been imported, and this repository otherwise has no reason to
# import it. Resolving through here means a cost heatmap does not depend on
# whether some earlier module happened to import seaborn.
_SEABORN_CMAPS = {
    "mako", "mako_r", "rocket", "rocket_r", "flare", "flare_r",
    "crest", "crest_r", "icefire", "icefire_r", "vlag", "vlag_r",
}


def get_cmap(name: str, fallback: str = "viridis"):
    """A colormap by name, from matplotlib or seaborn, with a stated fallback.

    Falling back is not silent: a missing colormap changes how every panel reads,
    so the substitution is warned about rather than made quietly.
    """
    import warnings

    try:
        return plt.get_cmap(name)
    except ValueError:
        pass
    if name in _SEABORN_CMAPS:
        try:
            import seaborn as sns

            return sns.color_palette(name, as_cmap=True)
        except Exception:  # pragma: no cover - seaborn absent
            pass
    warnings.warn(
        f"colormap {name!r} is not available; using {fallback!r} instead. "
        f"Install seaborn for the seaborn palettes, or set a matplotlib name.",
        stacklevel=2,
    )
    return plt.get_cmap(fallback)


def spread_labels(values: Sequence[float], gap: float, low: float,
                  high: float) -> np.ndarray:
    """Push overlapping label positions apart while keeping their order.

    Labels for stages that sit almost on top of each other in the embedding are
    unreadable if drawn at their centroid. This spaces them by at least `gap`,
    clamps them into `[low, high]`, and preserves the vertical order so a leader
    line never crosses another label. Returns a copy; a single value is returned
    unchanged.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return values.copy()
    order = np.argsort(values, kind="stable")
    sorted_values = values[order].copy()
    for i in range(1, len(sorted_values)):
        if sorted_values[i] < sorted_values[i - 1] + gap:
            sorted_values[i] = sorted_values[i - 1] + gap
    if sorted_values[-1] > high:
        sorted_values -= sorted_values[-1] - high
    for i in range(len(sorted_values) - 2, -1, -1):
        if sorted_values[i] > sorted_values[i + 1] - gap:
            sorted_values[i] = sorted_values[i + 1] - gap
    if sorted_values[0] < low:
        sorted_values += low - sorted_values[0]
    out = np.empty_like(sorted_values)
    out[order] = sorted_values
    return out


def legend_handles(color_map: Mapping[str, str], *, marker_size: float = 6,
                   extra: Iterable[tuple[str, str]] = ()) -> list[Line2D]:
    """Legend entries built from an explicit colour map.

    The order of `color_map` is the order of the legend -- callers pass it in the
    display order they want, so the figure does not depend on dict ordering.
    """
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=marker_size,
               markerfacecolor=color, markeredgecolor="none", label=label)
        for label, color in color_map.items()
    ]
    handles += [
        Line2D([0], [0], marker="o", linestyle="", markersize=marker_size,
               markerfacecolor=color, markeredgecolor="none", label=label)
        for label, color in extra
    ]
    return handles


def write_legend(path_stem: Path, color_map: Mapping[str, str], title: str,
                 *, width: float = 4.5, extra: Iterable[tuple[str, str]] = (),
                 marker_size: float = 6) -> tuple[Path, Path]:
    """A standalone legend, as its own PDF + PNG, with editable text."""
    handles = legend_handles(color_map, marker_size=marker_size, extra=extra)
    height = max(2.0, 0.28 * len(handles) + 0.8)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", frameon=False, title=title)
    return save_figure(fig, path_stem, pad=False)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_figure(fig: Figure, path_stem: str | Path, *, formats: Sequence[str] = ("pdf", "png"),
                pad: bool = False, dpi: int | None = None,
                transparent: bool = False,
                require_text: bool = True) -> tuple[Path, ...]:
    """Write `path_stem` as PDF (editable text) and PNG (raster).

    The PDF is checked for extractable text before this returns. A figure that
    legitimately carries no text -- a bare point cloud, say -- must pass
    `require_text=False`, so that "no text" is a decision the caller made rather
    than a failure nobody noticed.

    The PDF's creation and modification dates are REMOVED rather than set to some
    fixed value. Matplotlib writes the wall clock into them otherwise, which makes
    two runs of an unchanged analysis produce different bytes -- and "the figures
    are identical between runs" is then a claim nobody can check by hashing. A
    file with no date is honestly undated; a file with a falsified date is worse
    than either.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        target = stem.with_suffix("." + fmt)
        metadata = ({"CreationDate": None, "ModDate": None} if fmt == "pdf" else None)
        fig.savefig(target, format=fmt, dpi=dpi or DPI_PNG,
                    transparent=transparent, metadata=metadata,
                    bbox_inches="tight" if pad else None)
        written.append(target)
        if fmt == "pdf":
            verify_pdf_text(target, min_characters=1 if require_text else 0)
    plt.close(fig)
    return tuple(written)


# PDF font types whose glyphs cannot be edited by a vector editor.
#
# Type3 is what matplotlib's DEFAULT (`pdf.fonttype = 3`) produces. Its glyphs are
# stored as PDF drawing procedures rather than as a real font, so an editor can
# select the text but cannot change it -- retyping a label means redrawing it.
# The characters ARE still extractable, which is why extracting text alone is not
# a sufficient check; the font type has to be inspected too.
UNEDITABLE_FONT_TYPES = frozenset({"Type3", "Type 3"})


def verify_pdf_text(path: str | Path, *, min_characters: int = 1,
                    allow_uneditable_fonts: bool = False) -> int:
    """Assert a PDF's text is both present AND editable. Returns its length.

    Two independent failure modes, and both must be checked:

      1. **Text became an image.** Caught by extracting the text; the count is
         zero. Happens if an `Axes` carrying labels was rasterised.
      2. **Text became Type-3 glyph procedures.** The text still extracts -- so a
         naive check passes -- but no editor can modify it. This is matplotlib's
         default, and it is exactly what the `pdf.fonttype = 42` setting in
         `apply_style()` exists to prevent. Caught by reading the embedded font
         table.

    A figure that is legitimately text-free opts out with `min_characters=0`;
    `allow_uneditable_fonts` exists only so a caller can inspect a file it did not
    write.
    """
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"verify_pdf_text expects a PDF, got {path.name}")
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import pymupdf
    except ImportError:  # pragma: no cover - fallback for older installs
        import fitz as pymupdf

    with pymupdf.open(path) as document:
        text = "".join(page.get_text() for page in document)
        fonts = [font for page in document for font in page.get_fonts()]

    stripped = text.strip()
    if len(stripped) < min_characters:
        raise AssertionError(
            f"{path} contains no extractable text ({len(stripped)} characters). "
            f"Its labels have probably been rasterised or converted to outlines. "
            f"Check that no artist carrying text was saved with rasterized=True."
        )

    if not allow_uneditable_fonts:
        bad = sorted({font[3] for font in fonts
                      if str(font[2]) in UNEDITABLE_FONT_TYPES})
        if bad:
            raise AssertionError(
                f"{path} embeds Type-3 fonts ({', '.join(bad)}). Its text extracts, "
                f"so this is easy to miss, but it CANNOT be edited in Illustrator "
                f"or Inkscape. Set pdf.fonttype = 42 (see "
                f"plotting.apply_style(), or MATPLOTLIBRC for code that must not "
                f"be edited)."
            )
    return len(stripped)


def pdf_text_report(paths: Iterable[str | Path]) -> list[dict]:
    """Present-and-editable check over a whole results tree, for acceptance.

    `editable_text` is True only when the text both exists and is stored in a
    font an editor can modify. The two are reported separately, because "the
    characters are there" and "you can change them" are different properties and
    only the second one is the requirement.
    """
    rows = []
    for path in sorted(Path(p) for p in paths):
        try:
            characters = verify_pdf_text(path, min_characters=0)
            rows.append({"pdf": str(path), "characters": characters,
                         "extractable": characters > 0, "editable_text": True,
                         "fonts": _font_types(path)})
        except AssertionError as error:
            # text is present but in an uneditable font, or absent entirely
            rows.append({"pdf": str(path), "characters": _character_count(path),
                         "extractable": _character_count(path) > 0,
                         "editable_text": False, "fonts": _font_types(path),
                         "error": str(error)[:200]})
        except Exception as error:  # pragma: no cover - reported, not raised
            rows.append({"pdf": str(path), "characters": 0, "extractable": False,
                         "editable_text": False, "fonts": [], "error": str(error)[:200]})
    return rows


def _open_pdf(path: Path):
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        import fitz as pymupdf
    return pymupdf.open(path)


def _font_types(path: Path) -> list[str]:
    try:
        with _open_pdf(path) as document:
            return sorted({str(font[2]) for page in document for font in page.get_fonts()})
    except Exception:  # pragma: no cover
        return []


def _character_count(path: Path) -> int:
    try:
        with _open_pdf(path) as document:
            return len("".join(page.get_text() for page in document).strip())
    except Exception:  # pragma: no cover
        return 0


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

def categorical_colors(labels: Sequence[str], palette: Mapping[str, str],
                       *, fallback_cmap: str = "tab20",
                       name: str = "labels") -> dict[str, str]:
    """Assign colours from an explicit palette, filling gaps from a colormap.

    A label the palette does not know is a warning here rather than a silent
    default, because an unlisted label usually means the vocabulary changed and
    the figure is about to be drawn with an arbitrary colour.
    """
    import warnings

    result: dict[str, str] = {}
    cmap = plt.get_cmap(fallback_cmap)
    missing = [label for label in labels if label not in palette]
    for position, label in enumerate(labels):
        if label in palette:
            result[label] = palette[label]
    for offset, label in enumerate(missing):
        result[label] = matplotlib.colors.to_hex(cmap(offset % cmap.N))
    if missing:
        warnings.warn(
            f"{name}: {len(missing)} label(s) have no colour in the palette and "
            f"fell back to '{fallback_cmap}': {missing}. Add them to the palette "
            f"in src/common/labels.py so the colour is a choice, not a default.",
            stacklevel=2,
        )
    return result



"""
Single source of truth for the paper's figure style.

Every plotting script imports from here, so changing a font, size, or colour
in ONE place and re-running `make_figures.py` restyles every figure at once.

The knobs you are most likely to touch are grouped at the top.
"""

import matplotlib.pyplot as plt


# =============================================================================
# THE KNOBS  (edit these, then re-run make_figures.py)
# =============================================================================

FONT_FAMILY = "serif"     # "serif" matches a Computer/Latin Modern manuscript
BASE_FONTSIZE = 9         # body size; label/tick sizes are relative to this

# Figure widths, in inches. Size each figure to the physical width it will
# occupy in the PDF, and DO NOT let \includegraphics rescale it, or the font
# sizes will differ between figures. Measure your text width in LaTeX with
# \the\linewidth (pt) and divide by 72.27 to get inches.
SINGLE_COL = 3.40         # two-column-journal single-column width
DOUBLE_COL = 7.00         # two-column-journal full width
LINEWIDTH = 6.22          # 449.55322pt / 72.27
# Publication palette (shared by all figures for cross-figure consistency).
C_POS = "#1f4e79"         # C_L > 0 branch  (deep blue)
C_NEG = "#b3541e"         # C_L < 0 branch  (burnt orange)
C_SYM = "0.45"            # symmetric branch (mid grey)


# named presets accepted by resolve_width()
COLUMN_WIDTHS = {
    "single": SINGLE_COL,
    "double": DOUBLE_COL,
    "linewidth": LINEWIDTH,          # full text width, scale = 1
    "0.7linewidth": 0.7 * LINEWIDTH,  # 70% text width, scale = 1
}


def resolve_width(spec):
    """Turn a --width value into inches.

    Accepts a preset name ('single', 'double', 'linewidth', '0.7linewidth')
    or a plain number of inches ('5.79', 4.05). This lets each figure be
    generated at exactly the width it is included at, so text prints at the
    same size in every figure.
    """
    if isinstance(spec, str) and spec in COLUMN_WIDTHS:
        return COLUMN_WIDTHS[spec]
    return float(spec)


# =============================================================================
# STYLE
# =============================================================================

def set_style(usetex=False):
    """Apply the shared rcParams. Call once at the top of each figure script.

    usetex=False -> matplotlib's built-in Computer Modern mathtext (no LaTeX
                    needed, fast; good for drafting).
    usetex=True  -> render all text through a real LaTeX install so figure
                    fonts match the manuscript exactly (slower; needs texlive).
    """
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE + 1,
        "axes.titlesize": BASE_FONTSIZE + 1,
        "xtick.labelsize": BASE_FONTSIZE - 1,
        "ytick.labelsize": BASE_FONTSIZE - 1,
        "legend.fontsize": BASE_FONTSIZE - 1,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed editable (TrueType) text
        "ps.fonttype": 42,
    })
    if usetex:
        plt.rcParams.update({"text.usetex": True})
    else:
        plt.rcParams.update({"mathtext.fontset": "cm"})

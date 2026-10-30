"""The accepted figure style, independent of the original project package."""

import string
import warnings

import matplotlib as mpl
from matplotlib import font_manager
import scienceplots  # noqa: F401 - registers the pinned Nature/no-LaTeX styles

INK = '#222222'
LIGHT_INK = '#9A9A9A'


def apply_figure_style():
    """Reapply the publication settings; never use a silent style fallback."""
    mpl.rcdefaults()
    mpl.style.use(['nature', 'no-latex'])
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7.0, 'text.color': INK,
        'axes.labelcolor': INK, 'axes.titlecolor': INK,
        'axes.titlesize': 7.0, 'axes.titleweight': 'bold',
        'axes.labelsize': 7.0, 'axes.linewidth': 0.45,
        'axes.edgecolor': '#444444', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.axisbelow': True,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
        'xtick.major.width': 0.45, 'ytick.major.width': 0.45,
        'legend.fontsize': 6.5, 'legend.title_fontsize': 6.5,
        'legend.frameon': False, 'lines.linewidth': 0.9,
        'lines.solid_capstyle': 'round', 'patch.linewidth': 0.0,
        'errorbar.capsize': 1.5, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'savefig.transparent': False, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def check_reference_font():
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
    except ValueError:
        warnings.warn('Arial is unavailable. A fallback font will be used; exact layout agreement with the reference is not expected.', stacklevel=2)


def panel_label(ax, index, *, x=-0.10, y=1.07):
    label = string.ascii_lowercase[index] if isinstance(index, int) else str(index)
    ax.text(x, y, label, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8, fontweight='bold', color=INK, clip_on=False)

"""Publication plotting style for the m61-tng disk-ISM / HVC figures.

Typography & axis conventions follow /scratch/tsingh65/OVI_CGM_compare (Computer-Modern
serif, inward ticks on all four sides, thick spines, faint grids, framed legends,
pdf+png at high dpi). Palette is chosen here for the ISM/IVC/HVC + inflow/outflow science.

Use:  import m61_style as S ;  S.set_style() ;  ... ;  S.save(fig, "figname")
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FIGDIR = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/paper_figures")

# ── palette ─────────────────────────────────────────────────────────────────────
# kinematic class of an absorber by |dv| = |v_los - v_ISM|
CLASS = {"ISM": "#4C4C4C", "IVC": "#E19A3C", "HVC": "#B02418"}   # grey / amber / brick-red
CLASS_LABEL = {"ISM": r"ISM ($|\Delta v|<40$)", "IVC": r"IVC ($40\!-\!100$)",
               "HVC": r"HVC ($|\Delta v|>100$)"}
# inflow / outflow (diverging): blue = inflow (v_r<0), red = outflow (v_r>0)
INFLOW = "#2166AC"
OUTFLOW = "#B2182B"
DIVCMAP = "RdBu_r"        # signed maps: v_r, dv
SEQCMAP = "viridis"       # density / counts
DENSCMAP = "magma"        # 2D histograms
# sightline v_ISM provenance
VMODE = {"direct_cool": "#1B9E77", "R95-edge": "#7B6FB0"}
ACCENT = "#0B7285"

CLASS_BINS = [40.0, 100.0]   # |dv| thresholds: ISM | IVC | HVC

# ionization sequence (neutral/cool -> warm-hot), for multi-ion / multi-phase figures.
# color runs cool(black/blue) -> hot(red).
IONS = [("HI",    r"$\mathrm{H\,I}$",    "#222222"),
        ("CII",   r"$\mathrm{C\,II}$",   "#2C7FB8"),
        ("SiII",  r"$\mathrm{Si\,II}$",  "#33A895"),
        ("SiIII", r"$\mathrm{Si\,III}$", "#8C6BB1"),
        ("SiIV",  r"$\mathrm{Si\,IV}$",  "#EE9A2E"),
        ("NV",    r"$\mathrm{N\,V}$",    "#C0392B")]
ION_KEYS = [i[0] for i in IONS]
ION_LAB = {k: l for k, l, _ in IONS}
ION_COL = {k: c for k, _, c in IONS}


def classify(dv):
    """Kinematic class from dv = v_los - v_ISM (array or scalar)."""
    a = np.abs(np.asarray(dv, float))
    out = np.where(a < CLASS_BINS[0], "ISM", np.where(a < CLASS_BINS[1], "IVC", "HVC"))
    return out


def set_style(usetex=None, base=12.5):
    """Apply the publication rcParams. Auto-detects a working LaTeX; falls back to
    mathtext with the Computer-Modern font set (visually near-identical)."""
    if usetex is None:
        usetex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    rc = {
        "font.family": "serif",
        "font.size": base, "axes.labelsize": base + 2.0, "axes.titlesize": base + 1.0,
        "xtick.labelsize": base - 1.5, "ytick.labelsize": base - 1.5,
        "legend.fontsize": base - 2.0,
        "axes.linewidth": 1.3,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 6.5, "ytick.major.size": 6.5,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
        "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "0.6",
        "legend.handlelength": 1.7, "legend.borderpad": 0.5, "legend.columnspacing": 1.4,
        "axes.grid": False, "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "figure.dpi": 120,
    }
    if usetex:
        rc.update({"text.usetex": True, "font.serif": ["Computer Modern Roman"],
                   "text.latex.preamble": r"\usepackage{amsmath}\usepackage{textcomp}"})
    else:
        rc.update({"text.usetex": False, "mathtext.fontset": "cm",
                   "font.serif": ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": True})
    plt.rcParams.update(rc)
    return usetex


def grid(ax, which="major"):
    ax.grid(which=which, ls="-", lw=0.5, alpha=0.16, zorder=0)


def shade_classes(ax, xmax=560.0, alpha=0.06):
    """Shade the ISM/IVC/HVC |dv| regions symmetrically on a signed-dv axis."""
    ax.axvspan(-CLASS_BINS[0], CLASS_BINS[0], color=CLASS["ISM"], alpha=alpha, lw=0, zorder=0)
    for s in (1, -1):
        ax.axvspan(s * CLASS_BINS[0], s * CLASS_BINS[1], color=CLASS["IVC"], alpha=alpha, lw=0, zorder=0)
        ax.axvspan(s * CLASS_BINS[1], s * xmax, color=CLASS["HVC"], alpha=alpha, lw=0, zorder=0)


def tag(ax, text, corner="ul", fs=None, alpha=0.92):
    """Rounded corner tag box (OVI place_tag)."""
    xy = {"ul": (0.035, 0.965, "left", "top"), "ur": (0.965, 0.965, "right", "top"),
          "ll": (0.035, 0.035, "left", "bottom"), "lr": (0.965, 0.035, "right", "bottom")}
    x, y, ha, va = xy[corner]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fs or plt.rcParams["legend.fontsize"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=alpha), zorder=8)


def panel_label(ax, lab, fs=None):
    ax.text(0.02, 0.98, lab, transform=ax.transAxes, ha="left", va="top", fontweight="bold",
            fontsize=fs or (plt.rcParams["axes.labelsize"]))


def save(fig, name, dpi=240, formats=("pdf", "png")):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return FIGDIR / f"{name}.png"


def diverging_norm(vmax, vcenter=0.0):
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=vcenter, vmax=vmax)

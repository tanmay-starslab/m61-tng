#!/usr/bin/env python3
"""Shared loaders for the GAS-catalog (column-weighted) figures.

!! DETECTION IS NOT DEFINED HERE. USE m61_voigt.py. !!
------------------------------------------------------
The project's detection product is the pyGad Voigt fit of every mock spectrum, line by
line (`m61_voigt.py`): a DETECTION is a fitted component with uplim == False. That is the
pipeline's definition and the only one used in the paper.

A curve-of-growth `N_limit()` / `W_from_N()` / `N_from_W()` used to live here, to synthesise
a detection threshold from oscillator strengths. **They have been deleted.** They were not
how detection is defined, they gave the wrong answer (they flattened the Si II triplet's
covering-fraction spread from a factor 1.22 to ~1.1), and leaving them exported was a
foot-gun for the next author. If you find yourself wanting them, you want `m61_voigt` instead.

What this module IS for: loading the per-cell absorbing-gas catalog and computing
column-weighted, ION-level physical quantities (v_r, metallicity, temperature, phase
structure) that exist only in the simulation and have no counterpart in the fits.

For ION-level gas statistics the three Si II transitions are mathematically DEGENERATE --
they share the single column N_SiII, so any column-weighted statistic is identical for all
three. Gas figures are therefore labelled by ION, never by transition, and all
transition-resolved results come from the Voigt catalog. Do not plot three identical Si II
curves off N_SiII.

Atomic data (lambda_0, f) are Trident's `lines.txt` values -- the ones used to synthesise
the spectra. NOTE these are the *synthesis* wavelengths; pyGad fit Si II 1260 at a different
rest wavelength (1260.522 vs 1260.422), which is corrected in the Voigt catalog. See
VOIGT_CATALOG.md.

CAVEAT on denominators: `load_cells` reports 14,400 sightlines with a finite v_ISM, but only
14,205 appear in the cell catalog -- 195 sightlines intersect no absorbing cell at all. That
is harmless for the column-weighted gas statistics here, but `n_sl` must NOT be used as a
covering-fraction denominator. Covering fractions live in `m61_voigt`.

Usage:
    import m61_lines as L
    cells = L.load_cells("v3b")            # catalog + dv from the chosen v_ISM
    for ln in L.LINES: ...                 # 9 COS-G130M transitions
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity")
CATDIR = BASE / "absorber_catalog"
MASTER_V3 = BASE / "vism_tables_v3/vism_master_v3.csv"
MASTER_V1 = BASE / "vism_tables/vism_master_all_sightlines.csv"

# kinematic class thresholds on |dv| = |v_los - v_ISM|  [km/s]
IVC = 40.0
HVC = 100.0

# sightline identity in the catalog
SLKEY = ["sid", "mode", "alpha"]

# NOTE: an equivalent-width "detection limit" constant used to live here. Removed --
# detection comes from the Voigt fits (m61_voigt.py), never from a synthesised threshold.

# ---------------------------------------------------------------------------------
# atomic data -- taken from Trident's line list (the same data that made the spectra)
# ---------------------------------------------------------------------------------
# key            = HDF5 group name under spectrum_by_line/
# ion            = catalog column suffix (N_<ion>) -- the ion that produces the line
# lam0 [A], fosc = Trident lines.txt values (verified 2026-08-03)
_LINEDATA = [
    # key,          ion,     lam0,       fosc,   label,                              ls,   lw
    ("H_I_1216",    "HI",    1215.6700, 0.41600, r"$\mathrm{H\,I}\ \lambda1216$",    "-",  2.0),
    ("O_I_1302",    "OI",    1302.1680, 0.05200, r"$\mathrm{O\,I}\ \lambda1302$",    "-",  2.0),
    ("C_II_1335",   "CII",   1334.5320, 0.12900, r"$\mathrm{C\,II}\ \lambda1335$",   "-",  2.0),
    ("Si_II_1190",  "SiII",  1190.4160, 0.27700, r"$\mathrm{Si\,II}\ \lambda1190$",  ":",  1.7),
    ("Si_II_1193",  "SiII",  1193.2900, 0.57500, r"$\mathrm{Si\,II}\ \lambda1193$",  "--", 1.7),
    ("Si_II_1260",  "SiII",  1260.4220, 1.18000, r"$\mathrm{Si\,II}\ \lambda1260$",  "-",  2.0),
    ("Si_III_1206", "SiIII", 1206.5000, 1.63000, r"$\mathrm{Si\,III}\ \lambda1206$", "-",  2.0),
    ("Si_IV_1403",  "SiIV",  1402.7700, 0.25500, r"$\mathrm{Si\,IV}\ \lambda1403$",  "-",  2.0),
    ("N_V_1239",    "NV",    1238.8210, 0.15600, r"$\mathrm{N\,V}\ \lambda1239$",    "-",  2.0),
]

# ionization sequence -> colour. Ordered neutral/cool to warm-hot; all three Si II
# transitions share the Si II colour and are separated by linestyle.
ION_SEQ = ["HI", "OI", "CII", "SiII", "SiIII", "SiIV", "NV"]
_TURBO = plt.get_cmap("turbo")
ION_COL = {k: mcolors.to_hex(_TURBO(0.06 + 0.88 * i / (len(ION_SEQ) - 1)))
           for i, k in enumerate(ION_SEQ)}
ION_LAB = {"HI": r"$\mathrm{H\,I}$", "OI": r"$\mathrm{O\,I}$", "CII": r"$\mathrm{C\,II}$",
           "SiII": r"$\mathrm{Si\,II}$", "SiIII": r"$\mathrm{Si\,III}$",
           "SiIV": r"$\mathrm{Si\,IV}$", "NV": r"$\mathrm{N\,V}$"}


class Line:
    """One transition: atomic data + plotting style. No detection logic -- see m61_voigt."""

    __slots__ = ("key", "ion", "lam0", "fosc", "label", "ls", "lw", "color")

    def __init__(self, key, ion, lam0, fosc, label, ls, lw):
        self.key, self.ion, self.lam0, self.fosc = key, ion, lam0, fosc
        self.label, self.ls, self.lw = label, ls, lw
        self.color = ION_COL[ion]

    # ---- curve of growth (linear/optically-thin regime) --------------------------
    # W_lambda[mA] = 8.8524e-18 * N[cm^-2] * f * lambda[A]^2
    @property
    def strength(self):
        """f * lambda -- descriptive only (orders the Si II triplet). NOT a detection scale."""
        return self.fosc * self.lam0

    def __repr__(self):
        return f"<Line {self.key} f*lam={self.strength:.1f}>"


LINES = [Line(*d) for d in _LINEDATA]
LINE_BY_KEY = {L.key: L for L in LINES}
LINE_KEYS = [L.key for L in LINES]

# convenience: the Si II triplet, the whole point of the per-line treatment
SIII_LINES = [L for L in LINES if L.ion == "SiII"]


def lines_for_ion(ion):
    return [L for L in LINES if L.ion == ion]


# ---------------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------------
def load_master_v3():
    """v3 master table (one row per sightline) with both v3a and v3b."""
    return pd.read_csv(MASTER_V3)


def load_cells(variant="v3b", clean=True, master=None):
    """Absorbing-cell catalog with dv recomputed against the chosen v_ISM.

    variant : 'v3a' | 'v3b' | 'v1' (v_ism_direct_cool) | 'v2' (v_ism_v2)
    clean   : drop Tier-0b periodic-wrap / hypervelocity cells (published sample)

    Returns (cells, n_sl) where n_sl = number of sightlines with a finite v_ISM.
    """
    vcol = {"v3a": "v_ism_v3a", "v3b": "v_ism_v3b",
            "v1": "v_ism_direct_cool", "v2": "v_ism_v2"}[variant]
    files = sorted(glob.glob(str(CATDIR / "absorbers_sid*.parquet")))
    if not files:
        raise FileNotFoundError(f"no absorber catalog under {CATDIR}")
    cells = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if clean and "wrapped" in cells and "hypervel" in cells:
        n0 = len(cells)
        cells = cells[~(cells.wrapped | cells.hypervel)].reset_index(drop=True)
        print(f"[m61_lines] clean mask dropped {n0 - len(cells)} "
              f"({100 * (n0 - len(cells)) / n0:.3f}%) -> {len(cells)} cells")
    m = load_master_v3() if master is None else master
    v = m[["sid", "mode", "alpha_deg", vcol]].rename(columns={"alpha_deg": "alpha"})
    cells = cells.merge(v, on=SLKEY, how="left")
    cells["v_ISM"] = cells[vcol]
    cells["dv"] = cells["v_rest"] - cells[vcol]
    cells = cells[cells["dv"].notna()].reset_index(drop=True)
    n_sl = int(m[vcol].notna().sum())
    print(f"[m61_lines] variant={variant} ({vcol}): {len(cells)} cells, {n_sl} sightlines")
    return cells, n_sl


def sightline_columns(cells, mask=None, ions=None):
    """Per-sightline summed column for each ion, over the cells selected by `mask`.

    Returns a DataFrame indexed by (sid, mode, alpha) with one N_<ion> column per ion.
    Sightlines with no selected cell are present with 0 (not dropped), so covering
    fractions have the correct denominator.
    """
    ions = ions or ION_SEQ
    sub = cells if mask is None else cells[mask]
    cols = [f"N_{i}" for i in ions if f"N_{i}" in cells.columns]
    tot = sub.groupby(SLKEY)[cols].sum()
    allsl = cells[SLKEY].drop_duplicates().set_index(SLKEY)
    return tot.reindex(allsl.index, fill_value=0.0)


def covering_fraction(Nser, N_min):
    """f_c(>N_min): fraction of sightlines whose column exceeds the floor."""
    N = np.asarray(Nser, float)
    N = N[np.isfinite(N)]
    return float(np.mean(N > N_min)) if len(N) else np.nan


def cumulative_cf(Nser, grid):
    """Cumulative covering fraction f_c(>N) evaluated on `grid`."""
    N = np.asarray(Nser, float)
    N = N[np.isfinite(N)]
    if not len(N):
        return np.full(len(grid), np.nan)
    return np.array([np.mean(N > g) for g in grid], float)


def wilson(k, n, z=1.0):
    """Wilson score interval for a binomial covering fraction (z=1 -> 68%).
    Returns (lo, hi). Correct near f=0 and f=1 where the normal interval fails."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1.0 + z ** 2 / n
        c = (p + z ** 2 / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def combined_ray_path(sid):
    """The COMBINED ray product -- the one v_ISM was built on. Do NOT use the per-SID
    original_rays/*.h5, which are a different product (different ncells & velocity_los)."""
    p = glob.glob(f"/scratch/tsingh65/m61-tng/outputs/sid{sid}/"
                  f"rays_and_spectra_sid{sid}_snap99_L2Rvir/combined/all_rays_L2Rvir.h5")
    if not p:
        raise FileNotFoundError(f"no combined ray for sid {sid}")
    return p[0]


if __name__ == "__main__":
    print(f"{'line':14s} {'ion':6s} {'lam0':>9s} {'f':>7s} {'f*lam':>8s} {'rel':>6s}")
    ref = LINE_BY_KEY["Si_II_1260"].strength
    for L in LINES:
        print(f"{L.key:14s} {L.ion:6s} {L.lam0:9.3f} {L.fosc:7.3f} {L.strength:8.1f} "
              f"{L.strength / ref:6.3f}")

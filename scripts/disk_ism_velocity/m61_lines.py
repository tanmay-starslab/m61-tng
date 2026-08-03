#!/usr/bin/env python3
"""Shared loaders for the GAS-catalog (column-weighted) figures.

!! DETECTION IS NOT DEFINED HERE. USE m61_voigt.py. !!
------------------------------------------------------
The project's detection product is the pyGad Voigt fit of every mock spectrum, line by
line (`m61_voigt.py`): a DETECTION is a fitted component with uplim == False. That is the
pipeline's definition and the only one used in the paper.

The curve-of-growth machinery below (`Line.N_limit`, `W_from_N`, `N_from_W`) was an
earlier attempt to synthesise a detection threshold from oscillator strengths. It is NOT
how detection is defined and must NOT be used for covering fractions, detection rates or
any per-transition statistic. It is kept only because it is harmless as a descriptive
scaling, and removing it would break callers mid-run.

What this module IS for: loading the per-cell absorbing-gas catalog and computing
column-weighted, ION-level physical quantities (v_r, metallicity, temperature, phase
structure) that exist only in the simulation and have no counterpart in the fits.
For ION-level gas statistics the three Si II transitions are mathematically degenerate
(they share N_SiII) -- so gas figures are labelled by ION, and the transition-resolved
results come from the Voigt catalog.

Original rationale (superseded, kept for context)
-------------------------------------------------
The v1/v2/v3 paper figures are per-ION: they weight by the gas column N_ion from the
absorber catalog. But the observable is a *transition*, and Si II is observed through
three lines in the COS-G130M band (lambda 1190, 1193, 1260) whose strengths differ by
f*lambda = 1 : 0.46 : 0.22. Those three lines share the SAME N_SiII, so any statistic
that depends only on the column is identical for all three; what differs is how much of
that column is *detectable*.

So there are exactly two honest ways a Si II transition can enter a figure, and this
module supplies both:

  (1) SPECTRUM-BASED  -- statistics measured off the mock LSF spectra
      (`spectrum_by_line/<key>/lsf`). Saturation, blending and line strength all enter
      naturally, so the three Si II lines are genuinely independent curves.

  (2) DETECTION-LIMITED COLUMN  -- gas-catalog statistics restricted to the column that
      the given transition could actually detect, N > N_limit(line). N_limit follows the
      linear curve of growth for a fixed equivalent-width limit, so the three Si II
      transitions give three distinct curves that differ *only* through line strength.

Never plot three identical Si II curves from raw N_SiII: use (1) or (2).

Atomic data (lambda_0, f) are read straight out of Trident's `lines.txt`, i.e. exactly
the values used to synthesise the mock spectra, so the registry cannot drift from the data.

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

# equivalent-width limit adopted for "detectable" (COS G130M, S/N ~ 10 per resel).
# 3-sigma W limit; used to convert a line's f*lambda^2 into a column floor.
W_LIMIT_MA = 50.0

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
    """One transition: atomic data + plotting style + curve-of-growth helpers."""

    __slots__ = ("key", "ion", "lam0", "fosc", "label", "ls", "lw", "color")

    def __init__(self, key, ion, lam0, fosc, label, ls, lw):
        self.key, self.ion, self.lam0, self.fosc = key, ion, lam0, fosc
        self.label, self.ls, self.lw = label, ls, lw
        self.color = ION_COL[ion]

    # ---- curve of growth (linear/optically-thin regime) --------------------------
    # W_lambda[mA] = 8.8524e-18 * N[cm^-2] * f * lambda[A]^2
    _COG = 8.8524e-18

    @property
    def strength(self):
        """f * lambda -- the quantity that sets optical depth. Si II 1260 is the ref."""
        return self.fosc * self.lam0

    def W_from_N(self, N):
        """Linear-COG equivalent width [mA] for column N [cm^-2] (upper bound: real
        lines saturate, so this over-predicts W once tau_0 >~ 1)."""
        return self._COG * np.asarray(N, float) * self.fosc * self.lam0 ** 2

    def N_from_W(self, W_mA):
        """Column [cm^-2] that produces W_mA on the linear part of the COG."""
        return np.asarray(W_mA, float) / (self._COG * self.fosc * self.lam0 ** 2)

    def N_limit(self, W_mA=W_LIMIT_MA):
        """Detection floor: the smallest column this transition can reveal at W_mA."""
        return float(self.N_from_W(W_mA))

    def tau_int(self, N):
        """Velocity-integrated apparent optical depth [km/s] for column N.
        AOD:  N = 3.768e14 * int(tau) dv / (f * lambda)."""
        return np.asarray(N, float) * self.fosc * self.lam0 / 3.768e14

    def __repr__(self):
        return f"<Line {self.key} f*lam={self.strength:.1f} Nlim={self.N_limit():.2e}>"


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
    print(f"{'line':14s} {'ion':6s} {'lam0':>9s} {'f':>7s} {'f*lam':>8s} "
          f"{'rel':>6s} {'N_lim(50mA)':>12s}")
    ref = LINE_BY_KEY["Si_II_1260"].strength
    for L in LINES:
        print(f"{L.key:14s} {L.ion:6s} {L.lam0:9.3f} {L.fosc:7.3f} {L.strength:8.1f} "
              f"{L.strength / ref:6.3f} {L.N_limit():12.3e}")

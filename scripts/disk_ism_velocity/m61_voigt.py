#!/usr/bin/env python3
"""Voigt-fit (pyGad) component catalog -- the PIPELINE'S OWN detection product.

THIS is where "detection" is defined for this project. Do not invent one.

The spectral fitting stage already fit every mock COS-G130M spectrum, line by line, with
pyGad Voigt profiles at S/N=10, bin=3:
    outputs/sid<SID>/rays_and_spectra_sid<SID>_snap99_L2Rvir/
        fitted_individual_line_spectra_parallel_snr10_bin3/per_spectrum_h5/<mode>/alpha<NNN>/
            L2Rvir_sid<SID>_J122138+043026_<mode>_alpha<A>_spectrum_individual_line_fits.h5
720 orientations per galaxy x 20 galaxies = 14,400 sightlines, each with 9 fitted lines
(H I 1216, O I 1302, C II 1335, Si II 1190, Si II 1193, Si II 1260, Si III 1206,
 Si IV 1403, N V 1239).

Per line the file holds:
  lines/<LINE>/fit_rows      one row per fitted Voigt COMPONENT (fields in result_fieldnames)
  line_status_rows           per-line fit_status / non_detection / n_components
  lines/<LINE>/{obs,model,raw,lsf}   the spectra themselves

DETECTION  = a fitted component with UpLim == False (pyGad flags upper limits, i.e.
             non-detections, with UpLim True; a line with no detected component has
             non_detection == True in line_status_rows).
HVC        = a DETECTED component whose velocity offset from the disk ISM satisfies
             |dv| >= 100 km/s.  IVC 40-100, ISM < 40.

Because the fit is done per transition, the three Si II lines (1190/1193/1260) are three
INDEPENDENT measurements of the same gas: the weaker transitions simply fail to fit
components that the strong one recovers. That is the honest, pipeline-native way the Si II
triplet differs -- no curve-of-growth modelling or oscillator-strength thresholding is
needed or wanted.

CAVEATS THAT CHANGE RESULTS -- read before quoting any number:
  * `sat` is a PER-FIT flag, not per-component: it is constant across every component of
    a (sightline, line) fit (verified: 100% of groups). It means "this fit region contains
    saturation", NOT "this component is saturated". `Chisq` is likewise per fit region
    (constant in 78% of groups; the remainder have several regions). Never quote a
    "fraction of saturated components".
  * H I 1216 IS NOT COMPARABLE TO THE METAL LINES. It was fit over a +-1200 km/s window
    with b_max = 150 km/s, while the metals used +-800 with b_max 50-100. As a result 59%
    of H I components sit AT the b ceiling and 61% lie beyond |dv| > 500 km/s -- these are
    multi-component fits to the damped, saturated Lyman-alpha profile, not physical
    high-velocity clouds. That artefact alone drives H I's ~0.93 "HVC covering fraction".
    Use the `b_at_ceiling` and `beyond_common_window` flags to exclude them, and qualify
    every H I number. The metals are clean (b well inside bounds, ~1-1.6% beyond 500 km/s).

TWO FIELD-NAME TRAPS -- read before using:
  * pyGad's `dv_kms` is the UNCERTAINTY on `v_kms`, NOT a velocity offset. It is stored
    here as `v_err_kms`. The kinematic offset from the ISM is `dv_v3a` / `dv_v3b`.
  * `v_kms` is the RAW observed velocity c(lambda/lambda0 - 1). The rest-frame velocity on
    the same convention as v_ISM and the ray velocities is
        v_rest = -v_kms - v_sys
    VERIFIED 2026-08-03 against the Si II 1260 flux-minimum (SiII_dip) over 4 galaxies:
    this convention gives sigma = 7-33 km/s; every sign/offset alternative gives 200-640.
    It is the same convention as fig10_accumulate.py.

Usage:
    import m61_voigt as V
    comp = V.load_components()            # all fitted components, both v3 variants
    stat = V.load_line_status()           # per (sightline, line) fit status -- DENOMINATORS
    cf   = V.covering_fraction(comp, stat, "v3b", vmin=100)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity")
VOIGT_DIR = BASE / "voigt_catalog"
MASTER_V3 = BASE / "vism_tables_v3/vism_master_v3.csv"

RAY_ID = "J122138+043026"
RUN_LABEL = "L2Rvir"          # matches the combined ray that v_ISM was built on
FIT_TAG = "fitted_individual_line_spectra_parallel_snr10_bin3"

# kinematic classes on |dv| = |v_rest - v_ISM|  [km/s]
IVC = 40.0
HVC = 100.0

SLKEY = ["sid", "mode", "alpha"]

# the nine fitted transitions, in ionization order; ion = species, used for colour.
# lam0/fosc are METADATA ONLY (identification + optional curve-of-growth diagnostics).
# They play NO role in defining a detection -- pyGad's UpLim flag does that.
LINE_TABLE = [
    # key,          ion,     lam0,      fosc,    label,                              ls,   lw
    ("H_I_1216",    "HI",    1215.670, 0.41600, r"$\mathrm{H\,I}\ \lambda1216$",    "-",  2.0),
    ("O_I_1302",    "OI",    1302.168, 0.05200, r"$\mathrm{O\,I}\ \lambda1302$",    "-",  2.0),
    ("C_II_1335",   "CII",   1334.532, 0.12900, r"$\mathrm{C\,II}\ \lambda1335$",   "-",  2.0),
    ("Si_II_1190",  "SiII",  1190.416, 0.27700, r"$\mathrm{Si\,II}\ \lambda1190$",  ":",  1.7),
    ("Si_II_1193",  "SiII",  1193.290, 0.57500, r"$\mathrm{Si\,II}\ \lambda1193$",  "--", 1.7),
    ("Si_II_1260",  "SiII",  1260.422, 1.18000, r"$\mathrm{Si\,II}\ \lambda1260$",  "-",  2.0),
    ("Si_III_1206", "SiIII", 1206.500, 1.63000, r"$\mathrm{Si\,III}\ \lambda1206$", "-",  2.0),
    ("Si_IV_1403",  "SiIV",  1402.770, 0.25500, r"$\mathrm{Si\,IV}\ \lambda1403$",  "-",  2.0),
    ("N_V_1239",    "NV",    1238.821, 0.15600, r"$\mathrm{N\,V}\ \lambda1239$",    "-",  2.0),
]

ION_SEQ = ["HI", "OI", "CII", "SiII", "SiIII", "SiIV", "NV"]


def _ion_colors():
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    t = plt.get_cmap("turbo")
    return {k: mcolors.to_hex(t(0.06 + 0.88 * i / (len(ION_SEQ) - 1)))
            for i, k in enumerate(ION_SEQ)}


ION_COL = _ion_colors()
ION_LAB = {"HI": r"$\mathrm{H\,I}$", "OI": r"$\mathrm{O\,I}$", "CII": r"$\mathrm{C\,II}$",
           "SiII": r"$\mathrm{Si\,II}$", "SiIII": r"$\mathrm{Si\,III}$",
           "SiIV": r"$\mathrm{Si\,IV}$", "NV": r"$\mathrm{N\,V}$"}


class Line:
    __slots__ = ("key", "ion", "lam0", "fosc", "label", "ls", "lw", "color")

    def __init__(self, key, ion, lam0, fosc, label, ls, lw):
        self.key, self.ion, self.lam0, self.fosc = key, ion, lam0, fosc
        self.label, self.ls, self.lw = label, ls, lw
        self.color = ION_COL[ion]

    @property
    def strength(self):
        """f*lambda -- metadata for ORDERING/annotating the Si II triplet only."""
        return self.fosc * self.lam0

    def __repr__(self):
        return f"<Line {self.key}>"


LINES = [Line(*d) for d in LINE_TABLE]
LINE_BY_KEY = {L.key: L for L in LINES}
LINE_KEYS = [L.key for L in LINES]
SIII_LINES = [L for L in LINES if L.ion == "SiII"]        # the Si II triplet


def fit_h5_path(sid, mode, alpha):
    """Per-spectrum pyGad individual-line fit file."""
    return (Path(f"/scratch/tsingh65/m61-tng/outputs/sid{sid}/"
                 f"rays_and_spectra_sid{sid}_snap99_{RUN_LABEL}/{FIT_TAG}/per_spectrum_h5")
            / mode / f"alpha{alpha:03d}"
            / f"{RUN_LABEL}_sid{sid}_{RAY_ID}_{mode}_alpha{alpha}_spectrum_individual_line_fits.h5")


# ---------------------------------------------------------------------------------
# loaders (read the catalog built by build_voigt_catalog.py)
# ---------------------------------------------------------------------------------
def load_components(variant=None, detected_only=True):
    """All fitted Voigt components.

    variant : None keeps both dv_v3a and dv_v3b; 'v3a'/'v3b' also adds a plain `dv`
              column for that variant (convenience).
    detected_only : drop UpLim (upper-limit / non-detection) rows. Default True --
              those rows are NOT detections and must never be counted as absorbers.
    """
    f = VOIGT_DIR / "voigt_components.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing -- run build_voigt_catalog.py first")
    c = pd.read_parquet(f)
    if detected_only:
        n0 = len(c)
        c = c[~c.uplim].reset_index(drop=True)
        print(f"[m61_voigt] detections only: {n0} -> {len(c)} components "
              f"(dropped {n0 - len(c)} upper limits)")
    if variant:
        c["dv"] = c[f"dv_{variant}"]
    return c


def load_line_status():
    """Per (sightline, line) fit status. THE DENOMINATOR for covering fractions:
    a sightline counts only if that line was actually fit there."""
    f = VOIGT_DIR / "voigt_line_status.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing -- run build_voigt_catalog.py first")
    return pd.read_parquet(f)


def load_master_v3():
    return pd.read_csv(MASTER_V3)


# ---------------------------------------------------------------------------------
# covering fractions -- unweighted SIGHTLINE COUNTS (no column weighting)
# ---------------------------------------------------------------------------------
# line_status.fit_status is {'detection','non_detection'} -- i.e. the fit RAN and either
# found >=1 real component or returned only upper limits. Both are valid denominators;
# anything else (missing file / error) is not.
FIT_RAN = ("detection", "non_detection")


def line_denominator(stat, line):
    """Number of sightlines on which `line` was actually fit (the denominator).
    Includes non_detections -- a sightline where the line was fit and nothing was found
    is a genuine zero, not a missing measurement."""
    s = stat[(stat.line == line) & (stat.fit_status.isin(FIT_RAN))]
    return int(len(s.drop_duplicates(SLKEY)))


def sightlines_with(comp, line, variant, vmin=HVC, vmax=np.inf, signed=None):
    """Set of sightlines having >=1 DETECTED component of `line` with
    vmin <= |dv| < vmax.  signed: None=|dv|, 'blue'=dv<0, 'red'=dv>0."""
    c = comp[comp.line == line]
    dv = c[f"dv_{variant}"]
    m = (dv.abs() >= vmin) & (dv.abs() < vmax)
    if signed == "blue":
        m &= dv < 0
    elif signed == "red":
        m &= dv > 0
    return c[m].drop_duplicates(SLKEY)[SLKEY]


def covering_fraction(comp, stat, variant, line=None, vmin=HVC, vmax=np.inf, signed=None):
    """f_c = (# sightlines with >=1 detected component in the velocity window)
             / (# sightlines where that line was fit).
    Pure sightline counting -- NOT weighted by column, EW or anything else.
    Returns (f, k, n). If `line` is None, returns a dict keyed by line."""
    if line is None:
        return {L.key: covering_fraction(comp, stat, variant, L.key, vmin, vmax, signed)
                for L in LINES}
    n = line_denominator(stat, line)
    k = len(sightlines_with(comp, line, variant, vmin, vmax, signed))
    return (k / n if n else np.nan), k, n


def wilson(k, n, z=1.0):
    """Wilson score interval (z=1 -> 68%). Correct near f=0 and f=1."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1.0 + z ** 2 / n
        c = (p + z ** 2 / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def cf_vs_velocity(comp, stat, variant, line, edges):
    """Unweighted detection fraction per SIGNED dv bin: fraction of sightlines with
    >=1 detected component whose dv falls in the bin. Returns (k, n) arrays."""
    n = line_denominator(stat, line)
    c = comp[comp.line == line]
    dv = c[f"dv_{variant}"].to_numpy()
    sl = c[SLKEY].astype(str).agg("|".join, axis=1).to_numpy()
    idx = np.digitize(dv, edges) - 1
    k = np.zeros(len(edges) - 1, float)
    for b in range(len(edges) - 1):
        sel = idx == b
        k[b] = len(np.unique(sl[sel])) if sel.any() else 0
    return k, n


def cf_vs_threshold(comp, stat, variant, line, vthr):
    """Cumulative: fraction of sightlines with >=1 detected component at |dv| >= v,
    for each v in vthr. Returns (k, n)."""
    n = line_denominator(stat, line)
    c = comp[comp.line == line]
    adv = c[f"dv_{variant}"].abs().to_numpy()
    sl = c[SLKEY].astype(str).agg("|".join, axis=1).to_numpy()
    k = np.array([len(np.unique(sl[adv >= v])) if (adv >= v).any() else 0 for v in vthr], float)
    return k, n


def cf_vs_logN(comp, stat, variant, line, grid, vmin=HVC, signed=None):
    """Cumulative covering fraction vs FITTED column: fraction of sightlines with
    >=1 detected component in the velocity window having logN >= grid[i].
    Uses the pyGad-fitted logN -- the observable column, not the gas column."""
    n = line_denominator(stat, line)
    c = comp[comp.line == line]
    dv = c[f"dv_{variant}"]
    m = dv.abs() >= vmin
    if signed == "blue":
        m &= dv < 0
    elif signed == "red":
        m &= dv > 0
    c = c[m]
    sl = c[SLKEY].astype(str).agg("|".join, axis=1).to_numpy()
    lN = c["logN"].to_numpy()
    k = np.array([len(np.unique(sl[lN >= g])) if (lN >= g).any() else 0 for g in grid], float)
    return k, n


if __name__ == "__main__":
    comp = load_components()
    stat = load_line_status()
    print(f"\n{len(comp)} detected components, {len(stat)} (sightline,line) fits\n")
    print(f"{'line':14s} {'n_fit':>7s} {'HVC f_c(v3a)':>13s} {'HVC f_c(v3b)':>13s}")
    for L in LINES:
        fa, ka, na = covering_fraction(comp, stat, "v3a", L.key)
        fb, kb, nb = covering_fraction(comp, stat, "v3b", L.key)
        print(f"{L.key:14s} {na:7d} {fa:13.3f} {fb:13.3f}")

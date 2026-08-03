#!/usr/bin/env python3
r"""Properties of the fitted Voigt absorption components -- the observational
diagnostic layer of the m61-tng disk-ISM / HVC study.

Everything here is built from the pipeline's OWN detection product: a fitted pyGad
Voigt component with ``UpLim == False`` (see m61_voigt.py).  No curve-of-growth
modelling, no oscillator-strength column floor, no re-definition of "detection".

  fig14_component_properties      logN / b / EW distributions, ISM vs HVC, per transition
  fig15_b_vs_logN                 the b-logN component phase diagram per kinematic class,
                                  the saturation flag per FIT, and the fitter-boundary
                                  artefacts (b ceiling, +-500 km/s common window)
  fig16_components_per_sightline   absorber multiplicity, velocity spread, N_HVC vs N_ISM
  fig17_siII_triplet_consistency   Si II 1190/1193/1260 cross-transition systematics

Usage:
    python fig_voigt_components.py            # v3b (default)
    python fig_voigt_components.py v3a
    python fig_voigt_components.py all        # both variants

FOUR THINGS ABOUT THE CATALOG THAT ARE HONOURED THROUGHOUT (see m61_voigt docstring):
  * ``v_err_kms`` is pyGad's uncertainty on the fitted velocity, NOT a velocity offset.
    The kinematic offset from the disk ISM is ``dv_v3a`` / ``dv_v3b``.
  * ``sat`` is a PER-FIT flag: it is constant across every component of a given
    (sightline, transition) fit and means "this fit region contains saturation".  No
    per-component saturated fraction is computed or quoted anywhere in this module.
    ``Chisq`` is likewise a per-fit-region quantity.
  * H I 1216 was fit over +-1200 km/s with b_max = 150 km/s while the metals used +-800
    with b_max 50-100, so a large fraction of H I "components" are damping-wing pieces of
    the saturated Lyman-alpha profile pinned at the b ceiling and/or beyond 500 km/s.
    The DEFAULT SCIENCE SAMPLE here is therefore
        clean = ~b_at_ceiling & ~beyond_common_window
    Raw H I is shown alongside the clean curves wherever it changes the picture, and every
    printed table gives the raw and the cleaned number.
  * f*lambda is used ONLY to state the known Si II strength ordering
    (1260 : 1193 : 1190 = 1487 : 686 : 330 A) when discussing which transition saturates.
    It never enters a detection criterion.

FIFTH THING -- Si II 1260 VELOCITY ZERO-POINT, FIXED UPSTREAM AND VERIFIED HERE:
pyGad's line database (pygad/analysis/line_data.csv) has SiII1260 at 1260.5220 A while the
spectra were synthesised by Trident at 1260.4220 A, so every fitted lambda1260 velocity was
23.8 km/s too blue in v_obs, i.e. +23.8 km/s too RED in dv.  build_voigt_catalog.py now
recovers lam0_pygad = lambda_A/(1 + v_obs/c) per row and removes the offset
(`v_zeropoint_kms`).  It mattered most here in fig17: the cross-transition matcher uses a
15-40 km/s tolerance, so the old constant ~24 km/s offset rejected genuine pairs and the
completeness came out 0.62 / 0.60 instead of ~0.70 / ~0.69 (the median dlogN(weak - 1260)
was robust to it, +0.212 vs +0.215).  Nothing is corrected inside THIS file -- that would
double-count -- but fig17 re-measures the residual zero-point every run and will say so if
the catalog is stale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy.stats import ks_2samp, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m61_style as S          # noqa: E402
import m61_voigt as V          # noqa: E402

OUTBASE = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity")

CLS_SEQ = ["ISM", "IVC", "HVC"]
HI = "H_I_1216"
RAW_COL = "0.45"               # colour of the "raw H I" reference curves

CLEAN_TAG = (r"clean sample: $b$ not at the fitter ceiling," "\n"
             r"$|\Delta v|\leq500\ \mathrm{km\,s^{-1}}$")

# ── quantities shown in fig14 ────────────────────────────────────────────────────
QUANTS = [
    ("logN",  r"$\log_{10} N\ \ [\mathrm{cm^{-2}}]$", np.linspace(11.5, 19.8, 60), False),
    ("b_kms", r"$b\ \ [\mathrm{km\,s^{-1}}]$",        np.linspace(0.0, 156.0, 53), False),
    ("EW_mA", r"$W_\lambda\ \ [\mathrm{m\AA}]$",      np.logspace(0.2, 3.75, 52),  True),
]
QSHORT = {"logN": "logN", "b_kms": "b", "EW_mA": "EW"}
# y-limits of the fig14 median/IQR row -- headroom so the panel label and legend are clear
ROW2_YLIM = {"logN": (12.2, 21.4), "b_kms": (0.0, 205.0), "EW_mA": (18.0, 4.0e4)}

# ── Si II cross-transition matching ──────────────────────────────────────────────
# Two fitted components in two different transitions are the same absorber if their
# ISM-frame velocities agree to within
#     tol = clip( NSIG * sqrt(sig_1^2 + sig_2^2), TOL_MIN, TOL_MAX )
# TOL_MIN = 15 km/s ~ one COS G130M resolution element (FWHM ~ 17 km/s), so components
#   closer together than the instrument can separate are never split by the matcher.
# TOL_MAX = 40 km/s = the ISM/IVC class boundary: a pair wider than this spans more than
#   one kinematic class and must not be called the same absorber.  The cap is REQUIRED
#   because pyGad's velocity uncertainty has a pathological tail (median ~11 km/s but a
#   75th percentile of ~120 km/s and outliers > 1e6 km/s), so 3-sigma alone is unusable.
TOL_MIN, TOL_MAX, NSIG = 15.0, 40.0, 3.0

# Si II strength ordering (f*lambda, Angstrom) -- annotation only, never a threshold.
FLAM = {"Si_II_1260": 1487.3, "Si_II_1193": 686.1, "Si_II_1190": 329.7}
# Si II 1260 velocity zero-point (see the module docstring): pyGad fit at 1260.5220 A,
# Trident synthesised at 1260.4220 A. Removed in the catalog; kept here only to VERIFY.
LAM_1260_PYGAD, LAM_1260_TRUE = 1260.5220, 1260.4220
DV_1260_ZP = 299792.458 * (LAM_1260_PYGAD / LAM_1260_TRUE - 1.0)      # +23.8 km/s
ZP_TOL = 5.0        # km/s; residual |median dv(1260) - dv(weak)| above this = stale catalog
# column band used to separate the velocity trend from the column trend in fig17(d)
NBAND = (13.5, 14.5)


# =================================================================================
# data preparation
# =================================================================================
def prepare(variant):
    """Load the catalog; attach dv, |dv|, kinematic class and the clean-sample flag."""
    comp = V.load_components()
    stat = V.load_line_status()
    comp = comp.copy()

    for c in ("b_at_ceiling", "beyond_common_window"):
        if c not in comp.columns:
            raise KeyError(f"voigt_components.parquet is missing `{c}` -- rebuild the "
                           f"catalog with the current build_voigt_catalog.py")

    comp["dv"] = comp[f"dv_{variant}"]
    bad = ~np.isfinite(comp["dv"])
    if bad.any():
        print(f"[prepare] dropping {int(bad.sum())} components with non-finite dv_{variant}")
        comp = comp[~bad]
    comp["adv"] = comp["dv"].abs()
    comp["cls"] = np.where(comp.adv < V.IVC, "ISM",
                           np.where(comp.adv < V.HVC, "IVC", "HVC"))
    comp["clean"] = (~comp.b_at_ceiling) & (~comp.beyond_common_window)
    comp = comp.reset_index(drop=True)
    comp["cidx"] = np.arange(len(comp))
    stat = stat[stat.fit_status.isin(V.FIT_RAN)].copy()
    return comp, stat


def sightline_base(stat):
    """One row per (sightline, line) that was actually fit -- the denominator table."""
    return stat[V.SLKEY + ["line"]].drop_duplicates().reset_index(drop=True)


def pdf(x, bins):
    """Fraction of the SAMPLE per bin. Normalised by the number of finite values, not by
    the in-range count, so anything falling outside `bins` shows up as missing area rather
    than silently renormalising the curve upwards."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    h, _ = np.histogram(x, bins=bins)
    return (h / x.size) if x.size else h.astype(float)


def med_iqr(x):
    """(median, q25, q75) -- NaN-safe."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return q50, q25, q75


def _levels_from_frac(H, fracs=(0.90, 0.50)):
    """Contour levels enclosing the given fractions of the total counts."""
    h = np.sort(H.ravel())[::-1]
    cs = np.cumsum(h)
    if cs[-1] <= 0:
        return []
    lv = []
    for f in fracs:
        i = int(np.searchsorted(cs, f * cs[-1]))
        lv.append(h[min(i, len(h) - 1)])
    return sorted({float(v) for v in lv if v > 0})


def line_ticks(ax, keys, rot=32, fs=8.5):
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([V.LINE_BY_KEY[k].label for k in keys],
                       rotation=rot, ha="right", fontsize=fs)


LKEYS = [L.key for L in V.LINES]


# =================================================================================
# fig14 -- component properties, ISM vs HVC
# =================================================================================
def fig14_component_properties(comp, stat, variant):
    cc = comp[comp.clean]
    fig, ax = plt.subplots(3, 3, figsize=(15.2, 13.4))

    rows = [("ISM", S.CLASS_LABEL["ISM"]), ("HVC", S.CLASS_LABEL["HVC"])]
    labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"], ["(g)", "(h)", "(i)"]]

    # ── rows 0-1: the distributions, one curve per transition (clean sample) ─────
    for r, (cname, clab) in enumerate(rows):
        sub = cc[cc.cls == cname]
        rawhi = comp[(comp.cls == cname) & (comp.line == HI)]
        for q, (col, xlab, bins, logx) in enumerate(QUANTS):
            a = ax[r, q]
            xc = np.sqrt(bins[:-1] * bins[1:]) if logx else 0.5 * (bins[:-1] + bins[1:])
            a.step(xc, pdf(rawhi[col], bins), where="mid", color=RAW_COL, ls=(0, (1, 1.4)),
                   lw=1.5, zorder=1,
                   label=r"$\mathrm{H\,I}\ \lambda1216$ (raw)" if (r == 0 and q == 0) else None)
            for L in V.LINES:
                y = pdf(sub[sub.line == L.key][col], bins)
                a.step(xc, y, where="mid", color=L.color, ls=L.ls, lw=L.lw * 0.95, zorder=3,
                       label=L.label if (r == 0 and q == 0) else None)
            if logx:
                a.set_xscale("log")
            a.set_yscale("log")           # H I piles up at its logN/b ceilings
            a.set_xlim(bins[0], bins[-1])
            a.set_ylim(2e-4, 1.6)
            a.set_xlabel(xlab)
            a.set_ylabel("fraction of components / bin")
            S.grid(a)
            S.panel_label(a, labels[r][q])
            S.tag(a, clab, "ur", fs=9)
    ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), ncol=2, fontsize=7.2,
                    handlelength=2.2)
    S.tag(ax[1, 2], CLEAN_TAG, "ll", fs=8)

    # ── row 2: median + IQR per transition, ISM vs HVC ──────────────────────────
    summary = {}
    for q, (col, xlab, bins, logx) in enumerate(QUANTS):
        a = ax[2, q]
        x = np.arange(len(LKEYS))
        for cname, clab in rows:
            sub = cc[cc.cls == cname]
            m = np.array([med_iqr(sub[sub.line == k][col]) for k in LKEYS])
            off = -0.14 if cname == "ISM" else 0.14
            a.errorbar(x + off, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]],
                       fmt="o" if cname == "ISM" else "^", ms=6.5, lw=1.5, capsize=3,
                       color=S.CLASS[cname], label=clab,
                       mfc="white" if cname == "ISM" else S.CLASS[cname])
            summary[(col, cname)] = m
            # raw counterpart, per transition (open grey markers)
            subr = comp[comp.cls == cname]
            mr = np.array([med_iqr(subr[subr.line == k][col])[0] for k in LKEYS])
            a.plot(x + off, mr, ls="none", marker="x", ms=5.5, color=RAW_COL, zorder=5,
                   label=("raw (no clean cut)" if cname == "ISM" else None))
            summary[(col, cname, "raw")] = mr
        if logx:
            a.set_yscale("log")
        line_ticks(a, LKEYS)
        a.set_ylabel(xlab)
        a.set_xlim(-0.6, len(LKEYS) - 0.4)
        a.set_ylim(*ROW2_YLIM[col])          # headroom for the panel label / legend
        a.legend(loc="upper right", fontsize=8.0)
        S.grid(a)
        S.panel_label(a, labels[2][q])

    fig.tight_layout()
    p = S.save(fig, "fig14_component_properties")

    # ── printout ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print(f"fig14  component properties, ISM (|dv|<{V.IVC:.0f}) vs HVC (|dv|>={V.HVC:.0f})"
          f"   [{variant}]   CLEAN sample (b not at ceiling, |dv|<=500)")
    print("=" * 118)
    hdr = (f"{'line':13s} {'N_ISM':>6s} {'N_HVC':>6s} | "
           f"{'logN ISM':>8s} {'logN HVC':>8s} {'shift':>6s} | "
           f"{'b ISM':>6s} {'b HVC':>6s} {'shift':>6s} | "
           f"{'EW ISM':>7s} {'EW HVC':>7s} {'ratio':>6s}")
    print(hdr)
    for i, k in enumerate(LKEYS):
        nI = int(((cc.cls == "ISM") & (cc.line == k)).sum())
        nH = int(((cc.cls == "HVC") & (cc.line == k)).sum())
        lN = summary[("logN", "ISM")][i, 0], summary[("logN", "HVC")][i, 0]
        bb = summary[("b_kms", "ISM")][i, 0], summary[("b_kms", "HVC")][i, 0]
        ew = summary[("EW_mA", "ISM")][i, 0], summary[("EW_mA", "HVC")][i, 0]
        print(f"{k:13s} {nI:6d} {nH:6d} | {lN[0]:8.2f} {lN[1]:8.2f} {lN[1]-lN[0]:+6.2f} | "
              f"{bb[0]:6.1f} {bb[1]:6.1f} {bb[1]-bb[0]:+6.1f} | "
              f"{ew[0]:7.1f} {ew[1]:7.1f} {ew[1]/ew[0] if ew[0] else np.nan:6.2f}")
    print("\nsame medians on the RAW sample (no clean cut):")
    print(hdr)
    for i, k in enumerate(LKEYS):
        nI = int(((comp.cls == "ISM") & (comp.line == k)).sum())
        nH = int(((comp.cls == "HVC") & (comp.line == k)).sum())
        lN = summary[("logN", "ISM", "raw")][i], summary[("logN", "HVC", "raw")][i]
        bb = summary[("b_kms", "ISM", "raw")][i], summary[("b_kms", "HVC", "raw")][i]
        ew = summary[("EW_mA", "ISM", "raw")][i], summary[("EW_mA", "HVC", "raw")][i]
        print(f"{k:13s} {nI:6d} {nH:6d} | {lN[0]:8.2f} {lN[1]:8.2f} {lN[1]-lN[0]:+6.2f} | "
              f"{bb[0]:6.1f} {bb[1]:6.1f} {bb[1]-bb[0]:+6.1f} | "
              f"{ew[0]:7.1f} {ew[1]:7.1f} {ew[1]/ew[0] if ew[0] else np.nan:6.2f}")

    print("\nKS test ISM vs HVC (two-sided, clean sample):")
    print(f"{'line':13s} " + " ".join(f"{QSHORT[c]:>22s}" for c, _, _, _ in QUANTS))
    for k in LKEYS:
        cells = []
        for col, _, _, _ in QUANTS:
            a_ = cc[(cc.cls == "ISM") & (cc.line == k)][col].to_numpy()
            b_ = cc[(cc.cls == "HVC") & (cc.line == k)][col].to_numpy()
            if len(a_) > 5 and len(b_) > 5:
                st = ks_2samp(a_, b_)
                cells.append(f"D={st.statistic:5.3f} p={st.pvalue:8.1e}")
            else:
                cells.append("n/a")
        print(f"{k:13s} " + " ".join(f"{c:>22s}" for c in cells))
    return p


# =================================================================================
# fig15 -- b-logN phase diagram, saturation per FIT, fitter-boundary artefacts
# =================================================================================
def fit_table(comp):
    """One row per (sightline, transition) FIT: the sat flag (constant within the fit)
    and which kinematic classes the fit's components fall into."""
    g = comp.groupby(V.SLKEY + ["line"], sort=False)
    t = g.agg(sat=("sat", "first"), n=("sat", "size")).reset_index()
    for cn in CLS_SEQ:
        has = (comp[comp.cls == cn].groupby(V.SLKEY + ["line"], sort=False)
               .size().rename(f"n_{cn}").reset_index())
        t = t.merge(has, on=V.SLKEY + ["line"], how="left")
        t[f"n_{cn}"] = t[f"n_{cn}"].fillna(0).astype(int)
    return t


def fig15_b_vs_logN(comp, stat, variant):
    XB = np.linspace(11.5, 19.8, 84)
    YB = np.linspace(0.0, 156.0, 79)
    xc, yc = 0.5 * (XB[:-1] + XB[1:]), 0.5 * (YB[:-1] + YB[1:])
    bmax_line = comp.groupby("line")["fit_b_max_kms"].median()

    fig, axes = plt.subplots(2, 3, figsize=(19.2, 11.6))
    ax = axes.ravel()

    # ── (a)(b)(c) raw density, clean-sample contours, fitter b ceilings ──────────
    for i, cname in enumerate(CLS_SEQ):
        a = ax[i]
        sub = comp[comp.cls == cname]
        H, _, _ = np.histogram2d(sub.logN, sub.b_kms, bins=[XB, YB])
        pm = a.pcolormesh(XB, YB, H.T, cmap=S.DENSCMAP,
                          norm=LogNorm(vmin=1, vmax=max(H.max(), 10.0)), rasterized=True)
        cb = fig.colorbar(pm, ax=a, fraction=0.046, pad=0.02)
        cb.set_label(r"$N_{\rm components}$ (raw)")

        for bm in np.unique(bmax_line.dropna().to_numpy()):
            a.axhline(bm, color="0.75", ls=(0, (1, 2)), lw=1.0, zorder=2)

        uns = sub[sub.clean]
        if len(uns) > 50:
            Hu, _, _ = np.histogram2d(uns.logN, uns.b_kms, bins=[XB, YB])
            Hu = gaussian_filter(Hu, 1.1)
            lv = _levels_from_frac(Hu, (0.90, 0.50))
            if lv:
                a.contour(xc, yc, Hu.T, levels=lv, colors=S.ACCENT, linewidths=1.7,
                          linestyles=["--", "-"][: len(lv)], zorder=4)

        for L in V.LINES:                       # per-transition medians (clean)
            s = uns[uns.line == L.key]
            if len(s) < 20:
                continue
            a.plot(s.logN.median(), s.b_kms.median(), marker="o", ms=7.5, mfc=L.color,
                   mec="white", mew=1.2, ls="none", zorder=6)
        a.plot(sub.logN.median(), sub.b_kms.median(), marker="X", ms=12, mfc="0.25",
               mec="white", mew=1.4, ls="none", zorder=7)
        if len(uns):
            a.plot(uns.logN.median(), uns.b_kms.median(), marker="P", ms=13,
                   mfc=S.CLASS[cname], mec="white", mew=1.6, ls="none", zorder=7)

        a.set_xlim(XB[0], XB[-1])
        a.set_ylim(YB[0], YB[-1])
        a.set_xlabel(r"$\log_{10} N\ \ [\mathrm{cm^{-2}}]$")
        a.set_ylabel(r"$b\ \ [\mathrm{km\,s^{-1}}]$")
        S.tag(a, S.CLASS_LABEL[cname] + "\n" + r"$n_{\rm raw}=%d$, $n_{\rm clean}=%d$"
              % (len(sub), len(uns)), "ur", fs=8.5)
        S.panel_label(a, "(%s)" % "abc"[i])
    S.tag(ax[0], "dotted: fitter $b$ ceilings\n"
                 "cyan: clean-sample 50/90 per cent contours\n"
                 r"$\mathbf{+}$ clean median, $\mathbf{\times}$ raw median", "ll", fs=8)

    # ── (d) fraction of FITS whose region contains saturation ────────────────────
    ft = fit_table(comp)
    a = ax[3]
    x = np.arange(len(LKEYS))
    w = 0.26
    satfit = {}
    for j, cname in enumerate(CLS_SEQ):
        f, n = [], []
        for k in LKEYS:
            s = ft[(ft.line == k) & (ft[f"n_{cname}"] > 0)]
            f.append(s.sat.mean() if len(s) else np.nan)
            n.append(len(s))
        satfit[cname] = (np.array(f, float), np.array(n, int))
        a.bar(x + (j - 1) * w, satfit[cname][0], w, color=S.CLASS[cname],
              label=S.CLASS_LABEL[cname], alpha=0.92, edgecolor="white", lw=0.6)
    satfit["all"] = (np.array([ft[ft.line == k].sat.mean() for k in LKEYS], float),
                     np.array([int((ft.line == k).sum()) for k in LKEYS], int))
    a.plot(x, satfit["all"][0], ls="none", marker="_", ms=17, mew=2.2, color="0.15",
           label="all fits", zorder=5)
    line_ticks(a, LKEYS)
    a.set_ylabel("fraction of FITS with saturation in the fitted region")
    a.set_ylim(0, 1.30)
    a.set_xlim(-0.6, len(LKEYS) - 0.4)
    S.tag(a, "RAW components; denominator = fits with\n"
             r"$\geq1$ detected component (counts printed)", "ur", fs=8)
    a.legend(loc="lower left", fontsize=8.0, ncol=2)
    S.grid(a)
    S.panel_label(a, "(d)")

    # ── (e) fitter-boundary artefacts, per transition ────────────────────────────
    a = ax[4]
    fc = np.array([comp[comp.line == k].b_at_ceiling.mean() for k in LKEYS])
    fw = np.array([comp[comp.line == k].beyond_common_window.mean() for k in LKEYS])
    fcl = np.array([1.0 - comp[comp.line == k].clean.mean() for k in LKEYS])
    a.bar(x - 0.26, fc, 0.26, color=S.VMODE["R95-edge"], label=r"$b$ at fitter ceiling",
          edgecolor="white", lw=0.6)
    a.bar(x, fw, 0.26, color=S.OUTFLOW, label=r"$|\Delta v|>500\ \mathrm{km\,s^{-1}}$",
          edgecolor="white", lw=0.6)
    a.bar(x + 0.26, fcl, 0.26, color="0.45", label="removed by the clean cut",
          edgecolor="white", lw=0.6)
    line_ticks(a, LKEYS)
    a.set_ylabel("fraction of detected components")
    a.set_ylim(0, 1.05)
    a.set_xlim(-0.6, len(LKEYS) - 0.4)
    a.legend(loc="upper right", fontsize=8.5)
    S.grid(a)
    S.panel_label(a, "(e)")

    # ── (f) median b per transition, raw vs clean, with the ceiling marked ───────
    a = ax[5]
    braw = np.array([comp[comp.line == k].b_kms.median() for k in LKEYS])
    bcln = np.array([comp[(comp.line == k) & comp.clean].b_kms.median() for k in LKEYS])
    bmx = np.array([bmax_line.get(k, np.nan) for k in LKEYS], float)
    a.plot(x, bmx, ls="none", marker="_", ms=19, mew=2.0, color="0.72",
           label=r"fitter $b_{\max}$")
    a.plot(x, braw, ls="none", marker="x", ms=8, mew=2.0, color=RAW_COL, label="raw median")
    a.plot(x, bcln, ls="none", marker="o", ms=8, mfc=S.ACCENT, mec="white", mew=1.2,
           label="clean median")
    for i in range(len(LKEYS)):
        if np.isfinite(braw[i]) and np.isfinite(bcln[i]):
            a.plot([x[i], x[i]], [braw[i], bcln[i]], color="0.6", lw=1.0, zorder=1)
    line_ticks(a, LKEYS)
    a.set_ylabel(r"median $b\ \ [\mathrm{km\,s^{-1}}]$")
    a.set_ylim(0, 196)
    a.set_xlim(-0.6, len(LKEYS) - 0.4)
    a.legend(loc="upper right", fontsize=8.5)
    S.grid(a)
    S.panel_label(a, "(f)")

    fig.tight_layout()
    p = S.save(fig, "fig15_b_vs_logN")

    # ── printout ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print(f"fig15  b-logN phase diagram, saturation per FIT, boundary artefacts  [{variant}]")
    print("=" * 118)
    print("per kinematic class (raw | clean):")
    print(f"{'class':6s} {'n_raw':>8s} {'n_clean':>8s} {'logN raw':>9s} {'logN cln':>9s} "
          f"{'b raw':>7s} {'b cln':>7s}")
    for cname in CLS_SEQ:
        s = comp[comp.cls == cname]
        u = s[s.clean]
        print(f"{cname:6s} {len(s):8d} {len(u):8d} {s.logN.median():9.2f} "
              f"{u.logN.median():9.2f} {s.b_kms.median():7.1f} {u.b_kms.median():7.1f}")

    print("\nFRACTION OF FITS WHOSE REGION CONTAINS SATURATION (per fit, NOT per component;")
    print("`sat` is constant across all components of a (sightline, transition) fit).")
    print("RAW sample; the denominator is fits with >=1 detected component (n columns), not "
          "all 14400\nsightlines, and the class columns select fits having >=1 RAW component "
          "of that class:")
    print(f"{'line':13s} {'all fits':>9s} {'n_fits':>7s} " +
          " ".join(f"{c+' fits':>10s} {'n':>6s}" for c in CLS_SEQ))
    for i, k in enumerate(LKEYS):
        row = f"{k:13s} {satfit['all'][0][i]:9.3f} {satfit['all'][1][i]:7d} "
        row += " ".join(f"{satfit[c][0][i]:10.3f} {satfit[c][1][i]:6d}" for c in CLS_SEQ)
        print(row)
    mix = comp.groupby(V.SLKEY + ["line"])["sat"].nunique()
    print(f"(sightline,line) fits with a mixed sat flag: {int((mix > 1).sum())} / {len(mix)}"
          "  -- confirms sat is a per-fit property")

    print("\nFITTER-BOUNDARY ARTEFACTS per transition:")
    print(f"{'line':13s} {'b_max':>6s} {'f(b at ceil)':>13s} {'f(|dv|>500)':>12s} "
          f"{'f(removed)':>11s} {'med b raw':>10s} {'med b clean':>12s}")
    for i, k in enumerate(LKEYS):
        print(f"{k:13s} {bmx[i]:6.0f} {fc[i]:13.3f} {fw[i]:12.3f} {fcl[i]:11.3f} "
              f"{braw[i]:10.1f} {bcln[i]:12.1f}")
    return p


# =================================================================================
# fig16 -- multiplicity, velocity spread, HVC vs ISM structure
# =================================================================================
def _counts_table(comp, base, mask=None):
    """Per (sightline, line) number of detected components, zeros included."""
    c = comp if mask is None else comp[mask]
    n = c.groupby(V.SLKEY + ["line"]).size().rename("n").reset_index()
    t = base.merge(n, on=V.SLKEY + ["line"], how="left")
    t["n"] = t["n"].fillna(0).astype(int)
    return t


def _mult_panel(a, tab, nmax, xlab, raw_tab=None):
    edges = np.arange(-0.5, nmax + 1.5, 1.0)
    xc = np.arange(0, nmax + 1)
    if raw_tab is not None:
        n = np.clip(raw_tab[raw_tab.line == HI]["n"].to_numpy(), 0, nmax)
        h, _ = np.histogram(n, bins=edges)
        if h.sum():
            a.step(xc, h / h.sum(), where="mid", color=RAW_COL, ls=(0, (1, 1.4)), lw=1.5,
                   zorder=1, label=r"$\mathrm{H\,I}\ \lambda1216$ (raw)")
    for L in V.LINES:
        n = tab[tab.line == L.key]["n"].to_numpy()
        if n.size == 0:
            continue
        h, _ = np.histogram(np.clip(n, 0, nmax), bins=edges)
        a.step(xc, h / h.sum(), where="mid", color=L.color, ls=L.ls, lw=L.lw * 0.95,
               zorder=3, label=L.label)
    a.set_yscale("log")
    a.set_xlim(-0.6, nmax + 0.6)
    a.set_ylim(2e-5, 1.8)
    # the counts are clipped INTO the last bin, so label it as such (raw H I puts 3.3 per
    # cent of its HVC sightlines there); an unlabelled pile-up reads as a real feature
    a.set_xticks(xc)
    a.set_xticklabels([str(i) for i in range(nmax)] + [r"$\geq%d$" % nmax])
    a.set_xlabel(xlab)
    a.set_ylabel("fraction of sightlines")
    S.grid(a)


def _wstd_table(comp):
    """Column-weighted sigma of the component velocities, per (sightline, line)."""
    d = comp[V.SLKEY + ["line", "dv", "logN"]].copy()
    d["w"] = np.power(10.0, np.clip(d.logN.to_numpy(), 0.0, 25.0))
    d["wd"] = d.w * d.dv
    d["wd2"] = d.w * d.dv ** 2
    g = d.groupby(V.SLKEY + ["line"]).agg(sw=("w", "sum"), swd=("wd", "sum"),
                                          swd2=("wd2", "sum"), n=("dv", "size")).reset_index()
    mu = g.swd / g.sw
    g["wstd"] = np.sqrt(np.clip(g.swd2 / g.sw - mu ** 2, 0.0, None))
    return g


def fig16_components_per_sightline(comp, stat, variant):
    cc = comp[comp.clean]
    base = sightline_base(stat)
    all_t = _counts_table(cc, base)
    hvc_t = _counts_table(cc, base, cc.cls == "HVC")
    ism_t = _counts_table(cc, base, cc.cls == "ISM")
    all_raw = _counts_table(comp, base)
    hvc_raw = _counts_table(comp, base, comp.cls == "HVC")

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 11.2))
    ax = axes.ravel()

    _mult_panel(ax[0], all_t, 12, r"$N_{\rm comp}$ per sightline (all detected)",
                raw_tab=all_raw)
    ax[0].legend(loc="upper right", ncol=2, fontsize=7.2, handlelength=2.2)
    S.tag(ax[0], CLEAN_TAG, "ll", fs=8)
    S.panel_label(ax[0], "(a)")

    _mult_panel(ax[1], hvc_t, 8,
                r"$N_{\rm comp}$ per sightline (HVC only, $|\Delta v|\geq100$)",
                raw_tab=hvc_raw)
    ax[1].legend(loc="upper right", ncol=2, fontsize=7.2, handlelength=2.2)
    S.panel_label(ax[1], "(b)")

    # ── (c) velocity spread of the component ensemble ────────────────────────────
    a = ax[2]
    g = cc.groupby(V.SLKEY + ["line"])
    spread = (g["dv"].max() - g["dv"].min()).rename("range").reset_index()
    spread["n"] = g.size().to_numpy()
    multi = spread[spread.n >= 2]
    ws = _wstd_table(cc)
    ws = ws[ws.n >= 2]
    sbins = np.linspace(0, 800, 55)
    sc = 0.5 * (sbins[:-1] + sbins[1:])
    for L in V.LINES:
        x = multi[multi.line == L.key]["range"].to_numpy()
        if x.size < 20:
            continue
        a.step(sc, pdf(x, sbins), where="mid", color=L.color, ls=L.ls, lw=L.lw * 0.95,
               label=L.label)
    a.set_xlim(0, 800)
    a.set_xlabel(r"$\Delta v_{\max}-\Delta v_{\min}$ per sightline "
                 r"$[\mathrm{km\,s^{-1}}]$")
    a.set_ylabel("fraction of multi-component sightlines")
    a.legend(ncol=2, fontsize=7.2, loc="upper right", handlelength=2.2)
    S.grid(a)
    S.panel_label(a, "(c)")

    # ── (d) mean N_HVC vs N_ISM ─────────────────────────────────────────────────
    a = ax[3]
    j = (ism_t.rename(columns={"n": "n_ism"})
         .merge(hvc_t.rename(columns={"n": "n_hvc"}), on=V.SLKEY + ["line"]))
    NIMAX = 5
    j["n_ism_c"] = np.clip(j.n_ism, 0, NIMAX)
    rho = {}
    for L in V.LINES:
        s = j[j.line == L.key]
        if len(s) < 50:
            continue
        gg = s.groupby("n_ism_c")["n_hvc"]
        m, e, n = gg.mean(), gg.std(), gg.size()
        ok = (n >= 20).to_numpy()
        a.errorbar(np.asarray(m.index)[ok], m.to_numpy()[ok],
                   yerr=e.to_numpy()[ok] / np.sqrt(n.to_numpy()[ok]),
                   color=L.color, ls=L.ls, lw=L.lw * 0.95, marker="o", ms=4.5, capsize=2.5,
                   label=L.label)
        rho[L.key] = spearmanr(s.n_ism, s.n_hvc)
    a.set_xticks(np.arange(NIMAX + 1))
    a.set_xticklabels([str(i) for i in range(NIMAX)] + [r"$\geq%d$" % NIMAX])
    a.set_xlabel(r"$N_{\rm comp}$ per sightline with $|\Delta v|<40\ \mathrm{km\,s^{-1}}$"
                 r" (ISM)")
    a.set_ylabel(r"$\langle N_{\rm comp}\rangle$ with $|\Delta v|\geq100$ (HVC)")
    a.set_ylim(0.0, 1.62)
    a.legend(ncol=2, fontsize=7.2, loc="upper right", handlelength=2.2)
    S.tag(a, r"the fitter allows at most $3-6$ components per region," "\n"
             r"so part of this anti-correlation is a component budget", "ll", fs=8)
    S.grid(a)
    S.panel_label(a, "(d)")

    fig.tight_layout()
    p = S.save(fig, "fig16_components_per_sightline")

    # ── printout ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print(f"fig16  component multiplicity and velocity structure   [{variant}]  "
          f"CLEAN sample")
    print("=" * 118)
    print(f"{'line':13s} {'n_sl_fit':>8s} {'<N_all>':>8s} {'medN':>5s} {'f(N>=2)':>8s} "
          f"{'<N_HVC>':>8s} {'f(HVC>=1)':>10s} {'med range':>10s} {'med wsigma':>11s} "
          f"{'rho':>7s} {'p':>10s}")
    for L in V.LINES:
        k = L.key
        aa = all_t[all_t.line == k]["n"].to_numpy()
        hh = hvc_t[hvc_t.line == k]["n"].to_numpy()
        rg = multi[multi.line == k]["range"].to_numpy()
        wk = ws[ws.line == k]["wstd"].to_numpy()
        r = rho.get(k)
        rs = f"{r.statistic:7.3f} {r.pvalue:10.1e}" if r is not None else f"{'n/a':>7s} {'':>10s}"
        print(f"{k:13s} {len(aa):8d} {aa.mean():8.2f} {np.median(aa):5.1f} "
              f"{(aa >= 2).mean():8.3f} {hh.mean():8.2f} {(hh >= 1).mean():10.3f} "
              f"{(np.median(rg) if rg.size else np.nan):10.1f} "
              f"{(np.median(wk) if wk.size else np.nan):11.1f} {rs}")
    print("\nRAW sample (no clean cut) for comparison:")
    print(f"{'line':13s} {'<N_all>':>8s} {'<N_HVC>':>8s} {'f(HVC>=1)':>10s}")
    for L in V.LINES:
        aa = all_raw[all_raw.line == L.key]["n"].to_numpy()
        hh = hvc_raw[hvc_raw.line == L.key]["n"].to_numpy()
        print(f"{L.key:13s} {aa.mean():8.2f} {hh.mean():8.2f} {(hh >= 1).mean():10.3f}")
    print("\nmaximum number of Voigt components the fitter may place in one region "
          "(fit_max_lines):")
    print("  " + ", ".join(f"{L.key}={comp[comp.line == L.key].fit_max_lines.median():.0f}"
                           for L in V.LINES))
    print("  -> the N_HVC vs N_ISM anti-correlation in panel (d) is partly a component "
          "budget effect,\n     not purely physical: ISM components consume the same "
          "per-region allowance as HVC ones.")
    print("\nNOTE: the velocity spread and the column-weighted sigma are DIFFERENCES of dv "
          "within a\n      sightline, so they are identical for v3a and v3b (v_ISM cancels).")
    return p


# =================================================================================
# fig17 -- Si II 1190 / 1193 / 1260 cross-transition systematics
# =================================================================================
def match_transitions(comp, key_a, key_b):
    """One-to-one greedy velocity match of detected components between two transitions.

    Matching runs inside each sightline, cheapest (smallest velocity separation) pair
    first; a component may be used at most once.  Returns (matched, n_a, n_b).
    """
    cols = ["cidx", "dv", "adv", "logN", "b_kms", "EW_mA", "v_err_kms", "sat", "cls",
            "b_at_ceiling"]
    a = comp[comp.line == key_a][V.SLKEY + cols]
    b = comp[comp.line == key_b][V.SLKEY + cols]
    m = a.merge(b, on=V.SLKEY, suffixes=("_a", "_b"))
    if m.empty:
        return m, len(a), len(b)
    m = m.assign(sep=(m.dv_a - m.dv_b).abs())
    sig = np.hypot(np.clip(m.v_err_kms_a, 0, 1e4), np.clip(m.v_err_kms_b, 0, 1e4))
    m = m.assign(tol=np.clip(NSIG * sig, TOL_MIN, TOL_MAX))
    m = m[m.sep <= m.tol].sort_values("sep", kind="mergesort")
    used_a, used_b, keep = set(), set(), []
    for ia, ib in zip(m.cidx_a.to_numpy(), m.cidx_b.to_numpy()):
        if ia in used_a or ib in used_b:
            keep.append(False)
        else:
            used_a.add(ia)
            used_b.add(ib)
            keep.append(True)
    return m[np.array(keep, bool)].reset_index(drop=True), len(a), len(b)


def _running_median(x, y, edges, nmin=25):
    xc, med = [], []
    idx = np.digitize(x, edges) - 1
    for i in range(len(edges) - 1):
        s = idx == i
        if s.sum() >= nmin:
            xc.append(0.5 * (edges[i] + edges[i + 1]))
            med.append(np.median(y[s]))
    return np.array(xc), np.array(med)


def _completeness(ref, matched_sets, edges, xcol, nmin=25):
    """Fraction of `ref` components present in each matched set, per bin of xcol."""
    x = ref[xcol].to_numpy()
    idx = np.digitize(x, edges) - 1
    out = {}
    for name, ids in matched_sets.items():
        hit = ref.cidx.isin(ids).to_numpy()
        k, n, xc = [], [], []
        for i in range(len(edges) - 1):
            s = idx == i
            if s.sum() >= nmin:
                xc.append(0.5 * (edges[i] + edges[i + 1]))
                k.append(hit[s].sum())
                n.append(s.sum())
        out[name] = (np.array(xc, float), np.array(k, float), np.array(n, float))
    return out


def fig17_siII_triplet_consistency(comp, stat, variant):
    K90, K93, K60 = "Si_II_1190", "Si_II_1193", "Si_II_1260"
    TRIP = [K90, K93, K60]

    # only sightlines where ALL THREE Si II transitions were actually fit, so a missing
    # fit can never masquerade as a non-detection of a weak transition
    s3 = stat[stat.line.isin(TRIP)]
    nl = s3.groupby(V.SLKEY)["line"].nunique()
    good = set(map(tuple, nl[nl == 3].reset_index()[V.SLKEY].to_numpy()))
    keys = list(map(tuple, comp[V.SLKEY].to_numpy()))
    sel = np.array([t in good for t in keys], bool)
    comp3 = comp[sel & comp.line.isin(TRIP) & comp.clean]
    print(f"\n[fig17] {len(good)} of {int(nl.size)} sightlines have all three Si II "
          f"transitions fit; clean Si II components used: {len(comp3)}")
    win = {k: float(comp[comp.line == k].fit_velocity_window_kms.median()) for k in TRIP}
    bmx = {k: float(comp[comp.line == k].fit_b_max_kms.median()) for k in TRIP}
    print("[fig17] fitter settings per transition (these are NOT identical -- see below):")
    for k in TRIP:
        r = comp[comp.line == k]
        print(f"    {k:12s} window=+-{win[k]:.0f} km/s  b_max={bmx[k]:.0f} km/s  "
              f"f(b at ceiling)={r.b_at_ceiling.mean():.3f}  "
              f"p99 |v_rest|={r.v_rest.abs().quantile(0.99):.0f} km/s")

    # ── velocity zero-point regression check (see the module docstring) ─────────────
    # The same absorber seen in two Si II transitions must sit at the same dv. pyGad fit
    # lambda1260 at 1260.5220 A against a spectrum synthesised at 1260.4220 A, which used to
    # put a constant +23.8 km/s between them -- comparable to the matching tolerance, so it
    # rejected genuine pairs. build_voigt_catalog.py now removes it; verify that here,
    # because a stale catalog would silently depress every completeness below.
    zp = {}
    for kw in (K93, K90):
        pr = (comp3[comp3.line == K60][V.SLKEY + ["dv"]]
              .merge(comp3[comp3.line == kw][V.SLKEY + ["dv"]], on=V.SLKEY,
                     suffixes=("_a", "_b")))
        s = (pr.dv_a - pr.dv_b).to_numpy()
        s = s[np.abs(s) < 100.0]
        zp[kw] = float(np.median(s)) if s.size else np.nan
    print("[fig17] VELOCITY ZERO-POINT CHECK: median dv(1260) - dv(weak) over same-sightline "
          "pairs\n        within 100 km/s (must be ~0; it was "
          f"{DV_1260_ZP:+.1f} before the catalog fix):")
    for kw in (K93, K90):
        print(f"    1260 - {kw:12s} = {zp[kw]:+6.2f} km/s")
    if max(abs(zp[K93]), abs(zp[K90])) > ZP_TOL:
        raise SystemExit(
            f"Si II 1260 velocity zero-point is NOT corrected (residual {zp} km/s vs "
            f"expected {DV_1260_ZP:+.2f}). Rebuild voigt_components.parquet with the "
            "current build_voigt_catalog.py -- the completeness below would be biased low.")
    print(f"    -> within {ZP_TOL:.0f} km/s of zero, so the "
          f"{TOL_MIN:.0f}-{TOL_MAX:.0f} km/s matching tolerance is not being spent on a "
          "constant offset.")

    m93, _, n93b = match_transitions(comp3, K60, K93)
    m90, _, n90b = match_transitions(comp3, K60, K90)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 11.4))
    ax = axes.ravel()

    XE = np.arange(11.75, 16.51, 0.25)
    for i, (m, kw, lab) in enumerate([(m93, K93, "(a)"), (m90, K90, "(b)")]):
        a = ax[i]
        hb = a.hexbin(m.logN_a, m.logN_b, gridsize=42, cmap=S.DENSCMAP, bins="log",
                      mincnt=1, extent=(11.7, 16.6, 11.7, 16.6))
        cb = fig.colorbar(hb, ax=a, fraction=0.046, pad=0.02)
        cb.set_label(r"$N_{\rm matched\ pairs}$")
        a.plot([11.7, 16.6], [11.7, 16.6], ls="-", lw=2.6, color="0.1", zorder=3)
        a.plot([11.7, 16.6], [11.7, 16.6], ls="--", lw=1.3, color="w", zorder=4)
        for satflag, ls, cc_, nm in (
                (True, "-", S.CLASS["HVC"], r"saturation in the $\lambda1260$ fit region"),
                (False, "--", S.ACCENT, r"no saturation flagged")):
            s = m[m.sat_a == satflag]
            if len(s) < 50:
                continue
            xc, md = _running_median(s.logN_a.to_numpy(), s.logN_b.to_numpy(), XE)
            a.plot(xc, md, ls=ls, lw=2.4, color=cc_, marker="o", ms=4.5, label=nm)
        a.set_xlim(11.7, 16.6)
        a.set_ylim(11.7, 16.6)
        a.set_xlabel(r"$\log_{10} N$ from $\mathrm{Si\,II}\ \lambda1260$"
                     r"  ($f\lambda=1487$)")
        a.set_ylabel(r"$\log_{10} N$ from $\mathrm{Si\,II}\ \lambda%s$  ($f\lambda=%d$)"
                     % (kw.split("_")[-1], round(FLAM[kw])))
        a.legend(loc="lower right", fontsize=8.0)
        med = float(np.median(m.logN_b - m.logN_a)) if len(m) else np.nan
        S.tag(a, r"$n_{\rm match}=%d$" % len(m) + "\n" +
              r"median $\Delta\log N=%+.2f$" % med, "ur", fs=8.5)
        S.panel_label(a, lab)

    # ── completeness of the weak transitions, vs logN(1260) and vs |dv| ─────────
    ref = comp3[comp3.line == K60]
    sets = {K93: set(m93.cidx_a), K90: set(m90.cidx_a)}
    styles = {K93: (V.LINE_BY_KEY[K93].color, "--", "o",
                    r"also fit in $\mathrm{Si\,II}\ \lambda1193$"),
              K90: (S.VMODE["R95-edge"], ":", "s",
                    r"also fit in $\mathrm{Si\,II}\ \lambda1190$")}
    comp_res = {}

    for i, (edges, xcol, xlab, lab) in enumerate([
            (XE, "logN", r"$\log_{10} N$ of the $\mathrm{Si\,II}\ \lambda1260$ component",
             "(c)"),
            (np.arange(0.0, 511.0, 30.0), "adv",
             r"$|\Delta v|$ of the $\mathrm{Si\,II}\ \lambda1260$ component "
             r"$[\mathrm{km\,s^{-1}}]$", "(d)")]):
        a = ax[2 + i]
        res = _completeness(ref, sets, edges, xcol)
        for kk, (xc, k, n) in res.items():
            c_, ls, mk, lb = styles[kk]
            lo, hi = V.wilson(k, n)
            f = k / n
            a.errorbar(xc, f, yerr=[f - lo, hi - f], color=c_, ls=ls, lw=2.0, marker=mk,
                       ms=5.5, capsize=2.5, label=lb)
            comp_res[(xcol, kk)] = (xc, k, n)
        if xcol == "adv":
            # column-controlled control sample: HVC components have systematically lower
            # logN, so part of the |dv| trend is a column effect, not a velocity effect
            refb = ref[(ref.logN >= NBAND[0]) & (ref.logN < NBAND[1])]
            for kk, (xc, k, n) in _completeness(refb, sets, edges, xcol, nmin=15).items():
                c_, ls, mk, lb = styles[kk]
                a.plot(xc, k / n, color=c_, ls=ls, lw=1.1, marker=mk, ms=4.0, mfc="white",
                       alpha=0.95, label=lb + r", $%.1f\leq\log N<%.1f$" % NBAND)
                comp_res[("adv_band", kk)] = (xc, k, n)
            for v, cn in ((V.IVC, "IVC"), (V.HVC, "HVC")):
                a.axvline(v, color=S.CLASS[cn], ls=":", lw=1.5)
            a.set_xlim(0, edges[-1])
            S.tag(a, (r"$\lambda1190,\lambda1193$ fit over $\pm%d\ \mathrm{km\,s^{-1}}$,"
                      "\n" r"$\lambda1260$ over $\pm%d$ -- part of the high-$|\Delta v|$"
                      "\n" "incompleteness is a fitting-window effect."
                      % (win[K90], win[K60])) +
                  "\n" r"Velocities are on the corrected zero-point"
                  "\n" r"(residual $\lambda1260-\lambda1193 = %+.1f\ \mathrm{km\,s^{-1}}$)"
                  % zp[K93], "ur", fs=8)
        else:
            a.set_xlim(edges[0], edges[-1])
        a.set_ylim(0, 1.28)
        a.set_xlabel(xlab)
        a.set_ylabel(r"fraction of $\lambda1260$ components recovered")
        a.legend(loc="lower right" if xcol == "logN" else "lower left",
                 fontsize=8.5 if xcol == "logN" else 7.4)
        S.grid(a)
        S.panel_label(a, lab)

    fig.tight_layout()
    p = S.save(fig, "fig17_siII_triplet_consistency")

    # ── printout ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print(f"fig17  Si II triplet cross-transition consistency   [{variant}]")
    print("=" * 118)
    print(f"matching tolerance: clip(3 x sigma_v, {TOL_MIN:.0f}, {TOL_MAX:.0f}) km/s, "
          f"sigma_v = quadrature sum of pyGad v_err_kms (clipped at 1e4 km/s).")
    print(f"  {TOL_MIN:.0f} km/s ~ one COS G130M resolution element; the {TOL_MAX:.0f} km/s "
          f"cap = the ISM/IVC boundary,\n  needed because v_err_kms has a pathological tail "
          f"(median {comp3.v_err_kms.median():.1f}, "
          f"p75 {comp3.v_err_kms.quantile(0.75):.0f}, max {comp3.v_err_kms.max():.3g} km/s).")
    print(f"f*lambda ordering: 1260 : 1193 : 1190 = {FLAM[K60]:.0f} : {FLAM[K93]:.0f} : "
          f"{FLAM[K90]:.0f} A  (annotation only -- no threshold is derived from it).")
    n60 = int((comp3.line == K60).sum())
    for m, kw, nb in ((m93, K93, n93b), (m90, K90, n90b)):
        print(f"\n  {K60} vs {kw}:  {len(m)} matched pairs")
        print(f"    {K60:12s}: {n60} components, {n60 - len(m)} unmatched "
              f"({100*(n60-len(m))/max(n60,1):.1f} per cent)")
        print(f"    {kw:12s}: {nb} components, {nb - len(m)} unmatched "
              f"({100*(nb-len(m))/max(nb,1):.1f} per cent)")
        if not len(m):
            continue
        d = (m.logN_b - m.logN_a).to_numpy()
        print(f"    median dlogN (weak - 1260) = {np.median(d):+.3f}  "
              f"[16-84 pct: {np.percentile(d,16):+.2f}, {np.percentile(d,84):+.2f}]")
        for sf in (True, False):
            s = m[m.sat_a == sf]
            if len(s):
                dd = (s.logN_b - s.logN_a).to_numpy()
                tag = "saturation in the 1260 fit region" if sf else "no saturation flagged"
                print(f"      {tag:34s}: n={len(s):6d}  median dlogN = {np.median(dd):+.3f}")
        for cn in CLS_SEQ:
            s = m[m.cls_a == cn]
            if len(s):
                dd = (s.logN_b - s.logN_a).to_numpy()
                print(f"      class {cn:4s}{'':25s}: n={len(s):6d}  "
                      f"median dlogN = {np.median(dd):+.3f}")
        for lo, hi in ((11.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.0), (15.0, 17.0)):
            s = m[(m.logN_a >= lo) & (m.logN_a < hi)]
            if len(s) >= 20:
                dd = (s.logN_b - s.logN_a).to_numpy()
                print(f"      logN(1260) in [{lo:.1f},{hi:.1f}){'':13s}: n={len(s):6d}  "
                      f"median dlogN = {np.median(dd):+.3f}")

    print("\n  overall completeness of the weak transitions "
          "(fraction of 1260 components matched):")
    for kk, ids in sets.items():
        hit = ref.cidx.isin(ids)
        print(f"    {kk:12s}: {hit.mean():.3f}  ({int(hit.sum())}/{len(ref)})")
        for cn in CLS_SEQ:
            r = ref[ref.cls == cn]
            if len(r):
                print(f"        {cn}: {r.cidx.isin(ids).mean():.3f}  (n={len(r)})")
    print("\n  reverse check -- fraction of weak-transition components matched in 1260:")
    for m, kw in ((m93, K93), (m90, K90)):
        r = comp3[comp3.line == kw]
        print(f"    {kw:12s}: {r.cidx.isin(set(m.cidx_b)).mean():.3f}  (n={len(r)})")
    print("\n  completeness vs logN(1260)   [bin centre : fraction]")
    for kk in sets:
        xc, k, n = comp_res[("logN", kk)]
        print(f"    {kk:12s}: " + "  ".join(f"{a_:.2f}:{b_/c_:.2f}"
                                            for a_, b_, c_ in zip(xc, k, n)))
    print("  completeness vs |dv|(1260)   [bin centre : fraction]")
    for kk in sets:
        xc, k, n = comp_res[("adv", kk)]
        print(f"    {kk:12s}: " + "  ".join(f"{a_:.0f}:{b_/c_:.2f}"
                                            for a_, b_, c_ in zip(xc, k, n)))
    print(f"  completeness vs |dv| at CONTROLLED column {NBAND[0]:.1f}<=logN(1260)<"
          f"{NBAND[1]:.1f}  -- separates the velocity trend from the column trend:")
    for kk in sets:
        if ("adv_band", kk) in comp_res:
            xc, k, n = comp_res[("adv_band", kk)]
            print(f"    {kk:12s}: " + "  ".join(f"{a_:.0f}:{b_/c_:.2f}"
                                                for a_, b_, c_ in zip(xc, k, n)))
    return p


# =================================================================================
def run(variant):
    S.set_style()
    S.FIGDIR = OUTBASE / f"paper_figures_{variant}"
    S.FIGDIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 118)
    print(f"# variant {variant}   ->  {S.FIGDIR}")
    print("#" * 118)
    comp, stat = prepare(variant)
    print(f"{len(comp)} detected components over "
          f"{comp[V.SLKEY].drop_duplicates().shape[0]} sightlines; "
          f"{stat[V.SLKEY].drop_duplicates().shape[0]} sightlines with fits")
    print(f"clean sample (~b_at_ceiling & ~beyond_common_window): "
          f"{int(comp.clean.sum())} ({100*comp.clean.mean():.1f} per cent)")
    print("class counts (raw): " +
          ", ".join(f"{c}={int((comp.cls == c).sum())}" for c in CLS_SEQ))
    print("class counts (clean): " +
          ", ".join(f"{c}={int(((comp.cls == c) & comp.clean).sum())}" for c in CLS_SEQ))
    for fn in (fig14_component_properties, fig15_b_vs_logN,
               fig16_components_per_sightline, fig17_siII_triplet_consistency):
        p = fn(comp, stat, variant)
        print(f"\n  saved {p}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        variants = ["v3b"]
    elif args[0] == "all":
        variants = ["v3a", "v3b"]
    else:
        variants = args
    for v in variants:
        if v not in ("v3a", "v3b"):
            raise SystemExit(f"unknown variant {v!r}; expected v3a or v3b")
        run(v)


if __name__ == "__main__":
    main()
